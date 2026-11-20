"""E4 ingestion and freshness validation (PRD.md:278; #44).

"Modify one document and run it through the complete ingestion pipeline": fetch -> versioned raw store ->
parse/chunk/embed -> index -> snapshot activation, then check the change is queryable.

Everything runs against a fake copy of the CDS site (real page content from tests/pages/, served through
httpx.MockTransport) in a temporary data folder, so no institute server is contacted and the real index
under data/ is never touched. The site mirrors sources.yaml: the schedule and M.Tech pages are listed pages
AND crawl seeds; the course descriptions page is found by the crawl and sends no ETag (content-hash path).

Steps
  1. initial fetch + ingest
  2. refresh with nothing changed       (expect 0 new/changed, index unchanged, no new registry rows)
  3. change the M.Tech page ("minimum of 64 credits" -> "72"), new ETag, refresh with ingest
  4. ask "What is the minimum number of credits an M.Tech (CDS) student must earn?" and time until a passage
     from the M.Tech URL says 72 and none says 64

    python tools/eval_e4.py --out eval_e4            # configured embedder (bge-m3)
    python tools/eval_e4.py --out eval_e4 --hash     # plumbing check only; timings are not representative

Writes <out>/e4_summary.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HOST = "https://cds.iisc.ac.in/"
INDEX = HOST + "courses/"
SCHEDULE = HOST + "courses/schedule/"
DESCRIPTIONS = HOST + "courses/descriptions/"
MTECH = HOST + "courses/m-tech-course-requirements/"
QUESTION = "What is the minimum number of credits an M.Tech (CDS) student must earn?"
OLD, NEW = "minimum of 64 credits", "minimum of 72 credits"

INDEX_HTML = f"""<html><head><title>Courses | Department of Computational and Data Sciences</title></head><body>
<nav><a href="/">Home</a></nav><div class="entry-content"><h2>Courses offered by CDS</h2>
<p>The Department of Computational and Data Sciences offers courses for its M.Tech., M.Tech. (Research) and
Ph.D. programmes. Course schedules are published each term, and course descriptions list the syllabus and
prerequisites of every course.</p>
<ul><li><a href="{SCHEDULE}">Course schedule</a></li><li><a href="{DESCRIPTIONS}">Course descriptions</a></li>
<li><a href="{MTECH}">M.Tech. course requirements</a></li></ul></div></body></html>""".encode()


class Site:
    """Fake CDS site. Pages with an ETag answer a matching If-None-Match with 304."""

    def __init__(self):
        pages = ROOT / "tests" / "pages"
        self.pages = {
            INDEX: [INDEX_HTML, '"idx-1"'],
            SCHEDULE: [(pages / "schedule.html").read_bytes(), '"sch-1"'],
            DESCRIPTIONS: [(pages / "descriptions.html").read_bytes(), None],
            MTECH: [(pages / "mtech.html").read_bytes(), '"mt-1"'],
        }
        self.log = []                              # (url, conditional?, status) per page request

    def handler(self, request):
        import httpx
        url = str(request.url)
        if url.endswith("/robots.txt"):
            return httpx.Response(404)
        conditional = "if-none-match" in request.headers or "if-modified-since" in request.headers
        if url not in self.pages:
            self.log.append((url, conditional, 404))
            return httpx.Response(404)
        body, etag = self.pages[url]
        if etag and request.headers.get("if-none-match") == etag:
            self.log.append((url, conditional, 304))
            return httpx.Response(304, headers={"etag": etag})
        self.log.append((url, conditional, 200))
        return httpx.Response(200, content=body, headers={"content-type": "text/html; charset=utf-8",
                                                          **({"etag": etag} if etag else {})})

    def take(self) -> dict:
        log, self.log = self.log, []
        per_url = Counter(u for u, _, _ in log)
        return {"requests": len(log), "per_url": dict(per_url),
                "duplicate_requests": sum(n - 1 for n in per_url.values()),
                "conditional": sum(c for _, c, _ in log), "not_modified_304": sum(s == 304 for _, _, s in log),
                "full_200": sum(s == 200 for _, _, s in log)}


def setup_env(work: Path, use_hash: bool):
    """Point every path the pipeline uses at `work` before askiisc_retrieval is imported (config reads
    these at import time)."""
    (work / "data").mkdir(parents=True)
    (work / "sources.yaml").write_text(f"""department: CDS
pages:
  - {{url: '{SCHEDULE}', department: CDS, doc_type: schedule, scope: public}}
  - {{url: '{MTECH}', department: CDS, doc_type: policy, scope: public}}
crawls:
  - {{seeds: ['{INDEX}', '{SCHEDULE}', '{MTECH}'], department: CDS, max_depth: 1, max_pages: 20, delay_s: 0}}
""", encoding="utf-8")
    (work / "no_private_documents.json").write_text('{"documents": []}', encoding="utf-8")
    os.environ.update({"ASKIISC_DATA_DIR": str(work / "data"), "ASKIISC_SOURCES_FILE": str(work / "sources.yaml"),
                       "ASKIISC_SOURCE_DEPARTMENTS": "CDS",
                       "ASKIISC_PRIVATE_MANIFEST": str(work / "no_private_documents.json")})
    os.environ.pop("QDRANT_URL", None)                 # local Qdrant folder under the temp data dir only
    if use_hash:
        os.environ["ASKIISC_EMBEDDER"] = "hash"


def registry(raw: Path) -> list[dict]:
    with sqlite3.connect(raw / "versions" / "registry.sqlite3") as db:
        rows = db.execute("SELECT url, version, content_hash, object, superseded_at FROM versions "
                          "ORDER BY url, version").fetchall()
    return [{"url": u, "version": v, "content_hash": h[:12], "object": o, "superseded": bool(s)}
            for u, v, h, o, s in rows]


def run(work: Path, use_hash: bool) -> dict:
    setup_env(work, use_hash)
    import httpx

    from askiisc_retrieval import cli, config, crawl as crawl_mod, fetch as fetch_mod, refresh
    from askiisc_retrieval.ingest_stub import load_page_inputs
    from askiisc_retrieval.retriever import Retriever
    from askiisc_retrieval.snapshot import JsonSnapshotProvider
    from askiisc_retrieval.stores import JsonGraphStore

    assert config.DATA == (work / "data").resolve(), "ASKIISC_DATA_DIR was not applied"
    site = Site()
    real_client = httpx.Client
    fetch_mod.httpx.Client = lambda **kw: real_client(transport=httpx.MockTransport(site.handler), **kw)
    fetch_mod.time.sleep = crawl_mod.time.sleep = lambda s: None
    emb = cli.make_embedder()
    timings = {}

    def fetch():
        t = time.perf_counter()
        out = fetch_mod.fetch_all(refresh=True, delay_s=0)
        timings["fetch_s"] = round(time.perf_counter() - t, 3)
        return out

    def rebuild():
        t = time.perf_counter()
        rep = cli.safe_rebuild(load_page_inputs(include_fixtures=False), emb)
        timings["rebuild_s"] = round(time.perf_counter() - t, 3)
        return {"snapshot": rep["snapshot"], "chunks": rep["chunks"], "embedded": rep["embedded"],
                "failures": rep["failures"], "not_indexed": [p for p in rep["pages"] if "version_id" not in p]}

    def snapshot_id():
        return json.loads(config.SNAPSHOT_PATH.read_text(encoding="utf-8")).get("snapshot_id")

    result = {"config": {"embedder": emb.model_id, "plumbing_check_only": use_hash, "question": QUESTION,
                         "pages": {"listed_and_seed": [SCHEDULE, MTECH], "crawled_only": [DESCRIPTIONS, INDEX]}}}

    # 1. initial fetch + ingest
    fetch_mod.fetch_all(refresh=False, delay_s=0)
    initial = {"site": site.take()}
    initial["index"] = rebuild()
    initial["snapshot"] = snapshot_id()
    initial["registry_rows"] = len(registry(config.RAW_DIR))
    result["1_initial"] = initial

    # 2. refresh, nothing changed
    rows_before = registry(config.RAW_DIR)
    status = refresh.run_refresh(ingest=True, fetch=fetch, rebuild=rebuild)
    result["2_unchanged_refresh"] = {"site": site.take(), "ok": status["ok"], "stats": status.get("stats"),
                                     "index": status.get("index"), "snapshot": snapshot_id(),
                                     "new_registry_rows": len(registry(config.RAW_DIR)) - len(rows_before)}

    # 3. modify one document and refresh with ingest
    old_mtech = site.pages[MTECH][0]
    assert OLD.encode() in old_mtech
    site.pages[MTECH] = [old_mtech.replace(OLD.encode(), NEW.encode()), '"mt-2"']
    snap_before = snapshot_id()
    timings.clear()
    t_change = time.perf_counter()
    status = refresh.run_refresh(ingest=True, fetch=fetch, rebuild=rebuild)
    t_indexed = time.perf_counter()

    # 4. query until the change is visible (one query is enough when the snapshot switched)
    retriever = Retriever(emb, cli.make_store(), JsonSnapshotProvider(config.SNAPSHOT_PATH),
                          JsonGraphStore.load(config.GRAPH_PATH), today_term="2026-AUG")
    try:
        res = retriever.retrieve(QUESTION, ["public"], use_graph=True, use_rerank=False)
    finally:
        retriever.store.close()
    t_queryable = time.perf_counter()
    mtech_text = [f"{p.chunk.section_heading} {p.chunk.text}" for p in res.passages if p.chunk.source_url == MTECH]
    new_visible = any("72 credits" in t for t in mtech_text)
    old_visible = any("64 credits" in t for t in mtech_text)

    rows = registry(config.RAW_DIR)
    mtech_rows = [r for r in rows if r["url"] == MTECH]
    v1_kept = bool(mtech_rows) and (config.RAW_DIR / "versions" / mtech_rows[0]["object"]).read_bytes() == old_mtech
    failures_file = config.DATA / "extraction_failures.json"
    result["3_modified_refresh"] = {
        "site": site.take(), "ok": status["ok"], "stats": status.get("stats"), "index": status.get("index"),
        "snapshot_before": snap_before, "snapshot_after": snapshot_id(),
        "registry": {"mtech_versions": mtech_rows,
                     "other_urls_versions": {r["url"]: r["version"] for r in rows if r["url"] != MTECH},
                     "mtech_v1_object_kept_byte_for_byte": v1_kept},
        "extraction_failures": json.loads(failures_file.read_text()) if failures_file.exists() else None}
    result["4_queryable"] = {
        "mtech_passages_returned": len(mtech_text), "new_value_72_visible": new_visible,
        "old_value_64_visible": old_visible, "passed": new_visible and not old_visible,
        "write_to_queryable_s": round(t_queryable - t_change, 3),
        "breakdown_s": {"fetch": timings.get("fetch_s"), "rebuild": timings.get("rebuild_s"),
                        "open_index_and_query": round(t_queryable - t_indexed, 3)},
        "worst_case_with_24h_schedule": "24 h + write_to_queryable_s (maintainer trigger: write_to_queryable_s "
                                        "plus up to 30 s for the scheduler to notice the trigger)"}
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("eval_e4"))
    ap.add_argument("--hash", action="store_true", help="test-only hash embedder: plumbing check, not results")
    a = ap.parse_args(argv)
    out = a.out.resolve()
    work = Path(tempfile.mkdtemp(prefix="askiisc-e4-"))
    try:
        result = run(work, a.hash)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e4_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    i, u, m, q = result["1_initial"], result["2_unchanged_refresh"], result["3_modified_refresh"], result["4_queryable"]
    print(f"\nE4 ingestion and freshness, embedder {result['config']['embedder']}"
          + ("  (HASH: plumbing check only, do not report)" if a.hash else ""))
    print(f"{'step':24} {'requests':>8} {'dup':>4} {'cond':>5} {'304':>4} {'200':>4}  result")
    for name, step, extra in (("1 initial fetch+ingest", i, f"snapshot {i['snapshot']}, {i['index']['chunks']} chunks, "
                                                             f"{i['registry_rows']} registry rows"),
                              ("2 refresh, unchanged", u, f"stats {u['stats']}, index: {u['index']}, "
                                                          f"new rows {u['new_registry_rows']}"),
                              ("3 refresh, M.Tech changed", m, f"stats {m['stats']}, snapshot {m['snapshot_before']} -> "
                                                               f"{m['snapshot_after']}, index {m['index']}")):
        s = step["site"]
        print(f"{name:24} {s['requests']:>8} {s['duplicate_requests']:>4} {s['conditional']:>5} "
              f"{s['not_modified_304']:>4} {s['full_200']:>4}  {extra}")
    print(f"M.Tech registry: {[(r['version'], r['superseded']) for r in m['registry']['mtech_versions']]}, "
          f"v1 kept byte-for-byte: {m['registry']['mtech_v1_object_kept_byte_for_byte']}; "
          f"extraction failures: {m['extraction_failures']}")
    print(f"Query after change: 72 visible {q['new_value_72_visible']}, 64 visible {q['old_value_64_visible']} -> "
          f"{'PASS' if q['passed'] else 'FAIL'}; write-to-queryable {q['write_to_queryable_s']} s {q['breakdown_s']}")
    print(f"Wrote {out / 'e4_summary.json'}")
    ok = q["passed"] and u["new_registry_rows"] == 0 and all(
        s["site"]["duplicate_requests"] == 0 for s in (i, u, m))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
