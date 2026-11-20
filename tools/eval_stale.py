"""Stale-version evaluation (PRD.md:258 "10% stale-version queries"; FR-9; #44).

Builds a private index in a temporary folder: the current CDS pages from tests/pages/ plus an older,
superseded version of the same three URLs (fixtures/superseded_*.html). The superseded versions are embedded
but are not in the active snapshot. Then it asks the 12 questions of test_stale12.yaml, in vector mode and in
full (graph) mode, and reports for each:

  superseded_returned   passages from an inactive version (must be 0: FR-9)
  current_rank          rank of the first passage from the expected URL that contains the current answer
  control               the same search WITHOUT the active-version filter: does the superseded answer exist in
                        the index and rank in the top k? (shows the test is not vacuous)

    python tools/eval_stale.py --out eval_stale            # configured embedder (bge-m3) and reranker
    python tools/eval_stale.py --out eval_stale --hash     # plumbing check only; numbers are meaningless

The real CDS index under data/ is never touched. Writes <out>/stale_summary.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

QUESTIONS = ROOT / "test_stale12.yaml"
CURRENT = [  # (url, file under tests/pages, doc_type)
    ("https://cds.iisc.ac.in/courses/schedule/", "schedule.html", "schedule"),
    ("https://cds.iisc.ac.in/courses/descriptions/", "descriptions.html", "catalogue"),
    ("https://cds.iisc.ac.in/courses/m-tech-course-requirements/", "mtech.html", "policy"),
]
SUPERSEDED = [  # same URLs, older conflicting versions (active: false)
    ("https://cds.iisc.ac.in/courses/schedule/", "superseded_course_schedule.html", "schedule"),
    ("https://cds.iisc.ac.in/courses/descriptions/", "superseded_course_descriptions.html", "catalogue"),
    ("https://cds.iisc.ac.in/courses/m-tech-course-requirements/", "superseded_mtech_requirements.html", "policy"),
]


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


def make_models(use_hash: bool):
    """(embedder, reranker). --hash: the test-only HashEmbedder and keyword reranker (plumbing check)."""
    if use_hash:
        os.environ["ASKIISC_EMBEDDER"] = "hash"
    from askiisc_retrieval import cli
    return cli.make_embedder(), cli.make_reranker()


def build(workdir: Path, embedder, reranker=None, *, today_term: str = "2026-AUG", graph_fail: bool = False):
    """Index the current pages and their superseded versions under `workdir`; return the parts.

    Used by tools/eval_e6.py as well."""
    from askiisc_retrieval.ingest_stub import PageInput, build_index
    from askiisc_retrieval.retriever import Retriever
    from askiisc_retrieval.snapshot import JsonSnapshotProvider
    from askiisc_retrieval.stores import JsonGraphStore, QdrantStore

    workdir.mkdir(parents=True, exist_ok=True)
    pages = [PageInput(url, (ROOT / "tests" / "pages" / f).read_text(encoding="utf-8"), t, "public")
             for url, f, t in CURRENT]
    pages += [PageInput(url, (ROOT / "fixtures" / f).read_text(encoding="utf-8"), t, "public",
                        active=False, version_tag="stale-eval-" + f) for url, f, t in SUPERSEDED]
    store = QdrantStore("stale_eval", len(embedder.embed_query("dimension probe")), path=workdir / "qdrant")
    report = build_index(pages, embedder, store, workdir / "graph.json", workdir / "snapshot.json",
                         workdir / "chunks.jsonl")
    missing = [p for p in report["pages"] if "version_id" not in p]
    if missing:                                    # a page below the thin-page rule would make the test vacuous
        raise RuntimeError(f"pages not indexed: {missing}")
    graph = JsonGraphStore.load(workdir / "graph.json")
    graph.fail = graph_fail
    snapshots = JsonSnapshotProvider(workdir / "snapshot.json")
    retriever = Retriever(embedder, store, snapshots, graph, today_term=today_term, reranker=reranker)
    inactive = {p["version_id"] for p in report["pages"] if not p["active"]}
    return {"retriever": retriever, "store": store, "report": report, "snapshot": snapshots.active(),
            "inactive_versions": inactive, "graph": graph}


def _text(chunk) -> str:
    return f"{chunk.section_heading} {chunk.text}"


def evaluate(parts: dict, questions: list[dict], k: int) -> dict:
    r, snap, inactive = parts["retriever"], parts["snapshot"], parts["inactive_versions"]
    all_versions = sorted(snap.active_version_ids | inactive)
    rows = []
    for q in questions:
        row = {"id": q["id"], "question": q["question"]}
        for mode, use_graph in (("vector", False), ("full", True)):
            t0 = time.perf_counter()
            res = r.retrieve(q["question"], ["public"], use_graph=use_graph)
            ms = round((time.perf_counter() - t0) * 1000, 1)
            passages = res.passages[:k]
            stale = [p for p in passages if p.chunk.version_id not in snap.active_version_ids
                     or p.chunk.version_id in inactive]
            rank = next((i + 1 for i, p in enumerate(passages) if p.chunk.source_url == q["url"]
                         and re.search(q["current"], _text(p.chunk), re.S)), None)
            row[mode] = {"superseded_returned": len(stale), "current_rank": rank, "passages": len(passages),
                         "degraded": res.degraded, "latency_ms": ms}
        # Control: same query embedding, no active-version filter (both versions eligible).
        vec = r.embedder.embed_query(q["question"])
        hits = parts["store"].search(vec, k, all_versions, ["public"], snap.embedding_model)
        old_rank = next((i + 1 for i, (c, _) in enumerate(hits) if c.version_id in inactive
                         and re.search(q["superseded"], _text(c), re.S)), None)
        row["control_unfiltered"] = {"superseded_answer_rank": old_rank,
                                     "superseded_in_top_k": sum(c.version_id in inactive for c, _ in hits)}
        rows.append(row)

    def summary(mode):
        ranks = [row[mode]["current_rank"] for row in rows]
        return {"questions": len(rows),
                "superseded_passages_returned": sum(row[mode]["superseded_returned"] for row in rows),
                f"current_answer_recall_at_{k}": sum(x is not None for x in ranks),
                "current_answer_at_rank_1": sum(x == 1 for x in ranks),
                "mean_reciprocal_rank": round(sum(1 / x for x in ranks if x) / len(rows), 3)}
    return {"rows": rows, "vector": summary("vector"), "full": summary("full"),
            "control_unfiltered": {
                "questions_whose_superseded_answer_is_in_top_k": sum(
                    row["control_unfiltered"]["superseded_answer_rank"] is not None for row in rows),
                "superseded_passages_in_top_k": sum(row["control_unfiltered"]["superseded_in_top_k"] for row in rows)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("eval_stale"))
    ap.add_argument("--k", type=int, default=5, help="passages per question that count (default 5)")
    ap.add_argument("--hash", action="store_true", help="test-only hash embedder: plumbing check, not results")
    a = ap.parse_args(argv)
    questions = load_questions()
    embedder, reranker = make_models(a.hash)
    with tempfile.TemporaryDirectory(prefix="askiisc-stale-") as tmp:
        parts = build(Path(tmp), embedder, reranker)
        try:
            result = evaluate(parts, questions, a.k)
        finally:
            parts["store"].close()
    result["config"] = {"embedder": embedder.model_id, "reranker": getattr(reranker, "model_id", None),
                        "rerank_min": parts["retriever"].rerank_min,
                        "k": a.k, "questions": str(QUESTIONS.relative_to(ROOT)), "today_term": "2026-AUG",
                        "plumbing_check_only": a.hash,
                        "active_versions": len(parts["snapshot"].active_version_ids),
                        "superseded_versions": len(parts["inactive_versions"]),
                        "chunks": parts["report"]["chunks"]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "stale_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nStale-version evaluation: {len(questions)} questions, k={a.k}, embedder {embedder.model_id}"
          + ("  (HASH: plumbing check only, do not report)" if a.hash else ""))
    print(f"{'id':5} {'vector: stale/rank':>19} {'full: stale/rank':>17} {'control: old answer rank':>25}")
    for row in result["rows"]:
        v, f, c = row["vector"], row["full"], row["control_unfiltered"]
        print(f"{row['id']:5} {v['superseded_returned']:>12}/{str(v['current_rank']):<6}"
              f"{f['superseded_returned']:>10}/{str(f['current_rank']):<6}{str(c['superseded_answer_rank']):>20}")
    for mode in ("vector", "full"):
        print(f"{mode:6}: {json.dumps(result[mode])}")
    print(f"control: {json.dumps(result['control_unfiltered'])}")
    print(f"Wrote {a.out / 'stale_summary.json'}")
    bad = result["vector"]["superseded_passages_returned"] + result["full"]["superseded_passages_returned"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
