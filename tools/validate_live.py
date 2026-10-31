"""Validate the WHOLE stack on real data and real models, and write a report to share.

Runs against the live stores the ingestion pipeline filled from cds.iisc.ac.in (Qdrant, Postgres, Neo4j),
the real BGE-M3 (TEI) and the real reranker. Nothing is faked. Run it on the machine where the stack
runs, after `python pipeline.py`:

    python tools/validate_live.py --env-file ..\\ingestion\\.env
    python tools/validate_live.py --env-file ..\\ingestion\\.env --serving     # + serving checks (server on 8100)

Writes live_validation_<date-time>.md (no passwords in it). Uses the DEV questions only: the held-out
test set is run separately, once (RUN_INTEGRATED.md step 5).
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPOT = [  # (question, what a correct result looks like)
    ("Who teaches DS 252 this semester?", "the current-term DS 252 schedule row first"),
    ("Which course have Sathish Vadhiyar and Yogesh Simmhan taught together?", "a DS 221 row, found via graph"),
    ("Who works on computer vision at CDS?", "faculty research interests / lab listings"),
    ("What is the monthly mess fee at the IISc hostels?", "refused: score below the threshold"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", help="the ingestion .env (credentials are read, never printed)")
    ap.add_argument("--serving", action="store_true", help="also run tools/check_serving.py against the server")
    ap.add_argument("--serving-url", default="http://127.0.0.1:8100")
    ap.add_argument("--skip-eval", action="store_true", help="skip the dev-set evaluation (takes minutes on CPU)")
    a = ap.parse_args()
    os.environ.setdefault("ASKIISC_BACKEND", "ingestion")
    if a.env_file:
        os.environ["ASKIISC_ENV_FILE"] = str(Path(a.env_file).resolve())
    sys.path.insert(0, str(ROOT))
    from askiisc_retrieval import config

    out: list[str] = [f"# Live validation, {datetime.now():%Y-%m-%d %H:%M}", ""]
    status: list[tuple[str, str, str]] = []

    def check(name, fn):
        t = time.perf_counter()
        try:
            detail = fn()
            status.append((name, "PASS", f"{detail} ({(time.perf_counter() - t) * 1000:.0f} ms)"))
            return True
        except Exception as e:
            status.append((name, "FAIL", " ".join(f"{type(e).__name__}: {e}".split())[:200]))
            return False

    # ---------------------------------------------------------------- 1. services
    import httpx

    def qdrant():
        r = httpx.get(f"{config.INGEST_QDRANT_URL.replace('localhost', '127.0.0.1')}/collections/{config.INGEST_COLLECTION}",
                      timeout=10)
        r.raise_for_status()
        res = r.json()["result"]
        n = res.get("points_count")
        if not n:
            raise RuntimeError("collection is empty: run python pipeline.py")
        return f"{n} points in {config.INGEST_COLLECTION}"

    embedder = (os.environ.get("ASKIISC_EMBEDDER") or os.environ.get("EMBEDDER") or "tei").lower()

    def tei():
        if embedder == "local":
            return "not used (EMBEDDER=local: BGE-M3 runs in-process)"
        info = httpx.get(f"{config.TEI_URL}/info", timeout=10).json()
        model = info.get("model_id", "?")
        v = httpx.post(f"{config.TEI_URL}/embed", json={"inputs": ["health check"]}, timeout=30).json()[0]
        if model != config.EMBED_MODEL:
            raise RuntimeError(f"TEI serves {model}, the index needs {config.EMBED_MODEL}")
        return f"{model}, dim {len(v)}"

    def postgres():
        import psycopg2
        if not config.PG_DSN:
            raise RuntimeError("no Postgres credentials (PG_USER/PG_PASSWORD or ASKIISC_PG_DSN)")
        with psycopg2.connect(config.PG_DSN, connect_timeout=5) as c, c.cursor() as cur:
            cur.execute("SELECT count(*) FILTER (WHERE is_active), count(*) FROM document_versions")
            act, tot = cur.fetchone()
            cur.execute("SELECT count(*) FROM document_chunks dc JOIN document_versions dv USING (version_id) "
                        "WHERE dv.is_active")
            ch = cur.fetchone()[0]
        return f"{act} active versions ({tot} total), {ch} active chunks"

    def neo4j():
        from neo4j import GraphDatabase
        if not config.NEO4J_PASSWORD:
            raise RuntimeError("NEO4J_PASSWORD not set")
        d = GraphDatabase.driver(config.NEO4J_URI.replace("localhost", "127.0.0.1"),
                                 auth=(config.NEO4J_USER, config.NEO4J_PASSWORD), connection_timeout=5)
        try:
            with d.session() as s:
                rows = s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS n ORDER BY n DESC").data()
        finally:
            d.close()
        return ", ".join(f"{r['t']} {r['n']}" for r in rows) or "no edges"

    for name, fn in (("Qdrant", qdrant), ("TEI (BGE-M3)", tei), ("PostgreSQL", postgres), ("Neo4j", neo4j)):
        check(name, fn)
    out += ["## 1. Services", "", "| Service | Result | Detail |", "|---|---|---|"]
    out += [f"| {n} | {s} | {d} |" for n, s, d in status]
    if any(s == "FAIL" for n, s, d in status if n in ("Qdrant", "TEI (BGE-M3)")):
        return finish(out, "Qdrant or TEI is not usable; fix that first (RUN_INTEGRATED.md steps 1-3).")

    # ---------------------------------------------------------------- 2. ingestion report
    from askiisc_retrieval.cli import cmd_check_ingestion
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_check_ingestion(argparse.Namespace(json=False))
    out += ["", "## 2. Ingestion check", "", "```", buf.getvalue().strip(), "```"]

    # ---------------------------------------------------------------- 3-4. retrieval with the real models
    print("Loading retrieval with the real reranker ...", file=sys.stderr)
    from askiisc_retrieval.service import build_ingestion_retriever
    out += report_retrieval(build_ingestion_retriever(), run_eval=not a.skip_eval)

    # ---------------------------------------------------------------- 5. serving
    if a.serving:
        p = subprocess.run([sys.executable, str(ROOT / "tools" / "check_serving.py"), "--url", a.serving_url],
                           capture_output=True, text=True, cwd=ROOT, timeout=3600)
        out += ["", "## 5. Serving checks", "", "```", (p.stdout + p.stderr).strip()[-6000:], "```"]
    return finish(out, None)


def report_retrieval(r, run_eval: bool = True, dev_csv: Path | None = ROOT / "dev_live.csv") -> list[str]:
    """Sections 3 (spot questions, graph on/off) and 4 (dev set) for a built retriever."""
    from askiisc_retrieval import config
    out = ["", f"## 3. Spot questions (reranker {config.RERANK_MODEL}, threshold {config.RERANK_MIN})", ""]
    for q, expect in SPOT:
        for graph in (True, False):
            res = r.retrieve(q, ["public"], use_graph=graph)
            best = res.rerank_top if res.rerank_top is not None else res.top_score
            out.append(f"**{q}**  graph {'on' if graph else 'off'}: best score {best:.3f}, "
                       f"{'answer' if res.evidence_sufficient else 'REFUSE'}, {res.timings_ms.get('total', 0):.0f} ms "
                       f"(expected: {expect})")
            for p in res.passages[:4]:
                rr = f"{p.rerank_score:.3f}" if p.rerank_score is not None else "-"
                out.append(f"- [{p.marker}] rerank {rr} via {p.via} | {p.chunk.section_heading[:90]} | "
                           f"term {p.chunk.academic_term} | {p.chunk.source_url}")
            out.append("")
    if run_eval:
        from askiisc_retrieval.eval_recall import run
        print("Running the dev questions (minutes on CPU) ...", file=sys.stderr)
        buf = io.StringIO()
        with redirect_stdout(buf):
            summary = run(r, k=5, out=str(dev_csv) if dev_csv else None)["summary"]
        out += ["## 4. Dev questions (real models)", "", "```", buf.getvalue().strip()[-6000:], "```", "",
                "Summary: " + str({k: v for k, v in summary.items() if not str(k).startswith("calib")}),
                f"Abstention calibration (reranker): {summary.get('calib_rerank')}", "",
                "If `best_threshold` differs clearly from the current threshold, set ASKIISC_RERANK_MIN to it "
                "(dev only; never from the test set)."]
    return out


def finish(out: list[str], error: str | None):
    if error:
        out += ["", f"**Stopped:** {error}"]
    path = ROOT / f"live_validation_{datetime.now():%Y%m%d_%H%M}.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nReport written to {path}")
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
