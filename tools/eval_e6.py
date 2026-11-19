"""E6 failure handling (PRD.md:280; #44): generation-service outage and graph-store outage.

Builds the same private index as tools/eval_stale.py (current CDS test pages + superseded versions) and runs
the real gateway (askiisc_serving.create_app) in process with fastapi's TestClient:

  generation outage   VLLM_URL points at a closed port (127.0.0.1:9), so the model is unreachable. Every
                      question of test_stale12.yaml is asked non-streaming and streaming. Expected for each
                      (FR-11): HTTP 200, status "fallback", reason "generation_unavailable", at least one source
                      excerpt, and no generated answer text.
  graph outage        the graph store raises on every call. Full (graph) mode must still return passages,
                      marked degraded (FR-9), both from the retriever and through the gateway.

    python tools/eval_e6.py --out eval_e6            # configured embedder and reranker
    python tools/eval_e6.py --out eval_e6 --hash     # plumbing check only

The real index under data/ is never touched. Writes <out>/e6_summary.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import eval_stale  # noqa: E402

CLOSED_PORT_URL = "http://127.0.0.1:9"


def _sse_done(raw: str) -> dict | None:
    """The `done` event's JSON from an SSE body."""
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        lines = block.strip().splitlines()
        if lines and lines[0] == "event: done":
            return json.loads("".join(line[len("data: "):] for line in lines[1:] if line.startswith("data: ")))
    return None


def _client(parts):
    from fastapi.testclient import TestClient

    from askiisc_serving.app import create_app
    from askiisc_serving.retrieval_client import LocalRetrieval
    from askiisc_serving.settings import Settings
    s = Settings(vllm_url=CLOSED_PORT_URL, served_model="test-model", public_chat=True, email_auth_enabled=False,
                 keys_file=None, rate_per_min=10000, gen_connect_timeout_s=2)
    return TestClient(create_app(s, retrieval=LocalRetrieval(parts["retriever"])))


def ask(c, question: str, stream: bool, mode: str = "full") -> tuple[int, dict | None, float]:
    t0 = time.perf_counter()
    if stream:
        with c.stream("POST", "/v1/ask", json={"question": question, "mode": mode, "stream": True}) as r:
            body = r.read().decode("utf-8")
            status = r.status_code
        payload = _sse_done(body) if status == 200 else None
    else:
        r = c.post("/v1/ask", json={"question": question, "mode": mode, "stream": False})
        status, payload = r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else None)
    return status, payload, round((time.perf_counter() - t0) * 1000, 1)


def generation_outage(parts, questions) -> dict:
    rows = []
    with _client(parts) as c:
        for q in questions:
            for stream in (False, True):
                status, p, ms = ask(c, q["question"], stream)
                p = p or {}
                rows.append({"id": q["id"], "stream": stream, "http": status, "status": p.get("status"),
                             "reason": p.get("reason"), "excerpts": len(p.get("excerpts") or []),
                             "answer_is_none": p.get("answer") is None, "latency_ms": ms,
                             "ok": (status == 200 and p.get("status") == "fallback"
                                    and p.get("reason") == "generation_unavailable"
                                    and bool(p.get("excerpts")) and p.get("answer") is None)})
    lat = [r["latency_ms"] for r in rows]
    # A question whose evidence is insufficient abstains before the model is called; that is not a
    # generation-outage case, so it is counted separately rather than as a failure.
    early = [r for r in rows if r["status"] == "abstained"]
    return {"requests": len(rows), "expected_fallback": sum(r["ok"] for r in rows),
            "abstained_before_model_call": len(early),
            "http_200": sum(r["http"] == 200 for r in rows),
            "fabricated_answers": sum(r["status"] == "answered" for r in rows),
            "statuses": {f"{r['status']}/{r['reason']}": sum(1 for x in rows if (x['status'], x['reason']) == (r['status'], r['reason']))
                         for r in rows},
            "latency_ms_p50": round(statistics.median(lat), 1), "latency_ms_max": max(lat), "rows": rows}


def graph_outage(parts, questions) -> dict:
    parts["graph"].fail = True
    rows = []
    try:
        for q in questions:                               # retrieval layer
            res = parts["retriever"].retrieve(q["question"], ["public"], use_graph=True)
            rows.append({"id": q["id"], "layer": "retriever", "passages": len(res.passages),
                         "degraded": res.degraded, "reason": res.degraded_reason,
                         "ok": bool(res.passages) and res.degraded})
        with _client(parts) as c:                         # through the gateway (model also down)
            for q in questions:
                status, p, ms = ask(c, q["question"], stream=False)
                trace = (p or {}).get("trace") or {}
                rows.append({"id": q["id"], "layer": "gateway", "http": status, "status": (p or {}).get("status"),
                             "degraded": trace.get("degraded"), "excerpts": len((p or {}).get("excerpts") or []),
                             "latency_ms": ms,
                             "ok": status == 200 and bool(trace.get("degraded")) and bool((p or {}).get("excerpts"))})
    finally:
        parts["graph"].fail = False
    by = lambda layer: [r for r in rows if r["layer"] == layer]         # noqa: E731
    return {"retriever": {"queries": len(by("retriever")), "degraded_with_passages": sum(r["ok"] for r in by("retriever"))},
            "gateway": {"requests": len(by("gateway")), "http_200_degraded_with_excerpts": sum(r["ok"] for r in by("gateway"))},
            "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("eval_e6"))
    ap.add_argument("--hash", action="store_true", help="test-only hash embedder: plumbing check, not results")
    a = ap.parse_args(argv)
    questions = eval_stale.load_questions()
    embedder, reranker = eval_stale.make_models(a.hash)
    with tempfile.TemporaryDirectory(prefix="askiisc-e6-") as tmp:
        parts = eval_stale.build(Path(tmp), embedder, reranker)
        try:
            gen = generation_outage(parts, questions)
            graph = graph_outage(parts, questions)
        finally:
            parts["store"].close()
    result = {"config": {"embedder": embedder.model_id, "reranker": getattr(reranker, "model_id", None),
                         "model_endpoint": CLOSED_PORT_URL + " (closed port: generation service unavailable)",
                         "questions": len(questions), "plumbing_check_only": a.hash},
              "generation_outage": gen, "graph_outage": graph}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "e6_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nE6 failure handling, {len(questions)} questions, embedder {embedder.model_id}"
          + ("  (HASH: plumbing check only, do not report)" if a.hash else ""))
    print(f"Generation outage: {gen['expected_fallback']}/{gen['requests']} requests returned HTTP 200 + fallback/"
          f"generation_unavailable + excerpts + no answer text; {gen['abstained_before_model_call']} abstained "
          f"before the model call; answers generated {gen['fabricated_answers']}; "
          f"latency p50 {gen['latency_ms_p50']} ms, max {gen['latency_ms_max']} ms")
    print(f"  statuses: {gen['statuses']}")
    print(f"Graph outage: retriever {graph['retriever']['degraded_with_passages']}/{graph['retriever']['queries']} "
          f"degraded with passages; gateway {graph['gateway']['http_200_degraded_with_excerpts']}/"
          f"{graph['gateway']['requests']} HTTP 200, degraded, with excerpts")
    print(f"Wrote {a.out / 'e6_summary.json'}")
    ok = gen["expected_fallback"] > 0 and gen["fabricated_answers"] == 0 and \
        gen["expected_fallback"] + gen["abstained_before_model_call"] == gen["requests"] and \
        graph["retriever"]["degraded_with_passages"] == graph["retriever"]["queries"] and \
        graph["gateway"]["http_200_degraded_with_excerpts"] == graph["gateway"]["requests"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
