"""End-to-end evaluation through the running serving tier: every question goes through /v1/ask
(retrieval + prompt + LLM + citation check), one at a time.

    python tools/eval_e2e.py --out e2e_3b.csv                       # all 120 questions
    python tools/eval_e2e.py --out e2e_1b_dry.csv --limit 10        # quick dry run
    python tools/eval_e2e.py --out e2e_3b_b0.csv --mode none        # baseline B0 (no retrieval)

Resumable: questions already in --out are skipped, so a long CPU run can be stopped and restarted.
Records per question: status, reason, answer, cited pages, whether the expected page is cited, the
expected answer (for judging), timings and tokens. Answer CORRECTNESS is not scored here: judge the
`answer` column against `expected_answer` (by hand or with the evaluation team's method).
Run it once per model variant with the same pinned snapshot, and never tune on the results.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import httpx
import yaml

FIELDS = ["id", "category", "question", "status", "reason", "answer", "expected_answer", "gold_url",
          "cites_gold", "used_citations", "verification_ok", "attempts", "evidence_sufficient", "rerank_top",
          "retrieval_ms", "ttft_ms", "generation_ms", "total_ms", "prompt_tokens", "completion_tokens",
          "snapshot_id", "model", "http"]


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--keys", default="serving_keys.yaml")
    ap.add_argument("--role", default="guest")
    ap.add_argument("--questions", default="test_cds120.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "vector", "none"])
    ap.add_argument("--pinned-snapshot", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    key = next(k for k, v in yaml.safe_load(Path(a.keys).read_text(encoding="utf-8"))["keys"].items()
               if v["role"] == a.role)
    qs = yaml.safe_load(Path(a.questions).read_text(encoding="utf-8"))["questions"]
    if a.limit:
        qs = qs[: a.limit]
    out = Path(a.out)
    done = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            done = {r["id"] for r in csv.DictReader(f)}
    new = not out.exists()
    c = httpx.Client(base_url=a.url, timeout=900, headers={"Authorization": f"Bearer {key}"})
    with out.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, q in enumerate(qs, 1):
            if q["id"] in done:
                continue
            gold = (q.get("gold") or [{}])[0].get("url", "") if q.get("gold") else ""
            body = {"question": q["question"], "mode": a.mode}
            if a.pinned_snapshot:
                body["pinned_snapshot"] = a.pinned_snapshot
            for attempt in range(5):                       # wait out 429s politely
                r = c.post("/v1/ask", json=body)
                if r.status_code != 429:
                    break
                time.sleep(int(r.headers.get("retry-after", "5")))
            d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            used = [x["url"] for x in d.get("citations") or [] if x.get("used")]
            tr, t, u = d.get("trace") or {}, (d.get("trace") or {}).get("timings_ms", {}), d.get("usage") or {}
            w.writerow({
                "id": q["id"], "category": q["category"], "question": q["question"], "status": d.get("status"),
                "reason": d.get("reason") or d.get("error"), "answer": d.get("answer"),
                "expected_answer": q.get("expected_answer", ""), "gold_url": gold,
                "cites_gold": (gold.rstrip("/") in {x.rstrip("/") for x in used}) if gold else "",
                "used_citations": " ; ".join(used), "verification_ok": (d.get("verification") or {}).get("ok"),
                "attempts": d.get("attempts"), "evidence_sufficient": tr.get("evidence_sufficient"),
                "rerank_top": tr.get("rerank_top"), "retrieval_ms": t.get("retrieval_ms"), "ttft_ms": t.get("ttft_ms"),
                "generation_ms": t.get("generation_ms"), "total_ms": t.get("total_ms"),
                "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                "snapshot_id": tr.get("snapshot_id"), "model": tr.get("model"), "http": r.status_code})
            f.flush()
            print(f"{i:>4}/{len(qs)} {q['id']:<8} {str(d.get('status')):<10} {str(d.get('reason') or ''):<22} "
                  f"{(t.get('total_ms') or 0) / 1000:6.1f}s  {str(d.get('answer') or '')[:60]!r}", flush=True)
    summarize(out)


def summarize(path: Path):
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    ans = [r for r in rows if r["category"] != "unanswerable"]
    una = [r for r in rows if r["category"] == "unanswerable"]
    st = lambda rs, s: sum(r["status"] == s for r in rs)
    print(f"\n{path.name}: {len(rows)} questions (model {rows[0]['model'] if rows else '?'})")
    if ans:
        print(f"Answerable ({len(ans)}): answered {st(ans, 'answered')}, abstained {st(ans, 'abstained')}, "
              f"fallback {st(ans, 'fallback')}; answered AND cites the expected page "
              f"{sum(r['status'] == 'answered' and r['cites_gold'] == 'True' for r in ans)}")
    if una:
        refused = [r for r in una if r["status"] != "answered"]
        print(f"Must-refuse ({len(una)}): refused {len(refused)} ({len(refused) / len(una):.0%})"
              + (f"; ANSWERED: {', '.join(r['id'] for r in una if r['status'] == 'answered')}"
                 if len(refused) < len(una) else ""))
    num = lambda k: [float(r[k]) for r in rows if r[k] not in ("", "None")]
    for k in ("total_ms", "ttft_ms", "retrieval_ms"):
        xs = num(k)
        if xs:
            print(f"{k:<13} p50 {pct(xs, .5):8.0f}  p95 {pct(xs, .95):8.0f}  p99 {pct(xs, .99):8.0f}")
    print("Judge correctness: compare `answer` with `expected_answer` in the CSV.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--summary":
        summarize(Path(sys.argv[2]))
    else:
        main()
