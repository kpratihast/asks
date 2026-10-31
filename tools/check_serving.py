"""Final end-to-end check of a RUNNING serving tier (python -m askiisc_serving).

    python tools/check_serving.py                         # the standard checks
    python tools/check_serving.py --rate-test 3           # ONLY the 429 check; server started with RATE_PER_MIN=3
    python tools/check_serving.py --outage-test           # with the model endpoint stopped (Ollama/vLLM)

Prints PASS / WARN / FAIL per check. WARN = behaviour depends on the model or corpus (not a code fault).
Uses the guest and student keys from serving_keys.yaml. Each answered question can take ~1 min on a CPU.
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import httpx
import yaml

ROWS = []


def row(name, verdict, detail=""):
    ROWS.append((name, verdict, detail))
    print(f"[{verdict:<4}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def keys(path: Path) -> dict:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))["keys"]
    out = {}
    for k, v in d.items():
        out.setdefault(v["role"], k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--keys", default="serving_keys.yaml")
    ap.add_argument("--rate-test", type=int, default=0, help="RATE_PER_MIN the server was started with")
    ap.add_argument("--outage-test", action="store_true", help="the model endpoint is stopped on purpose")
    a = ap.parse_args()
    K = keys(Path(a.keys))
    guest, student = K.get("guest"), K.get("student")
    if not guest:
        sys.exit("no guest key in the key file")
    c = httpx.Client(base_url=a.url, timeout=600)
    H = lambda k: {"Authorization": f"Bearer {k}"}

    def ask(q, key=guest, expect_ok=True, **body):
        t = time.perf_counter()
        r = c.post("/v1/ask", json={"question": q, **body}, headers=H(key) if key else {})
        if expect_ok and r.status_code != 200:
            sys.exit(f"[FAIL] {q!r}: HTTP {r.status_code} {r.text[:200]}  (429? restart the server without a "
                     "small RATE_PER_MIN, or wait a minute)")
        return r, time.perf_counter() - t

    if a.rate_test:
        key = student or guest
        n = a.rate_test + 1
        with cf.ThreadPoolExecutor(n) as ex:
            res = list(ex.map(lambda i: ask("Say OK.", key=key, expect_ok=False, mode="none")[0], range(n)))
        codes = [x.status_code for x in res]
        r429 = [x for x in res if x.status_code == 429]
        ok = bool(r429) and r429[0].headers.get("retry-after")
        row("over the rate limit -> 429 + Retry-After (FR-7)", "PASS" if ok else "FAIL",
            f"codes {sorted(codes)}" + (f", Retry-After {r429[0].headers.get('retry-after')}s" if r429 else ""))
        return finish()

    if a.outage_test:
        r, s = ask("Who teaches DS 252 this semester?")
        d = r.json()
        ok = r.status_code == 200 and d.get("status") == "fallback" and d.get("reason") == "generation_unavailable" \
            and d.get("excerpts")
        row("model down -> source excerpts, no error (FR-11)", "PASS" if ok else "FAIL",
            f"HTTP {r.status_code}, status={d.get('status')}, reason={d.get('reason')}, {len(d.get('excerpts') or [])} excerpts")
        return finish()

    # 1-2 health
    r = c.get("/healthz")
    row("server up (/healthz)", "PASS" if r.status_code == 200 else "FAIL", str(r.status_code))
    r = c.get("/readyz")
    d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    row("retrieval + model reachable (/readyz)", "PASS" if r.status_code == 200 else "FAIL",
        f"retrieval={d.get('retrieval')}, model={d.get('model')} reachable={d.get('model_endpoint')}")

    # 3-4 auth
    r, _ = ask("hi", key=None, expect_ok=False)
    row("no API key -> 401 (FR-6)", "PASS" if r.status_code == 401 else "FAIL", str(r.status_code))
    r, _ = ask("hi", key="wrong-key-0000000000000000", expect_ok=False)
    row("wrong API key -> 401", "PASS" if r.status_code == 401 else "FAIL", str(r.status_code))

    # 5 grounded answer
    r, s = ask("Who teaches DS 252 this semester?")
    d = r.json()
    used = [x for x in d.get("citations", []) if x.get("used")]
    ok = d.get("status") == "answered" and (d.get("verification") or {}).get("ok") and \
        any("courses/schedule" in x["url"] for x in used)
    row("answerable question -> cited answer", "PASS" if ok else ("WARN" if d.get("status") == "answered" else "FAIL"),
        f"{s:.0f}s, status={d.get('status')}, answer={str(d.get('answer'))[:90]!r}")
    first_chunk = (used[0]["chunk_ids"][0] if used and used[0].get("chunk_ids") else None)
    t = (d.get("trace") or {}).get("timings_ms", {})
    if t:
        row("timings recorded", "PASS", f"retrieval {t.get('retrieval_ms')} ms, first token {t.get('ttft_ms')} ms, "
            f"total {t.get('total_ms')} ms, prompt tokens {(d.get('usage') or {}).get('prompt_tokens')}")

    # 6 early abstention (evidence too weak: model not called)
    r, s = ask("What is the monthly mess fee at the IISc hostels?")
    d = r.json()
    row("out-of-corpus question -> refused before the model", "PASS" if d.get("reason") == "insufficient_evidence"
        else ("WARN" if d.get("status") == "abstained" else "FAIL"),
        f"{s:.0f}s, status={d.get('status')}, reason={d.get('reason')}")

    # 7 model abstention (on-topic pages, fact absent)
    r, s = ask("What is the exact GATE cutoff score for M.Tech CDS in 2026?")
    d = r.json()
    row("on-topic but unanswerable -> refused (model decides)", "PASS" if d.get("status") == "abstained" else "WARN",
        f"{s:.0f}s, status={d.get('status')}, reason={d.get('reason')} (WARN = model answered: known hard case)")

    # 8 access scope
    q = "By when must CDS lab access cards be renewed?"
    leaks = lambda d: [x["url"] for x in (d.get("citations") or []) + (d.get("sources_that_may_help") or [])
                       + (d.get("excerpts") or []) if "student-notices" in x.get("url", "")]
    r, s = ask(q, key=guest)
    row("guest key never sees student-only content", "PASS" if not leaks(r.json()) else "FAIL",
        f"{s:.0f}s, student-only sources returned: {len(leaks(r.json()))}")
    if student:
        r, s = ask(q, key=student)
        found = leaks(r.json())
        row("student key can see student-only content", "PASS" if found else "WARN",
            f"{s:.0f}s, {'found' if found else 'not found (fixture not in this corpus?)'}; status={r.json().get('status')}")

    # 9 streaming
    t0 = time.perf_counter()
    events, first_tok = [], None
    with c.stream("POST", "/v1/ask", json={"question": "Who teaches DS 252 this semester?", "stream": True},
                  headers=H(guest)) as resp:
        ev = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: ") and ev:
                events.append((ev, json.loads(line[6:])))
                if ev == "token" and first_tok is None:
                    first_tok = time.perf_counter() - t0
    kinds = [e for e, _ in events]
    done = events[-1][1] if events else {}
    ok = kinds[:1] == ["meta"] and "token" in kinds and kinds[-1] == "done"
    row("streaming: meta -> tokens -> done (FR-12)", "PASS" if ok else "FAIL",
        f"{kinds.count('token')} tokens, first after {first_tok or 0:.0f}s, done.status={done.get('status')}")

    # 10 citation resolution
    if first_chunk:
        r = c.get(f"/v1/sources/{first_chunk}", headers=H(guest))
        row("open a cited source (/v1/sources)", "PASS" if r.status_code == 200 and r.json().get("text") else "FAIL",
            str(r.status_code))
        r = c.get(f"/v1/sources/{first_chunk}")
        row("sources need a key", "PASS" if r.status_code == 401 else "FAIL", str(r.status_code))

    # 11 metrics
    m = c.get("/metrics").text
    row("metrics (/metrics)", "PASS" if "askiisc_requests_total" in m and "askiisc_ttft_ms" in m else "FAIL",
        next((l for l in m.splitlines() if l.startswith('askiisc_requests_total{mode="full",status="answered"}')), ""))

    return finish()


def finish():
    n = {v: sum(1 for _, x, _ in ROWS if x == v) for v in ("PASS", "WARN", "FAIL")}
    print(f"\n{n['PASS']} passed, {n['WARN']} warnings, {n['FAIL']} failed")
    return 1 if n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
