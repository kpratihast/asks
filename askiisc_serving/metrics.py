"""Minimal in-process metrics in Prometheus text format (GET /metrics), plus one JSON log line per
request for the benchmark (latency, TTFT, tokens, status, snapshot). No extra dependency."""
from __future__ import annotations

import json
import logging
from collections import Counter, deque

log = logging.getLogger("askiisc.requests")


def _q(xs, q):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


class Metrics:
    def __init__(self, window: int = 5000):
        self.requests = Counter()          # (mode, status)
        self.http = Counter()              # status code
        self.tokens = Counter()            # prompt / completion
        self.lat = {"total_ms": deque(maxlen=window), "ttft_ms": deque(maxlen=window),
                    "retrieval_ms": deque(maxlen=window), "queue_ms": deque(maxlen=window)}

    def record(self, rec: dict):
        self.requests[(rec.get("mode", "?"), rec.get("status", "?"))] += 1
        self.http[rec.get("http", 200)] += 1
        for k in self.lat:
            v = rec.get("timings_ms", {}).get(k.replace("_ms", "") + "_ms")
            if isinstance(v, (int, float)):
                self.lat[k].append(v)
        u = rec.get("usage") or {}
        self.tokens["prompt"] += u.get("prompt_tokens") or 0
        self.tokens["completion"] += u.get("completion_tokens") or 0
        log.info(json.dumps(rec, default=str))

    def reject(self, code: int, reason: str, tenant: str | None = None):
        self.http[code] += 1
        self.requests[("-", f"rejected:{reason}")] += 1
        log.info(json.dumps({"http": code, "status": f"rejected:{reason}", "tenant": tenant}))

    def prometheus(self, admission: dict | None = None) -> str:
        L = ["# TYPE askiisc_requests_total counter"]
        L += [f'askiisc_requests_total{{mode="{m}",status="{s}"}} {n}' for (m, s), n in sorted(self.requests.items())]
        L.append("# TYPE askiisc_http_responses_total counter")
        L += [f'askiisc_http_responses_total{{code="{c}"}} {n}' for c, n in sorted(self.http.items())]
        L.append("# TYPE askiisc_tokens_total counter")
        L += [f'askiisc_tokens_total{{kind="{k}"}} {n}' for k, n in sorted(self.tokens.items())]
        for name, xs in self.lat.items():
            L.append(f"# TYPE askiisc_{name} summary")
            for q in (0.5, 0.95, 0.99):
                L.append(f'askiisc_{name}{{quantile="{q}"}} {_q(list(xs), q)}')
            L.append(f"askiisc_{name}_count {len(xs)}")
        if admission:
            L.append(f"askiisc_inflight {admission.get('global_active', 0)}")
            for t, v in admission.get("tenants", {}).items():
                L.append(f'askiisc_tenant_active{{tenant="{t}"}} {v["active"]}')
                L.append(f'askiisc_tenant_waiting{{tenant="{t}"}} {v["waiting"]}')
        return "\n".join(L) + "\n"
