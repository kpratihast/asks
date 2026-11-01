# Admission limits and load testing

The public AWS endpoint is https://13.203.0.141/v1/ask. Its live settings use the small-server deployment profile rather than the serving package's PRD defaults.

The load-test observations have three causes:

- The AWS launcher sets **global concurrency = 1**, **tenant concurrency = 1**, **tenant queue = 4**, and **queue wait = 120 s** unless explicitly configured otherwise. Changing the serving package's default C=20 / Q=10 alone does not change AWS.
- All anonymous browsers use the same `public-browser` rate bucket and `public` tenant. Requests from a phone and laptop share that budget. A token bucket of R requests/minute starts with R tokens and refills one token every 60/R seconds; rejection attempts also consume any available token. A refill of one every five seconds is consistent with R=12. Concurrent traffic and timing affect an observed burst count. Inspection of the deployed runtime confirmed **R=10/minute**, refilling one token every six seconds; the approximate 12-request observation is not the configured limit.
- The earlier admission controller checked the global cap before entering the tenant queue. With global=1, the second request immediately returned `429/server_busy`; the configured queue was never reached. It also failed to recheck the global cap after a queued request woke, which could exceed the global cap across tenants.

The deployed `/readyz` counters captured during diagnosis were 304 admitted, 3,605 rate-limited, 31 globally rejected, and zero queue-full/queue-timeout rejections. Those counters describe that process at the time of capture; they are not a complete latency or throughput benchmark.

## Fix

The controller now queues requests when either concurrency limit is full, preserves FIFO order within each tenant, rechecks both limits under one lock, and removes waiting requests on timeout or cancellation. Slots remain held through the full response or stream. A full queue returns `429` with `reason=tenant_queue_full`; a wait that expires returns `queue_timeout`. With the queue explicitly disabled (`TENANT_QUEUE=0`), global saturation still returns `server_busy`. All rejection responses include `Retry-After`.

The AWS configuration now accepts the concurrency and queue settings below. Missing values keep the existing small-server defaults. This change does not raise the live limits automatically, change the model, grant document access or expose the internal `/metrics` endpoint.

`/readyz` now includes `admission.limits` with the effective settings and `anonymous_rate_scope=shared`. Check this before a benchmark:

```bash
curl --fail --silent --show-error https://13.203.0.141/readyz | python3 -m json.tool
```

Interpret the response `reason`, not only HTTP 429:

| Reason | Limiter reached | Next action |
| --- | --- | --- |
| `rate_limited` | Per-key token bucket; shared by anonymous users | Allow refill or configure the intended rate for the controlled test. |
| `tenant_queue_full` | This tenant already has its allowed number of waiters | Expected load shedding beyond admitted capacity plus Q. |
| `queue_timeout` | A queued request waited longer than `QUEUE_WAIT_S` | Measure service time and queue delay before increasing the timeout. |
| `server_busy` | Global limit reached with queuing disabled | Check `TENANT_QUEUE` and the global limit. |

## Controlled C=20 / Q=10 profile

For one tenant's 31-request burst, use these fields in the **current, reconciled private AWS runtime JSON**, retaining every model, SMTP and authentication setting:

```json
{
  "RATE_PER_MIN": 60,
  "GLOBAL_CONCURRENCY": 20,
  "TENANT_CONCURRENCY": 20,
  "TENANT_QUEUE": 10,
  "QUEUE_WAIT_S": 120
}
```

This is a patch illustration, not a complete runtime file. Do not replace the full SSM parameter with it or an example containing placeholder secrets. Existing AWS configuration is captured per immutable release, so changing environment variables in a local shell does not change the running app. Apply a reconciled full runtime configuration through the existing named deployer workflow and deploy a new release containing the queue fix. Local/manual deployment and rollback instructions are in `AWS_DEPLOYMENT.md`; automatic GitHub deployment remains gated by its IAM setup and `AWS_DEPLOY_ENABLED`.

Supported AWS bounds: rate 1–100/min, global and tenant concurrency 1–32, queue 0–100, queue wait 1–300 seconds. They validate configuration; they are not a claim that the current EC2 instance or upstream LLM can sustain those values. Twenty admitted application requests may still queue or be rejected inside the model provider. A real end-to-end test consumes model requests and should use a controlled window followed by restoration of the small-server profile.

Before each burst, allow the shared rate bucket to refill and pause other traffic. For 31 attempts, at least 31 tokens must be available. With an otherwise unused tenant and sufficiently slow requests, the expected sequence is 20 active, 10 queued, and one queue-full rejection; queued requests succeed only if service completes before their deadlines. A 30/minute bucket would reject request 31 at the rate limiter first, masking the queue-full boundary. A 12/minute bucket prevents reaching 20 active requests even with C=20 configured.

Use `/readyz` sampling to capture peak active/waiting counts, record each response's HTTP code, body `status`/`reason`, `Retry-After` and latency, and distinguish admission success from usable cited answers. An HTTP 200 fallback or abstention is not evidence of successful model generation or answer quality. Admission limits are per process: keep the current single application worker for these measurements.

## Reproducible offline checks

```bash
python3 bench/admission_probe.py --concurrency 20 --queue 10 --requests 31 --rate-per-min 60
python3 -m pytest -q tests/test_admission_limits.py
```

The probe holds synthetic slots and makes **zero model calls**. In a matched before/after burst with synthetic work, the results were:

| Configuration and burst | Before | After |
| --- | --- | --- |
| C=1, global=1, Q=4, 12 attempts, R=60/min | 1 completed, 0 queued, 11 busy | 5 completed, peak 4 queued, 7 queue-full |
| C=20, global=20, Q=10, 31 attempts, R=60/min | 20 completed, 0 queued, 11 busy | 30 completed, peak 10 queued, 1 queue-full |
| C=20, global=20, Q=10, 31 attempts, R=12/min | 12 completed, 19 rate-limited | 12 completed, 19 rate-limited |

The HTTP regression test also runs 31 simultaneous requests through the actual FastAPI `/v1/ask` route with fake retrieval/model responses: 20 hold active slots, 10 wait, one receives HTTP 429, and the remaining 30 return cited answers after the fake model is released. These results verify queue policy and HTTP integration. They do not measure production LLM latency, AWS throughput, or a live 20-way model workload.

## Live AWS verification — 1 November 2026

Release `aws-20261101-admission-r10` was deployed with the exact previous runtime configuration preserved: R=10/min shared by anonymous clients, global=1, C=1, Q=4, queue wait=120 s, and the same OpenRouter model/authentication settings.

A six-request simultaneous burst produced **five HTTP 200 cited answers and one HTTP 429 `tenant_queue_full`**, with observed peaks of one active and four waiting. The first answer arrived in 1.92 s and the last in 12.12 s; the rejected request returned in about 95 ms with `Retry-After: 2`. The queue drained to zero. Protected answers and the private PDF remained inaccessible without sign-in. This was a live C=1 / Q=4 test; C=20 / Q=10 was verified separately through the offline HTTP regression test, not enabled on the live app.
