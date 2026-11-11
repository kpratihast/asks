# Serving tier: how to run and use it

The serving tier answers a question with a grounded, cited answer:

```
client --(API key)--> gateway --> admission control --> retrieval.run --> generator.prepare --> vLLM (stream)
                                    401 / 429                 |                  token budget          |
                                                    abstain early if evidence     fixed prompt prefix    v
                                                    is insufficient (no model)    (prefix caching)   citation check
                                                                                                        |
                                                                    answered | abstained | fallback (source excerpts)
```

| PRD item | Where |
|---|---|
| FR-6: API keys, 401, scope from the key (guest / student / faculty / admin) | `askiisc_serving/auth.py` |
| FR-7: per-key rate limit, per-tenant concurrency C, bounded queue Q, HTTP 429 + Retry-After | `admission.py` |
| FR-8: 3B / 8B / 8B-AWQ, one at a time | `deploy/vllm_variants.sh` |
| FR-10: fixed prompt prefix, length bound enforced once | `generator.py` (`SYSTEM`, `prepare`) |
| FR-11: answer only from evidence, cite every claim, abstain, excerpt fallback on outage | `generator.py`, `app.py` |
| FR-12: token streaming (SSE), citation resolution | `app.py` (`stream: true`, `GET /v1/sources/{chunk_id}`) |
| Baselines B0 / B2 / full | request `mode`: `none` / `vector` / `full` |
| Baseline B1 (no batching, no prefix cache) | `deploy/vllm_variants.sh b1-3b` |
| Load test (7.1): Poisson arrivals, 3 tenants | `bench/locustfile.py` |
| Metrics (7.3): TTFT, p50/p95/p99, tokens, 429 rate | `GET /metrics` and one JSON log line per request |

---

## 1. On the laptop (CPU, Ollama)

Ollama serves the same OpenAI-compatible API as vLLM, so the code does not change, only two variables.

```powershell
# once: install Ollama from ollama.com, then
ollama pull llama3.2:1b          # 1.3 GB; use llama3.2:3b (2 GB) if you have the memory
# keys (never commit the real file)
copy serving_keys.example.yaml serving_keys.yaml   # then edit: put long random keys
conda activate askiisc
docker compose up -d qdrant
$env:QDRANT_URL       = "http://127.0.0.1:6333"
$env:SERVING_KEYS_FILE = "serving_keys.yaml"
$env:VLLM_URL         = "http://127.0.0.1:11434"
$env:SERVED_MODEL     = "llama3.2:1b"
$env:MAX_MODEL_LEN    = "8192"
python -m askiisc_serving --port 8100
```

Ask (second window):

```powershell
$key = "<a key from serving_keys.yaml>"
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8100/v1/ask -Headers @{Authorization="Bearer $key"} `
  -ContentType "application/json" -Body '{"question":"Who teaches DS 252 this semester?"}' | ConvertTo-Json -Depth 6
```

Or open http://127.0.0.1:8100/docs. Memory: retrieval models (~3.5 GB) + the 1B model (~1.3 GB) run in
one process plus Ollama; close the browser first. On CPU, expect tens of seconds per answer.

## 2. On the GPU (vLLM)

```bash
./deploy/vllm_variants.sh 3b            # or 8b | 8b-awq | b1-3b  (one at a time)
export SERVED_MODEL=llama-3.2-3b VLLM_URL=http://127.0.0.1:8000 MAX_MODEL_LEN=8192
export ASKIISC_DEVICE=cuda ASKIISC_RERANK_MODEL=BAAI/bge-reranker-v2-m3
export ASKIISC_BACKEND=ingestion ...   # retrieval on the ingestion stores: see INTEGRATION.md
export SERVING_KEYS_FILE=serving_keys.yaml
python -m askiisc_serving --host 0.0.0.0 --port 8100
```

- The Llama models are gated on Hugging Face: accept the licence and set `HF_TOKEN` once to download,
  then run with `HF_HUB_OFFLINE=1` so nothing leaves the cloud at runtime (NFR-8).
- If you set `ASKIISC_RERANK_MIN`, calibrate it on the indexed departments with your selected
  reranker. By default, reranking orders the evidence without applying a score cutoff.
- Set `SERVING_TOKENIZER=meta-llama/Llama-3.2-3B-Instruct` to count prompt tokens exactly (default: a
  conservative estimate).

## 3. API

### `POST /v1/ask` (header `Authorization: Bearer <key>`)

```json
{"question": "Who teaches DS 252 this semester?",
 "programme": null, "academic_year": "2026-2027",   // optional filters / prompt context
 "mode": "full",                                     // full | vector (B2) | none (B0)
 "pinned_snapshot": null,                            // benchmarks: 409 if the corpus changed
 "stream": false}
```

Response (`status` is what the client must act on):

| status | meaning | client shows |
|---|---|---|
| `answered` | answer passed the citation check | `answer`; `citations` resolve [S#] to title, URL, version, excerpt |
| `abstained` | `insufficient_evidence` (model not called) or `model_abstained` | "I could not find this in the CDS pages." (+ `sources_that_may_help`) |
| `fallback` | `generation_unavailable` or `citation_check_failed` | `excerpts`: source titles, links, excerpts; no generated text |

Always present: `trace` (snapshot id, mode, `timings_ms`: queue, retrieval, prompt, ttft, generation,
total), `usage` (prompt/completion tokens), `budget` (prompt tokens, evidence kept/dropped).

Errors: 401 no/invalid key, 422 bad request, 429 over limit (`Retry-After` header and `reason`:
`rate_limited`, `tenant_queue_full`, `queue_timeout`, `server_busy`), 409 pinned snapshot not active,
503 retrieval down.

### Streaming (`"stream": true`, Server-Sent Events)
`event: meta` (citation metadata for [S1].., budget, snapshot) -> `event: token` (`{"t": "..."}`) ...
-> `event: done` (the same object as the JSON response). If `done.status` is `fallback` with
`replace_streamed_text: true`, the client must replace the streamed text with the excerpts.
Streaming does not regenerate; the JSON path regenerates once when citations are invalid.

### Others
`GET /v1/sources/{chunk_id}` (full cited text, only if active and within the key's scope),
`GET /healthz`, `GET /readyz` (retrieval + model endpoint), `GET /metrics` (Prometheus text).

## 4. Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `VLLM_URL` / `SERVED_MODEL` | `http://127.0.0.1:8000` / `meta-llama/Llama-3.2-3B-Instruct` | model endpoint |
| `MAX_MODEL_LEN` / `MAX_OUTPUT_TOKENS` | 8192 / 384 | prompt budget = len - output - 128 safety |
| `MAX_EVIDENCE` | 6 | most sources in a prompt |
| `RATE_PER_MIN` | 30 | per API key |
| `TENANT_CONCURRENCY` / `TENANT_QUEUE` / `QUEUE_WAIT_S` | 20 / 10 / 20 | C / Q / max wait in queue |
| `GLOBAL_CONCURRENCY` | 32 | per process |
| `GEN_TIMEOUT_S` | 120 | model call timeout, then excerpt fallback |
| `REGENERATE` | 1 | one retry when citations are invalid (JSON path) |
| `SERVING_RETRIEVAL` / `RETRIEVAL_URL` | `local` / `http://127.0.0.1:8090` | in-process, or the retrieval service |
| `SERVING_KEYS_FILE` | (required) | API keys, tenants, roles |

Calibrate C, Q and the rate on the RTX 4090 and report them with every benchmark (NFR-9).

## 5. Checks and the end-to-end evaluation

With the server running (second window, project folder):

```
python tools/check_serving.py                 # 14 checks: health, 401, cited answer, refusals, scope, streaming, sources, metrics
python tools/check_serving.py --rate-test 3   # only the 429 check; start the server with RATE_PER_MIN=3 for it
python tools/check_serving.py --outage-test   # only the fallback check; stop Ollama/vLLM first
python tools/eval_e2e.py --out e2e_<model>.csv [--limit 10]   # the 120 questions through the full system
```

`eval_e2e.py` is resumable (re-run the same command to continue) and prints refusal accuracy, answered
questions that cite the expected page, and latency p50/p95/p99. Answer correctness: judge the `answer`
column against `expected_answer`.

## 6. Running the experiments

- **E1 (3B vs 8B vs 8B-AWQ):** start one variant, run `bench/locustfile.py` at the concurrency sweep and the
  120-question accuracy run, stop it, start the next. Same `pinned_snapshot` for all.
- **Baselines:** B0 = `"mode": "none"`; B2 = `"mode": "vector"`; B1 = `vllm_variants.sh b1-3b`.
- **E2 (overload):** Locust past capacity; count 429s (`askiisc_http_responses_total{code="429"}`) and check
  admitted requests still complete.
- **E6 (fallback):** stop Neo4j -> `trace.degraded: true`, answers still come; stop vLLM -> `fallback` with excerpts.

## 7. Known limits

- On the 120-question set, retrieval alone would answer 3 of the 20 must-refuse questions (topic pages
  without the exact fact). The prompt rule 3 and the model's abstention must catch them: measure
  abstention end to end, never on retrieval alone.
- Streaming shows text before the citation check; a failed check is signalled in `done` and the client
  must replace the text.
- One process, models loaded once; scale by running more processes behind the gateway.
- Kubernetes manifests are not included yet (PRD lists Kubernetes); the services run as plain processes or
  Docker containers.
