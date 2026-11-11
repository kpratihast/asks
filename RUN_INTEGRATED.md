# Running ingestion + retrieval + serving together (laptop, Windows PowerShell)

Ingestion v3 (Smit's pipeline with the integration fixes, `ingestion-v3.zip`; changes in its
`CHANGES.md`) fills the stores; retrieval reads them (`ASKIISC_BACKEND=ingestion`); serving answers on
top. Tested end to end with that `pipeline.py` (`tests/test_e2e_smit_pipeline.py`, 14 tests).

Folders assumed (adjust the paths):

```
...\Capstone Project\askiisc-retrieval     this repo (retrieval + serving)
...\Capstone Project\ingestion             ingestion v3 (pipeline.py, docker-compose.yml, .env, ...)
```

## Ports

| Port  | What                                   |
|-------|----------------------------------------|
| 5432  | PostgreSQL (registry)                  |
| 6333  | Qdrant (one server for both collections) |
| 7474 / 7687 | Neo4j (browser / bolt)           |
| 8080  | TEI (BGE-M3 embeddings)                |
| 11434 | Ollama (laptop LLM)                    |
| 8000  | vLLM (GPU LLM)                         |
| **8100** | **serving (`/v1/ask`)**: changed from 8080, which TEI uses |

## Memory (16 GB laptop)

TEI ~2.5 GB, Neo4j ~1 GB, Postgres/Qdrant ~0.2 GB, serving (reranker) ~1.2 GB, Ollama 1B ~1.3 GB:
about 6.2 GB. Close the browser and other apps; do not start Ollama/serving while ingesting.
In ingestion mode retrieval does **not** load BGE-M3 itself (it asks TEI), which saves ~2.3 GB.

---

## Low-memory laptop: run without the TEI container

If TEI never becomes ready (or Docker becomes slow while it loads), run the same BGE-M3 in Python
instead. Add this line to `..\ingestion\.env`:

```
EMBEDDER=local
```

and start only the three other containers: `docker compose up -d postgres qdrant neo4j`
(stop TEI with `docker stop iisc_tei`). The pipeline and retrieval both read `EMBEDDER` from that file.
The model is the one already in your Hugging Face cache from the stand-in runs. Do not mix: a corpus
embedded with one embedder should be queried with the same one (re-ingest after switching).

## 1. Start the stores (once per boot)

```powershell
cd "..\ingestion"
copy .env.example .env       # first time only; set the passwords (or reuse your existing .env)
docker ps                    # if your OLD Qdrant container is running on 6333, stop it first:
docker stop <old-qdrant-container-name>
docker compose up -d
docker compose ps            # 4 containers: iisc_postgres, iisc_qdrant, iisc_tei, iisc_neo4j
Invoke-RestMethod http://127.0.0.1:8080/health     # TEI ready? (first start downloads BGE-M3, minutes)
```

Coming from v2 data? The graph schema changed, so start empty once: `docker compose down -v`, then
`docker compose up -d`. (Passwords in `.env` take effect only when a volume is first created.)

## 2. Create tables, bucket, collection, constraints (once; safe to repeat)

```powershell
conda activate askiisc
pip install -r requirements.txt
python setup_stores.py       # Postgres tables, snapshot folder, Qdrant collection + payload indexes
python setup_graph.py        # Neo4j constraints
```

## 3. Ingest

```powershell
python pipeline.py           # seeds.json: CDS up to ~420 pages; embeds on CPU via TEI
```

Expect tens of minutes on CPU the first time. Re-running skips unchanged pages (content hash) and
replaces changed ones safely (new version written first, then activated, then the old one removed).

## 4. Point retrieval at the stores and check them

```powershell
cd "..\askiisc-retrieval"
$env:ASKIISC_BACKEND  = "ingestion"
$env:ASKIISC_ENV_FILE = "..\ingestion\.env"     # same credentials file as ingestion; nothing retyped
python -m askiisc_retrieval.cli check-ingestion
```

With v3 the report should end with "No problems found" and show
`Graph: ... (N from Neo4j, 0 from rules)`. Anything it lists is a real gap in the crawled data.

Quick checks (no LLM, ~5 s each on CPU):

```powershell
python -m askiisc_retrieval.cli ask "Who teaches DS 252 this semester?"
python -m askiisc_retrieval.cli ask "What is the monthly mess fee at the IISc hostels?"
python -m askiisc_retrieval.cli ask "Which course have Sathish Vadhiyar and Yogesh Simmhan taught together?"
```

Expect: the Aug 2026 DS 252 row first; the mess fee refused (score < threshold); `via graph` lines.
The header shows `layout full contract payload` and `Graph: ... from Neo4j, ... from rules`.

## 4b. One command: validate everything on the real data and models

```powershell
python tools/validate_live.py --env-file ..\ingestion\.env
# with serving running in another window (step 6), add --serving
```

It checks the four services (and that TEI really serves BAAI/bge-m3), prints the ingestion report,
asks four spot questions with the graph on and off using the real reranker, runs the dev questions, and
writes `live_validation_<date>.md`. Send that file: it holds no passwords.

## 5. Recalibrate on dev, then run the held-out set once

The corpus is chunked differently from the stand-in, so the scores move. Dev questions only:

```powershell
python -m askiisc_retrieval.cli eval --out dev_ingestion.csv
# Set a cutoff only after calibrating it on this corpus and reranker.
$env:ASKIISC_RERANK_MIN = "<suggested value>"
```

Then the 120 held-out questions, once, recorded as they come out (do not tune on them):

```powershell
python -m askiisc_retrieval.cli eval --questions test_cds120.yaml --out test_cds120_ingestion.csv
python tools/score_heldout.py test_cds120_ingestion.csv
```

Compare with the stand-in result (96/100 right page in top 5). Pages the crawl did not reach show
up as misses: that is coverage, not ranking (raise `max_pages` in `seeds.json`).

## 6. Serving on top

Same window (the two variables from step 4 stay set), plus the usual serving ones:

```powershell
$env:SERVING_KEYS_FILE = "serving_keys.yaml"
$env:VLLM_URL          = "http://127.0.0.1:11434"
$env:SERVED_MODEL      = "llama3.2:1b"            # or askiisc-1b
$env:MAX_MODEL_LEN     = "8192"
python -m askiisc_serving --port 8100
```

Second window:

```powershell
cd "...\askiisc-retrieval"; conda activate askiisc
python tools/check_serving.py                      # now defaults to http://127.0.0.1:8100
```

Your `Ask` helper: change `8080` to `8100` in its URL.

## 7. Switching back to the stand-in corpus

```powershell
Remove-Item Env:ASKIISC_BACKEND, Env:ASKIISC_ENV_FILE
docker compose -f "..\ingestion\docker-compose.yml" stop qdrant    # free 6333
docker start <old-qdrant-container-name>
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Cannot read collection 'iisc_policy_passages'` | Smit's Qdrant not running, or your old Qdrant is on 6333 instead |
| `Embedder not reachable` | TEI still starting (first run downloads the model) or not on 8080 |
| `Registry: unavailable` | Postgres down or tables missing (step 2); retrieval still runs, active set from Qdrant |
| `Graph: none` / `degraded` | `NEO4J_PASSWORD` not in the env file, or Neo4j down; vector-only results |
| Every score very low | TEI is not serving BAAI/bge-m3, or the index was built with another model |
| `address already in use` on 8080 | serving now uses 8100; an old serving process may still hold 8080 |
