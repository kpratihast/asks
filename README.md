# Local Llama chat

Use `./model.sh 3b`, `./model.sh 8b`, or `./model.sh 8b-quantized` to switch the inference model. See [MODELS.md](MODELS.md) for precision choices and lifecycle commands.

AWS provisioning, releases and rollback: [deployment guide](AWS_DEPLOYMENT.md). AWS uses its own configuration and entry point.

For the working Mac setup and browser chat UI, see [LOCAL_CHAT.md](LOCAL_CHAT.md). Run `./run-local.sh` and open http://127.0.0.1:8100.

# Ask IISc: retrieval (dev slice)

Connected course/instructor → timetable answers: [design and validation](CONNECTED_QUESTIONS.md), [local-model comparison](output/connected-comparison/2026-10-29/REPORT.md).

Graph evidence selection and validation: [EVIDENCE_SELECTION.md](EVIDENCE_SELECTION.md). Latest [before/after evaluation](output/graph-comparison/2026-10-28-evidence-fix/BEFORE_AFTER.md).

**Serving tier (answers with citations): [SERVING.md](SERVING.md).** **Integrating retrieval? [INTEGRATION.md](INTEGRATION.md)**: running on the ingestion stores, the HTTP API (`serve`), how generation should use the results, and known limitations.

Retrieval for Ask IISc, developed on a laptop against a fixed snapshot of CDS pages.
Question + user scopes in, ranked citable passages out, from one active snapshot.

## Setup (Windows, Anaconda, CPU only)

```powershell
conda create -n askiisc python=3.11 -y
conda activate askiisc
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q          # 15 tests, ~2 s, no network or model needed
```

## Qdrant in Docker (recommended once set up)

The code runs against either backend: **local mode** (a folder, default) or the **Qdrant server**
in Docker when `QDRANT_URL` is set. Use the server for any latency number you report (NFR-4):
local mode has no payload indexes and allows only one process at a time.

1. Cap Docker's memory so it cannot starve BGE-M3. Create `C:\Users\<you>\.wslconfig`:
   ```ini
   [wsl2]
   memory=3GB
   swap=2GB
   ```
   then run `wsl --shutdown` in PowerShell and restart Docker Desktop.
2. Start Qdrant and point the code at it (set the variable in every new terminal):
   ```powershell
   docker compose up -d qdrant
   $env:QDRANT_URL = "http://127.0.0.1:6333"   # not "localhost": on Windows that adds ~2 s per request
   python -m askiisc_retrieval.cli ingest
   ```
   Dashboard: http://localhost:6333/dashboard. Stop with `docker compose stop`; data persists.
3. To go back to local mode: `Remove-Item Env:QDRANT_URL`.

## Run

```powershell
python -m askiisc_retrieval.cli fetch       # saves the pages in sources.yaml to data/raw/ (once)
python -m askiisc_retrieval.cli ingest      # parse, chunk, embed (BGE-M3, ~2.3 GB first time), index
python -m askiisc_retrieval.cli ask "Who teaches DS 252?"
python -m askiisc_retrieval.cli ask "lab access cards" --scope public student
python -m askiisc_retrieval.cli eval --k 5 --out eval_baseline.csv
python -m askiisc_retrieval.cli show-chunks --grep "DS 252"     # see exactly what was indexed
python -m askiisc_retrieval.cli ask "Who teaches DS 252?" --context   # the [S1]-labelled block for the generator
python -m askiisc_retrieval.cli explain "question" https://page/url/   # where a page ranks: vector rank + rerank
python -m askiisc_retrieval.cli rebuild-graph   # after graph-rule changes: rebuild edges, no re-embedding
python -m askiisc_retrieval.cli serve --port 8090   # HTTP: POST /retrieve, POST /verify, GET /health
python -m askiisc_retrieval.cli check-ingestion     # (ASKIISC_BACKEND=ingestion) gaps in the ingestion output
```

Close browser tabs before `ingest`: BGE-M3 needs about 3 GB of free RAM while embedding.
`ask`/`eval` also load the reranker (~1.1 GB more; downloaded on first use). `--no-rerank` skips it,
`$env:ASKIISC_RERANKER="none"` disables it entirely.
To check the plumbing without downloading the model: `$env:ASKIISC_EMBEDDER="hash"` before
`ingest`/`ask` (meaningless results; remove it with `Remove-Item Env:ASKIISC_EMBEDDER` and re-ingest).

## Corpus = listed pages + a bounded crawl of the CDS site

`sources.yaml` lists the 9 original pages (frozen) and a `crawl:` block: breadth-first from the homepage,
2 clicks deep, max 200 pages, HTML only, robots.txt respected, lab subdomains as leaves. The corpus is set
by the site's links, never by a question set. `fetch` runs the crawl and reuses any page already saved.

## Held-out test set (`test_questions.yaml`)

Aditya's questions, converted; gold labels exactly as he wrote them (my notes are marked `suggestion:`).
They are never used for tuning, choosing pages or changing labels: run once on a fixed configuration.
```powershell
python -m askiisc_retrieval.cli eval --k 5 --questions test_questions.yaml --out test_v1.csv
```
`gold_found` shows how many of a question's cited sources were retrieved (multi-source questions).
Check new pages with `ingest` output and `diagnose` BEFORE running the test, never in response to it.

## What each file does

| File | Role |
|---|---|
| `contract.py` | **The interface.** `ChunkPayload` (what ingestion writes to Qdrant) and `RetrievalResult` (what serving gets). Share with the team. |
| `retriever.py` | The retriever: filters, course-code pass, term logic, ranking, graph expansion, fallback, audit |
| `query_analysis.py` | Course codes and explicit term ("Aug-Dec 2025", "January 2026", "last semester") from the question |
| `stores.py` | Embedders (BGE-M3 local, TEI, fake), Qdrant (local folder or Docker server), JSON graph store (Neo4j stand-in) |
| `docker-compose.yml` | Qdrant now; Neo4j (profile `graph`) in phase 2 |
| `snapshot.py` | Active snapshot provider (JSON now; PostgreSQL manifest later) |
| `parse.py`, `chunking.py`, `graph_rules.py`, `ingest_stub.py`, `fetch.py` | **Stub ingestion**, replaced by the ingestion team's pipeline |
| `eval_recall.py`, `dev_questions.yaml` | Hit@k, MRR, leak count, latency on the dev set |
| `fixtures/` | Synthetic superseded and student-only documents that test the filters on every run |

## Retrieval design (current)

1. **Snapshot + scope filter** inside the Qdrant search: only active `version_id`s, only the user's scopes,
   only the pinned embedding model (a mismatch raises, FR-5).
2. **Course-code pass**: if the question names a course, a second search restricted to chunks tagged with it.
   **People pass**: if the question asks for a person or lab ("who", "whom", "professor", "faculty",
   "advisor", "lab", "group"), a search restricted to `doc_type: people` chunks (profiles with research
   interests). Its top 5 (`ASKIISC_PEOPLE_K`, 0 = off) always go to the reranker, which decides: short
   profile chunks are otherwise cut by long general pages before they are ever scored.
3. **Term logic**: schedule rows carry `academic_term`. Explicit term in the question -> only that term.
   Otherwise one row per course: the latest term not after the current one. (Version filtering cannot do
   this: an old term's schedule is a valid page, not a superseded version.)
4. **Ranking**: chunks whose heading names the asked course first, then by similarity.
5. **Graph expansion** (2 hops, only through Course/Person nodes, same snapshot and scope filters):
   - Seeds: courses and people named in the question (names resolved with the same alias keys as the
     graph), plus courses/people in the top-3 passages.
   - Term rules: with no term in the question, each course's latest term up to now is used. With a term
     ("Jan-Apr 2026", "this semester"), only that term is cited, but current facts may bridge hops
     ("the DS 252 instructor" is resolved as of now). Bridge edges are never cited.
   - Edges (`graph_rules.py`, `rules-v2`): Course -TAUGHT_BY/OFFERED_IN/HELD_IN-> from schedule rows,
     Course -REQUIRES-> Course from the catalogue, Person -WORKS_ON-> Topic from a profile's "Research
     Interests" section, Person -HEADS-> Lab from lab listings ("Faculty: ..."). After changing the rules,
     `rebuild-graph` rebuilds `graph.json` from the saved chunks without re-embedding.
   - For person/lab questions without a course code, a person's own facts (their lab) come before the
     courses they taught.
   - Adds up to 3 supporting chunks, those linked to more of the question's entities first. Edges are
     never cited. Any error or a 0.5 s timeout -> vector-only result with `degraded=True` (FR-9, E6).
6. **Reranking (FR-13)**: a cross-encoder (`bge-reranker-base` on the laptop, `bge-reranker-v2-m3` on the
   GPU) scores the top 10 candidates and the graph additions against the question. Order = asked-course
   match first, then reranker score. Schedule rows are scored as a sentence ("In the Aug-Dec 2024 term,
   DS 221 ... was taught by ...", `rerank_text.py`): cross-encoders rank terse key-value rows too low.
   Stored and cited text is unchanged. A reranker failure keeps vector order and never fails the request.
7. **Abstention signal (FR-11)**: `rerank_top` (best reranker score) and `evidence_sufficient`
   (`rerank_top >= RERANK_MIN`). Serving abstains when it is False. `eval` prints how well vector
   similarity and the reranker each separate answerable from must-abstain questions, and suggests a
   threshold. Default 0.45 (first dev calibration). On the crawl corpus the dev set separates at 0.584 /
   0.797, so use 0.69: `$env:ASKIISC_RERANK_MIN = "0.69"` (`"none"` switches the signal off). Never
   calibrate on the held-out test questions.
8. **Audit**: every returned passage is re-checked against snapshot and scope; violations are dropped and counted.
9. **Citations** (`citations.py`): `result.citations` has one entry per page section ([S1], [S2], ... in rank
   order) with URL, title, section, version and an excerpt; each passage carries its `marker`. For serving:
   `format_context(result)` builds the labelled prompt block (lowest-ranked sources dropped first under a
   length budget), `check_answer(answer, result)` reports markers used, unknown (hallucinated) markers and
   uncited answers, and `render_sources(result, used)` gives the user-facing source list. Only audited
   passages are cited, so a citation can never point to a superseded or out-of-scope document.
   `eval` reports `cite_precision`: the share of cited pages in the top k that are gold pages.

## Open items

- **Abstention threshold** rests on few dev must-abstain questions (q08, q12, q17, q26). Add more
  self-written ones; Aditya's questions stay test-only.
- **Research-area / lab facts** come from a stand-in rule extractor; the ingestion team's extractor (FR-2)
  should produce the same WORKS_ON / HEADS edges. Topic nodes are leaves (no Person -> Topic -> Person hops yet).
- **Neo4j adapter** must implement the same term rules as `JsonGraphStore.expand` (current-as-of-now,
  evidence vs bridge edges) in Cypher.

- **Pages whose content is in linked documents** (e.g. M.Tech requirements: two links to embedded docs)
  are reported as NOT INDEXED by `ingest`. The ingestion team's crawler should follow those links.

- Verify the gold labels in `dev_questions.yaml` against the fetched pages (`verified: false` entries).
- Calibrate the abstention signal: compare `top_score` of answerable vs unanswerable questions (q17), or add a reranker (FR-13).
- Phase 2: Neo4j adapter behind the same `GraphStore` interface; PostgreSQL snapshot provider.
- Agree `ChunkPayload` (especially per-section `academic_term` and `course_codes`) with the ingestion team.


## Email login and access levels

The chat supports IISc email OTP login for students and CDS professors. Public documents need no login. Student and faculty attachments are classified during ingestion and filtered before vector retrieval, graph expansion, prompts, citations, and file downloads. See [EMAIL_ACCESS.md](EMAIL_ACCESS.md) for email setup, document imports, access rules, and validation.

Automatic AWS deployments from GitHub: see [GITHUB_DEPLOYMENT.md](deploy/aws/GITHUB_DEPLOYMENT.md).

Load-control diagnosis, effective AWS limits and a reproducible C=20 / Q=10 check: [LOAD_TESTING.md](LOAD_TESTING.md).

Private document imports: [single-command student/professor ingestion](DOCUMENT_INGESTION.md). Security controls and assessment limitations: [security review](SECURITY_REVIEW.md).

## Adding IISc departments

The default source selection is CDS. Use [the department rollout guide](DEPARTMENTS.md) to build a separate CDS/CSA/ECE pilot index, add a department, or point the cluster retrieval service at a wider corpus.
