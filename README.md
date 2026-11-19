# Ask IISc

Ask questions about IISc courses, admissions and academic policies. Answers cite the indexed pages, and every cited claim is checked against its source. Every request needs a credential: a guest key, an IISc email sign-in, or a student/faculty key. Student and faculty documents require the configured role.

## Install and run

Use Python 3.12 (the version CI tests and the EC2 host runs; other versions are untested), Git and a model server that exposes the OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints. The app uses Qdrant in a local folder by default. The first index build downloads a sentence embedding model.

```bash
git clone https://github.com/kpratihast/asks.git
cd asks
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a CPU-only Linux install, install PyTorch from the [CPU wheel index](https://download.pytorch.org/whl/cpu) before installing the requirements. Start a compatible model server and set its URL and model name. See [model setup](MODELS.md) for the Mac and CUDA launch scripts.

```bash
export VLLM_URL=http://127.0.0.1:8000/v1
export SERVED_MODEL=askiisc
export ASKIISC_DATA_DIR="$PWD/data/public"
export ASKIISC_SOURCE_DEPARTMENTS=CDS,CSA,ECE
mkdir -p .local
printf '%s\n' '{"documents":[]}' > .local/public-documents.json
export ASKIISC_PRIVATE_MANIFEST="$PWD/.local/public-documents.json"
./run-local.sh --prepare --auth off
./run-local.sh --auth off
```

Replace `askiisc` with the model ID your server reports. Open <http://127.0.0.1:8100>. If the model is unavailable, the app can return matching source excerpts.

**Access (FR-6).** With sign-in off, the launcher creates a private guest key in `.local/serving_keys.yaml` (git-ignored, mode 600) and prints its path. Paste the key into the chat page's **API access** box, or send `Authorization: Bearer <key>`. Requests without a valid key get HTTP 401. To use your own keys, set `SERVING_KEYS_FILE` (see [serving_keys.example.yaml](serving_keys.example.yaml)). `PUBLIC_CHAT=1` allows anonymous guests for a demo only; it is off by default and logs a warning.

**Answer checks (FR-11).** The launcher runs the cross-encoder reranker with the 0.69 abstention gate on, as in the evaluated configuration. After generation, every number and specific name or term in a cited claim must appear in the passage it cites; unsupported claims are removed, a reply that says the sources lack the answer is returned as a refusal, and a reply that both answers and says it found nothing returns source excerpts instead. Questions that ask the assistant to use its general knowledge or not to cite sources are refused before the model is called.

**Metadata and degraded mode.** Every response's `trace.model_info` records the served checkpoint, precision (for example `bf16`, `awq-4bit`, or Ollama's `Q4_K_M`), digest when available, and the model id the server reported for that answer (FR-8). If `graph.json` is missing or unreadable, the app starts vector-only and marks graph-mode answers `degraded` with the reason; `/readyz` reports it (FR-9). `ASKIISC_GRAPH_TIMEOUT_S` sets the graph time limit (default 0.5 s). A request's `programme` filter (for example `M.Tech.`, `M.Tech. (Research)`, `PhD`) drops sources that name a different programme; `trace.programme_filter` shows whether it was applied. `/readyz` gives anonymous callers liveness only; admission counters, limits, tenants and model details need a key (an admin key when guests are allowed), as for `/metrics`.

**Index rebuilds (FR-1, FR-2, FR-4).** The crawl follows each page's own content links, not the shared navigation, header or footer, and downloads linked PDFs within the depth limit for the PDF extractor. A page or PDF that fails extraction is listed with its URL and reason in `data/extraction_failures.json` and the rest is indexed. `./run-local.sh --rebuild` and `python -m askiisc_retrieval.cli ingest` add the new version next to the old one and switch the snapshot only after the build succeeds (graph, chunk list and snapshot are written atomically, snapshot last), so a failed or killed rebuild (error, Ctrl-C, out-of-memory kill) keeps the previous index serving. Restart the app after a rebuild to load the new graph; until then graph-mode answers are vector-only and marked `degraded`.

General admissions and fee questions use IISc sources. Name a department when asking about its requirements, faculty or labs.

## What the quickstart indexes

| Source | Coverage |
| --- | --- |
| IISc admissions | Research and course admissions, CFTI entry, FAQ, postgraduate application fees, regular PhD and M.Tech./M.Des. fee tables, and selected scholarship and fellowship rates. |
| Scheme of Instruction | Common guidance and the chapters for CDS, CSA and ECE in the August 2026 edition. Citations include the PDF page number. |
| Department sites | CDS course and research pages. CSA and ECE admissions, faculty and research or lab listings. |

The command above enables all three departments. If you omit `ASKIISC_SOURCE_DEPARTMENTS`, the bundled configuration selects CDS and shared IISc sources. The active index reflects the last build. To add these sources to an existing index, stop the app and run `./run-local.sh --rebuild` with the same settings. It fetches missing configured sources and reuses cached downloads. Use `--refresh` to re-check the configured sources: pages are requested conditionally (ETag/Last-Modified) and stored as a new version only when their content changed. For the 24-hour scheduled refresh of listed, crawled and linked sources and the maintainer trigger, see [source refresh](INGESTION_REFRESH.md).

## Add sources

Add reviewed URLs to [sources.yaml](sources.yaml). Enable departments with `ASKIISC_SOURCE_DEPARTMENTS=CDS,CSA,ECE`. Add a new department code and its names to [config/departments.yaml](config/departments.yaml). Use a fresh data directory to test the resulting index before switching the app. [Department and cluster setup](DEPARTMENTS.md)

When replacing the Scheme of Instruction with a new edition, update the PDF URL, title and department page ranges together. [Official IISc student resources](https://www.iisc.ac.in/about/student-corner/)

To fetch linked course or staff pages, configure `linked_pages` in `sources.yaml` with approved hosts, source types and a page limit. Fetching runs at ingestion time. A linked page can supply an email only if it actually publishes the address and the evidence identifies the instructor. [Developer guide](dev.md)

## Sign-in and private documents

Provide the matching private documents and reviewed text, configure SMTP on the server, and start with `--auth on`. Unset `ASKIISC_PRIVATE_MANIFEST` to use the checked-in catalogue. See [document ingestion](DOCUMENT_INGESTION.md) and [server settings](config/email-auth.env.example).

To test professor access, start the backend with an exact IISc test address:

```bash
AUTH_PROFESSOR_TEST_EMAILS=your-test-address@iisc.ac.in ./run-local.sh --auth on
```

The default override is empty. The test address must still complete OTP verification. The verified faculty directory controls normal professor access. Addresses found in documents are answer evidence and have no effect on roles.

Use a loopback model for restricted answers, or set `ASKIISC_TRUSTED_MODEL_URL` to the exact URL of a private model service you operate. The app returns authorized excerpts and document links when an untrusted model is selected. OpenRouter handles public evidence only.

## Compare the two retrieval modes

Send the same question twice to `POST /v1/ask` with `"mode":"vector"` and `"mode":"full"`. Both modes use the same index and model. Vector RAG ranks passages by relevance. Graph RAG can also follow a source link or an indexed relationship to find a passage that ranks poorly on its own.

An instructor email lookup may need a course schedule to identify the instructor and a faculty page to find the published address. Graph RAG can connect the two and require citations for both. A fee or credit question answered by one table can work in either mode.

Run `python -m pytest -q` for the regression suite. See [dev.md](dev.md) for the code map and evaluation steps. See [AWS deployment](AWS_DEPLOYMENT.md) for packaging and release commands.
