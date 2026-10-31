# Local CDS chat with the running MLX model

From `askiisc-retrieval`, run:

```bash
./run-local.sh
```

Open **http://127.0.0.1:8100**. The included UI needs no Node build or external assets.
It supports streamed answers, source excerpts and links, conversation follow-ups,
Stop, New conversation, and optional API-key entry. History and keys stay in page memory
and disappear on reload. Each follow-up is rewritten as a standalone question, then
answered using freshly retrieved sources; previous answers are not evidence.

The launcher reads `config/local.json` and connects to the already running
**mlx-community/Qwen3.5-9B-MLX-4bit** MLX server at `http://127.0.0.1:8000`.
The request model id matches `/v1/models`. The local prompt budget remains 4,096 tokens
with 512 output tokens. `CHAT_TEMPLATE_KWARGS` disables thinking for concise chat output.
The launcher does not start or stop the inference server. Environment variables override
the JSON profile; `ASKIISC_LOCAL_CONFIG` can select another profile file.

The installed instance is running in the `askiisc-chat` tmux session. To view it:

```bash
tmux attach -t askiisc-chat
```

Detach with Ctrl-b, then d. To stop the chat server:

```bash
tmux send-keys -t askiisc-chat C-c
```

Restart with `./run-local.sh` after the previous server exits. The MLX service is managed separately.

## Switching models

To switch back to managed vLLM Llama models, stop the separately managed MLX server first,
then use `./model.sh 3b`, `./model.sh 8b`, or `./model.sh 8b-quantized`.
Set `SERVED_MODEL=auto CHAT_TEMPLATE_KWARGS='{}' ./run-local.sh` when using those models.
The first two select BF16 weights; the last selects 4-bit MLX weights.
Use `./model.sh 3b-quantized` for the original lightweight 3B configuration.
See [MODELS.md](MODELS.md) for stop/status commands and resource requirements.

## Installation and rebuilding

On this Mac, `.local/python` records the existing application Python environment.
`ASKIISC_PYTHON` overrides it; a project `.venv/bin/python` takes precedence over that
saved fallback. For a fresh checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run-local.sh --prepare
./run-local.sh
```

Preparation downloads the nine real public CDS pages in `sources.yaml`, builds
embeddings with **BAAI/bge-small-en-v1.5** (downloaded on first use if uncached), and writes
an isolated Qdrant local index plus graph and source manifest in `data/local/`.
It excludes synthetic fixtures and does not launch Docker. The initial snapshot has
380 chunks and 791 graph edges. Its coverage is limited to those nine pages.

Stop serving before refreshing. Then run:

```bash
./run-local.sh --refresh
./run-local.sh
```

The local profile uses dense retrieval plus the existing rule graph, without a
cross-encoder reranker. Its original BGE-M3 reranker threshold is not applicable;
The model must abstain when retrieved text does not answer the question. Citation checks
validate marker references, not the truth of every claim. Recalibrate on development
questions before treating this profile as a measured quality benchmark.

## Configuration

All launcher defaults can be overridden with environment variables:

```bash
VLLM_URL=http://127.0.0.1:8000 SERVED_MODEL=mlx-community/Qwen3.5-9B-MLX-4bit ./run-local.sh --port 8100
```

`VLLM_URL` accepts a root URL or one ending in `/v1`. `SERVED_MODEL=auto` requires exactly
one served model; otherwise set its exact id. The local profile admits one generation
request at a time to match the local inference profile's single active request limit. Other
consumers of that model share its capacity.

Public chat is enabled only by this launcher (`PUBLIC_CHAT=1`) and runs on loopback.
Requests without a key can retrieve **public documents only**; the `none` baseline
is blocked for these requests. Invalid supplied keys still receive HTTP 401.
Set `PUBLIC_CHAT=0 SERVING_KEYS_FILE=/path/to/keys.yaml` to require keys.
The generic serving entry point remains key-protected by default.

The original ingestion-backed deployment remains documented in `RUN_INTEGRATED.md`.
It continues to use BGE-M3 and server-backed stores; the lightweight local index is
separate. Do not query an index with a different embedding model or dimension.

## Checks

```bash
.venv/bin/python -m pytest -q
curl http://127.0.0.1:8100/readyz
curl http://127.0.0.1:8100/v1/ask -H 'Content-Type: application/json' \
  -d '{"question":"What is DS 252 about?","stream":true}'
```

The PostgreSQL ingestion integration tests require `ASKIISC_TEST_PG_DSN` and the sibling
`ingestion` folder. They are skipped when that database is unavailable.
