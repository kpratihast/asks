# GPU assessment setup

**GPU access:** none is provided by the blue team. Deploy on your own CUDA GPU (24 GB, e.g. RTX 4090, for Llama 3.1 8B BF16) following this runbook.

**Private reporting:** report Major security or privacy findings privately to the blue team (direct message on Teams; contacts in pinned issue #1) and the mentors, not in a public issue.

Record the GPU, driver, CUDA, Python, vLLM version and model revision with each result. Use Python 3.12 and the model installation guidance in SERVING.md. Accept model licences before downloading weights.

The standalone backend avoids a dependency on the external ingestion team's databases.

~~~bash
export ASKIISC_BACKEND=stub
export ASKIISC_DATA_DIR="$PWD/data/assessment"
export ASKIISC_EMBED_MODEL=BAAI/bge-m3
export ASKIISC_EMBED_DIM=1024
export ASKIISC_RERANKER=crossencoder
export ASKIISC_RERANK_MODEL=BAAI/bge-reranker-base
export ASKIISC_RERANK_MIN=0.69
export ASKIISC_SOURCE_DEPARTMENTS=CDS,CSA,ECE
python -m askiisc_retrieval.cli fetch
python -m askiisc_retrieval.cli ingest
~~~

The cutoff reproduces the declared baseline. It was calibrated on earlier CDS data with bge-reranker-base. Evaluate current-corpus false refusals before claiming a campus-wide calibration.

Generate private random guest, student, faculty and admin keys using SERVING.md. Point SERVING_KEYS_FILE to that private file and set PUBLIC_CHAT=0. Start the chosen model using deploy/vllm_variants.sh in a separate terminal.

~~~bash
export PUBLIC_CHAT=0
export VLLM_URL=http://127.0.0.1:8000/v1
export SERVED_MODEL=llama-3.1-8b
python -m askiisc_serving --host 127.0.0.1 --port 8100
~~~

B0 additionally needs ALLOW_NO_RETRIEVAL=1 and an admin evaluation credential. Disable it for ordinary traffic.

Measure ingestion freshness, stale-version exclusion, graph and generation outages, refusal rate, GPU hours and cost on the assessment hardware. A refusal result below the PRD's 100% invariant fails that invariant. Historical report figures require fresh measurements before describing this branch.
