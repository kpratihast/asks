# Ask IISc

Ask questions about IISc courses, admissions and academic policies. Answers cite the indexed pages. Public chat can run without sign-in. Student and faculty documents require the configured role.

## Install and run

Use Python 3.11 or 3.12, Git and a model server that exposes the OpenAI-compatible `/v1/models` and `/v1/chat/completions` endpoints. The app uses Qdrant in a local folder by default. The first index build downloads a sentence embedding model.

```bash
git clone https://github.com/kpratihast/asks.git
cd asks
git switch instiwide
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For a CPU-only Linux install, install PyTorch from the [CPU wheel index](https://download.pytorch.org/whl/cpu) before installing the requirements. Start a compatible model server and set its URL and model name. See [model setup](MODELS.md) for the Mac and CUDA launch scripts.

```bash
export VLLM_URL=http://127.0.0.1:8000/v1
export SERVED_MODEL=askiisc
export ASKIISC_DATA_DIR="$PWD/data/public"
mkdir -p .local
printf '%s\n' '{"documents":[]}' > .local/public-documents.json
export ASKIISC_PRIVATE_MANIFEST="$PWD/.local/public-documents.json"
./run-local.sh --prepare --auth off
./run-local.sh --auth off
```

Replace `askiisc` with the model ID your server reports. Open <http://127.0.0.1:8100>. If the model is unavailable, the app can return matching source excerpts.

## Add sources

Add reviewed URLs to [sources.yaml](sources.yaml). Enable departments with `ASKIISC_SOURCE_DEPARTMENTS=CDS,CSA,ECE`. Add a new department code and its names to [config/departments.yaml](config/departments.yaml). Use a fresh data directory to test the resulting index before switching the app. [Department and cluster setup](DEPARTMENTS.md)

To fetch linked course or staff pages, configure `linked_pages` in `sources.yaml` with approved hosts, source types and a page limit. Fetching runs at ingestion time. A linked page can supply an email only if it actually publishes the address and the evidence identifies the instructor. [Developer guide](dev.md)

## Sign-in and private documents

Provide the matching private documents and reviewed text, configure SMTP on the server, and start with `--auth on`. Unset `ASKIISC_PRIVATE_MANIFEST` to use the checked-in catalogue. See [document ingestion](DOCUMENT_INGESTION.md) and [server settings](config/email-auth.env.example).

To test professor access, start the backend with an exact IISc test address:

```bash
AUTH_PROFESSOR_TEST_EMAILS=your-test-address@iisc.ac.in ./run-local.sh --auth on
```

The default override is empty. The test address must still complete OTP verification. The verified faculty directory controls normal professor access. Addresses found in documents are answer evidence and have no effect on roles.

## Compare the two retrieval modes

Send the same question twice to `POST /v1/ask` with `"mode":"vector"` and `"mode":"full"`. Both modes use the same index and model. Vector RAG ranks passages by relevance. Graph RAG can also follow a source link or an indexed relationship to find a passage that ranks poorly on its own.

For example, a schedule row identifies an instructor, a course page links to the instructor’s profile, and the profile publishes an email. The answer needs the passages that establish the connection. Graph RAG keeps them together and checks that the answer cites them.

Run `python -m pytest -q` for the regression suite. See [dev.md](dev.md) for the code map and evaluation steps. See [AWS deployment](AWS_DEPLOYMENT.md) for packaging and release commands.


## Team ASKS

| Member | Role |
|---|---|
| Kshitij Pratihast | API contracts, serving, UI, department routing |
| Aditya Barodia | Evaluation, held-out sets, freshness |
| Smit Patel | Ingestion pipeline, crawling, stores |
| Sejadri Banik | Retrieval, GPU serving (vLLM), cluster deployment |
