# OpenRouter local checks

The serving adapter accepts `MODEL_API_KEY` or `OPENROUTER_API_KEY` for Bearer authentication.
`MODEL_REASONING` supplies the provider's reasoning object, for example `{"enabled":true}`.
Model discovery and readiness use the same authentication as generation. Stream errors trigger the
existing source-excerpt fallback. The key is omitted from settings representations and redacted from
upstream error messages.

The existing `config/local.json` still selects the local MLX model. Separate hosted profiles are
`config/openrouter.json` for Liquid and `config/openrouter-gemma.json` for Gemma. Both preserve the
current local retrieval backend and clear the MLX-specific chat template options.

## Reproduce the check

From the project directory, using the Python environment with the project dependencies installed:

```bash
python tools/check_openrouter.py \
  --model google/gemma-4-26b-a4b-it:free \
  --out output/openrouter/gemma-recheck \
  --serve --port 8101
```

The runner prompts for the key with input hidden. It also accepts `OPENROUTER_API_KEY` from the
environment. It never writes the key to the profile or result files. Use a fresh output directory
for each run. Stopping the process ends the local chat server and releases its in-memory key.

The runner verifies zero input/output pricing, tries the strawberry prompt with reasoning enabled,
and tests the real local application against a consistent copy of the frozen index. A successful
smoke check leads to eight questions in both vector and graph modes. During an upstream outage,
only one question is attempted in both modes to verify fallback without repeatedly hitting a busy
provider. It does not switch to paid models. Citation regeneration is disabled for this bounded check.

The test uses a 4,096-token context limit and 1,024 output tokens, including reasoning, with the same
six-source cap. These settings differ from the earlier local Qwen run's 512-token output limit.

## Results on 29 October 2026

| Model | Direct inference | Upstream provider | Local application |
|---|---|---|---|
| `liquid/lfm-2.5-2.6b:free` | HTTP 429 after bounded retries | Liquid shared free pool | Both RAG modes returned source excerpts |
| `google/gemma-4-26b-a4b-it:free` | HTTP 429 after bounded retries | Google AI Studio shared free pool | Both RAG modes returned source excerpts |

The key authenticated successfully. OpenRouter reported 50 free requests remaining before and after
these rejected requests. No successful model answer was obtained, including for the strawberry
prompt, so this run supplies no answer-accuracy or model-speed comparison. The proposed eight-question
evaluation was not run in full.

The browser check confirmed that Gemma is selected at `http://127.0.0.1:8101` and that an unavailable
model displays the fallback message and source cards. The key is held by the server, not the browser.

Local regression suite: **152 passed, 17 skipped**. The skipped tests require the external PostgreSQL
integration environment. The new tests cover authenticated requests and catalog calls, reasoning
options, HTTP 429 fallback, secret redaction, mid-stream errors and unchanged unauthenticated local calls.

Raw sanitized results and response excerpts are saved in:

- `output/openrouter/2026-10-29-liquid/`
- `output/openrouter/2026-10-29-gemma/`
