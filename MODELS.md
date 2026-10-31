# Switching the local Llama server

Run these commands from `askiisc-retrieval`:

```bash
./model.sh 3b               # Llama 3.2 3B, BF16
./model.sh 8b               # Llama 3.1 8B, BF16
./model.sh 8b-quantized     # Llama 3.1 8B, 4-bit MLX weights
```

`--model` is also accepted: `./model.sh --model 8b-quantized`.

Your earlier local 3B setup uses 4-bit weights. To select that same configuration:

```bash
./model.sh 3b-quantized
```

Every choice serves the alias `askiisc` on `http://127.0.0.1:8000`, with a 4,096-token
context, prefix caching, and one active sequence. The chat application on port 8100
can stay running. Requests made during the switch may return source excerpts until
the new model is ready.

## What each parameter selects

| Parameter | Checkpoint | Precision | Default Metal memory fraction |
|---|---|---|---|
| `3b` | `mlx-community/Llama-3.2-3B-Instruct` | BF16 | 0.45 |
| `8b` | `mlx-community/Meta-Llama-3.1-8B-Instruct-bf16` | BF16 | 0.85 |
| `8b-quantized` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | 4-bit | 0.45 |
| `3b-quantized` | `mlx-community/Llama-3.2-3B-Instruct-4bit` | 4-bit | 0.25 |

The BF16 checkpoints are unquantized. The quantized profiles use MLX's 4-bit checkpoint
format on Apple Silicon. The CUDA deployment's AWQ/Marlin profile remains in
`deploy/vllm_variants.sh`; it is not the format used by this Mac switcher.

The BF16 8B weights alone occupy about 16 GB. This is a tight configuration on a 24 GB
Mac: close other memory-heavy applications first, and use `8b-quantized` for routine
local work. Memory fractions are starting budgets, not guarantees that every workload
will fit. Failed model startup reports the log location; no different-precision model
is silently substituted.

Checkpoint sources: [3B BF16](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct),
[8B BF16](https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-bf16),
[8B 4-bit](https://huggingface.co/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit).

## Stop, inspect, restart

```bash
./model.sh stop
./model.sh status
./model.sh logs
./model.sh 3b-quantized --restart
```

Repeating the currently ready profile does nothing unless `--restart` is supplied.
Use `tmux attach -t askiisc-model` to see the server pane; Ctrl-b then d detaches.
The script can also stop the earlier server in `askiisc-cds-local:vllm`, preserving
that session's separate API window. New model launches use `askiisc-model:vllm`.
It refuses to interrupt a port listener or pane process outside these managed servers.

Before a switch, the requested checkpoint is downloaded/cached and checked for missing
weight files and its expected quantization setting. Download failure leaves the old
server running. The old server then receives Ctrl-C, and the replacement starts only
after graceful shutdown and release of port 8000. Simultaneous switch commands are
rejected using a process lock. Model files remain cached after stopping.

The script waits for the exact requested checkpoint to appear at `/v1/models` before
reporting readiness. Logs and selected-profile information are under `.local/model/`.
If startup fails after the old server has stopped, inspect `./model.sh logs` and select
another profile. A readiness timeout leaves the loading server available for inspection;
`./model.sh stop` stops it.

## Options

```bash
./model.sh 8b --dry-run                  # show the command; no downloads or service changes
./model.sh 3b-quantized --offline        # require a complete cached checkpoint
./model.sh 8b-quantized --timeout 900     # allow 15 minutes for model loading
./model.sh 8b-quantized --memory-fraction 0.5
./model.sh 3b --vllm-env "$HOME/.venv-vllm-metal"
```

The default inference environment is `~/.venv-vllm-metal`; `ASKIISC_VLLM_ENV` also
overrides it. The launcher uses `ASKIISC_PYTHON`, a project `.venv`, the saved
`.local/python`, or `python3` to run the controller. The controller itself uses only
Python's standard library plus the `tmux`, `lsof`, and `ps` commands. Checkpoint download
uses `huggingface_hub` from the inference environment. First-time downloads require
network access and sufficient disk space; offline mode uses cached files only.
