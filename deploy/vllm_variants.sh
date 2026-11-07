#!/usr/bin/env bash
# The three model variants of PRD FR-8 / experiment E1, one at a time on the RTX 4090 (24 GB), plus the
# B1 systems baseline. Run ONE of these, then start the serving tier with the matching SERVED_MODEL.
# Flags change between vLLM releases: check `vllm serve --help` for your pinned version and record it.
# --gpu-memory-utilization leaves room on the same GPU for the reranker (and TEI if it shares the GPU).
set -euo pipefail
VARIANT=${1:?usage: vllm_variants.sh 3b|8b|8b-awq|b1-3b}
MODEL_HOST=${ASKIISC_VLLM_HOST:-127.0.0.1}
case "$MODEL_HOST" in
  127.0.0.1|::1|localhost) ;;
  *) : "${VLLM_API_KEY:?Set VLLM_API_KEY before binding the model outside loopback}" ;;
esac
COMMON=(--host "$MODEL_HOST" --port 8000 --max-model-len 8192)
case "$VARIANT" in
  3b)     exec vllm serve meta-llama/Llama-3.2-3B-Instruct --served-model-name llama-3.2-3b \
                 --gpu-memory-utilization 0.80 --enable-prefix-caching "${COMMON[@]}" ;;
  8b)     exec vllm serve meta-llama/Llama-3.1-8B-Instruct --served-model-name llama-3.1-8b --dtype bfloat16 \
                 --gpu-memory-utilization 0.88 --enable-prefix-caching "${COMMON[@]}" ;;
  8b-awq) exec vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --served-model-name llama-3.1-8b-awq \
                 --quantization awq_marlin --gpu-memory-utilization 0.80 --enable-prefix-caching "${COMMON[@]}" ;;
  # B1 (PRD 7.2): serving optimizations off -> one sequence at a time (no batching), no prefix cache
  b1-3b)  exec vllm serve meta-llama/Llama-3.2-3B-Instruct --served-model-name llama-3.2-3b \
                 --gpu-memory-utilization 0.80 --no-enable-prefix-caching --max-num-seqs 1 "${COMMON[@]}" ;;
  *) echo "unknown variant $VARIANT" >&2; exit 2 ;;
esac
