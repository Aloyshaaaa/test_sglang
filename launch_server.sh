#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sglang_benchmark_common.sh"

MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-0.6B}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
SGLANG_PORT="${SGLANG_PORT:-30001}"
HOST="${HOST:-0.0.0.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-256}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-1}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"

RESOLVED_MODEL_PATH="$(resolve_model_dir "$MODEL_PATH")"
export_default_server_env
run_sglang_runtime_preflight "$PYTHON_EXECUTABLE"

cmd=(
    "$PYTHON_EXECUTABLE" -m sglang.launch_server
    --model "$RESOLVED_MODEL_PATH"
    --port "$SGLANG_PORT"
    --host "$HOST"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --attention-backend "$ATTENTION_BACKEND"
)

if [ "$TRUST_REMOTE_CODE" = "1" ]; then
    cmd+=(--trust-remote-code)
fi
if [ "$DISABLE_RADIX_CACHE" = "1" ]; then
    cmd+=(--disable-radix-cache)
fi
if [ "$DISABLE_OVERLAP_SCHEDULE" = "1" ]; then
    cmd+=(--disable-overlap-schedule)
fi
if [ "$DISABLE_CUDA_GRAPH" = "1" ]; then
    cmd+=(--disable-cuda-graph)
fi
if [ -n "$MAX_PREFILL_TOKENS" ]; then
    cmd+=(--max-prefill-tokens "$MAX_PREFILL_TOKENS")
fi

print_section "SGLang Server Startup"
echo "模型目录: $RESOLVED_MODEL_PATH"
echo "监听地址: ${HOST}:${SGLANG_PORT}"
echo "TP: $TENSOR_PARALLEL_SIZE"
echo "禁用 CUDA Graph: $DISABLE_CUDA_GRAPH"
echo "GLOO 网卡: ${GLOO_SOCKET_IFNAME:-<未设置>}"
echo "TP 网卡: ${TP_SOCKET_IFNAME:-<未设置>}"
echo "命令: ${cmd[*]}"
echo ""

exec "${cmd[@]}"
