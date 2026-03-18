#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sglang_benchmark_common.sh"

MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-0.6B}"
VLLM_EXECUTABLE="${VLLM_EXECUTABLE:-vllm}"
VLLM_PORT="${VLLM_PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-256}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
DISABLE_LOG_STATS="${DISABLE_LOG_STATS:-1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"

RESOLVED_MODEL_PATH="$(resolve_model_dir "$MODEL_PATH")"
export_default_server_env

cmd=(
    "$VLLM_EXECUTABLE" serve
    --model "$RESOLVED_MODEL_PATH"
    --host "$HOST"
    --port "$VLLM_PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE"
)

if [ "$TRUST_REMOTE_CODE" = "1" ]; then
    cmd+=(--trust-remote-code)
fi
if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
    cmd+=(--enable-prefix-caching)
else
    cmd+=(--no-enable-prefix-caching)
fi
if [ "$DISABLE_LOG_STATS" = "1" ]; then
    cmd+=(--disable-log-stats)
fi
if [ -n "$SERVED_MODEL_NAME" ]; then
    cmd+=(--served-model-name "$SERVED_MODEL_NAME")
fi

print_section "vLLM Server Startup"
echo "模型目录: $RESOLVED_MODEL_PATH"
echo "监听地址: ${HOST}:${VLLM_PORT}"
echo "TP: $TENSOR_PARALLEL_SIZE"
echo "GLOO 网卡: ${GLOO_SOCKET_IFNAME:-<未设置>}"
echo "TP 网卡: ${TP_SOCKET_IFNAME:-<未设置>}"
echo "命令: ${cmd[*]}"
echo ""

exec "${cmd[@]}"
