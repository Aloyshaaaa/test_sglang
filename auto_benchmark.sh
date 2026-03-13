#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sglang_benchmark_common.sh"

MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-0.6B}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
SGLANG_PORT="${SGLANG_PORT:-30001}"
HOST="${HOST:-0.0.0.0}"
HEALTH_HOST="${HEALTH_HOST:-127.0.0.1}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-2400}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-256}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-1}"
DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-1}"
INPUT_LENGTH="${INPUT_LENGTH:-3000}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-500}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1.0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
REQUEST_RATE="${REQUEST_RATE:-1.6}"
MODEL_TYPE="${MODEL_TYPE:-auto}"
DATASET_NAME="${DATASET_NAME:-}"
DATASET_PATH="${DATASET_PATH:-}"
RANDOM_IMAGE_NUM_IMAGES="${RANDOM_IMAGE_NUM_IMAGES:-1}"
RANDOM_IMAGE_RESOLUTION="${RANDOM_IMAGE_RESOLUTION:-1148x112}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"
USE_EXISTING_SERVER="${USE_EXISTING_SERVER:-0}"
KEEP_SERVER="${KEEP_SERVER:-0}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/results}"
RESULT_DIR="${RESULT_DIR:-${RESULT_ROOT}/benchmark_${TIMESTAMP}}"
SERVER_LOG_FILE="${SERVER_LOG_FILE:-${RESULT_DIR}/server.log}"
BENCH_LOG_FILE="${BENCH_LOG_FILE:-${RESULT_DIR}/bench.log}"
RAW_OUTPUT_FILE="${RAW_OUTPUT_FILE:-${RESULT_DIR}/bench_raw.json}"
CONFIG_FILE="${CONFIG_FILE:-${RESULT_DIR}/run_config.env}"

mkdir -p "$RESULT_DIR"
RESOLVED_MODEL_PATH="$(resolve_model_dir "$MODEL_PATH")"

print_section "SGLang Auto Benchmark"
echo "原始模型路径: $MODEL_PATH"
echo "解析后模型目录: $RESOLVED_MODEL_PATH"
echo "结果目录: $RESULT_DIR"
echo "Server 日志: $SERVER_LOG_FILE"
echo "Bench 日志: $BENCH_LOG_FILE"
echo "Bench 原始 JSON: $RAW_OUTPUT_FILE"
echo ""

cat > "$CONFIG_FILE" <<EOF
MODEL_PATH=$MODEL_PATH
RESOLVED_MODEL_PATH=$RESOLVED_MODEL_PATH
PYTHON_EXECUTABLE=$PYTHON_EXECUTABLE
SGLANG_PORT=$SGLANG_PORT
HOST=$HOST
HEALTH_HOST=$HEALTH_HOST
SERVER_TIMEOUT=$SERVER_TIMEOUT
TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE
MEM_FRACTION_STATIC=$MEM_FRACTION_STATIC
CUDA_GRAPH_MAX_BS=$CUDA_GRAPH_MAX_BS
ATTENTION_BACKEND=$ATTENTION_BACKEND
MAX_PREFILL_TOKENS=$MAX_PREFILL_TOKENS
INPUT_LENGTH=$INPUT_LENGTH
OUTPUT_LENGTH=$OUTPUT_LENGTH
NUM_PROMPTS=$NUM_PROMPTS
WARMUP_REQUESTS=$WARMUP_REQUESTS
RANDOM_RANGE_RATIO=$RANDOM_RANGE_RATIO
MAX_CONCURRENCY=$MAX_CONCURRENCY
REQUEST_RATE=$REQUEST_RATE
MODEL_TYPE=$MODEL_TYPE
DATASET_NAME=$DATASET_NAME
DATASET_PATH=$DATASET_PATH
RANDOM_IMAGE_NUM_IMAGES=$RANDOM_IMAGE_NUM_IMAGES
RANDOM_IMAGE_RESOLUTION=$RANDOM_IMAGE_RESOLUTION
APPLY_CHAT_TEMPLATE=$APPLY_CHAT_TEMPLATE
USE_EXISTING_SERVER=$USE_EXISTING_SERVER
KEEP_SERVER=$KEEP_SERVER
EOF

SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ] && [ "$KEEP_SERVER" != "1" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo ""
        echo "停止脚本拉起的 SGLang Server: $SERVER_PID"
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

if curl -fsS "http://${HEALTH_HOST}:${SGLANG_PORT}/health" >/dev/null 2>&1; then
    echo "检测到 ${HEALTH_HOST}:${SGLANG_PORT} 已有健康的 SGLang 服务。"
    USE_EXISTING_SERVER="1"
fi

if [ "$USE_EXISTING_SERVER" != "1" ]; then
    print_section "Start Server"
    (
        export MODEL_PATH="$RESOLVED_MODEL_PATH"
        export PYTHON_EXECUTABLE
        export SGLANG_PORT
        export HOST
        export TENSOR_PARALLEL_SIZE
        export MEM_FRACTION_STATIC
        export CUDA_GRAPH_MAX_BS
        export ATTENTION_BACKEND
        export MAX_PREFILL_TOKENS
        export TRUST_REMOTE_CODE
        export DISABLE_RADIX_CACHE
        export DISABLE_OVERLAP_SCHEDULE
        exec "${SCRIPT_DIR}/launch_server.sh"
    ) >"$SERVER_LOG_FILE" 2>&1 &
    SERVER_PID="$!"
    echo "Server PID: $SERVER_PID"
    echo "等待 /health 就绪..."
    start_time="$(date +%s)"
    server_ready="0"
    while [ "$(( $(date +%s) - start_time ))" -lt "$SERVER_TIMEOUT" ]; do
        if wait_for_health "$HEALTH_HOST" "$SGLANG_PORT" 5; then
            server_ready="1"
            break
        fi
        if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
            echo "错误: SGLang Server 在健康检查通过前已退出。" >&2
            echo "请检查日志: $SERVER_LOG_FILE" >&2
            echo "最近日志:" >&2
            tail -n 40 "$SERVER_LOG_FILE" >&2 || true
            exit 1
        fi
    done

    if [ "$server_ready" != "1" ]; then
        echo "错误: SGLang 服务在 ${SERVER_TIMEOUT}s 内未就绪。" >&2
        echo "请检查日志: $SERVER_LOG_FILE" >&2
        exit 1
    fi
else
    print_section "Reuse Server"
    echo "复用现有服务: ${HEALTH_HOST}:${SGLANG_PORT}"
fi

print_section "Run Benchmark"
(
    export MODEL_PATH="$RESOLVED_MODEL_PATH"
    export PYTHON_EXECUTABLE
    export SGLANG_PORT
    export INPUT_LENGTH
    export OUTPUT_LENGTH
    export NUM_PROMPTS
    export WARMUP_REQUESTS
    export RANDOM_RANGE_RATIO
    export MAX_CONCURRENCY
    export REQUEST_RATE
    export MODEL_TYPE
    export DATASET_NAME
    export DATASET_PATH
    export RANDOM_IMAGE_NUM_IMAGES
    export RANDOM_IMAGE_RESOLUTION
    export APPLY_CHAT_TEMPLATE
    export RAW_OUTPUT_FILE
    export BENCH_LOG_FILE
    exec "${SCRIPT_DIR}/run_benchmark.sh"
)

print_section "Completed"
echo "Benchmark 已完成。"
echo "结果目录: $RESULT_DIR"
echo "Server 日志: $SERVER_LOG_FILE"
echo "Bench 日志: $BENCH_LOG_FILE"
echo "Bench 原始 JSON: $RAW_OUTPUT_FILE"
echo ""
echo "如果你只想复用现有服务压测，可以设置:"
echo "USE_EXISTING_SERVER=1"
