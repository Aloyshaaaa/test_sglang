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
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
INPUT_LENGTH="${INPUT_LENGTH:-3500}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-1500}"
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
BENCHMARK_MODE="${BENCHMARK_MODE:-sweep}"
SWEEP_VALUES="${SWEEP_VALUES:-1,2,4,8,16,32,50,64,100,128,256}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="${RESULT_ROOT:-${SCRIPT_DIR}/results}"
RESULT_DIR="${RESULT_DIR:-${RESULT_ROOT}/benchmark_${TIMESTAMP}}"
SERVER_LOG_FILE="${SERVER_LOG_FILE:-${RESULT_DIR}/server.log}"
BENCH_LOG_FILE="${BENCH_LOG_FILE:-${RESULT_DIR}/bench.log}"
RAW_OUTPUT_FILE="${RAW_OUTPUT_FILE:-${RESULT_DIR}/bench_raw.json}"
CONFIG_FILE="${CONFIG_FILE:-${RESULT_DIR}/run_config.env}"
SWEEP_SUMMARY_FILE="${SWEEP_SUMMARY_FILE:-${RESULT_DIR}/sweep_summary.csv}"

mkdir -p "$RESULT_DIR"
RESOLVED_MODEL_PATH="$(resolve_model_dir "$MODEL_PATH")"
export_default_server_env

print_section "SGLang Auto Benchmark"
echo "原始模型路径: $MODEL_PATH"
echo "解析后模型目录: $RESOLVED_MODEL_PATH"
echo "GLOO 网卡: ${GLOO_SOCKET_IFNAME:-<未设置>}"
echo "TP 网卡: ${TP_SOCKET_IFNAME:-<未设置>}"
echo "压测模式: $BENCHMARK_MODE"
echo "结果目录: $RESULT_DIR"
echo "Server 日志: $SERVER_LOG_FILE"
if [ "$BENCHMARK_MODE" = "sweep" ]; then
    echo "Sweep 档位: $SWEEP_VALUES"
    echo "Sweep 汇总: $SWEEP_SUMMARY_FILE"
else
    echo "Bench 日志: $BENCH_LOG_FILE"
    echo "Bench 原始 JSON: $RAW_OUTPUT_FILE"
fi
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
DISABLE_CUDA_GRAPH=$DISABLE_CUDA_GRAPH
GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}
TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-}
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
BENCHMARK_MODE=$BENCHMARK_MODE
SWEEP_VALUES=$SWEEP_VALUES
SWEEP_SUMMARY_FILE=$SWEEP_SUMMARY_FILE
EOF

SERVER_PID=""

validate_benchmark_mode() {
    case "$BENCHMARK_MODE" in
        single|sweep) ;;
        *)
            echo "无效的 BENCHMARK_MODE: $BENCHMARK_MODE，可选值: single|sweep" >&2
            exit 1
            ;;
    esac
}

run_benchmark_case() {
    local bench_log_file="$1"
    local raw_output_file="$2"
    local case_num_prompts="$3"
    local case_max_concurrency="$4"

    (
        export MODEL_PATH="$RESOLVED_MODEL_PATH"
        export PYTHON_EXECUTABLE
        export SGLANG_PORT
        export INPUT_LENGTH
        export OUTPUT_LENGTH
        export NUM_PROMPTS="$case_num_prompts"
        export WARMUP_REQUESTS
        export RANDOM_RANGE_RATIO
        export MAX_CONCURRENCY="$case_max_concurrency"
        export REQUEST_RATE
        export MODEL_TYPE
        export DATASET_NAME
        export DATASET_PATH
        export RANDOM_IMAGE_NUM_IMAGES
        export RANDOM_IMAGE_RESOLUTION
        export APPLY_CHAT_TEMPLATE
        export RAW_OUTPUT_FILE="$raw_output_file"
        export BENCH_LOG_FILE="$bench_log_file"
        exec "${SCRIPT_DIR}/run_benchmark.sh"
    )
}

run_benchmark_sweep() {
    local raw_values=()
    local sweep_values=()
    local item=""
    local trimmed=""
    local total_cases=0
    local case_index=0
    local value=""
    local status=""
    local case_name=""
    local case_dir=""
    local case_bench_log=""
    local case_raw_output=""

    IFS=',' read -r -a raw_values <<< "$SWEEP_VALUES"
    for item in "${raw_values[@]}"; do
        trimmed="${item//[[:space:]]/}"
        if [ -z "$trimmed" ]; then
            continue
        fi
        if ! [[ "$trimmed" =~ ^[0-9]+$ ]] || [ "$trimmed" -le 0 ]; then
            echo "SWEEP_VALUES 里存在无效档位: $item" >&2
            exit 1
        fi
        sweep_values+=("$trimmed")
    done

    if [ "${#sweep_values[@]}" -eq 0 ]; then
        echo "SWEEP_VALUES 不能为空" >&2
        exit 1
    fi

    total_cases="${#sweep_values[@]}"
    mkdir -p "$(dirname "$SWEEP_SUMMARY_FILE")"
    printf 'case_index,max_concurrency,num_prompts,status,result_dir,bench_log,raw_output\n' > "$SWEEP_SUMMARY_FILE"

    print_section "Run Benchmark Sweep"
    echo "输入长度: $INPUT_LENGTH"
    echo "输出长度: $OUTPUT_LENGTH"
    echo "Sweep 档位: ${sweep_values[*]}"
    echo "Sweep 汇总: $SWEEP_SUMMARY_FILE"
    echo ""

    for value in "${sweep_values[@]}"; do
        case_index=$((case_index + 1))
        printf -v case_name 'case_%03d_c%s_n%s' "$case_index" "$value" "$value"
        case_dir="${RESULT_DIR}/${case_name}"
        case_bench_log="${case_dir}/bench.log"
        case_raw_output="${case_dir}/bench_raw.json"
        mkdir -p "$case_dir"

        echo "[$case_index/$total_cases] MAX_CONCURRENCY=$value NUM_PROMPTS=$value"
        if run_benchmark_case "$case_bench_log" "$case_raw_output" "$value" "$value"; then
            status="completed"
        else
            status="failed"
        fi

        printf '%s,%s,%s,%s,%s,%s,%s\n' \
            "$case_index" \
            "$value" \
            "$value" \
            "$status" \
            "$case_dir" \
            "$case_bench_log" \
            "$case_raw_output" >> "$SWEEP_SUMMARY_FILE"

        if [ "$status" != "completed" ]; then
            echo "错误: sweep case 失败: $case_name" >&2
            echo "请检查日志: $case_bench_log" >&2
            exit 1
        fi
    done
}

cleanup() {
    if [ -n "$SERVER_PID" ] && [ "$KEEP_SERVER" != "1" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo ""
        echo "停止脚本拉起的 SGLang Server: $SERVER_PID"
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT
validate_benchmark_mode

if curl -fsS "http://${HEALTH_HOST}:${SGLANG_PORT}/health" >/dev/null 2>&1; then
    echo "检测到 ${HEALTH_HOST}:${SGLANG_PORT} 已有健康的 SGLang 服务。"
    USE_EXISTING_SERVER="1"
fi

if [ "$USE_EXISTING_SERVER" != "1" ]; then
    run_sglang_runtime_preflight "$PYTHON_EXECUTABLE"
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

if [ "$BENCHMARK_MODE" = "sweep" ]; then
    run_benchmark_sweep
else
    print_section "Run Benchmark"
    run_benchmark_case "$BENCH_LOG_FILE" "$RAW_OUTPUT_FILE" "$NUM_PROMPTS" "$MAX_CONCURRENCY"
fi

print_section "Completed"
echo "Benchmark 已完成。"
echo "结果目录: $RESULT_DIR"
echo "Server 日志: $SERVER_LOG_FILE"
if [ "$BENCHMARK_MODE" = "sweep" ]; then
    echo "Sweep 汇总: $SWEEP_SUMMARY_FILE"
else
    echo "Bench 日志: $BENCH_LOG_FILE"
    echo "Bench 原始 JSON: $RAW_OUTPUT_FILE"
fi
echo ""
echo "如果你只想复用现有服务压测，可以设置:"
echo "USE_EXISTING_SERVER=1"
