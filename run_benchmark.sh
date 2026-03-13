#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sglang_benchmark_common.sh"

MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-0.6B}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
SGLANG_PORT="${SGLANG_PORT:-30001}"
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
RAW_OUTPUT_FILE="${RAW_OUTPUT_FILE:-}"
BENCH_LOG_FILE="${BENCH_LOG_FILE:-}"

RESOLVED_MODEL_PATH="$(resolve_model_dir "$MODEL_PATH")"
DETECTED_MODEL_TYPE="$(detect_model_type "$RESOLVED_MODEL_PATH" "$MODEL_TYPE")"

if [ -z "$DATASET_NAME" ]; then
    if [ -n "$DATASET_PATH" ]; then
        DATASET_NAME="sharegpt"
    elif [ "$DETECTED_MODEL_TYPE" = "vl" ]; then
        DATASET_NAME="random-image"
    else
        DATASET_NAME="random"
    fi
fi

if [ "$DATASET_NAME" = "sharegpt" ] && [ -z "$DATASET_PATH" ]; then
    echo "DATASET_NAME=sharegpt 时必须同时提供 DATASET_PATH" >&2
    exit 1
fi
if [ -n "$DATASET_PATH" ] && [ ! -f "$DATASET_PATH" ]; then
    echo "DATASET_PATH 不存在或不是文件: $DATASET_PATH" >&2
    echo "如果你在容器里运行脚本，这里必须传容器内路径，而不是宿主机路径。" >&2
    exit 1
fi

cmd=(
    "$PYTHON_EXECUTABLE" -u -m sglang.bench_serving
    --backend sglang
    --port "$SGLANG_PORT"
    --model "$RESOLVED_MODEL_PATH"
    --num-prompts "$NUM_PROMPTS"
    --random-output-len "$OUTPUT_LENGTH"
    --dataset-name "$DATASET_NAME"
    --random-input-len "$INPUT_LENGTH"
    --warmup-requests "$WARMUP_REQUESTS"
    --random-range-ratio "$RANDOM_RANGE_RATIO"
    --max-concurrency "$MAX_CONCURRENCY"
)

if [ "$APPLY_CHAT_TEMPLATE" = "1" ]; then
    cmd+=(--apply-chat-template)
fi
if [ -n "$DATASET_PATH" ]; then
    cmd+=(--dataset-path "$DATASET_PATH")
fi
if [ -n "$REQUEST_RATE" ]; then
    cmd+=(--request-rate "$REQUEST_RATE")
fi
if [ "$DATASET_NAME" = "random-image" ]; then
    cmd+=(--random-image-num-images "$RANDOM_IMAGE_NUM_IMAGES")
    cmd+=(--random-image-resolution "$RANDOM_IMAGE_RESOLUTION")
fi
if [ -n "$RAW_OUTPUT_FILE" ]; then
    mkdir -p "$(dirname "$RAW_OUTPUT_FILE")"
    cmd+=(--output-file "$RAW_OUTPUT_FILE")
fi

print_section "SGLang Serving Benchmark"
echo "模型目录: $RESOLVED_MODEL_PATH"
echo "模型类型: $DETECTED_MODEL_TYPE"
echo "数据集: $DATASET_NAME"
if [ -n "$DATASET_PATH" ]; then
    echo "数据集路径: $DATASET_PATH"
fi
echo "输入长度: $INPUT_LENGTH"
echo "输出长度: $OUTPUT_LENGTH"
echo "请求数: $NUM_PROMPTS"
echo "最大并发: $MAX_CONCURRENCY"
echo "命令: ${cmd[*]}"
echo ""

if [ -n "$BENCH_LOG_FILE" ]; then
    mkdir -p "$(dirname "$BENCH_LOG_FILE")"
    "${cmd[@]}" 2>&1 | tee "$BENCH_LOG_FILE"
else
    exec "${cmd[@]}"
fi
