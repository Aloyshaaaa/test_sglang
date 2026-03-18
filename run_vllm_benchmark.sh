#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/sglang_benchmark_common.sh"

MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-0.6B}"
VLLM_EXECUTABLE="${VLLM_EXECUTABLE:-vllm}"
BENCH_HOST="${BENCH_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
INPUT_LENGTH="${INPUT_LENGTH:-3500}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-1500}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
DATASET_NAME="${DATASET_NAME:-sharegpt}"
DATASET_PATH="${DATASET_PATH:-}"
TOKENIZER="${TOKENIZER:-}"
REQUEST_RATE="${REQUEST_RATE:-}"
RESULT_DIR="${RESULT_DIR:-}"
RESULT_FILENAME="${RESULT_FILENAME:-bench_result.json}"
BENCH_LOG_FILE="${BENCH_LOG_FILE:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
SKIP_CHAT_TEMPLATE="${SKIP_CHAT_TEMPLATE:-0}"
READY_CHECK_TIMEOUT_SEC="${READY_CHECK_TIMEOUT_SEC:-600}"

RESOLVED_MODEL_PATH="$(resolve_model_dir "$MODEL_PATH")"

if [ "$DATASET_NAME" = "sharegpt" ] && [ -z "$DATASET_PATH" ]; then
    echo "DATASET_NAME=sharegpt 时必须同时提供 DATASET_PATH" >&2
    exit 1
fi
if [ -n "$DATASET_PATH" ] && [ ! -f "$DATASET_PATH" ]; then
    echo "DATASET_PATH 不存在或不是文件: $DATASET_PATH" >&2
    echo "如果你在容器里运行脚本，这里必须传容器内路径，而不是宿主机路径。" >&2
    exit 1
fi

if [ -z "$RESULT_DIR" ]; then
    RESULT_DIR="${SCRIPT_DIR}/results"
fi
mkdir -p "$RESULT_DIR"

cmd=(
    "$VLLM_EXECUTABLE" bench serve
    --backend vllm
    --host "$BENCH_HOST"
    --port "$VLLM_PORT"
    --model "$RESOLVED_MODEL_PATH"
    --dataset-name "$DATASET_NAME"
    --num-prompts "$NUM_PROMPTS"
    --max-concurrency "$MAX_CONCURRENCY"
    --input-len "$INPUT_LENGTH"
    --output-len "$OUTPUT_LENGTH"
    --ready-check-timeout-sec "$READY_CHECK_TIMEOUT_SEC"
    --save-result
    --result-dir "$RESULT_DIR"
    --result-filename "$RESULT_FILENAME"
)

if [ -n "$DATASET_PATH" ]; then
    cmd+=(--dataset-path "$DATASET_PATH")
fi
if [ -n "$TOKENIZER" ]; then
    cmd+=(--tokenizer "$TOKENIZER")
fi
if [ -n "$REQUEST_RATE" ]; then
    cmd+=(--request-rate "$REQUEST_RATE")
fi
if [ -n "$SERVED_MODEL_NAME" ]; then
    cmd+=(--served-model-name "$SERVED_MODEL_NAME")
fi
if [ "$SKIP_CHAT_TEMPLATE" = "1" ]; then
    cmd+=(--skip-chat-template)
fi

print_section "vLLM Serving Benchmark"
echo "模型目录: $RESOLVED_MODEL_PATH"
echo "数据集: $DATASET_NAME"
if [ -n "$DATASET_PATH" ]; then
    echo "数据集路径: $DATASET_PATH"
fi
echo "输入长度: $INPUT_LENGTH"
echo "输出长度: $OUTPUT_LENGTH"
echo "请求数: $NUM_PROMPTS"
echo "最大并发: $MAX_CONCURRENCY"
echo "结果目录: $RESULT_DIR"
echo "结果文件: $RESULT_FILENAME"
echo "命令: ${cmd[*]}"
echo ""

if [ -n "$BENCH_LOG_FILE" ]; then
    mkdir -p "$(dirname "$BENCH_LOG_FILE")"
    "${cmd[@]}" 2>&1 | tee "$BENCH_LOG_FILE"
else
    exec "${cmd[@]}"
fi
