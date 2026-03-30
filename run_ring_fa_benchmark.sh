#!/usr/bin/env bash
# Ring-FlashAttention 基准测试示例
# 用法: ./run_ring_fa_benchmark.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sglang_benchmark_common.sh"
source "${SCRIPT_DIR}/ring_fa_config.sh"

# ========== 基本配置 ==========
MODEL_PATH="${MODEL_PATH:-/data/model/Qwen2.5-7B-Instruct}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
INPUT_LENGTH="${INPUT_LENGTH:-2048}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-512}"

# ========== Ring-FA 配置 ==========
USE_RING_FLASH_ATTN="${USE_RING_FLASH_ATTN:-1}"
RING_PARALLEL_SIZE="${RING_PARALLEL_SIZE:-4}"  # 4卡组成1个ring
MAX_SEQ_LEN="${MAX_SEQ_LEN:-16384}"             # 目标16K序列

# ========== 测试矩阵 ==========
# 短序列 (基线)
echo "=========================================="
echo "测试1: 短序列基线 (4K)"
echo "=========================================="
MODEL_PATH="$MODEL_PATH" \
TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE \
INPUT_LENGTH=2048 \
OUTPUT_LENGTH=512 \
NUM_PROMPTS=64 \
MAX_CONCURRENCY=64 \
USE_RING_FLASH_ATTN=0 \
BENCHMARK_MODE=single \
"${SCRIPT_DIR}/run_all_tests.sh"

# 长序列 (Ring-FA)
echo ""
echo "=========================================="
echo "测试2: 长序列 Ring-FA (16K)"
echo "=========================================="
MODEL_PATH="$MODEL_PATH" \
TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE \
INPUT_LENGTH=8192 \
OUTPUT_LENGTH=512 \
NUM_PROMPTS=32 \
MAX_CONCURRENCY=32 \
USE_RING_FLASH_ATTN=1 \
RING_PARALLEL_SIZE=$RING_PARALLEL_SIZE \
MAX_SEQ_LEN=$MAX_SEQ_LEN \
BENCHMARK_MODE=single \
"${SCRIPT_DIR}/run_all_tests.sh"

# 超长序列 (Ring-FA)
echo ""
echo "=========================================="
echo "测试3: 超长序列 Ring-FA (32K)"
echo "=========================================="
MODEL_PATH="$MODEL_PATH" \
TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE \
INPUT_LENGTH=16384 \
OUTPUT_LENGTH=512 \
NUM_PROMPTS=16 \
MAX_CONCURRENCY=16 \
USE_RING_FLASH_ATTN=1 \
RING_PARALLEL_SIZE=$RING_PARALLEL_SIZE \
MAX_SEQ_LEN=32768 \
BENCHMARK_MODE=single \
"${SCRIPT_DIR}/run_all_tests.sh"

echo ""
echo "=========================================="
echo "测试完成，请对比结果:"
echo "  - 短序列基线: results/.../case_001_*"
echo "  - 长序列Ring-FA: results/.../case_002_*"
echo "  - 超长序列Ring-FA: results/.../case_003_*"
echo "=========================================="