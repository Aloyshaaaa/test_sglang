#!/bin/bash
# SGLang Qwen3-0.6B 推理测试脚本 - 摩尔线程 MUSA 环境
# 输入长度: 3000, 输出长度: 500

set -e

echo "=========================================="
echo "SGLang Qwen3-0.6B 推理测试"
echo "摩尔线程 MUSA 环境"
echo "=========================================="
echo ""

# 配置
MODEL_PATH="/workspace/models/Qwen3-0.6B"
INPUT_LENGTH=3000
OUTPUT_LENGTH=500
NUM_RUNS=10
WARMUP=2

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    echo "请检查模型路径是否正确"
    exit 1
fi

echo "模型路径: $MODEL_PATH"
echo "输入长度: $INPUT_LENGTH"
echo "输出长度: $OUTPUT_LENGTH"
echo "测试次数: $NUM_RUNS"
echo ""

# 创建输出目录
mkdir -p results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="results/test_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

echo "结果将保存到: $RESULT_DIR"
echo ""

# ============================================
# 步骤 1: 运行基准测试
# ============================================
echo "=========================================="
echo "步骤 1: 运行基准测试"
echo "=========================================="
echo ""

python benchmark_sglang.py \
    --model-path "$MODEL_PATH" \
    --input-length "$INPUT_LENGTH" \
    --output-length "$OUTPUT_LENGTH" \
    --num-runs "$NUM_RUNS" \
    --warmup "$WARMUP" \
    --backend sglang \
    --dataset-name random \
    --max-concurrency 64 \
    --port 30001 \
    --tensor-parallel-size 1 \
    --output "$RESULT_DIR/benchmark_results.json" \
    2>&1 | tee "$RESULT_DIR/benchmark_log.txt"

echo ""
echo "✓ 基准测试完成"
echo ""

# ============================================
# 步骤 2: 运行性能分析
# ============================================
echo "=========================================="
echo "步骤 2: 运行性能分析"
echo "=========================================="
echo ""

python profile_inference.py \
    --model-path "$MODEL_PATH" \
    --input-length "$INPUT_LENGTH" \
    --output-length "$OUTPUT_LENGTH" \
    --profile-iterations 5 \
    --output "$RESULT_DIR/profile_results.json" \
    --mode all \
    2>&1 | tee "$RESULT_DIR/profile_log.txt"

echo ""
echo "✓ 性能分析完成"
echo ""

# ============================================
# 步骤 3: 分析算子耗时
# ============================================
echo "=========================================="
echo "步骤 3: 分析算子耗时"
echo "=========================================="
echo ""

# 检查是否有 profiler 结果
if [ -f "$RESULT_DIR/profile_results.json" ]; then
    python analyze_operators.py \
        --input "$RESULT_DIR/profile_results.json" \
        --output "$RESULT_DIR/operator_analysis_report.txt" \
        --viz-output "$RESULT_DIR/operators_for_viz.json" \
        2>&1 | tee "$RESULT_DIR/analysis_log.txt"
    
    echo ""
    echo "✓ 算子分析完成"
else
    echo "警告: 未找到 profiler 结果，跳过算子分析"
fi

echo ""

# ============================================
# 生成汇总报告
# ============================================
echo "=========================================="
echo "生成汇总报告"
echo "=========================================="
echo ""

cat > "$RESULT_DIR/SUMMARY.md" << 'EOF'
# SGLang Qwen3-0.6B 推理测试报告

## 测试配置

- **模型**: Qwen3-0.6B
- **输入长度**: 3000 tokens
- **输出长度**: 500 tokens
- **后端**: 摩尔线程 MUSA
- **测试时间**: TIMESTAMP

## 生成的文件

### 基准测试
- `benchmark_results.json` - 详细的基准测试结果
- `benchmark_log.txt` - 基准测试日志

### 性能分析
- `profile_results.json` - Profiler 分析结果
- `profiler_trace_musa.json` - Chrome trace 文件（可用 Chrome 浏览器打开 chrome://tracing 查看）
- `profiler_stats_musa.txt` - 详细的 profiler 统计
- `profile_log.txt` - 性能分析日志

### 算子分析
- `operator_analysis_report.txt` - 算子耗时分析报告
- `operators_for_viz.json` - 可视化数据
- `analysis_log.txt` - 分析日志

## 如何查看结果

### 1. 查看基准测试结果
```bash
cat results/test_TIMESTAMP/benchmark_results.json | python -m json.tool
```

### 2. 查看性能分析
```bash
# 查看统计报告
cat results/test_TIMESTAMP/profiler_stats_musa.txt

# 使用 Chrome 可视化 trace
# 打开 Chrome 浏览器，访问 chrome://tracing
# 加载 results/test_TIMESTAMP/profiler_trace_musa.json
```

### 3. 查看算子分析
```bash
cat results/test_TIMESTAMP/operator_analysis_report.txt
```

## 关键指标说明

- **延迟 (Latency)**: 完成一次推理所需的时间
- **吞吐 (Throughput)**: 每秒生成的 token 数
- **P50/P99 延迟**: 50%/99% 分位的延迟

## 优化建议

根据算子分析报告中的瓶颈，可以考虑以下优化方向：

1. **矩阵运算优化**: 使用 FP16/BF16，启用 Tensor Core
2. **注意力优化**: 使用 FlashAttention
3. **内存优化**: 优化内存访问模式，使用 KV Cache
EOF

# 替换时间戳
sed -i "s/TIMESTAMP/$TIMESTAMP/g" "$RESULT_DIR/SUMMARY.md"

echo "✓ 汇总报告已生成: $RESULT_DIR/SUMMARY.md"
echo ""

# ============================================
# 显示结果摘要
# ============================================
echo "=========================================="
echo "测试完成!"
echo "=========================================="
echo ""
echo "结果目录: $RESULT_DIR"
echo ""
echo "生成的文件:"
ls -lh "$RESULT_DIR"
echo ""
echo "关键文件:"
echo "  - benchmark_results.json    基准测试结果"
echo "  - profile_results.json      性能分析结果"
echo "  - operator_analysis_report.txt  算子分析报告"
echo "  - SUMMARY.md                汇总报告"
echo ""
echo "查看算子分析报告:"
echo "  cat $RESULT_DIR/operator_analysis_report.txt"
echo ""
echo "使用 Chrome 查看 trace:"
echo "  1. 打开 Chrome 浏览器"
echo "  2. 访问 chrome://tracing"
echo "  3. 加载 $RESULT_DIR/profiler_trace_musa.json"
echo ""
