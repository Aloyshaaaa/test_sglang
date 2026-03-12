# SGLang Qwen3-0.6B 推理测试工具

针对摩尔线程 MUSA 环境的 Qwen3-0.6B 模型推理性能测试和分析工具。

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- torch_musa (摩尔线程 MUSA 支持)
- transformers
- SGLang (可选)
- vLLM (可选，用于对比测试)

## 安装依赖

```bash
# 基础依赖
pip install torch transformers accelerate

# 摩尔线程 MUSA 支持
pip install torch_musa

# SGLang (如果需要)
pip install sglang

# vLLM (如果需要对比测试)
pip install vllm
```

## 快速开始

### 一键运行所有测试

```bash
chmod +x run_all_tests.sh
./run_all_tests.sh
```

这将自动运行：
1. 基准测试
2. 性能分析
3. 算子耗时分析

结果将保存在 `results/test_YYYYMMDD_HHMMSS/` 目录下。

## 单独运行测试

### 1. 基准测试

```bash
python benchmark_sglang.py \
    --model-path /workspace/models/Qwen3-0.6B \
    --input-length 3000 \
    --output-length 500 \
    --num-runs 10 \
    --warmup 2 \
    --device auto \
    --backend both \
    --output benchmark_results.json
```

参数说明：
- `--model-path`: 模型路径
- `--input-length`: 输入长度 (默认: 3000)
- `--output-length`: 输出长度 (默认: 500)
- `--num-runs`: 测试运行次数 (默认: 10)
- `--warmup`: Warmup 次数 (默认: 2)
- `--device`: 推理设备类型 (auto/musa/cuda/cpu，默认: auto)
- `--backend`: 测试后端 (sglang/vllm/both)
- `--output`: 输出结果文件

设备说明：
- `--device auto`: 自动按 `torch_musa -> CUDA -> CPU` 顺序选择设备
- `--device musa`: 在摩尔线程环境中显式指定 MUSA，适合 vLLM / vllm_musa 自动探测异常时使用
- `--device cuda`: 在 NVIDIA CUDA 环境中显式指定 CUDA
- `--device cpu`: 仅用于无加速卡环境下的兼容性测试

### 2. 性能分析

```bash
python profile_inference.py \
    --model-path /workspace/models/Qwen3-0.6B \
    --input-length 3000 \
    --output-length 500 \
    --profile-iterations 5 \
    --mode all \
    --output profile_results.json
```

分析模式 (`--mode`)：
- `profiler`: 仅 PyTorch Profiler 分析
- `layer`: 仅逐层分析
- `memory`: 仅内存分析
- `all`: 全部分析

### 3. 算子分析

```bash
python analyze_operators.py \
    --input profile_results.json \
    --output operator_analysis_report.txt \
    --viz-output operators_for_viz.json
```

## 查看结果

### 基准测试结果

```bash
cat benchmark_results.json | python -m json.tool
```

关键指标：
- `avg_latency`: 平均延迟
- `avg_tokens_per_sec`: 平均吞吐量
- `p50_latency`: P50 延迟
- `p99_latency`: P99 延迟

### 性能分析结果

1. **Chrome Trace 可视化**
   - 打开 Chrome 浏览器
   - 访问 `chrome://tracing`
   - 加载 `profiler_trace_musa.json`

2. **统计报告**
   ```bash
   cat profiler_stats_musa.txt
   ```

### 算子分析报告

```bash
cat operator_analysis_report.txt
```

报告包含：
- 按类别统计的算子耗时
- Top 20 耗时算子排序
- 各类别详细算子列表
- 优化建议

## 输出文件说明

### 基准测试
- `benchmark_results.json` - JSON 格式的详细结果
- `benchmark_log.txt` - 测试日志

### 性能分析
- `profile_results.json` - Profiler 结果
- `profiler_trace_musa.json` - Chrome trace 文件
- `profiler_stats_musa.txt` - 统计报告

### 算子分析
- `operator_analysis_report.txt` - 详细分析报告
- `operators_for_viz.json` - 可视化数据

## 优化建议

根据算子分析报告，常见的优化方向：

### 1. 矩阵运算优化
- 使用 FP16/BF16 精度
- 启用 Tensor Core
- 使用优化的 GEMM 库

### 2. 注意力机制优化
- 使用 FlashAttention
- 使用稀疏注意力
- 优化 KV Cache

### 3. 内存优化
- 优化内存访问模式
- 减少数据拷贝
- 使用内存池

### 4. 计算图优化
- 算子融合
- 消除冗余计算
- 使用编译优化 (torch.compile)

## 摩尔线程 MUSA 特定优化

### 环境变量

```bash
# 启用 MUSA 特定优化
export MUSA_VISIBLE_DEVICES=0
export PYTORCH_MUSA_ALLOC_CONF=max_split_size_mb:512

# 调试信息
export MUSA_LAUNCH_BLOCKING=1  # 同步模式（调试时使用）
```

### 性能调优

1. **使用 FP16**
   ```python
   model = model.half()  # 转换为 FP16
   ```

2. **启用 TF32** (如果支持)
   ```python
   torch.backends.musa.matmul.allow_tf32 = True
   ```

3. **优化内存分配**
   ```python
   import torch_musa
   torch_musa.empty_cache()  # 清理缓存
   ```

## 故障排除

### 1. SGLang 无法加载
- 检查 SGLang 是否正确安装
- 当前脚本会自动兼容 `max_running_requests` 和 `max_num_reqs` 两种初始化参数
- 如果仍然报 `ServerArgs.__init__() got an unexpected keyword argument ...`，说明开发环境里的 SGLang 分支接口又有变化，需要根据实际分支继续对齐参数名
- 尝试使用 vLLM 后端进行对比测试

### 2. MUSA 设备不可用
- 检查 `torch_musa` 是否安装
- 运行 `python -c "import torch_musa; print(torch.musa.is_available())"` 检查
- 如果 vLLM 报 `Device string must not be empty`，优先显式传 `--device musa`

### 3. 内存不足
- 减少 `input-length` 或 `output-length`
- 使用 `torch.float16` 而不是 `torch.float32`
- 启用梯度检查点

### 4. 性能异常
- 检查是否启用了正确的设备同步
- 确认模型已加载到 MUSA 设备
- 检查是否有其他进程占用 GPU

## 贡献

欢迎提交 Issue 和 PR！

## 许可证

MIT License
