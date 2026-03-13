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
    --backend sglang \
    --dataset-name random \
    --max-concurrency 64 \
    --port 30001 \
    --tensor-parallel-size 1 \
    --output benchmark_results.json
```

参数说明：
- `--model-path`: 模型路径
- `--input-length`: `bench_serving --random-input-len`
- `--output-length`: `bench_serving --random-output-len`
- `--num-runs`: 兼容旧参数名，等价于 `--num-prompts`
- `--warmup`: 兼容旧参数名，等价于 `--warmup-requests`
- `--dataset-name`: 数据集类型，Dense 模型用 `random`，VL 模型用 `random-image`
- `--max-concurrency`: 压测最大并发
- `--port`: SGLang Server 监听端口
- `--tensor-parallel-size`: `launch_server` 的 TP 配置
- `--backend`: 保留旧接口；当前脚本仅实现文档口径的 `sglang`
- `--output`: 输出结果文件

执行流程：
- 脚本会先启动 `python -m sglang.launch_server`
- 然后轮询 `http://127.0.0.1:<port>/health`
- 健康检查通过后，执行 `python -m sglang.bench_serving`
- 最终将 `bench_serving` 原始 JSON 和封装后的结果写入输出文件附近

模型路径说明：
- `--model-path` 最终需要指向一个包含 `config.json` 的 Hugging Face 模型目录
- 如果传入的是上层目录，脚本会尝试自动向下查找唯一的 `config.json` 所在目录
- 如果目录下存在多个 `config.json`，脚本会打印候选目录并要求手动指定更精确的路径

数据集说明：
- Dense 模型建议使用 `--dataset-name random`
- VL 模型建议使用 `--dataset-name random-image --random-image-num-images 1 --random-image-resolution 1148x112`
- 如果要跑真实对话数据，可使用 `--dataset-name sharegpt --dataset-path /data/model/ShareGPT.json`

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
- `request_throughput`: 请求吞吐量 (req/s)
- `input_throughput`: 输入吞吐量 (tok/s)
- `output_throughput`: 输出吞吐量 (tok/s)
- `mean_ttft_ms`: 平均首字延迟
- `mean_e2e_latency_ms`: 平均端到端延迟
- `concurrency`: bench 实测并发

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
- `benchmark_results.json` - 封装后的完整结果，包含 server 配置和 bench 指标
- `benchmark_results.sglang.raw.json` - `sglang.bench_serving` 原始输出
- `benchmark_results.server.log` - `sglang.launch_server` 日志
- `benchmark_log.txt` - benchmark 脚本自身日志

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
- 当前脚本会直接调用 `python -m sglang.launch_server`
- 如果报 `Can't load the configuration of ...`，说明 `--model-path` 还没有指到真正包含 `config.json` 的模型目录
- 如果 `/health` 一直不返回 200，请检查 `benchmark_results.server.log`
- 对照 MUSA 文档检查环境变量：`SGLANG_USE_MTT=1`、`MUSA_VISIBLE_DEVICES=all`、`TP_SOCKET_IFNAME=bond0`

### 2. MUSA 设备不可用
- 检查 `torch_musa` 是否安装
- 运行 `python -c "import torch_musa; print(torch.musa.is_available())"` 检查
- 如果是多卡环境，检查 `MUSA_VISIBLE_DEVICES`、`tensor-parallel-size` 与实际卡数是否匹配
- 如果网络初始化失败，检查 `GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME`、`MCCL_IB_GID_INDEX`

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
