# SGLang MUSA 自动化压测脚本

这个仓库现在的推荐入口是 shell 脚本，不再要求你手工先起服务、再手工跑 `bench_serving`。

目标是只跑一个脚本，就能拿到类似下面这种原始压测输出：

```text
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    1.6
...
Mean TTFT (ms):                          286.31
P99 ITL (ms):                            380.31
==================================================
```

## 脚本说明

- `run_all_tests.sh`
  - 默认入口，当前直接转发到 `auto_benchmark.sh`
- `auto_benchmark.sh`
  - 一条命令完成：解析模型目录 -> 启动 `sglang.launch_server` -> 等 `/health` -> 执行 `sglang.bench_serving`
- `launch_server.sh`
  - 只负责启动 SGLang Server
- `run_benchmark.sh`
  - 只负责跑 `sglang.bench_serving`
- `benchmark_sglang.py`
  - 保留的 Python 包装脚本，已补成失败早报；如果模型目录不对或服务秒退，会直接报错

## 快速开始

先给脚本执行权限：

```bash
chmod +x run_all_tests.sh auto_benchmark.sh launch_server.sh run_benchmark.sh
```

然后直接跑：

```bash
MODEL_PATH=/workspace/models/Qwen3-0.6B ./run_all_tests.sh
```

如果你想显式指定更多参数，可以这样跑：

```bash
MODEL_PATH=/workspace/models/Qwen3-0.6B \
INPUT_LENGTH=3000 \
OUTPUT_LENGTH=500 \
NUM_PROMPTS=10 \
WARMUP_REQUESTS=2 \
MAX_CONCURRENCY=64 \
REQUEST_RATE=1.6 \
TENSOR_PARALLEL_SIZE=1 \
SGLANG_PORT=30001 \
./auto_benchmark.sh
```

VL 模型示例：

```bash
MODEL_PATH=/data/model/Qwen2___5-VL-72B-Instruct \
MODEL_TYPE=vl \
INPUT_LENGTH=1024 \
OUTPUT_LENGTH=1024 \
NUM_PROMPTS=128 \
MAX_CONCURRENCY=64 \
TENSOR_PARALLEL_SIZE=8 \
./auto_benchmark.sh
```

Dense 模型如果要跑 ShareGPT：

```bash
MODEL_PATH=/data/model/Qwen3-32B \
MODEL_TYPE=dense \
DATASET_PATH=/data/model/ShareGPT.json \
DATASET_NAME=sharegpt \
TENSOR_PARALLEL_SIZE=8 \
./auto_benchmark.sh
```

## 关键参数

- `MODEL_PATH`
  - 模型目录，既可以直接传真正的模型目录，也可以传它的上层目录
- `MODEL_TYPE`
  - `auto|dense|vl`
  - 默认 `auto`，如果模型名里带 `VL/vl` 会自动按 VL 处理
- `INPUT_LENGTH`
  - 对应 `bench_serving --random-input-len`
- `OUTPUT_LENGTH`
  - 对应 `bench_serving --random-output-len`
- `NUM_PROMPTS`
  - 对应 `bench_serving --num-prompts`
- `WARMUP_REQUESTS`
  - 对应 `bench_serving --warmup-requests`
- `MAX_CONCURRENCY`
  - 对应 `bench_serving --max-concurrency`
- `REQUEST_RATE`
  - 对应 `bench_serving --request-rate`
  - 默认 `1.6`
- `TENSOR_PARALLEL_SIZE`
  - 对应 `launch_server --tensor-parallel-size`
- `MEM_FRACTION_STATIC`
  - 对应 `launch_server --mem-fraction-static`
- `CUDA_GRAPH_MAX_BS`
  - 对应 `launch_server --cuda-graph-max-bs`
- `ATTENTION_BACKEND`
  - 对应 `launch_server --attention-backend`
- `MAX_PREFILL_TOKENS`
  - 可选，对应 `launch_server --max-prefill-tokens`
- `SGLANG_PORT`
  - 服务端口
- `SERVER_TIMEOUT`
  - 等待 `/health` 的超时时间，默认 `2400`
- `USE_EXISTING_SERVER`
  - 设成 `1` 时不启动新服务，直接复用已有服务压测
- `KEEP_SERVER`
  - 设成 `1` 时脚本退出后不自动停服务

## 默认环境变量

脚本已经按你给的文档内置了这些 MUSA 环境变量：

```bash
SGLANG_TORCH_PROFILER_DIR=/tmp/
MUSA_LAUNCH_BLOCKING=0
MCCL_IB_GID_INDEX=3
MCCL_NET_SHARED_BUFFERS=0
MCCL_PROTOS=2
GLOO_SOCKET_IFNAME=bond0
TP_SOCKET_IFNAME=bond0
SGL_DEEP_GEMM_BLOCK_M=128
MUSA_VISIBLE_DEVICES=all
SGLANG_USE_MTT=1
```

如果你们环境还需要：

```bash
export NVSHMEM_HCA_PE_MAPPING="mlx5_bond_2:1:2,mlx5_bond_3:1:2,mlx5_bond_4:1:2,mlx5_bond_5:1:2"
```

再执行脚本即可，脚本会继承它。

## `config.json` 是什么

`config.json` 是 Hugging Face 模型目录里的配置文件，`transformers`、`sglang`、`vllm` 都靠它识别模型结构。

常见情况：

- 你传的是正确模型目录
  - 目录下直接有 `config.json`
- 你传的是上层目录
  - 脚本会自动往下找 `config.json`
- 你传的目录里根本没有模型
  - 脚本会直接报错，不再傻等 `/health`

## 输出文件

每次运行会在 `results/benchmark_时间戳/` 下生成：

- `server.log`
  - `sglang.launch_server` 日志
- `bench.log`
  - `sglang.bench_serving` 控制台日志
- `bench_raw.json`
  - `bench_serving --output-file` 原始 JSON
- `run_config.env`
  - 本次运行参数快照

其中 `bench.log` 里就会有你要的 `Serving Benchmark Result` 原始汇总输出。

## 分步执行

如果你要手动拆开跑，也可以：

先起服务：

```bash
MODEL_PATH=/data/model/Qwen2___5-VL-72B-Instruct \
TENSOR_PARALLEL_SIZE=8 \
./launch_server.sh
```

再单独压测：

```bash
MODEL_PATH=/data/model/Qwen2___5-VL-72B-Instruct \
MODEL_TYPE=vl \
INPUT_LENGTH=1024 \
OUTPUT_LENGTH=1024 \
NUM_PROMPTS=128 \
MAX_CONCURRENCY=64 \
./run_benchmark.sh
```

## 故障排查

### 1. `/health` 一直不通

优先看：

```bash
tail -n 100 results/benchmark_*/server.log
```

现在脚本如果发现服务进程在健康检查通过前就退出，会直接打印 `server.log` 的最后 40 行。

### 2. 报模型目录没有 `config.json`

说明 `MODEL_PATH` 还不是最终模型目录。你需要：

- 传真正包含 `config.json` 的目录
- 或者传它的上层目录，让脚本自动向下查找

### 3. VL / Dense 数据集不对

- VL 模型默认走 `random-image`
- Dense 模型默认走 `random`
- 如果你要用真实对话数据，手动传：
  - `DATASET_NAME=sharegpt`
  - `DATASET_PATH=/data/model/ShareGPT.json`

### 4. 复用已有服务

如果你已经手工起好了服务，不想让脚本重复启动：

```bash
USE_EXISTING_SERVER=1 MODEL_PATH=/data/model/Qwen3-32B ./auto_benchmark.sh
```

## 其他脚本

- `profile_inference.py`
  - 还保留着，用于 PyTorch Profiler 分析
- `analyze_operators.py`
  - 还保留着，用于算子汇总分析

这两个脚本没有并入当前默认入口，因为你这次的核心诉求是先把 SGLang 服务压测流程跑通。
