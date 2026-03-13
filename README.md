# SGLang MUSA 自动化压测脚本

## 项目目的

这个仓库用于把 SGLang 压测流程收敛成一个统一入口，避免手工拆成“起服务 + 等健康检查 + 跑 `bench_serving` + 收集结果”几步。

默认流程是：

1. 解析 `MODEL_PATH`
2. 启动 `sglang.launch_server`
3. 等待 `/health`
4. 执行 `sglang.bench_serving`
5. 保存日志和原始结果

## 文件结构

- `run_all_tests.sh`
  - 默认入口，当前直接转发到 `auto_benchmark.sh`
- `auto_benchmark.sh`
  - 主流程脚本：解析模型目录、启动服务、等待健康检查、执行压测、保存结果
- `launch_server.sh`
  - 只负责启动 `sglang.launch_server`
- `run_benchmark.sh`
  - 只负责执行 `sglang.bench_serving`
- `sglang_benchmark_common.sh`
  - 公共函数和默认环境变量
- `benchmark_sglang.py`
  - Python 包装脚本，保留备用
- `patches/`
  - 补丁文件
- `results/`
  - 每次压测的输出目录

## 如何进行测试

先给脚本执行权限：

```bash
chmod +x run_all_tests.sh auto_benchmark.sh launch_server.sh run_benchmark.sh
```

最基础的跑法：

```bash
./run_all_tests.sh
```

这会优先使用内建 dense 预设：

- `MODEL_PATH=/workspace/Qwen3-0.6B-FP8`
- `DATASET_NAME=random`
- `DATASET_PATH=/workspace/aloysha/ShareGPT.json`
- 自动探测 `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME`

如果这些路径存在，就不需要再手工传。

VL 模型直接跑：

```bash
./run_all_tests.sh vl
```

这会优先使用内建 VL 预设：

- `MODEL_PATH=/data/model/Qwen2___5-VL-72B-Instruct`
- `MODEL_TYPE=vl`
- `INPUT_LENGTH=1024`
- `OUTPUT_LENGTH=1024`
- `NUM_PROMPTS=128`
- `MAX_CONCURRENCY=64`
- `TENSOR_PARALLEL_SIZE=8`

说明：

- 如果脚本跑在容器里，`DATASET_PATH` 必须传容器内路径
- `GLOO_SOCKET_IFNAME` 和 `TP_SOCKET_IFNAME` 可显式指定网卡；如果不传，脚本会自动探测默认路由网卡
- 如果你已经手工起好了服务，可以传 `USE_EXISTING_SERVER=1`
- 你显式传入的环境变量优先级更高，会覆盖内建预设

## 结果放在哪里

每次运行会生成一个结果目录：

```bash
results/benchmark_时间戳/
```

目录内主要文件：

- `server.log`
  - `sglang.launch_server` 日志
- `bench.log`
  - `sglang.bench_serving` 控制台输出
- `bench_raw.json`
  - `bench_serving --output-file` 生成的原始 JSON
- `run_config.env`
  - 本次运行的参数快照

其中 `bench.log` 里会包含 `Serving Benchmark Result` 汇总输出。
