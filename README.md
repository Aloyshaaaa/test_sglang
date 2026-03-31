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
- `launch_vllm_server.sh`
  - 只负责启动 `vllm serve`
- `run_vllm_benchmark.sh`
  - 只负责执行 `vllm bench serve`
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
chmod +x run_all_tests.sh auto_benchmark.sh launch_server.sh run_benchmark.sh launch_vllm_server.sh run_vllm_benchmark.sh
```

最基础的跑法：

```bash
MODEL_PATH=/workspace/Qwen3-0.6B-FP8 \
./run_all_tests.sh
```

现在 `run_all_tests.sh` 默认会做一轮单模型 sweep：

- `INPUT_LENGTH=3500`
- `OUTPUT_LENGTH=1500`
- `NUM_PROMPTS = 2 * MAX_CONCURRENCY`
- 档位依次为 `1,2,4,8,16,32,50,64,100,128,256`

也就是说，你只要指定一个模型路径，就会起一次服务，然后把这 11 档顺序全跑完。

Dense 模型使用本地 `ShareGPT.json` 做采样（Qwen3-32B-FP8 八卡示例）：

```bash
GLOO_SOCKET_IFNAME=ens19f0np0 \
TP_SOCKET_IFNAME=ens19f0np0 \
MODEL_PATH=/workspace/mochi/models/Qwen3-32B-FP8 \
TENSOR_PARALLEL_SIZE=8 \
DATASET_NAME=sharegpt \
DATASET_PATH=/workspace/aloysha/ShareGPT.json \
./run_all_tests.sh
```

如果当前容器里的外部 `sglang.bench_serving` 仍包含未修复的 `tokenizer.bos_token` 处理逻辑，仓库脚本会在真正开跑前直接报错，并提示你应用仓库内补丁 [patches/sglang_bench_serving_bos_token.patch](/home/aloysha/aloysha/test_sglang/patches/sglang_bench_serving_bos_token.patch)。

如果你只想跑单个固定档位，不做 sweep，可以显式切回单次模式：

```bash
GLOO_SOCKET_IFNAME=ens19f0np0 \
TP_SOCKET_IFNAME=ens19f0np0 \
MODEL_PATH=/workspace/mochi/models/Qwen3-32B-FP8 \
TENSOR_PARALLEL_SIZE=8 \
DATASET_NAME=sharegpt \
DATASET_PATH=/workspace/aloysha/ShareGPT.json \
BENCHMARK_MODE=single \
MAX_CONCURRENCY=64 \
./run_all_tests.sh
```

说明：当前脚本会自动把 `NUM_PROMPTS` 设为 `MAX_CONCURRENCY` 的两倍；上例里会实际使用 `NUM_PROMPTS=128`。

如果你后面还想改 sweep 档位，可以直接覆盖：

```bash
SWEEP_VALUES=1,2,4,8,16,32,64 \
./run_all_tests.sh
```

如果 MUSA 环境在服务启动阶段卡在 `Capture cuda graph`，或者日志里出现 `is_varlen_q must be equal to is_varlen_k`，可以先禁用 CUDA graph 验证：

```bash
DISABLE_CUDA_GRAPH=1 \
./run_all_tests.sh
```

如果禁用 CUDA graph 以后，服务能启动但真实请求阶段仍然在 `fa3` attention backend 报 `is_varlen_q must be equal to is_varlen_k`，可以先手工切换到更保守的 attention backend 验证：

```bash
python -m sglang.launch_server \
  --model /home/mccxadmin/kayce/models/Qwen2.5-0.5B \
  --host 0.0.0.0 \
  --port 30001 \
  --tensor-parallel-size 1 \
  --mem-fraction-static 0.9 \
  --disable-cuda-graph \
  --prefill-attention-backend torch_native \
  --decode-attention-backend torch_native \
  --trust-remote-code
```

如果上面命令能正常处理 `/generate` 请求，通常说明问题在镜像内 `fa3/MUSA` 的 attention 运行时兼容性，而不是本仓库脚本参数本身。

如果你希望继续使用仓库里的默认入口，也可以直接通过环境变量让 `./run_all_tests.sh` 带上这组更稳定的参数：

```bash
MODEL_PATH=/home/mccxadmin/kayce/models/Qwen2.5-0.5B \
DISABLE_CUDA_GRAPH=1 \
PREFILL_ATTENTION_BACKEND=torch_native \
DECODE_ATTENTION_BACKEND=torch_native \
./run_all_tests.sh
```

地址约定：

- `HOST` 只用于 `sglang.launch_server` 的监听地址，默认 `0.0.0.0`
- `HEALTH_HOST` 用于 `/health` 检查，默认 `127.0.0.1`
- `BENCH_HOST` 用于 `sglang.bench_serving` 真正发请求的目标地址，默认跟随 `HEALTH_HOST`

因此，看到服务监听在 `0.0.0.0:30001` 是正常的；但压测客户端不应该去连接 `0.0.0.0:30001`，而应该连 `127.0.0.1:30001` 或你实际可达的服务地址。

VL 模型示例：

```bash
MODEL_PATH=/data/model/Qwen2___5-VL-72B-Instruct \
MODEL_TYPE=vl \
INPUT_LENGTH=1024 \
OUTPUT_LENGTH=1024 \
MAX_CONCURRENCY=64 \
BENCHMARK_MODE=single \
TENSOR_PARALLEL_SIZE=8 \
./auto_benchmark.sh
```

说明：

- 如果脚本跑在容器里，`DATASET_PATH` 必须传容器内路径
- `GLOO_SOCKET_IFNAME` 和 `TP_SOCKET_IFNAME` 可显式指定网卡；如果不传，脚本会自动探测默认路由网卡
- 如果你已经手工起好了服务，可以传 `USE_EXISTING_SERVER=1`

## 结果放在哪里

每次运行会生成一个结果目录：

```bash
results/benchmark_时间戳/
```

目录内主要文件：

- `server.log`
  - `sglang.launch_server` 日志
- `bench.log`
  - 单次模式下的 `sglang.bench_serving` 控制台输出
- `bench_raw.json`
  - 单次模式下 `bench_serving --output-file` 生成的原始 JSON
- `sweep_summary.csv`
  - sweep 模式下每个档位的结果文件索引
- `case_*/bench.log`
  - sweep 模式下每个档位各自的压测日志
- `case_*/bench_raw.json`
  - sweep 模式下每个档位各自的原始 JSON
- `run_config.env`
  - 本次运行的参数快照

其中 `bench.log` 里会包含 `Serving Benchmark Result` 汇总输出。
