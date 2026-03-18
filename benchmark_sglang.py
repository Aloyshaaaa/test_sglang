#!/usr/bin/env python3
"""
SGLang benchmark script for MUSA environments.

Reference workflow:
1. Launch `python -m sglang.launch_server`
2. Wait for `/health`
3. Run `python -m sglang.bench_serving`
4. Parse the generated JSON report
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_SERVER_ENV = {
    "SGLANG_TORCH_PROFILER_DIR": "/tmp/",
    "MUSA_LAUNCH_BLOCKING": "0",
    "MCCL_IB_GID_INDEX": "3",
    "MCCL_NET_SHARED_BUFFERS": "0",
    "MCCL_PROTOS": "2",
    "SGL_DEEP_GEMM_BLOCK_M": "128",
    "MUSA_VISIBLE_DEVICES": "all",
    "SGLANG_USE_MTT": "1",
}
OPTIONAL_SERVER_ENV_KEYS = ("GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME")


def resolve_model_path(model_path: str) -> str:
    """Resolve the model path to a directory that contains config.json."""
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    if path.is_file():
        if path.name == "config.json":
            return str(path.parent)
        raise ValueError(f"--model-path 不是模型目录，也不是 config.json: {model_path}")

    if (path / "config.json").exists():
        return str(path)

    candidate_dirs = []
    seen = set()
    for config_file in sorted(path.rglob("config.json")):
        parent_dir = str(config_file.parent)
        if parent_dir in seen:
            continue
        seen.add(parent_dir)
        candidate_dirs.append(parent_dir)
        if len(candidate_dirs) >= 8:
            break

    if len(candidate_dirs) == 1:
        print(f"检测到嵌套模型目录，自动切换到: {candidate_dirs[0]}")
        return candidate_dirs[0]

    if len(candidate_dirs) > 1:
        print("警告: 模型路径下存在多个 config.json，无法自动选择:")
        for candidate_dir in candidate_dirs[:5]:
            print(f"  - {candidate_dir}")
        raise ValueError("请把 --model-path 指到更精确的单个模型目录。")

    raise FileNotFoundError(
        f"模型路径下未找到 config.json: {model_path}\n"
        "请把 --model-path 指到真正的 Hugging Face 模型目录，或者传它的上层目录让脚本自动向下查找。"
    )


def parse_key_value_pairs(items: list[str]) -> dict[str, str]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"无效的 KEY=VALUE 参数: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"无效的环境变量名: {item}")
        parsed[key] = value
    return parsed


def quote_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def network_interface_exists(ifname: str) -> bool:
    return bool(ifname) and Path("/sys/class/net", ifname).exists()


def detect_default_network_interface() -> str | None:
    try:
        completed = subprocess.run(
            ["ip", "route", "get", "1.1.1.1"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        completed = None

    if completed and completed.returncode == 0:
        fields = completed.stdout.split()
        for index, field in enumerate(fields[:-1]):
            if field == "dev":
                candidate = fields[index + 1]
                if network_interface_exists(candidate):
                    return candidate

    try:
        completed = subprocess.run(
            ["ip", "-o", "link", "show", "up"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if completed.returncode != 0:
        return None

    for line in completed.stdout.splitlines():
        parts = line.split(": ", 2)
        if len(parts) < 2:
            continue
        candidate = parts[1]
        if candidate in {"lo", "docker0"}:
            continue
        if network_interface_exists(candidate):
            return candidate
    return None


def build_server_command(args, model_path: str) -> list[str]:
    cmd = [
        args.python_executable,
        "-m",
        "sglang.launch_server",
        "--model",
        model_path,
        "--port",
        str(args.port),
        "--host",
        args.host,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--cuda-graph-max-bs",
        str(args.cuda_graph_max_bs),
        "--attention-backend",
        args.attention_backend,
    ]

    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.disable_radix_cache:
        cmd.append("--disable-radix-cache")
    if args.disable_overlap_schedule:
        cmd.append("--disable-overlap-schedule")
    if args.max_prefill_tokens is not None:
        cmd.extend(["--max-prefill-tokens", str(args.max_prefill_tokens)])
    return cmd


def build_bench_command(args, model_path: str, raw_output_file: str) -> list[str]:
    cmd = [
        args.python_executable,
        "-u",
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--port",
        str(args.port),
        "--model",
        model_path,
        "--num-prompts",
        str(args.num_prompts),
        "--random-input-len",
        str(args.input_length),
        "--random-output-len",
        str(args.output_length),
        "--dataset-name",
        args.dataset_name,
        "--warmup-requests",
        str(args.warmup_requests),
        "--random-range-ratio",
        str(args.random_range_ratio),
        "--max-concurrency",
        str(args.max_concurrency),
        "--output-file",
        raw_output_file,
    ]

    if args.apply_chat_template:
        cmd.append("--apply-chat-template")
    if args.dataset_path:
        cmd.extend(["--dataset-path", args.dataset_path])
    if args.request_rate is not None:
        cmd.extend(["--request-rate", str(args.request_rate)])
    if args.random_image_num_images is not None:
        cmd.extend(["--random-image-num-images", str(args.random_image_num_images)])
    if args.random_image_resolution:
        cmd.extend(["--random-image-resolution", args.random_image_resolution])
    return cmd


def build_server_env(args) -> dict[str, str]:
    env = os.environ.copy()
    env.update(DEFAULT_SERVER_ENV)
    env.update(parse_key_value_pairs(args.server_env))

    gloo_ifname = env.get("GLOO_SOCKET_IFNAME", "")
    tp_ifname = env.get("TP_SOCKET_IFNAME", "")

    if gloo_ifname and not network_interface_exists(gloo_ifname):
        raise ValueError(f"GLOO_SOCKET_IFNAME 指定的网卡不存在: {gloo_ifname}")
    if tp_ifname and not network_interface_exists(tp_ifname):
        raise ValueError(f"TP_SOCKET_IFNAME 指定的网卡不存在: {tp_ifname}")

    if not gloo_ifname and tp_ifname:
        env["GLOO_SOCKET_IFNAME"] = tp_ifname
        gloo_ifname = tp_ifname
    if not tp_ifname and gloo_ifname:
        env["TP_SOCKET_IFNAME"] = gloo_ifname
        tp_ifname = gloo_ifname

    if not gloo_ifname or not tp_ifname:
        detected_ifname = detect_default_network_interface()
        if detected_ifname:
            env.setdefault("GLOO_SOCKET_IFNAME", detected_ifname)
            env.setdefault("TP_SOCKET_IFNAME", detected_ifname)

    return env


def wait_for_health(health_host: str, port: int, timeout_seconds: int) -> bool:
    deadline = time.time() + timeout_seconds
    health_url = f"http://{health_host}:{port}/health"
    while time.time() < deadline:
        try:
            with urlopen(health_url, timeout=5) as response:
                if response.status == 200:
                    return True
        except (HTTPError, URLError, TimeoutError, OSError):
            time.sleep(5)
    return False


def terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        pass
    process.kill()
    process.wait(timeout=10)


def read_tail(path: Path, num_lines: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file_obj:
            lines = file_obj.readlines()
    except OSError:
        return ""
    return "".join(lines[-num_lines:]).strip()


def load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_results(path: str, results: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2, ensure_ascii=False)
    print(f"结果已保存到: {output_path}")


def benchmark_sglang(args, model_path: str) -> dict:
    print("\n" + "=" * 60)
    print("SGLang Serving Benchmark")
    print(f"模型路径: {model_path}")
    print(f"输入长度: {args.input_length}, 输出长度: {args.output_length}")
    print(f"请求数: {args.num_prompts}, 最大并发: {args.max_concurrency}")
    print("=" * 60 + "\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    server_log_path = Path(args.server_log) if args.server_log else output_path.with_suffix(".server.log")
    raw_result_path = output_path.with_suffix(".sglang.raw.json")
    raw_result_path.parent.mkdir(parents=True, exist_ok=True)

    server_env = build_server_env(args)
    server_command = build_server_command(args, model_path)
    bench_command = build_bench_command(args, model_path, str(raw_result_path))

    server_process = None
    server_log_handle = None
    server_started_by_script = False

    try:
        if not args.use_existing_server:
            server_log_handle = open(server_log_path, "a", encoding="utf-8")
            print("正在启动 SGLang Server...")
            print(f"Server 命令: {quote_command(server_command)}")
            print(f"Server 日志: {server_log_path}")
            server_process = subprocess.Popen(
                server_command,
                stdout=server_log_handle,
                stderr=subprocess.STDOUT,
                env=server_env,
                start_new_session=True,
                text=True,
            )
            server_started_by_script = True
        else:
            print(f"使用现有 SGLang Server: {args.health_host}:{args.port}")

        print("等待 /health 就绪...")
        start_time = time.time()
        while time.time() - start_time < args.server_timeout:
            if wait_for_health(args.health_host, args.port, 5):
                break
            if server_process is not None and server_process.poll() is not None:
                log_tail = read_tail(server_log_path)
                raise RuntimeError(
                    "SGLang Server 在健康检查通过前已退出。\n"
                    f"退出码: {server_process.returncode}\n"
                    f"日志文件: {server_log_path}\n"
                    f"最近日志:\n{log_tail or '(日志为空)'}"
                )
        else:
            raise RuntimeError(
                f"SGLang 服务在 {args.server_timeout}s 内未就绪，请检查日志: {server_log_path}"
            )

        print("SGLang 服务已就绪，开始压测...")
        print(f"Benchmark 命令: {quote_command(bench_command)}")
        completed = subprocess.run(bench_command, env=os.environ.copy(), check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"sglang.bench_serving 退出码非 0: {completed.returncode}")

        if not raw_result_path.exists():
            raise RuntimeError(f"未生成 bench_serving 结果文件: {raw_result_path}")

        benchmark_result = load_json_file(str(raw_result_path))
        result = {
            "backend": "sglang",
            "requested_model_path": args.model_path,
            "model": model_path,
            "dataset_name": args.dataset_name,
            "server_started_by_script": server_started_by_script,
            "server": {
                "host": args.host,
                "health_host": args.health_host,
                "port": args.port,
                "tensor_parallel_size": args.tensor_parallel_size,
                "mem_fraction_static": args.mem_fraction_static,
                "cuda_graph_max_bs": args.cuda_graph_max_bs,
                "attention_backend": args.attention_backend,
                "max_prefill_tokens": args.max_prefill_tokens,
                "disable_radix_cache": args.disable_radix_cache,
                "disable_overlap_schedule": args.disable_overlap_schedule,
                "trust_remote_code": args.trust_remote_code,
                "env": {
                    key: server_env[key]
                    for key in (*DEFAULT_SERVER_ENV, *OPTIONAL_SERVER_ENV_KEYS)
                    if key in server_env
                },
                "log_file": str(server_log_path),
                "launch_command": quote_command(server_command),
            },
            "benchmark_config": {
                "num_prompts": args.num_prompts,
                "input_length": args.input_length,
                "output_length": args.output_length,
                "warmup_requests": args.warmup_requests,
                "max_concurrency": args.max_concurrency,
                "random_range_ratio": args.random_range_ratio,
                "apply_chat_template": args.apply_chat_template,
                "dataset_path": args.dataset_path,
                "request_rate": args.request_rate,
                "random_image_num_images": args.random_image_num_images,
                "random_image_resolution": args.random_image_resolution,
                "bench_command": quote_command(bench_command),
                "raw_result_file": str(raw_result_path),
            },
            "metrics": benchmark_result,
        }
        return result
    finally:
        if server_log_handle is not None:
            server_log_handle.flush()
            server_log_handle.close()
        if server_started_by_script and not args.keep_server:
            terminate_process(server_process)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SGLang MUSA Serving Benchmark")
    parser.add_argument("--model-path", type=str, default="/workspace/models/Qwen3-0.6B", help="模型路径")
    parser.add_argument("--input-length", type=int, default=3000, help="随机输入长度")
    parser.add_argument("--output-length", type=int, default=500, help="随机输出长度")
    parser.add_argument("--num-runs", dest="num_prompts", type=int, default=10, help="兼容旧参数名，等价于 --num-prompts")
    parser.add_argument("--num-prompts", dest="num_prompts", type=int, help="压测请求数")
    parser.add_argument("--warmup", dest="warmup_requests", type=int, default=2, help="兼容旧参数名，等价于 --warmup-requests")
    parser.add_argument("--warmup-requests", dest="warmup_requests", type=int, help="bench_serving warmup 请求数")
    parser.add_argument("--backend", type=str, default="sglang", choices=["sglang", "vllm", "both"], help="保留旧接口；vllm 请改用 launch_vllm_server.sh + run_vllm_benchmark.sh")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="输出结果文件")

    parser.add_argument("--python-executable", type=str, default=sys.executable, help="用于启动 server 和 bench 的 Python")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="SGLang server host")
    parser.add_argument("--health-host", type=str, default="127.0.0.1", help="health check 使用的 host")
    parser.add_argument("--port", type=int, default=30001, help="SGLang server 端口")
    parser.add_argument("--server-timeout", type=int, default=2400, help="等待 server ready 的超时时间（秒）")
    parser.add_argument("--server-log", type=str, default=None, help="服务日志文件路径")
    parser.add_argument("--use-existing-server", action="store_true", help="不启动新服务，直接压测现有服务")
    parser.add_argument("--keep-server", action="store_true", help="脚本退出后保留当前脚本启动的服务")

    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--mem-fraction-static", type=float, default=0.9, help="launch_server --mem-fraction-static")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=256, help="launch_server --cuda-graph-max-bs")
    parser.add_argument("--attention-backend", type=str, default="fa3", help="launch_server --attention-backend")
    parser.add_argument("--max-prefill-tokens", type=int, default=None, help="可选的 launch_server --max-prefill-tokens")
    parser.add_argument("--server-env", action="append", default=[], metavar="KEY=VALUE", help="附加或覆盖 server 环境变量")

    parser.add_argument("--dataset-name", type=str, default="random", choices=["random", "random-image", "sharegpt"], help="bench_serving 数据集类型")
    parser.add_argument("--dataset-path", type=str, default=None, help="sharegpt 等数据集路径")
    parser.add_argument("--max-concurrency", type=int, default=64, help="bench_serving 最大并发")
    parser.add_argument("--random-range-ratio", type=float, default=1.0, help="bench_serving --random-range-ratio")
    parser.add_argument("--request-rate", type=float, default=1.6, help="可选的 bench_serving --request-rate")
    parser.add_argument("--random-image-num-images", type=int, default=None, help="VL 模型随机图片数")
    parser.add_argument("--random-image-resolution", type=str, default=None, help="VL 模型随机图片分辨率，例如 1148x112")

    parser.set_defaults(
        trust_remote_code=True,
        disable_radix_cache=True,
        disable_overlap_schedule=True,
        apply_chat_template=True,
    )
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", help="启用 trust remote code")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false", help="关闭 trust remote code")
    parser.add_argument("--disable-radix-cache", dest="disable_radix_cache", action="store_true", help="禁用 radix cache")
    parser.add_argument("--enable-radix-cache", dest="disable_radix_cache", action="store_false", help="启用 radix cache")
    parser.add_argument("--disable-overlap-schedule", dest="disable_overlap_schedule", action="store_true", help="禁用 overlap schedule")
    parser.add_argument("--enable-overlap-schedule", dest="disable_overlap_schedule", action="store_false", help="启用 overlap schedule")
    parser.add_argument("--apply-chat-template", dest="apply_chat_template", action="store_true", help="bench_serving 时应用 chat template")
    parser.add_argument("--no-apply-chat-template", dest="apply_chat_template", action="store_false", help="bench_serving 时不应用 chat template")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path)
    all_results = {}

    if args.backend in {"sglang", "both"}:
        all_results["sglang"] = benchmark_sglang(args, model_path)

    if args.backend in {"vllm", "both"}:
        all_results["vllm"] = {
            "backend": "vllm",
            "skipped": True,
            "reason": "当前脚本未实现 vLLM 服务流；请改用 launch_vllm_server.sh + run_vllm_benchmark.sh。",
        }
        if args.backend == "vllm":
            print("vLLM 基准测试未在这个入口实现；请改用 launch_vllm_server.sh + run_vllm_benchmark.sh。")

    if not all_results:
        raise SystemExit("没有可执行的 benchmark backend。")

    save_results(args.output, all_results)


if __name__ == "__main__":
    main()
