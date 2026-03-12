#!/usr/bin/env python3
"""
SGLang Benchmark Script for Qwen3-0.6B
输入长度: 3000, 输出长度: 500
"""

import argparse
import inspect
import json
import time
import torch
import statistics

# 摩尔线程 MUSA 环境检测和适配
MUSA_AVAILABLE = False
try:
    import torch_musa
    if torch.musa.is_available():
        MUSA_AVAILABLE = True
        print("检测到摩尔线程 MUSA 环境")
        device = torch.device("musa")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("检测到 CUDA 环境")
    else:
        print("使用 CPU 环境")


def get_device_type(explicit_device: str = "auto") -> str:
    """返回当前推理设备类型。"""
    if explicit_device != "auto":
        return explicit_device
    if MUSA_AVAILABLE:
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def synchronize_device():
    """同步设备（支持 MUSA 和 CUDA）"""
    if MUSA_AVAILABLE:
        torch.musa.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()

# 尝试导入 SGLang
# 如果未安装，可以使用 pip install sglang
try:
    import sglang as sgl
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    print("警告: SGLang 未安装，将使用 PyTorch 原生推理作为替代")
    print("安装命令: pip install sglang")

SGLANG_FRONTEND_AVAILABLE = False
if SGLANG_AVAILABLE:
    try:
        from sglang import function, system, user, assistant, gen, set_default_backend
        SGLANG_FRONTEND_AVAILABLE = True
    except ImportError:
        # 新版本可能只保留 Engine 离线接口，不强依赖前端 DSL。
        SGLANG_FRONTEND_AVAILABLE = False


class SGLangBenchmark:
    """SGLang 推理性能测试类"""
    
    def __init__(
        self,
        model_path: str,
        input_length: int = 3000,
        output_length: int = 500,
        device_type: str = "auto",
    ):
        self.model_path = model_path
        self.input_length = input_length
        self.output_length = output_length
        self.device_type = get_device_type(device_type)
        self.results = []
        
    def generate_prompt(self, length: int) -> str:
        """生成指定长度的测试 prompt"""
        # 使用重复文本来模拟长输入
        base_text = """人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。
        这些系统可以学习、推理、感知环境并做出决策。机器学习是AI的一个重要子领域，它使计算机能够从数据中学习而无需明确编程。
        深度学习是机器学习的一种方法，使用多层神经网络来模拟人脑的工作方式。"""
        
        # 计算需要重复的次数
        repeat_times = (length // len(base_text)) + 1
        prompt = (base_text * repeat_times)[:length]
        return prompt

    def _callable_supports_kwarg(self, callable_obj, arg_name: str) -> bool:
        """检查可调用对象是否显式声明了某个参数。"""
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False
        return arg_name in signature.parameters

    def _create_sglang_backend(self):
        """兼容不同版本 SGLang 的后端初始化方式。"""
        constructor_candidates = []
        runtime_ctor = getattr(sgl, "Runtime", None)
        engine_ctor = getattr(sgl, "Engine", None)
        if runtime_ctor is not None:
            constructor_candidates.append(("Runtime", runtime_ctor))
        if engine_ctor is not None:
            constructor_candidates.append(("Engine", engine_ctor))

        if not constructor_candidates:
            raise RuntimeError("当前 SGLang 版本未找到 Runtime 或 Engine 入口")

        request_keys = ["max_running_requests", "max_num_reqs", None]
        tp_keys = ["tp_size", "tensor_parallel_size"]
        model_keys = ["model_path", "model"]
        device_values = [self.device_type]
        if self.device_type != "cpu":
            device_values.append(None)

        errors = []
        for constructor_name, constructor in constructor_candidates:
            for model_key in model_keys:
                for tp_key in tp_keys:
                    base_kwargs = {model_key: self.model_path, tp_key: 1}
                    for request_key in request_keys:
                        kwargs = dict(base_kwargs)
                        if request_key is not None:
                            kwargs[request_key] = 1

                        for device_value in device_values:
                            attempt_kwargs = dict(kwargs)
                            if device_value is not None:
                                attempt_kwargs["device"] = device_value

                            try:
                                backend = constructor(**attempt_kwargs)
                                print(
                                    f"SGLang 后端已初始化: {constructor_name}"
                                    f" ({attempt_kwargs})"
                                )
                                return backend
                            except TypeError as exc:
                                errors.append(
                                    f"{constructor_name}{attempt_kwargs}: {exc}"
                                )
                                continue
                            except Exception as exc:
                                raise RuntimeError(
                                    f"{constructor_name} 初始化失败: {exc}"
                                ) from exc

        raise RuntimeError(" ; ".join(errors[-6:]))

    def _extract_sglang_text(self, output) -> str:
        """从不同版本 SGLang 返回值中提取文本。"""
        if isinstance(output, dict):
            for key in ("answer", "text", "generated_text", "output_text"):
                value = output.get(key)
                if isinstance(value, str):
                    return value
            return str(output)

        if isinstance(output, list) and output:
            return self._extract_sglang_text(output[0])

        if hasattr(output, "text") and isinstance(output.text, str):
            return output.text

        return str(output)

    def _build_sglang_runner(self, backend):
        """优先使用前端 DSL，失败时回退到 Engine.generate。"""
        if SGLANG_FRONTEND_AVAILABLE:
            try:
                set_default_backend(backend)

                @function
                def qa(s, question):
                    s += system("你是一个有用的AI助手。")
                    s += user(question)
                    s += assistant(gen("answer", max_tokens=self.output_length))

                def run_once(question: str) -> str:
                    state = qa.run(question=question)
                    return self._extract_sglang_text(state)

                return run_once
            except Exception as exc:
                print(f"SGLang 前端 DSL 不可用，回退到 Engine.generate: {exc}")

        if not hasattr(backend, "generate"):
            raise RuntimeError("当前 SGLang 后端既不支持前端 DSL，也不支持 generate 接口")

        def run_once(question: str) -> str:
            sampling_variants = [
                {"temperature": 0.7, "max_new_tokens": self.output_length},
                {"temperature": 0.7, "max_tokens": self.output_length},
            ]
            last_error = None
            for sampling_params in sampling_variants:
                try:
                    outputs = backend.generate([question], sampling_params)
                    return self._extract_sglang_text(outputs)
                except TypeError as exc:
                    last_error = exc
                    continue
                except Exception as exc:
                    error_text = str(exc)
                    if "max_new_tokens" in error_text or "max_tokens" in error_text:
                        last_error = exc
                        continue
                    raise
            raise RuntimeError(f"SGLang generate 调用失败: {last_error}")

        return run_once
    
    def benchmark_sglang(self, num_runs: int = 10, warmup: int = 2) -> dict:
        """使用 SGLang 进行推理测试"""
        if not SGLANG_AVAILABLE:
            print("SGLang 不可用，跳过 SGLang 测试")
            return {}
        
        print(f"\n{'='*60}")
        print(f"SGLang Benchmark - Qwen3-0.6B")
        print(f"输入长度: {self.input_length}, 输出长度: {self.output_length}")
        print(f"{'='*60}\n")
        
        # 初始化 SGLang 后端
        print("正在初始化 SGLang 后端...")
        try:
            backend = self._create_sglang_backend()
            run_once = self._build_sglang_runner(backend)
        except Exception as e:
            print(f"SGLang 初始化失败: {e}")
            return {}
        
        # 生成测试 prompt
        prompt = self.generate_prompt(self.input_length)
        print(f"Prompt 长度: {len(prompt)} 字符")
        
        # Warmup
        print(f"\nWarmup 运行 {warmup} 次...")
        for _ in range(warmup):
            try:
                _ = run_once(prompt)
            except Exception as e:
                print(f"Warmup 失败: {e}")
                return {}
        
        # 正式测试
        print(f"开始正式测试 {num_runs} 次...\n")
        latencies = []
        tokens_per_sec = []
        
        for i in range(num_runs):
            synchronize_device()
            start_time = time.perf_counter()
            
            try:
                answer = run_once(prompt)
                
                synchronize_device()
                end_time = time.perf_counter()
                
                latency = end_time - start_time
                latencies.append(latency)
                
                # 估算生成的 token 数
                output_tokens = len(answer) // 4  # 粗略估算
                tps = output_tokens / latency if latency > 0 else 0
                tokens_per_sec.append(tps)
                
                print(f"Run {i+1}/{num_runs}: {latency:.3f}s, {tps:.2f} tokens/s")
                
            except Exception as e:
                print(f"Run {i+1} 失败: {e}")
                continue
        
        # 计算统计指标
        if latencies:
            results = {
                "backend": "sglang",
                "model": self.model_path,
                "input_length": self.input_length,
                "output_length": self.output_length,
                "num_runs": len(latencies),
                "avg_latency": statistics.mean(latencies),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "p50_latency": statistics.median(latencies),
                "p99_latency": sorted(latencies)[int(len(latencies)*0.99)] if len(latencies) >= 100 else max(latencies),
                "avg_tokens_per_sec": statistics.mean(tokens_per_sec),
                "latencies": latencies,
            }
            
            print(f"\n{'='*60}")
            print("SGLang 测试结果:")
            print(f"{'='*60}")
            print(f"平均延迟: {results['avg_latency']:.3f}s")
            print(f"最小延迟: {results['min_latency']:.3f}s")
            print(f"最大延迟: {results['max_latency']:.3f}s")
            print(f"P50 延迟: {results['p50_latency']:.3f}s")
            print(f"P99 延迟: {results['p99_latency']:.3f}s")
            print(f"平均吞吐: {results['avg_tokens_per_sec']:.2f} tokens/s")
            print(f"{'='*60}\n")
            
            return results
        
        return {}
    
    def benchmark_vllm(self, num_runs: int = 10, warmup: int = 2) -> dict:
        """使用 vLLM 进行推理测试（作为对比）"""
        print(f"\n{'='*60}")
        print(f"vLLM Benchmark - Qwen3-0.6B")
        print(f"输入长度: {self.input_length}, 输出长度: {self.output_length}")
        print(f"{'='*60}\n")
        
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            print("vLLM 未安装，跳过 vLLM 测试")
            return {}
        
        print("正在加载 vLLM 模型...")
        try:
            llm_kwargs = dict(
                model=self.model_path,
                tensor_parallel_size=1,
                max_model_len=4096,
                device=self.device_type,
            )
            print(f"vLLM 设备类型: {self.device_type}")
            llm = LLM(**llm_kwargs)
        except Exception as e:
            print(f"vLLM 加载失败: {e}")
            return {}
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=self.output_length,
        )
        
        # 生成测试 prompt
        prompt = self.generate_prompt(self.input_length)
        prompts = [prompt]
        
        # Warmup
        print(f"\nWarmup 运行 {warmup} 次...")
        for _ in range(warmup):
            outputs = llm.generate(prompts, sampling_params)
        
        # 正式测试
        print(f"开始正式测试 {num_runs} 次...\n")
        latencies = []
        
        for i in range(num_runs):
            synchronize_device()
            start_time = time.perf_counter()
            
            outputs = llm.generate(prompts, sampling_params)
            
            synchronize_device()
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            latencies.append(latency)
            
            output_tokens = len(outputs[0].outputs[0].token_ids)
            tps = output_tokens / latency if latency > 0 else 0
            
            print(f"Run {i+1}/{num_runs}: {latency:.3f}s, {tps:.2f} tokens/s")
        
        if latencies:
            results = {
                "backend": "vllm",
                "model": self.model_path,
                "input_length": self.input_length,
                "output_length": self.output_length,
                "num_runs": len(latencies),
                "avg_latency": statistics.mean(latencies),
                "min_latency": min(latencies),
                "max_latency": max(latencies),
                "p50_latency": statistics.median(latencies),
                "p99_latency": sorted(latencies)[int(len(latencies)*0.99)] if len(latencies) >= 100 else max(latencies),
                "latencies": latencies,
            }
            
            print(f"\n{'='*60}")
            print("vLLM 测试结果:")
            print(f"{'='*60}")
            print(f"平均延迟: {results['avg_latency']:.3f}s")
            print(f"最小延迟: {results['min_latency']:.3f}s")
            print(f"最大延迟: {results['max_latency']:.3f}s")
            print(f"P50 延迟: {results['p50_latency']:.3f}s")
            print(f"P99 延迟: {results['p99_latency']:.3f}s")
            print(f"{'='*60}\n")
            
            return results
        
        return {}
    
    def save_results(self, results: dict, filename: str = "benchmark_results.json"):
        """保存测试结果到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {filename}")


def main():
    parser = argparse.ArgumentParser(description='SGLang Benchmark for Qwen3-0.6B')
    parser.add_argument('--model-path', type=str, default='/workspace/models/Qwen3-0.6B',
                        help='模型路径')
    parser.add_argument('--input-length', type=int, default=3000,
                        help='输入长度')
    parser.add_argument('--output-length', type=int, default=500,
                        help='输出长度')
    parser.add_argument('--num-runs', type=int, default=10,
                        help='测试运行次数')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Warmup 次数')
    parser.add_argument('--backend', type=str, default='sglang',
                        choices=['sglang', 'vllm', 'both'],
                        help='测试后端')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'musa', 'cuda', 'cpu'],
                        help='推理设备类型')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='输出结果文件')
    
    args = parser.parse_args()
    
    benchmark = SGLangBenchmark(
        model_path=args.model_path,
        input_length=args.input_length,
        output_length=args.output_length,
        device_type=args.device,
    )
    
    all_results = {}
    
    if args.backend in ['sglang', 'both']:
        sgl_results = benchmark.benchmark_sglang(
            num_runs=args.num_runs,
            warmup=args.warmup
        )
        if sgl_results:
            all_results['sglang'] = sgl_results
    
    if args.backend in ['vllm', 'both']:
        vllm_results = benchmark.benchmark_vllm(
            num_runs=args.num_runs,
            warmup=args.warmup
        )
        if vllm_results:
            all_results['vllm'] = vllm_results
    
    if all_results:
        benchmark.save_results(all_results, args.output)


if __name__ == '__main__':
    main()
