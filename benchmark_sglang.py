#!/usr/bin/env python3
"""
SGLang Benchmark Script for Qwen3-0.6B
输入长度: 3000, 输出长度: 500
"""

import argparse
import json
import time
import torch
from typing import List, Tuple
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
    from sglang import function, system, user, assistant, gen, set_default_backend
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False
    print("警告: SGLang 未安装，将使用 PyTorch 原生推理作为替代")
    print("安装命令: pip install sglang")


class SGLangBenchmark:
    """SGLang 推理性能测试类"""
    
    def __init__(self, model_path: str, input_length: int = 3000, output_length: int = 500):
        self.model_path = model_path
        self.input_length = input_length
        self.output_length = output_length
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
            backend = sgl.Runtime(
                model_path=self.model_path,
                tp_size=1,  # Tensor parallelism size
                max_num_reqs=1,
            )
            set_default_backend(backend)
        except Exception as e:
            print(f"SGLang 初始化失败: {e}")
            return {}
        
        @function
        def qa(s, question):
            s += system("你是一个有用的AI助手。")
            s += user(question)
            s += assistant(gen("answer", max_tokens=self.output_length))
        
        # 生成测试 prompt
        prompt = self.generate_prompt(self.input_length)
        print(f"Prompt 长度: {len(prompt)} 字符")
        
        # Warmup
        print(f"\nWarmup 运行 {warmup} 次...")
        for _ in range(warmup):
            try:
                state = qa.run(question=prompt)
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
                state = qa.run(question=prompt)
                answer = state["answer"]
                
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
            llm = LLM(
                model=self.model_path,
                tensor_parallel_size=1,
                max_model_len=4096,
            )
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
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='输出结果文件')
    
    args = parser.parse_args()
    
    benchmark = SGLangBenchmark(
        model_path=args.model_path,
        input_length=args.input_length,
        output_length=args.output_length,
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
