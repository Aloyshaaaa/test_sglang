#!/usr/bin/env python3
"""
PyTorch Profiler 性能分析脚本 - 摩尔线程 MUSA 环境
用于分析 Qwen3-0.6B 推理过程中的算子瓶颈
"""

import argparse
import json
import time
import torch
import torch.profiler
from torch.profiler import ProfilerActivity, profile, record_function
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 摩尔线程 MUSA 环境检测
try:
    import torch_musa
    if torch.musa.is_available():
        print("✓ 检测到摩尔线程 MUSA 环境")
        device = torch.device("musa")
        BACKEND = "musa"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        BACKEND = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BACKEND = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ 使用 {BACKEND.upper()} 后端")

def synchronize_device():
    """同步设备"""
    if BACKEND == "musa":
        torch.musa.synchronize()
    elif BACKEND == "cuda":
        torch.cuda.synchronize()


class InferenceProfiler:
    """推理性能分析器"""
    
    def __init__(self, model_path: str, input_length: int = 3000, output_length: int = 500):
        self.model_path = model_path
        self.input_length = input_length
        self.output_length = output_length
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """加载模型和分词器"""
        print(f"\n{'='*60}")
        print(f"加载模型: {self.model_path}")
        print(f"{'='*60}\n")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            print("正在加载模型到设备...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
            
            print(f"✓ 模型加载完成")
            print(f"  设备: {device}")
            print(f"  模型参数量: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
            
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise
    
    def generate_prompt(self, length: int) -> str:
        """生成测试 prompt"""
        base_text = """人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于创造能够模拟人类智能的系统。
        这些系统可以学习、推理、感知环境并做出决策。机器学习是AI的一个重要子领域，它使计算机能够从数据中学习而无需明确编程。
        深度学习是机器学习的一种方法，使用多层神经网络来模拟人脑的工作方式。"""
        
        repeat_times = (length // len(base_text)) + 1
        prompt = (base_text * repeat_times)[:length]
        return prompt
    
    def profile_with_pytorch_profiler(self, num_iterations: int = 5) -> Dict:
        """
        使用 PyTorch Profiler 进行详细性能分析
        """
        print(f"\n{'='*60}")
        print(f"PyTorch Profiler 性能分析")
        print(f"输入长度: {self.input_length}, 输出长度: {self.output_length}")
        print(f"迭代次数: {num_iterations}")
        print(f"{'='*60}\n")
        
        if self.model is None:
            self.load_model()
        
        # 准备输入
        prompt = self.generate_prompt(self.input_length)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"输入 tokens: {inputs['input_ids'].shape[1]}")
        
        # Warmup
        print("\nWarmup...")
        with torch.no_grad():
            for _ in range(2):
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
        synchronize_device()
        
        # 使用 PyTorch Profiler
        print("\n开始性能分析...")
        
        activities = [ProfilerActivity.CPU]
        if BACKEND == "musa":
            activities.append(ProfilerActivity.MUSA)
        elif BACKEND == "cuda":
            activities.append(ProfilerActivity.CUDA)
        
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            with torch.no_grad():
                for i in range(num_iterations):
                    with record_function(f"iteration_{i}"):
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.output_length,
                            do_sample=False,
                            use_cache=True
                        )
                    synchronize_device()
        
        # 导出结果
        print("\n导出分析结果...")
        
        # 1. 按 CPU 时间排序
        cpu_trace = prof.key_averages().table(
            sort_by="cpu_time_total", 
            row_limit=50
        )
        
        # 2. 按设备时间排序
        device_sort_key = "musa_time_total" if BACKEND == "musa" else "cuda_time_total"
        try:
            device_trace = prof.key_averages().table(
                sort_by=device_sort_key,
                row_limit=50
            )
        except:
            device_trace = "设备时间数据不可用"
        
        # 3. 导出 Chrome trace
        trace_file = f"profiler_trace_{BACKEND}.json"
        prof.export_chrome_trace(trace_file)
        print(f"✓ Chrome trace 已保存: {trace_file}")
        
        # 4. 导出详细统计
        stats_file = f"profiler_stats_{BACKEND}.txt"
        with open(stats_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"PyTorch Profiler Report - {BACKEND.upper()}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Input length: {self.input_length}, Output length: {self.output_length}\n")
            f.write("="*80 + "\n\n")
            
            f.write("【按 CPU 时间排序】\n")
            f.write(cpu_trace)
            f.write("\n\n")
            
            f.write(f"【按 {BACKEND.upper()} 时间排序】\n")
            f.write(device_trace)
        
        print(f"✓ 统计报告已保存: {stats_file}")
        
        # 解析关键指标
        key_averages = prof.key_averages()
        operator_stats = []
        
        for event in key_averages:
            op_stat = {
                "name": event.key,
                "cpu_time_ms": event.cpu_time_total / 1000.0,
                "cpu_time_percent": event.cpu_time_total / sum(e.cpu_time_total for e in key_averages) * 100 if sum(e.cpu_time_total for e in key_averages) > 0 else 0,
                "device_time_ms": getattr(event, f"{BACKEND}_time_total", 0) / 1000.0 if hasattr(event, f"{BACKEND}_time_total") else 0,
                "calls": event.count,
            }
            operator_stats.append(op_stat)
        
        # 按设备时间排序
        operator_stats.sort(key=lambda x: x["device_time_ms"], reverse=True)
        
        results = {
            "backend": BACKEND,
            "model": self.model_path,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "num_iterations": num_iterations,
            "top_operators": operator_stats[:30],
            "trace_file": trace_file,
            "stats_file": stats_file,
        }
        
        return results
    
    def profile_layer_by_layer(self) -> Dict:
        """
        逐层分析模型性能
        """
        print(f"\n{'='*60}")
        print(f"逐层性能分析")
        print(f"{'='*60}\n")
        
        if self.model is None:
            self.load_model()
        
        prompt = self.generate_prompt(self.input_length)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 获取模型层
        model_layers = []
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            model_layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            model_layers = self.model.transformer.h
        
        print(f"检测到 {len(model_layers)} 个 Transformer 层")
        
        layer_times = []
        
        # 分析每一层
        for idx, layer in enumerate(model_layers):
            # 使用 hooks 测量每层时间
            times = []
            
            def pre_hook(module, input):
                module.start_time = time.perf_counter()
            
            def post_hook(module, input, output):
                module.end_time = time.perf_counter()
                times.append(module.end_time - module.start_time)
            
            handle_pre = layer.register_forward_pre_hook(pre_hook)
            handle_post = layer.register_forward_hook(post_hook)
            
            # 运行几次取平均
            with torch.no_grad():
                for _ in range(3):
                    _ = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False
                    )
                    synchronize_device()
            
            handle_pre.remove()
            handle_post.remove()
            
            avg_time = sum(times) / len(times) if times else 0
            layer_times.append({
                "layer": idx,
                "avg_time_ms": avg_time * 1000,
            })
            
            print(f"Layer {idx}: {avg_time*1000:.2f} ms")
        
        # 找出最慢的层
        layer_times.sort(key=lambda x: x["avg_time_ms"], reverse=True)
        
        print(f"\n{'='*60}")
        print("最慢的 5 个层:")
        for i, layer_info in enumerate(layer_times[:5]):
            print(f"  {i+1}. Layer {layer_info['layer']}: {layer_info['avg_time_ms']:.2f} ms")
        print(f"{'='*60}\n")
        
        return {
            "layer_times": layer_times,
            "slowest_layers": layer_times[:10]
        }
    
    def analyze_memory_usage(self) -> Dict:
        """
        分析内存使用情况
        """
        print(f"\n{'='*60}")
        print(f"内存使用分析")
        print(f"{'='*60}\n")
        
        if self.model is None:
            self.load_model()
        
        # 获取初始内存
        if BACKEND == "musa":
            torch.musa.reset_peak_memory_stats()
            initial_memory = torch.musa.memory_allocated() / 1024**2
        elif BACKEND == "cuda":
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / 1024**2
        else:
            initial_memory = 0
        
        print(f"初始显存使用: {initial_memory:.2f} MB")
        
        # 运行推理
        prompt = self.generate_prompt(self.input_length)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.output_length,
                do_sample=False
            )
        
        synchronize_device()
        
        # 获取峰值内存
        if BACKEND == "musa":
            peak_memory = torch.musa.max_memory_allocated() / 1024**2
            current_memory = torch.musa.memory_allocated() / 1024**2
        elif BACKEND == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
            current_memory = torch.cuda.memory_allocated() / 1024**2
        else:
            peak_memory = 0
            current_memory = 0
        
        print(f"峰值显存使用: {peak_memory:.2f} MB")
        print(f"当前显存使用: {current_memory:.2f} MB")
        print(f"推理显存占用: {peak_memory - initial_memory:.2f} MB")
        
        return {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "current_memory_mb": current_memory,
            "inference_memory_mb": peak_memory - initial_memory,
        }
    
    def save_results(self, results: Dict, filename: str = "profile_results.json"):
        """保存分析结果"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 结果已保存到: {filename}")


def main():
    parser = argparse.ArgumentParser(description='PyTorch Profiler for Qwen3-0.6B on MUSA')
    parser.add_argument('--model-path', type=str, default='/workspace/models/Qwen3-0.6B',
                        help='模型路径')
    parser.add_argument('--input-length', type=int, default=3000,
                        help='输入长度')
    parser.add_argument('--output-length', type=int, default=500,
                        help='输出长度')
    parser.add_argument('--profile-iterations', type=int, default=5,
                        help='Profiler 迭代次数')
    parser.add_argument('--output', type=str, default='profile_results.json',
                        help='输出结果文件')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['profiler', 'layer', 'memory', 'all'],
                        help='分析模式')
    
    args = parser.parse_args()
    
    profiler = InferenceProfiler(
        model_path=args.model_path,
        input_length=args.input_length,
        output_length=args.output_length,
    )
    
    all_results = {
        "backend": BACKEND,
        "model": args.model_path,
        "input_length": args.input_length,
        "output_length": args.output_length,
    }
    
    if args.mode in ['profiler', 'all']:
        print("\n" + "="*60)
        print("模式 1: PyTorch Profiler 详细分析")
        print("="*60)
        profiler_results = profiler.profile_with_pytorch_profiler(args.profile_iterations)
        all_results["profiler"] = profiler_results
    
    if args.mode in ['layer', 'all']:
        print("\n" + "="*60)
        print("模式 2: 逐层性能分析")
        print("="*60)
        layer_results = profiler.profile_layer_by_layer()
        all_results["layer_analysis"] = layer_results
    
    if args.mode in ['memory', 'all']:
        print("\n" + "="*60)
        print("模式 3: 内存使用分析")
        print("="*60)
        memory_results = profiler.analyze_memory_usage()
        all_results["memory"] = memory_results
    
    # 保存结果
    profiler.save_results(all_results, args.output)
    
    print(f"\n{'='*60}")
    print("性能分析完成!")
    print(f"{'='*60}")
    print(f"\n生成的文件:")
    print(f"  - {args.output} (JSON 结果)")
    if args.mode in ['profiler', 'all']:
        print(f"  - profiler_trace_{BACKEND}.json (Chrome trace)")
        print(f"  - profiler_stats_{BACKEND}.txt (详细统计)")
    print(f"\n使用 Chrome 浏览器访问 chrome://tracing 加载 trace 文件进行可视化分析")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
