#!/usr/bin/env python3
"""
快速测试脚本 - 验证环境和模型加载
"""

import torch
import sys

print("="*60)
print("SGLang Qwen3-0.6B 快速测试")
print("="*60)
print()

# 检查 MUSA
print("【环境检查】")
try:
    import torch_musa
    if torch.musa.is_available():
        print(f"✓ MUSA 可用")
        print(f"  设备数量: {torch.musa.device_count()}")
        print(f"  当前设备: {torch.musa.current_device()}")
        print(f"  设备名称: {torch.musa.get_device_name(0)}")
        device = torch.device("musa")
    else:
        print("✗ MUSA 不可用")
        device = torch.device("cpu")
except ImportError:
    print("✗ torch_musa 未安装")
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("✗ CUDA 不可用，使用 CPU")
        device = torch.device("cpu")

print()

# 检查 transformers
print("【依赖检查】")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✓ transformers 已安装")
except ImportError:
    print("✗ transformers 未安装")
    print("  安装命令: pip install transformers")
    sys.exit(1)

try:
    import accelerate
    print("✓ accelerate 已安装")
except ImportError:
    print("⚠ accelerate 未安装 (推荐安装)")
    print("  安装命令: pip install accelerate")

print()

# 检查模型路径
print("【模型检查】")
import os
model_path = "/workspace/models/Qwen3-0.6B"
if os.path.exists(model_path):
    print(f"✓ 模型路径存在: {model_path}")
    files = os.listdir(model_path)
    print(f"  文件数: {len(files)}")
    if 'config.json' in files:
        print("  ✓ 找到 config.json")
    else:
        print("  ⚠ 未找到 config.json")
else:
    print(f"✗ 模型路径不存在: {model_path}")
    print("  请检查模型路径")
    sys.exit(1)

print()

# 尝试加载 tokenizer
print("【加载测试】")
try:
    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("✓ Tokenizer 加载成功")
    
    # 测试编码
    test_text = "你好，世界！"
    tokens = tokenizer.encode(test_text)
    print(f"  测试编码: '{test_text}' -> {len(tokens)} tokens")
    
except Exception as e:
    print(f"✗ Tokenizer 加载失败: {e}")
    sys.exit(1)

print()

# 尝试加载模型（可选，可能较慢）
print("【模型加载测试】")
print("是否加载完整模型进行测试? (y/n)", end=" ")
try:
    response = input().strip().lower()
    if response == 'y':
        print("正在加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✓ 模型加载成功")
        
        # 测试推理
        print("\n测试推理...")
        inputs = tokenizer("人工智能是", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"输入: '人工智能是'")
        print(f"输出: '{result}'")
        print("✓ 推理测试成功")
    else:
        print("跳过模型加载测试")
except KeyboardInterrupt:
    print("\n跳过模型加载")
except Exception as e:
    print(f"✗ 模型加载或推理失败: {e}")

print()
print("="*60)
print("快速测试完成!")
print("="*60)
print()
print("下一步:")
print("  1. 运行完整测试: ./run_all_tests.sh")
print("  2. 或运行基准测试: python benchmark_sglang.py")
print("  3. 或运行性能分析: python profile_inference.py")
print()
