#!/usr/bin/env python3
"""
算子耗时分析工具 - 摩尔线程 MUSA 环境
分析推理过程中的算子瓶颈，按耗时排序
"""

import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import os


class OperatorAnalyzer:
    """算子耗时分析器"""
    
    def __init__(self, profile_file: str = None):
        self.profile_file = profile_file
        self.operators = defaultdict(lambda: {
            "count": 0,
            "cpu_time_ms": 0.0,
            "device_time_ms": 0.0,
            "avg_cpu_ms": 0.0,
            "avg_device_ms": 0.0,
        })
    
    def parse_profiler_output(self, stats_file: str) -> List[Dict]:
        """
        解析 PyTorch Profiler 的文本输出
        """
        print(f"解析 profiler 输出: {stats_file}")
        
        if not os.path.exists(stats_file):
            print(f"警告: 文件不存在: {stats_file}")
            return []
        
        with open(stats_file, 'r') as f:
            content = f.read()
        
        # 提取算子统计表格
        operators = []
        
        # 查找表格部分
        lines = content.split('\n')
        in_table = False
        headers = []
        
        for line in lines:
            # 检测表格开始
            if 'Name' in line and 'CPU time' in line:
                in_table = True
                headers = [h.strip() for h in line.split('|') if h.strip()]
                continue
            
            if in_table and line.strip() and not line.startswith('-') and not line.startswith('='):
                # 解析表格行
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 4:
                    op_name = parts[0]
                    if op_name and not op_name.startswith('---'):
                        try:
                            # 尝试解析时间
                            cpu_time = self._parse_time(parts[1]) if len(parts) > 1 else 0
                            device_time = self._parse_time(parts[2]) if len(parts) > 2 else 0
                            count = int(parts[3]) if len(parts) > 3 else 1
                            
                            operators.append({
                                "name": op_name,
                                "cpu_time_ms": cpu_time,
                                "device_time_ms": device_time,
                                "count": count,
                            })
                        except:
                            pass
        
        return operators
    
    def _parse_time(self, time_str: str) -> float:
        """解析时间字符串为毫秒"""
        time_str = time_str.strip()
        if not time_str or time_str == '-':
            return 0.0
        
        # 处理不同单位
        try:
            if 'us' in time_str:
                return float(time_str.replace('us', '').strip()) / 1000.0
            elif 'ms' in time_str:
                return float(time_str.replace('ms', '').strip())
            elif 's' in time_str:
                return float(time_str.replace('s', '').strip()) * 1000.0
            else:
                return float(time_str)
        except:
            return 0.0
    
    def analyze_from_json(self, json_file: str) -> Dict:
        """
        从 JSON 结果文件分析算子
        """
        print(f"分析 JSON 结果: {json_file}")
        
        if not os.path.exists(json_file):
            print(f"警告: 文件不存在: {json_file}")
            return {}
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if 'profiler' in data and 'top_operators' in data['profiler']:
            operators = data['profiler']['top_operators']
        elif 'top_operators' in data:
            operators = data['top_operators']
        else:
            print("未找到算子数据")
            return {}
        
        # 按设备时间排序
        operators.sort(key=lambda x: x.get('device_time_ms', 0), reverse=True)
        
        return {
            "operators": operators,
            "total_device_time": sum(op.get('device_time_ms', 0) for op in operators),
            "total_cpu_time": sum(op.get('cpu_time_ms', 0) for op in operators),
        }
    
    def categorize_operators(self, operators: List[Dict]) -> Dict:
        """
        按类型对算子进行分类
        """
        categories = defaultdict(list)
        
        for op in operators:
            name = op.get('name', '')
            
            # 分类规则
            if any(kw in name.lower() for kw in ['matmul', 'mm', 'bmm', 'linear']):
                category = "矩阵运算 (GEMM)"
            elif any(kw in name.lower() for kw in ['attention', 'softmax', 'scale']):
                category = "注意力机制 (Attention)"
            elif any(kw in name.lower() for kw in ['layernorm', 'norm', 'rmsnorm']):
                category = "归一化 (Normalization)"
            elif any(kw in name.lower() for kw in ['embedding', 'embed']):
                category = "嵌入层 (Embedding)"
            elif any(kw in name.lower() for kw in ['activation', 'gelu', 'relu', 'silu', 'swiglu']):
                category = "激活函数 (Activation)"
            elif any(kw in name.lower() for kw in ['conv', 'pool']):
                category = "卷积/池化 (Conv/Pool)"
            elif any(kw in name.lower() for kw in ['copy', 'cat', 'stack', 'split', 'slice']):
                category = "内存操作 (Memory)"
            elif any(kw in name.lower() for kw in ['add', 'mul', 'div', 'sub']):
                category = "逐元素运算 (Element-wise)"
            elif any(kw in name.lower() for kw in ['transpose', 'permute', 'view', 'reshape']):
                category = "张量变换 (Tensor Transform)"
            else:
                category = "其他 (Other)"
            
            categories[category].append(op)
        
        return dict(categories)
    
    def generate_report(self, analysis_result: Dict, output_file: str = "operator_analysis_report.txt"):
        """
        生成详细的分析报告
        """
        operators = analysis_result.get('operators', [])
        
        if not operators:
            print("没有算子数据可分析")
            return
        
        # 分类
        categories = self.categorize_operators(operators)
        
        # 计算各类别总时间
        category_stats = []
        for category, ops in categories.items():
            total_device_time = sum(op.get('device_time_ms', 0) for op in ops)
            total_cpu_time = sum(op.get('cpu_time_ms', 0) for op in ops)
            total_calls = sum(op.get('calls', 1) for op in ops)
            category_stats.append({
                "category": category,
                "device_time_ms": total_device_time,
                "cpu_time_ms": total_cpu_time,
                "calls": total_calls,
                "operator_count": len(ops),
            })
        
        category_stats.sort(key=lambda x: x['device_time_ms'], reverse=True)
        
        # 生成报告
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("算子耗时分析报告")
        report_lines.append("="*80)
        report_lines.append("")
        
        # 总体统计
        total_device_time = sum(op.get('device_time_ms', 0) for op in operators)
        total_cpu_time = sum(op.get('cpu_time_ms', 0) for op in operators)
        total_calls = sum(op.get('calls', 1) for op in operators)
        
        report_lines.append(f"总体统计:")
        report_lines.append(f"  总设备时间: {total_device_time:.2f} ms")
        report_lines.append(f"  总 CPU 时间: {total_cpu_time:.2f} ms")
        report_lines.append(f"  总调用次数: {total_calls}")
        report_lines.append(f"  算子种类数: {len(operators)}")
        report_lines.append("")
        
        # 按类别统计
        report_lines.append("="*80)
        report_lines.append("按类别统计 (按设备时间排序)")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append(f"{'类别':<30} {'设备时间(ms)':<15} {'占比(%)':<10} {'调用次数':<10} {'算子数':<10}")
        report_lines.append("-"*80)
        
        for stat in category_stats:
            percentage = (stat['device_time_ms'] / total_device_time * 100) if total_device_time > 0 else 0
            report_lines.append(
                f"{stat['category']:<30} {stat['device_time_ms']:<15.2f} {percentage:<10.2f} "
                f"{stat['calls']:<10} {stat['operator_count']:<10}"
            )
        
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("Top 20 耗时算子 (按设备时间排序)")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append(f"{'排名':<6} {'算子名称':<50} {'设备时间(ms)':<15} {'占比(%)':<10} {'调用次数':<10}")
        report_lines.append("-"*100)
        
        for i, op in enumerate(operators[:20], 1):
            device_time = op.get('device_time_ms', 0)
            percentage = (device_time / total_device_time * 100) if total_device_time > 0 else 0
            name = op.get('name', 'Unknown')[:48]
            calls = op.get('calls', 1)
            report_lines.append(f"{i:<6} {name:<50} {device_time:<15.2f} {percentage:<10.2f} {calls:<10}")
        
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("各类别详细算子")
        report_lines.append("="*80)
        report_lines.append("")
        
        for stat in category_stats:
            category = stat['category']
            report_lines.append(f"\n【{category}】")
            report_lines.append("-"*80)
            report_lines.append(f"{'算子名称':<50} {'设备时间(ms)':<15} {'CPU时间(ms)':<15} {'调用次数':<10}")
            report_lines.append("-"*100)
            
            cat_ops = categories[category]
            cat_ops.sort(key=lambda x: x.get('device_time_ms', 0), reverse=True)
            
            for op in cat_ops[:10]:  # 每类只显示前10个
                name = op.get('name', 'Unknown')[:48]
                device_time = op.get('device_time_ms', 0)
                cpu_time = op.get('cpu_time_ms', 0)
                calls = op.get('calls', 1)
                report_lines.append(f"{name:<50} {device_time:<15.2f} {cpu_time:<15.2f} {calls:<10}")
        
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("优化建议")
        report_lines.append("="*80)
        report_lines.append("")
        
        # 生成优化建议
        top_category = category_stats[0]['category'] if category_stats else None
        if top_category:
            report_lines.append(f"1. 主要瓶颈类别: {top_category}")
            report_lines.append(f"   占总设备时间的 {(category_stats[0]['device_time_ms'] / total_device_time * 100):.2f}%")
            report_lines.append("")
        
        # 检查特定瓶颈
        gemm_ops = categories.get("矩阵运算 (GEMM)", [])
        if gemm_ops:
            gemm_time = sum(op.get('device_time_ms', 0) for op in gemm_ops)
            if gemm_time / total_device_time > 0.5:
                report_lines.append("2. 矩阵运算占比较高，建议:")
                report_lines.append("   - 检查是否使用了最优的 GEMM 实现")
                report_lines.append("   - 考虑使用 FP16/BF16 精度")
                report_lines.append("   - 评估是否可以使用 Tensor Core")
                report_lines.append("")
        
        attention_ops = categories.get("注意力机制 (Attention)", [])
        if attention_ops:
            attn_time = sum(op.get('device_time_ms', 0) for op in attention_ops)
            if attn_time / total_device_time > 0.3:
                report_lines.append("3. 注意力机制占比较高，建议:")
                report_lines.append("   - 使用 FlashAttention 优化")
                report_lines.append("   - 考虑使用稀疏注意力")
                report_lines.append("   - 检查 KV Cache 效率")
                report_lines.append("")
        
        memory_ops = categories.get("内存操作 (Memory)", [])
        if memory_ops:
            mem_time = sum(op.get('device_time_ms', 0) for op in memory_ops)
            if mem_time / total_device_time > 0.1:
                report_lines.append("4. 内存操作占比较高，建议:")
                report_lines.append("   - 优化内存访问模式")
                report_lines.append("   - 减少不必要的数据拷贝")
                report_lines.append("   - 使用内存池管理")
                report_lines.append("")
        
        report_lines.append("="*80)
        
        # 写入文件
        report_text = "\n".join(report_lines)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # 同时打印到控制台
        print(report_text)
        print(f"\n✓ 报告已保存到: {output_file}")
        
        return report_text
    
    def export_for_visualization(self, analysis_result: Dict, output_file: str = "operators_for_viz.json"):
        """
        导出数据用于可视化
        """
        operators = analysis_result.get('operators', [])
        
        # 准备可视化数据
        viz_data = {
            "top_operators": operators[:50],
            "categories": self.categorize_operators(operators),
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 可视化数据已导出: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='算子耗时分析工具')
    parser.add_argument('--input', type=str, required=True,
                        help='输入文件 (profile_results.json 或 profiler_stats_*.txt)')
    parser.add_argument('--output', type=str, default='operator_analysis_report.txt',
                        help='输出报告文件')
    parser.add_argument('--viz-output', type=str, default='operators_for_viz.json',
                        help='可视化数据输出文件')
    
    args = parser.parse_args()
    
    analyzer = OperatorAnalyzer()
    
    # 根据文件类型选择解析方式
    if args.input.endswith('.json'):
        result = analyzer.analyze_from_json(args.input)
    else:
        operators = analyzer.parse_profiler_output(args.input)
        result = {
            "operators": operators,
            "total_device_time": sum(op.get('device_time_ms', 0) for op in operators),
            "total_cpu_time": sum(op.get('cpu_time_ms', 0) for op in operators),
        }
    
    if result:
        # 生成报告
        analyzer.generate_report(result, args.output)
        
        # 导出可视化数据
        analyzer.export_for_visualization(result, args.viz_output)
    else:
        print("分析失败，请检查输入文件")


if __name__ == '__main__':
    main()
