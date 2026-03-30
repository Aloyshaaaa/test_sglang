#!/usr/bin/env bash
# Ring-FlashAttention 扩展配置
# 在 sglang_benchmark_common.sh 中添加

# ========== Ring-FlashAttention 配置 ==========
# 序列并行大小 (每个ring组的GPU数)
export RING_PARALLEL_SIZE="${RING_PARALLEL_SIZE:-1}"

# Ring通信模式: ring | pipeline
export RING_COMM_MODE="${RING_COMM_MODE:-ring}"

# KV Cache分布策略: layer | sequence
export KV_CACHE_SHARD_STRATEGY="${KV_CACHE_SHARD_STRATEGY:-layer}"

# 通信计算重叠
export ENABLE_OVERLAP="${ENABLE_OVERLAP:-1}"

# Ring通信buffer大小 (MB)
export RING_BUFFER_SIZE_MB="${RING_BUFFER_SIZE_MB:-256}"

# FlashAttention版本 (针对MUSA优化)
export FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-3}"

# 长序列支持
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

# Ring-FA开关
export USE_RING_FLASH_ATTN="${USE_RING_FLASH_ATTN:-0}"

# ========== 帮助函数 ==========
validate_ring_config() {
    local tp_size="${TENSOR_PARALLEL_SIZE:-1}"
    local ring_size="${RING_PARALLEL_SIZE:-1}"
    
    if [ "$USE_RING_FLASH_ATTN" = "1" ]; then
        if [ "$ring_size" -gt "$tp_size" ]; then
            echo "错误: RING_PARALLEL_SIZE ($ring_size) 不能大于 TENSOR_PARALLEL_SIZE ($tp_size)" >&2
            return 1
        fi
        
        if [ $((tp_size % ring_size)) -ne 0 ]; then
            echo "警告: TENSOR_PARALLEL_SIZE ($tp_size) 不是 RING_PARALLEL_SIZE ($ring_size) 的倍数" >&2
            echo "  这可能导致负载不均衡"
        fi
        
        echo "Ring-FlashAttention 配置:"
        echo "  - TP Size: $tp_size"
        echo "  - Ring Size: $ring_size"
        echo "  - Ring Groups: $((tp_size / ring_size))"
        echo "  - KV Cache策略: $KV_CACHE_SHARD_STRATEGY"
        echo "  - 通信模式: $RING_COMM_MODE"
        echo "  - 最大序列长度: $MAX_SEQ_LEN"
    fi
}

print_ring_config() {
    if [ "$USE_RING_FLASH_ATTN" = "1" ]; then
        print_section "Ring-FlashAttention Enabled"
        echo "序列并行大小: $RING_PARALLEL_SIZE"
        echo "通信模式: $RING_COMM_MODE"
        echo "KV分片策略: $KV_CACHE_SHARD_STRATEGY"
        echo "最大序列: $MAX_SEQ_LEN"
        echo ""
    fi
}