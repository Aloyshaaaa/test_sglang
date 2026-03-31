#!/bin/bash

# ================= 配置区域 =================
# 宿主机结果保存路径
HOST_RESULT_DIR="/home/mccxadmin/workspace/aloysha/qwen_result"
# 模型源路径
HOST_MODEL_DIR="/home/mccxadmin/workspace/aloysha"
# 容器名称
CONTAINER_NAME="sglang_qwen_test_auto"
# 镜像名称
IMAGE_NAME="registry.mthreads.com/mcconline/sglang:v0.5.2-ph1-4.3.2-kuae2.1-20251227-1614"

# 待测试模型列表 (请确保宿主机路径 HOST_MODEL_DIR 下有这些文件夹)
MODELS=("Qwen3-0.6B" "Qwen2.5-0.5B" "Qwen3-VL-8B-Instruct" "Qwen3.5-0.8B")
# MODELS=("Qwen3-8B-FP8")

# SGLang 服务端口
SGLANG_PORT=30001

# 服务启动超时时间 (秒) - 40分钟
SERVER_TIMEOUT=2400

# 每次运行生成一个新的文件名，避免覆盖
CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
CSV_FILENAME="benchmark_report_${CURRENT_TIME}.csv"
# ===========================================

# 1. 检查并启动容器
echo "[Step 1] Checking Container Status..."
mkdir -p "$HOST_RESULT_DIR"

if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        echo "Removing old stopped container..."
        docker rm $CONTAINER_NAME
    fi
    echo "Starting new container..."
    docker run -d \
        --name $CONTAINER_NAME \
        --runtime runc \
        --net host \
        --privileged \
        --pid=host \
        --shm-size 500g \
        -v "$HOST_MODEL_DIR":/data/model \
        -v "$HOST_RESULT_DIR":/sgl-workspace/results \
        $IMAGE_NAME \
        sleep inf
else
    echo "Container $CONTAINER_NAME is already running."
fi

# 2. 注入测试脚本 (增加数据集路径逻辑)
echo "[Step 2] Injecting benchmark script into container..."

cat << 'EOF' | docker exec -i $CONTAINER_NAME bash -c "cat > /sgl-workspace/run_bench_inner.sh"
#!/bin/bash
MODEL_NAME=$1
TARGET_CSV_NAME=$2
BACKEND="sglang"
PORT="30001"
WARMUP_REQUESTS="100"
RANDOM_RANGE_RATIO="1."
REQUEST_RATE="1.6"
RESULT_CSV="/sgl-workspace/results/${TARGET_CSV_NAME}"
TEMP_JSON="/sgl-workspace/results/temp_${MODEL_NAME}.json"

# === 自动判断模型类型并切换数据集 ===
if [[ "$MODEL_NAME" == *"VL"* ]] || [[ "$MODEL_NAME" == *"vl"* ]]; then
    echo "   -> [Type Check] Detected VL Model: Using random-image generation."
    # VL 模型专用参数
    DATASET_ARGS="--dataset-name random-image --random-image-num-images 1 --random-image-resolution 1148x112"
else
    echo "   -> [Type Check] Detected Dense Model: Using ShareGPT.json."
    # Dense 模型专用参数：指定 dataset-path
    # 注意：这里 dataset-name 改为 sharegpt 以匹配 json 格式
    DATASET_ARGS="--dataset-name random --dataset-path /data/model/ShareGPT.json"
fi

# === 测试参数配置 ===
MAX_CONCURRENCY_LIST=(1 2 4 8 16 32 50 64 100 128 256)
NUM_PROMPTS_LIST=(2 4 8 16 32 64 100 128 200 256 512)
INPUT_OUTPUT_PAIRS=("256:256" "512:512" "1024:1024" "2048:1024" "3072:1024" "4096:1024")

# CSV 表头初始化
if [ ! -f "$RESULT_CSV" ]; then
    HEADER="model_name,input_len,output_len,max_concurrency,num_prompts,"
    HEADER+="req_tp,in_tok_tp,out_tok_tp,"
    HEADER+="mean_ttft,median_ttft,p99_ttft,"
    HEADER+="mean_tpot,median_tpot,p99_tpot,"
    HEADER+="mean_itl,p99_itl,mean_e2e,real_concurrency, duration, total_input_tokens, total_output_tokens"
    echo "$HEADER" > "$RESULT_CSV"
fi

for pair in "${INPUT_OUTPUT_PAIRS[@]}"; do
    input_len=${pair%%:*}
    output_len=${pair#*:}

    for idx in "${!MAX_CONCURRENCY_LIST[@]}"; do
        max_concurrency=${MAX_CONCURRENCY_LIST[$idx]}
        num_prompts=${NUM_PROMPTS_LIST[$idx]}

        echo "   -> Benchmarking: In=$input_len | Out=$output_len | Conc=$max_concurrency"

        # 执行测试 (使用 $DATASET_ARGS 变量)
        python3 -m sglang.bench_serving \
            --backend $BACKEND --port $PORT \
            --num-prompts $num_prompts --random-output-len $output_len \
            --apply-chat-template \
            --random-input-len $input_len --warmup-requests $WARMUP_REQUESTS \
            --random-range-ratio $RANDOM_RANGE_RATIO \
            --max-concurrency $max_concurrency \
            $DATASET_ARGS \
            --output-file $TEMP_JSON > /dev/null 2>&1

        if [ -f "$TEMP_JSON" ]; then
             stats=$(python3 -c "
import json
try:
    with open('$TEMP_JSON') as f:
        d = json.load(f)
    fields = [
        'request_throughput', 'input_throughput', 'output_throughput',
        'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms',
        'mean_tpot_ms', 'median_tpot_ms', 'p99_tpot_ms',
        'mean_itl_ms', 'p99_itl_ms', 'mean_e2e_latency_ms', 'concurrency', 'duration', 'total_input_tokens', 'total_output_tokens'
    ]
    print(','.join(str(d.get(f, 0)) for f in fields))
except:
    print(','.join(['ERROR']*16))
")
            echo "$MODEL_NAME,$input_len,$output_len,$max_concurrency,$num_prompts,$stats" >> "$RESULT_CSV"
        else
            echo "$MODEL_NAME,$input_len,$output_len,$max_concurrency,$num_prompts,ERROR,,,,,,,,,,,," >> "$RESULT_CSV"
        fi
        [ -f "$TEMP_JSON" ] && rm "$TEMP_JSON"
    done
done
EOF
docker exec $CONTAINER_NAME chmod +x /sgl-workspace/run_bench_inner.sh


# ==============================================================================
# [Step 2.5] 修复 SGLang Qwen3 源码 Bug (针对非量化 Qwen3 模型)
# ==============================================================================
echo "[Step 2.5] Applying Hotfix for Qwen3 non-quantized models..."
docker exec $CONTAINER_NAME sed -i \
    "s/self.quant_name = quant_config.get_name()/self.quant_name = quant_config.get_name() if quant_config is not None else None/g" \
    /sgl-workspace/sglang/python/sglang/srt/models/qwen3.py
echo "   -> Patch applied."
# ==============================================================================


# 3. 主循环
echo "[Step 3] Starting Benchmark Loop..."

for model in "${MODELS[@]}"; do
    echo "============================================================"
    echo "Processing Model: $model"
    echo "============================================================"

    SERVER_LOG="/sgl-workspace/results/server_${model}.log"
    MODEL_PATH="/data/model/${model}"

    echo "   -> Cleaning up previous processes..."
    docker exec $CONTAINER_NAME bash -c "pkill -f 'sglang.launch_server' || true"
    sleep 5
    docker exec $CONTAINER_NAME bash -c "fuser -k -9 /dev/nvidia*" 2>/dev/null

    echo "   -> Starting SGLang Server..."
    # 启动 Server
    docker exec -d $CONTAINER_NAME bash -c "
        export NVSHMEM_HCA_PE_MAPPING='mlx5_bond_2:1:2,mlx5_bond_3:1:2,mlx5_bond_4:1:2,mlx5_bond_5:1:2'
        export SGLANG_TORCH_PROFILER_DIR=/tmp/
        export MUSA_LAUNCH_BLOCKING=0
        export MCCL_IB_GID_INDEX=3
        export MCCL_NET_SHARED_BUFFERS=0
        export MCCL_PROTOS=2
        export GLOO_SOCKET_IFNAME=bond0
        export TP_SOCKET_IFNAME=bond0
        export SGL_DEEP_GEMM_BLOCK_M=128
        export MUSA_VISIBLE_DEVICES=all
        export SGLANG_USE_MTT=1
        
        nohup python3 -m sglang.launch_server \
            --model $MODEL_PATH \
            --trust-remote-code \
            --cuda-graph-max-bs 256 \
            --tensor-parallel-size 8 \
            --mem-fraction-static 0.9 \
            --max-prefill-tokens 32768 \
            --disable-radix-cache \
            --disable-overlap-schedule \
            --port $SGLANG_PORT \
            --attention-backend fa3 \
            --host 0.0.0.0 > $SERVER_LOG 2>&1 &
    "

    echo "   -> Waiting for server to be ready (Timeout: ${SERVER_TIMEOUT}s)..."
    start_time=$(date +%s)
    server_ready=0
    while [ $(($(date +%s) - start_time)) -lt $SERVER_TIMEOUT ]; do
        http_code=$(docker exec $CONTAINER_NAME bash -c "curl -o /dev/null -s -w '%{http_code}' localhost:$SGLANG_PORT/health")
        if [ "$http_code" == "200" ]; then
            server_ready=1
            break
        fi
        echo -n "."
        sleep 5
    done
    echo ""

    if [ $server_ready -eq 0 ]; then
        echo "   [ERROR] Server failed to start for $model. Check logs at $HOST_RESULT_DIR/server_${model}.log"
        continue
    fi
    echo "   -> Server is READY!"

    echo "   -> Running Benchmark Script..."
    docker exec $CONTAINER_NAME bash /sgl-workspace/run_bench_inner.sh "$model" "$CSV_FILENAME"
    
    echo "   -> Model $model Finished."
done

echo "============================================================"
echo "All benchmarks completed!"