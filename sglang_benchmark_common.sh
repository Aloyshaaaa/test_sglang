#!/usr/bin/env bash

resolve_model_dir() {
    local input_path="$1"

    if [ -z "$input_path" ]; then
        echo "MODEL_PATH 不能为空" >&2
        return 1
    fi

    if [ ! -e "$input_path" ]; then
        echo "模型路径不存在: $input_path" >&2
        return 1
    fi

    if [ -f "$input_path" ]; then
        if [ "$(basename "$input_path")" = "config.json" ]; then
            dirname "$input_path"
            return 0
        fi
        echo "模型路径不是目录，也不是 config.json: $input_path" >&2
        return 1
    fi

    if [ -f "$input_path/config.json" ]; then
        echo "$input_path"
        return 0
    fi

    local configs=()
    while IFS= read -r line; do
        configs+=("$line")
    done < <(find "$input_path" -type f -name config.json | sort)

    if [ "${#configs[@]}" -eq 0 ]; then
        echo "模型路径下未找到 config.json: $input_path" >&2
        echo "config.json 是 Hugging Face 模型目录里的配置文件，SGLang/transformers 依赖它识别模型结构。" >&2
        return 1
    fi

    if [ "${#configs[@]}" -gt 1 ]; then
        echo "警告: 检测到多个 config.json，默认使用第一个候选目录:" >&2
        printf '  - %s\n' "${configs[@]:0:5}" >&2
    fi

    dirname "${configs[0]}"
}

detect_model_type() {
    local model_path="$1"
    local requested_type="${2:-auto}"
    local model_name
    model_name="$(basename "$model_path")"

    case "$requested_type" in
        dense|vl)
            echo "$requested_type"
            ;;
        auto)
            if [[ "$model_name" == *"VL"* || "$model_name" == *"vl"* ]]; then
                echo "vl"
            else
                echo "dense"
            fi
            ;;
        *)
            echo "无效的 MODEL_TYPE: $requested_type，可选值: auto|dense|vl" >&2
            return 1
            ;;
    esac
}

network_interface_exists() {
    local ifname="$1"
    [ -n "$ifname" ] && [ -d "/sys/class/net/$ifname" ]
}

detect_default_network_interface() {
    local ifname=""

    if command -v ip >/dev/null 2>&1; then
        ifname="$(ip route get 1.1.1.1 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "dev") {print $(i + 1); exit}}')"
        if [ -n "$ifname" ] && network_interface_exists "$ifname"; then
            echo "$ifname"
            return 0
        fi

        ifname="$(ip -o link show up 2>/dev/null | awk -F': ' '$2 != "lo" && $2 != "docker0" {print $2; exit}')"
        if [ -n "$ifname" ] && network_interface_exists "$ifname"; then
            echo "$ifname"
            return 0
        fi
    fi

    return 1
}

export_default_server_env() {
    export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/tmp/}"
    export MUSA_LAUNCH_BLOCKING="${MUSA_LAUNCH_BLOCKING:-0}"
    export MCCL_IB_GID_INDEX="${MCCL_IB_GID_INDEX:-3}"
    export MCCL_NET_SHARED_BUFFERS="${MCCL_NET_SHARED_BUFFERS:-0}"
    export MCCL_PROTOS="${MCCL_PROTOS:-2}"
    export SGL_DEEP_GEMM_BLOCK_M="${SGL_DEEP_GEMM_BLOCK_M:-128}"
    export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-all}"
    export SGLANG_USE_MTT="${SGLANG_USE_MTT:-1}"

    if [ -n "${GLOO_SOCKET_IFNAME:-}" ] && ! network_interface_exists "$GLOO_SOCKET_IFNAME"; then
        echo "GLOO_SOCKET_IFNAME 指定的网卡不存在: $GLOO_SOCKET_IFNAME" >&2
        return 1
    fi
    if [ -n "${TP_SOCKET_IFNAME:-}" ] && ! network_interface_exists "$TP_SOCKET_IFNAME"; then
        echo "TP_SOCKET_IFNAME 指定的网卡不存在: $TP_SOCKET_IFNAME" >&2
        return 1
    fi

    if [ -z "${GLOO_SOCKET_IFNAME:-}" ] && [ -n "${TP_SOCKET_IFNAME:-}" ]; then
        export GLOO_SOCKET_IFNAME="$TP_SOCKET_IFNAME"
    fi
    if [ -z "${TP_SOCKET_IFNAME:-}" ] && [ -n "${GLOO_SOCKET_IFNAME:-}" ]; then
        export TP_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME"
    fi

    if [ -z "${GLOO_SOCKET_IFNAME:-}" ] || [ -z "${TP_SOCKET_IFNAME:-}" ]; then
        local detected_iface=""
        detected_iface="$(detect_default_network_interface || true)"
        if [ -z "${GLOO_SOCKET_IFNAME:-}" ] && [ -n "$detected_iface" ]; then
            export GLOO_SOCKET_IFNAME="$detected_iface"
        fi
        if [ -z "${TP_SOCKET_IFNAME:-}" ] && [ -n "$detected_iface" ]; then
            export TP_SOCKET_IFNAME="$detected_iface"
        fi
    fi

    if [ -n "${NVSHMEM_HCA_PE_MAPPING:-}" ]; then
        export NVSHMEM_HCA_PE_MAPPING
    fi
}

wait_for_health() {
    local health_host="$1"
    local port="$2"
    local timeout_seconds="$3"
    local start_time
    start_time="$(date +%s)"

    while [ "$(( $(date +%s) - start_time ))" -lt "$timeout_seconds" ]; do
        if curl -fsS "http://${health_host}:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done

    return 1
}

print_section() {
    local title="$1"
    printf '\n============================================================\n'
    printf '%s\n' "$title"
    printf '============================================================\n\n'
}
