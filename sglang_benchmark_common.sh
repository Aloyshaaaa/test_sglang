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

run_sglang_runtime_preflight() {
    local python_executable="${1:-python3}"
    local skip_preflight="${SKIP_RUNTIME_PREFLIGHT:-0}"

    if [ "$skip_preflight" = "1" ]; then
        return 0
    fi

    "$python_executable" - <<'PY'
import importlib
import importlib.util
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


sglang_spec = importlib.util.find_spec("sglang")
if sglang_spec is None or sglang_spec.origin is None:
    fail(
        f"错误: 当前 Python 环境未安装 sglang: {sys.executable}\n"
        "请先确认运行容器/虚拟环境里的 sglang 安装是否完整。"
    )

sglang_root = Path(sglang_spec.origin).resolve().parent
fp8_utils = "sglang.srt.layers.quantization.fp8_utils"
has_mtt = importlib.util.find_spec("mtt_torch_ext") is not None

try:
    importlib.import_module(fp8_utils)
except ModuleNotFoundError as exc:
    if exc.name == "mtt_torch_ext":
        fail(
            "错误: 当前 sglang 运行时在导入量化模块时依赖 mtt_torch_ext，但运行环境里缺少该模块。\n"
            f"sglang 包路径: {sglang_root}\n"
            f"导入模块: {fp8_utils}\n"
            "这属于外部 sglang 运行环境问题，不是本仓库脚本参数问题。\n"
            "请安装与当前 sglang/MUSA 版本匹配的 mtt_torch_ext，或切换到不硬依赖它的 sglang 包。\n"
            "如需跳过该预检，可显式设置 SKIP_RUNTIME_PREFLIGHT=1。"
        )
    raise
except Exception:
    if not has_mtt:
        fp8_utils_path = sglang_root / "srt" / "layers" / "quantization" / "fp8_utils.py"
        try:
            fp8_utils_text = (
                fp8_utils_path.read_text(encoding="utf-8", errors="ignore")
                if fp8_utils_path.is_file()
                else ""
            )
        except OSError as exc:
            fail(f"错误: 无法读取 {fp8_utils_path}: {exc}")

        if "mtt_torch_ext" in fp8_utils_text:
            fail(
                "错误: 检测到当前 sglang 安装包硬依赖 mtt_torch_ext，但运行环境里缺少该模块。\n"
                f"sglang 包路径: {sglang_root}\n"
                f"检查文件: {fp8_utils_path}\n"
                "这属于外部 sglang 运行环境问题，不是本仓库脚本参数问题。\n"
                "请安装与当前 sglang/MUSA 版本匹配的 mtt_torch_ext，或切换到不硬依赖它的 sglang 包。\n"
                "如需跳过该预检，可显式设置 SKIP_RUNTIME_PREFLIGHT=1。"
            )
    raise
PY
}

run_sharegpt_bos_token_preflight() {
    local python_executable="${1:-python3}"
    local patch_file_path="$2"
    local skip_preflight="${SKIP_BENCH_BOS_TOKEN_PREFLIGHT:-0}"

    if [ "$skip_preflight" = "1" ]; then
        return 0
    fi

    PATCH_FILE_PATH="$patch_file_path" "$python_executable" - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


spec = importlib.util.find_spec("sglang.bench_serving")
if spec is None or spec.origin is None:
    fail(
        f"错误: 当前 Python 环境未安装 sglang.bench_serving: {sys.executable}\n"
        "请先确认运行容器/虚拟环境里的 sglang 安装是否完整。"
    )

bench_path = Path(spec.origin).resolve()
try:
    bench_text = bench_path.read_text(encoding="utf-8", errors="ignore")
except OSError as exc:
    fail(f"错误: 无法读取 {bench_path}: {exc}")

unsafe_snippet = 'prompt = prompt.replace(tokenizer.bos_token, "")'
safe_snippet = 'bos_token = getattr(tokenizer, "bos_token", None)'

if unsafe_snippet in bench_text and safe_snippet not in bench_text:
    patch_path = os.environ.get("PATCH_FILE_PATH", "")
    fail(
        "错误: 当前 sglang.bench_serving 仍包含未修复的 bos_token 处理逻辑。\n"
        f"检查文件: {bench_path}\n"
        "这属于外部 SGLang 代码兼容性问题，不是本仓库测试脚本参数问题。\n"
        "当 tokenizer.bos_token 为 None 时，ShareGPT 压测会在 sample_sharegpt_requests 阶段报错:\n"
        "TypeError: replace() argument 1 must be str, not None\n"
        f"请在外部 SGLang 代码树应用补丁: patch -p1 < {patch_path}\n"
        "如需跳过该预检，可显式设置 SKIP_BENCH_BOS_TOKEN_PREFLIGHT=1。"
    )
PY
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
