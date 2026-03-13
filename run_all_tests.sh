#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" != "" ]; then
    case "$1" in
        dense|vl)
            export MODEL_TYPE="${MODEL_TYPE:-$1}"
            ;;
        *)
            echo "用法: $0 [dense|vl]" >&2
            exit 1
            ;;
    esac
fi

exec "${SCRIPT_DIR}/auto_benchmark.sh"
