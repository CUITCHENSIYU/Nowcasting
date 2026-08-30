#!/bin/bash
# Launch MetNet training.
#
# Usage:
#   bash start_training.sh --config configs/rainfall_forecast.yaml --gpus 0,1 --output runs/exp1
#
# Required:
#   --config / -c   yaml config path
#   --gpus   / -g   GPU ids, e.g. 0 or 0,1,2 (or all)
#   --output / -o   directory for logs and checkpoints
#
set -euo pipefail

cd "$(dirname "$0")"

CONFIG=""
GPUS="0"
OUTPUT=""
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-}"
PYTHON="${PYTHON:-python}"

usage() {
    cat <<EOF
Usage:
  bash start_training.sh --config <yaml> --gpus <ids> --output <dir>

Examples:
  bash start_training.sh -c configs/rainfall_forecast.yaml -g 0 -o runs/debug
  bash start_training.sh -c configs/rainfall_forecast.yaml -g 0,1,2 -o runs/metnet_v1
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|-c)
            CONFIG="$2"
            shift 2
            ;;
        --gpus|-g)
            GPUS="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT="$2"
            shift 2
            ;;
        --python)
            PYTHON="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$CONFIG" || -z "$OUTPUT" ]]; then
    echo "Error: --config and --output are required."
    usage
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

if [[ "$GPUS" == "all" ]]; then
    NPROC="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC - 1)))"
else
    CUDA_VISIBLE_DEVICES="$GPUS"
    # count comma-separated ids
    IFS=',' read -r -a GPU_ARR <<< "$GPUS"
    NPROC="${#GPU_ARR[@]}"
fi

if [[ -z "$MASTER_PORT" ]]; then
    MASTER_PORT="$(shuf -i 12000-13000 -n 1)"
fi

mkdir -p "$OUTPUT"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

echo "========================================"
echo "Config : $CONFIG"
echo "GPUs   : $CUDA_VISIBLE_DEVICES  (nproc=$NPROC)"
echo "Output : $OUTPUT"
echo "Python : $PYTHON"
echo "Port   : $MASTER_PORT"
echo "========================================"

torchrun \
    --standalone \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --nproc_per_node "$NPROC" \
    train.py \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --gpus "$CUDA_VISIBLE_DEVICES"
