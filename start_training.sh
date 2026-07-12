#!/bin/bash
cd "$(dirname "$0")"

N_NODES="1"
NODE_RANK="0"
MASTER_ADDR="localhost"
GPUS_PER_NODE="-1"
GPUS="all"
WORKSPACE="./runs"
VERSION="metnet_v1"
PROJECTS_CONFIGS_DIR="./configs"

for ARGUMENT in "$@"; do
    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)

    case "$KEY" in
        MASTER_ADDR) MASTER_ADDR=${VALUE} ;;
        MASTER_PORT) MASTER_PORT=${VALUE} ;;
        N_NODES) N_NODES=${VALUE} ;;
        GPUS) GPUS=${VALUE} ;;
        GPUS_PER_NODE) GPUS_PER_NODE=${VALUE} ;;
        NODE_RANK) NODE_RANK=${VALUE} ;;
        WORKSPACE) WORKSPACE=${VALUE} ;;
        VERSION) VERSION=${VALUE} ;;
        PROJECTS_CONFIGS_DIR) PROJECTS_CONFIGS_DIR=${VALUE} ;;
    esac
done

if [[ "$GPUS_PER_NODE" == "-1" ]]; then
    GPUS_PER_NODE="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
    echo "GPUS_PER_NODE is set to -1, use all GPUs: $GPUS_PER_NODE"
fi

if [[ -z "${MASTER_PORT}" ]]; then
    MASTER_PORT=$(shuf -i 12000-13000 -n 1)
fi

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Number of nodes: $N_NODES"
echo "Number of GPUs per node: $GPUS_PER_NODE"
echo "Workspace: $WORKSPACE"
echo "Version: $VERSION"

export PYTHONPATH="$(pwd):${PYTHONPATH}"

NCCL_P2P_DISABLE=1 torchrun \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    --nnodes "$N_NODES" \
    --nproc_per_node "$GPUS_PER_NODE" \
    --node_rank "$NODE_RANK" \
    train.py \
    --workspace "$WORKSPACE" \
    --version "$VERSION" \
    --projects_configs_dir "$PROJECTS_CONFIGS_DIR" \
    --ngpus "$GPUS_PER_NODE" \
    --gpus "$GPUS"
