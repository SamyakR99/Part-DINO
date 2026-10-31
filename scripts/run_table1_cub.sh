#!/bin/bash
# Replicate Table 1: CUB-200-2011 Quantitative Evaluation (K=4, 8, 16)
# Usage:
#   bash scripts/run_table1_cub.sh        # Runs all K in (4, 8, 16)
#   bash scripts/run_table1_cub.sh 16     # Runs only K=16

set -e

# Default paths (override with environment variables if desired)
DATA_ROOT="${DATA_ROOT:-/scratch/bcyh/dataset/CUB_200_2011}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TARGET_KS=(4 8 16)
if [ -n "$1" ]; then
    TARGET_KS=("$1")
fi

echo "=========================================================="
echo "Part-DINO: Replicating Table 1 (CUB-200-2011)"
echo "Data Root: ${DATA_ROOT}"
echo "=========================================================="

for K in "${TARGET_KS[@]}"; do
    PROTO_PATH="${PROJECT_ROOT}/pretrained_centroids/cub/cub_global_centroids_k${K}.npy"
    echo ""
    echo ">>> Running CUB Evaluation with K=${K} <<<"
    echo ">>> Using Prototypes: ${PROTO_PATH}"
    
    python -u main.py \
        --dataset CUB \
        --data_root "${DATA_ROOT}" \
        --split test \
        --mode evaluate \
        --target_k "${K}" \
        --min_shard_size 10 \
        --alpha 0.0 \
        --skip_gradient_merge \
        --use_prototypes \
        --n_reference 6000 \
        --proto_path "${PROTO_PATH}" \
        --use_sam3 \
        --img_size 448 \
        --backbone dinov2
done

echo ""
echo "Table 1 evaluation complete!"

