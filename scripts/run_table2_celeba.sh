#!/bin/bash
# Replicate Table 2: CelebA / MAFL Quantitative Evaluation (K=4, 8, 16)
# Usage:
#   bash scripts/run_table2_celeba.sh        # Runs all K in (4, 8, 16)
#   bash scripts/run_table2_celeba.sh 4      # Runs only K=4

set -e

# Default paths (override with environment variables if desired)
DATA_ROOT="${DATA_ROOT:-/scratch/bcyh/dataset/CelebA}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TARGET_KS=(4 8 16)
if [ -n "$1" ]; then
    TARGET_KS=("$1")
fi

echo "=========================================================="
echo "Part-DINO: Replicating Table 2 (CelebA)"
echo "Data Root: ${DATA_ROOT}"
echo "=========================================================="

for K in "${TARGET_KS[@]}"; do
    PROTO_PATH="${PROJECT_ROOT}/pretrained_centroids/celeba/celeba_centroids_k${K}.npy"
    echo ""
    echo ">>> Running CelebA Evaluation with K=${K} <<<"
    echo ">>> Using Prototypes: ${PROTO_PATH}"
    
    python -u main.py \
        --dataset CelebA \
        --data_root "${DATA_ROOT}" \
        --split test \
        --mode evaluate \
        --target_k "${K}" \
        --min_shard_size 10 \
        --alpha 0.0 \
        --skip_gradient_merge \
        --use_prototypes \
        --n_reference 500 \
        --proto_path "${PROTO_PATH}" \
        --min_distance 1 \
        --use_sam3 \
        --sam3_prompt "head, face" \
        --img_size 224 \
        --backbone dinov2
done

echo ""
echo "Table 2 evaluation complete!"

