#!/bin/bash
# Replicate Table 5 (Right): PartImageNet Segmentation (PartImageNet-Seg) Evaluation (K=8, 25, 50)
# Usage:
#   bash scripts/run_table5_partimagenet_seg.sh        # Runs all K in (8, 25, 50)
#   bash scripts/run_table5_partimagenet_seg.sh 8      # Runs only K=8

set -e

# Default paths (override with environment variables if desired)
DATA_ROOT="${DATA_ROOT:-/scratch/bcyh/dataset/PartImageNet_Seg/PartImageNet}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

TARGET_KS=(8 25 50)
if [ -n "$1" ]; then
    TARGET_KS=("$1")
fi

echo "=========================================================="
echo "Part-DINO: Replicating Table 5 (PartImageNet-Seg)"
echo "Data Root: ${DATA_ROOT}"
echo "=========================================================="

for K in "${TARGET_KS[@]}"; do
    CENTROID_DIR="${PROJECT_ROOT}/pretrained_centroids/partimagenet_seg/k${K}"
    echo ""
    echo ">>> Running PartImageNet-Seg Evaluation with K=${K} <<<"
    echo ">>> Using Centroid Directory: ${CENTROID_DIR}"
    
    python -u main.py \
        --dataset PartImageNet_Seg \
        --data_root "${DATA_ROOT}" \
        --split test \
        --mode evaluate \
        --target_k "${K}" \
        --min_shard_size 10 \
        --alpha 0.0 \
        --skip_gradient_merge \
        --n_reference 5000 \
        --centroid_dir "${CENTROID_DIR}" \
        --max_samples 0 \
        --n_debug 0 \
        --img_size 224 \
        --backbone dinov2 \
        --use_sam3
done

echo ""
echo "Table 5 (PartImageNet-Seg) evaluation complete!"

