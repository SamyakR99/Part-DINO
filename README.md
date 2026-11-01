# Complementing DINO Features with Image Structure for Part Discovery

Official PyTorch implementation for **"Complementing DINO Features with Image Structure for Part Discovery"** (NeurIPS 2026).

---

[![Paper](https://img.shields.io/badge/Arxiv-b31b1b.svg)](https://arxiv.org/abs/2610.32293)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://samyakr99.github.io/Part_dino/)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-red)](https://neurips.cc/Conferences/2026)


## 📌 Overview

**Part-DINO** is a training-free framework for unsupervised semantic part discovery. Self-supervised model features (like DINO / DINOv2) encode rich semantics but produce coarse, blurry part boundaries due to their rigid ViT patch grid. Part-DINO overcomes this limitation by complementing dense semantic representations with high-frequency structural image gradients:

1. **Foreground Isolation**: Isolates the foreground object using text-guided SAM3 or unsupervised TokenCut.
2. **Boundary-Aware Oversegmentation**: Blends structural edge responses (Akbas-Ahuja RAMP filter) with DINO feature gradients to compute watershed seeds and generate edge-conforming atomic regions.
3. **Region Absorption & Global Prototype Mapping**: Region features are aggregated and mapped to global semantic prototypes via cosine similarity, guaranteeing consistent part indices across all images in the dataset without any training.

---

## 📂 Repository Structure

```
part_discovery_release/
├── README.md                      # Complete setup and replication instructions
├── requirements.txt               # Minimal pip dependencies
├── environment.yml                # Conda environment definition
├── main.py                        # Core pipeline & evaluation harness
├── dataset.py                     # Unified dataloaders (CUB, CelebA, PartImageNet)
├── global_kmeans.py               # Prototype discovery & region-to-centroid assignment
├── segmentation.py                # SAM3 / TokenCut foreground segmentation backends
├── scripts/                       # Reproduction scripts for paper tables
│   ├── run_table1_cub.sh          # Replicate Table 1 (CUB-200-2011, K=4, 8, 16)
│   ├── run_table2_celeba.sh       # Replicate Table 2 (CelebA / MAFL, K=4, 8, 16)
│   ├── run_table5_partimagenet_ood.sh # Replicate Table 5 Left (PartImageNet-OOD, K=8, 25, 50)
│   └── run_table5_partimagenet_seg.sh # Replicate Table 5 Right (PartImageNet-Seg, K=8, 25, 50)
└── pretrained_centroids/          # Precomputed global centroids for instant reproduction
    ├── cub/                       # K=4, 8, 16 centroids for CUB
    ├── celeba/                    # K=4, 8, 16 centroids for CelebA
    ├── partimagenet_ood/          # Per-supercategory centroids for OOD (K=8, 25, 50)
    └── partimagenet_seg/          # Per-supercategory centroids for Seg (K=8, 25, 50)
```

---

## Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# 1. Create and activate conda environment
conda env create -f environment.yml
conda activate part_dino

# 2. Log in to Hugging Face (Required for gated SAM3 model weights: facebook/sam3)
huggingface-cli login
```

### Option 2: Using Pip

```bash
conda create -n part_dino python=3.10 -y
conda activate part_dino

# Install PyTorch with CUDA support (e.g. CUDA 12.1+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install repository dependencies
pip install -r requirements.txt

# Log in to Hugging Face
huggingface-cli login
```


---

## Dataset Preparation

Set up the datasets following the directory layouts below. You can either place them in `./data/` or set the environment variable `DATA_ROOT` before running any script.

### 1. CUB-200-2011 (Birds)
Download the [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset. Expected structure:
```
<CUB_ROOT>/
├── images/                        # 200 bird species subfolders
├── parts/
│   └── part_locs.txt              # 15 keypoint landmark coordinates
├── bounding_boxes.txt
├── images.txt
├── test_list.txt                  # Test image list (5,794 images)
└── train_list.txt
```

### 2. CelebA / MAFL (Faces)
Download [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and MAFL annotations:
```
<CELEBA_ROOT>/
├── img_celeba/                    # Raw or aligned CelebA images
├── MAFL_test_lm.txt               # MAFL test split landmark annotations (283 images)
├── cele_train_lm.txt              # CelebA train split landmark annotations
└── list_landmarks_align_celeba.txt
```

### 3. PartImageNet-OOD (Out-of-Distribution split)
Download [PartImageNet](https://github.com/TACJu/PartImageNet) OOD split:
```
<PARTIMAGENET_OOD_ROOT>/
├── test.json                      # COCO annotations for test split
├── train.json                     # COCO annotations for train split
└── test/
    └── test/
        └── <synset>/<file>.JPEG   # Test images grouped by synset
```

### 4. PartImageNet-Seg (Segmentation split)
Download PartImageNet segmentation split:
```
<PARTIMAGENET_SEG_ROOT>/
├── annotations/
│   ├── test_whole/test.json       # COCO annotations
│   └── train_whole/train.json
└── images/
    ├── test/<file>.JPEG           # Test images
    └── train/<file>.JPEG
```

---

## Results

Precomputed global prototypes/centroids are included in `pretrained_centroids/`, allowing you to run evaluation immediately without waiting hours for reference clustering.

### CUB-200-2011 ($K \in \{4, 8, 16\}$)

Replicate quantitative results on the CUB dataset across granularities:

```bash
# Run all granularities (K=4, 8, 16):
DATA_ROOT=/path/to/CUB_200_2011 bash scripts/run_table1_cub.sh

# Or run an individual granularity (e.g. K=16):
DATA_ROOT=/path/to/CUB_200_2011 bash scripts/run_table1_cub.sh 16
```

**Expected Results (from Paper Table 1):**
| Granularity | NME (↓) | NMI (↑) | ARI (↑) |
|:---:|:---:|:---:|:---:|
| **$K = 4$**  | 8.21% | 53.93% | 23.01% |
| **$K = 8$**  | 6.00% | 61.84% | 37.92% |
| **$K = 16$** | 4.35% | 65.45% | 46.00% |

---

### CelebA / MAFL ($K \in \{4, 8, 16\}$)

Replicate quantitative comparisons on CelebA across granularities:

```bash
# Run all granularities (K=4, 8, 16):
DATA_ROOT=/path/to/CelebA bash scripts/run_table2_celeba.sh

# Or run an individual granularity (e.g. K=4):
DATA_ROOT=/path/to/CelebA bash scripts/run_table2_celeba.sh 4
```

**Expected Results (from Paper Table 2):**
| Granularity | NME (↓) | NMI (↑) | ARI (↑) |
|:---:|:---:|:---:|:---:|
| **$K = 4$**  | 10.81% | 45.07% | 32.20% |
| **$K = 8$**  | 10.59% | 41.53% | 28.59% |
| **$K = 16$** | 9.15%  | 41.04% | 25.20% |

---

### PartImageNet Out-of-Distribution ($PartImage_O$) & Segmentation ($PartImage_S$)

Replicate pixel-level NMI and ARI on both PartImageNet benchmark splits:

#### 1. PartImageNet-OOD ($K \in \{8, 25, 50\}$)
```bash
# Run all granularities:
DATA_ROOT=/path/to/PartImageNet_OOD bash scripts/run_table5_partimagenet_ood.sh

# Or run K=50:
DATA_ROOT=/path/to/PartImageNet_OOD bash scripts/run_table5_partimagenet_ood.sh 50
```

#### 2. PartImageNet-Seg ($K \in \{8, 25, 50\}$)
```bash
# Run all granularities:
DATA_ROOT=/path/to/PartImageNet_Seg/PartImageNet bash scripts/run_table5_partimagenet_seg.sh

# Or run K=8:
DATA_ROOT=/path/to/PartImageNet_Seg/PartImageNet bash scripts/run_table5_partimagenet_seg.sh 8
```

**Expected Results (from Paper Table 5):**
| Dataset | Metric | $K = 8$ | $K = 25$ | $K = 50$ |
|:---|:---:|:---:|:---:|:---:|
| **PartImageNet-OOD** | NMI (↑) | 30.07% | 31.14% | 30.13% |
| | ARI (↑) | 63.27% | 64.93% | 65.42% |
| **PartImageNet-Seg** | NMI (↑) | 37.21% | 41.54% | 41.80% |
| | ARI (↑) | 66.10% | 66.94% | 66.97% |

---

## Discovering Centroids from Scratch

If you wish to discover global centroids from scratch instead of using precomputed ones:

1. Simply specify a new `--proto_path` (for CUB/CelebA) or `--centroid_dir` (for PartImageNet) pointing to a non-existent file or directory.
2. The pipeline will automatically sample reference images from the training split, extract foreground atomic region features, fit `MiniBatchKMeans`, and cache the discovered centroids to disk for subsequent evaluation:

```bash
# Example: Discover K=4 centroids on CUB from 6000 reference images
python main.py \
    --dataset CUB \
    --data_root /path/to/CUB_200_2011 \
    --split test \
    --mode evaluate \
    --target_k 4 \
    --use_prototypes \
    --n_reference 6000 \
    --proto_path ./my_centroids/cub_k4.npy \
    --alpha 0.0 \
    --skip_gradient_merge \
    --use_sam3 \
    --img_size 448
```

---

## Discovery & Visualization on Custom Images

To run unsupervised part discovery and save visual part segmentations:

```bash
python main.py \
    --dataset CUB \
    --data_root /path/to/CUB_200_2011 \
    --split test \
    --mode evaluate \
    --target_k 16 \
    --use_prototypes \
    --proto_path pretrained_centroids/cub/cub_global_centroids_k16.npy \
    --use_sam3 \
    --img_size 448 \
    --save_vis \
    --vis_dir ./visualizations
```

---

## 📖 Citation

If you find this work or code useful in your research, please cite:

```bibtex
@inproceedings{rawlekar2026complementing,
  title={Complementing {DINO} Features with Image Structure for Part Discovery},
  author={Rawlekar, Samyak and Paleti, Nikhil Chowdary and Gupta, Amey and Ahuja, Narendra},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

