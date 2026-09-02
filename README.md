# LeafSentinel
> **Computer Vision System for In-the-Wild Plant Disease Detection, Fine-Grained Lesion Segmentation, Severity Estimation & Crop-Health Analytics.**

---

## 📌 Project Overview
**LeafSentinel** is a computer vision pipeline engineered to detect, segment, and quantify foliar crop diseases from real-world agricultural RGB imagery.

The core pipeline is organized around two foundational milestones:
1. **Dataset Discovery & Leakage Audit**: Non-destructive data profiling, validation, duplicate detection, and leakage-free benchmark split generation.
2. **Lesion Segmentation Baseline**: Pixel-level disease localization using a U-Net architecture with ImageNet-pretrained ResNet-18 feature extraction.

---

## 🔬 Benchmark Dataset & Leakage-Free Preparation

### 📊 Dataset Profiling & Verification
* **Source Dataset**: PlantSeg (7,774 high-resolution leaf images across 34 crop hosts and 115 pathology categories).
* **Two-Stage Duplicate Verification**:
  * **Stage 1 (Exact)**: MD5 byte-exact matching automatically groups identical files.
  * **Stage 2 (Near-Duplicates)**: 256-bit Difference Hash (dHash) candidates ($\text{dist} \le 6$) undergo secondary Structural Similarity Index (SSIM $\ge 0.85$) verification.
* **Zero-Leakage Stratified Splitting**: Disjoint Set Union (Union-Find) connected-component grouping guarantees that **no duplicate group spans across training, validation, and test splits**.

### 🎯 Benchmark Class Selection
The initial lesion segmentation benchmark targets 10 high-priority agricultural crops (1,304 total verified samples):

| Crop Host | Disease Pathology | Display Label | Samples |
|---|---|---|---|
| **Citrus** | Citrus Canker | Citrus — Citrus Canker | 323 |
| **Grape** | Downy Mildew | Grape — Downy Mildew | 211 |
| **Soybean** | Frogeye Leaf Spot | Soybean — Frogeye Leaf Spot | 153 |
| **Tomato** | Early Blight | Tomato — Early Blight | 153 |
| **Banana** | Black Sigatoka | Banana — Black Sigatoka | 114 |
| **Potato** | Late Blight | Potato — Late Blight | 78 |
| **Corn** | Gray Leaf Spot | Corn — Gray Leaf Spot | 76 |
| **Wheat** | Leaf Rust | Wheat — Leaf Rust | 75 |
| **Apple** | Black Rot | Apple — Black Rot | 63 |
| **Bell Pepper** | Bacterial Spot | Bell Pepper — Bacterial Spot | 53 |
| **Controls** | Healthy Foliage | Crop — Healthy | 8 |

---

## 📁 Repository Structure

```
LeafSentinel/
├── configs/
│   ├── dataset_audit.yaml        # Dataset profiling and validation configuration
│   └── segmentation.yaml         # Lesion segmentation model & training configuration
├── data/
│   └── raw/plantseg/             # Unmodified raw PlantSeg dataset (.gitignored)
├── outputs/
│   ├── audit/                    # Discovery summaries, statistics, and duplicate reports (.gitignored)
│   ├── dataset/                  # Authoritative manifest.csv and exclusions.csv (.gitignored)
│   ├── figures/                  # Publication-quality distribution plots (.gitignored)
│   ├── training/                 # Model checkpoints, training history, and loss curves (.gitignored)
│   └── evaluation/               # Test metrics, per-class metrics, 5-panel predictions (.gitignored)
├── src/
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── inspect.py            # Dynamic dataset auto-discovery & metadata parsing
│   │   ├── validate.py           # Incremental stream validation 
│   │   ├── duplicates.py         # Exact (MD5) & Perceptual (dHash) duplicate detection
│   │   ├── leakage.py            # Two-stage duplicate verification & Union-Find grouping
│   │   ├── statistics.py         # Statistical profiling & cross-tabulations
│   │   ├── feasibility.py        # Class feasibility heuristic tier scoring
│   │   └── visualize.py          # Matplotlib distribution chart & sample card generators
│   └── segmentation/
│       ├── __init__.py
│       ├── model.py              # ResNet18-UNet architecture (14.3M parameters)
│       ├── dataset.py            # PyTorch Dataset with synchronized spatial transforms
│       ├── metrics.py            # Dice, IoU, Precision, Recall, FP Area Ratio
│       ├── train.py              # BCE+Dice loss, AdamW, validation, checkpointing & early stopping
│       └── evaluate.py           # Test set evaluation & 5-panel qualitative prediction cards
├── scripts/
│   ├── audit_dataset.py          # End-to-end dataset discovery & audit runner
│   ├── prepare_dataset.py        # Leakage-free benchmark dataset preparation & manifest generator
│   ├── train_segmentation.py     # Training runner CLI (with --smoke-test mode)
│   └── evaluate_segmentation.py  # Test set evaluation runner CLI
├── tests/
│   ├── __init__.py
│   ├── test_dataset_audit.py     # Unit tests for discovery, validation, and duplicates
│   └── test_segmentation.py      # Unit tests for zero leakage, U-Net forward pass, loss, metrics
├── .gitignore                    # Excludes weights, virtual env, dataset, and outputs
├── requirements.txt              # Standard dependencies
└── README.md
```

---

## 🚀 Execution Guide

### 1. Environment Setup
```bash
python -m venv .venv
# Activate virtual environment
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m unittest tests/test_dataset_audit.py tests/test_segmentation.py
```

### 3. Run Dataset Preparation & Manifest Generation
```bash
python scripts/prepare_dataset.py --config configs/segmentation.yaml
```

### 4. Run CPU Smoke Test
```bash
python scripts/train_segmentation.py --config configs/segmentation.yaml --smoke-test
```

### 5. Run Full Baseline Training
```bash
python scripts/train_segmentation.py --config configs/segmentation.yaml
```

### 6. Evaluate on Test Split
```bash
python scripts/evaluate_segmentation.py \
    --config configs/segmentation.yaml \
    --checkpoint outputs/training/<run_name>/best_model.pth
```
