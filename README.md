# Leaf Sentinel
> **Portfolio-Grade Computer Vision System for Real-World Plant Disease Detection, Fine-Grained Segmentation, Severity Estimation & Crop-Health Analytics.**

---

## 📌 Project Overview
**Leaf Sentinel** is an enterprise-grade precision agriculture computer vision pipeline engineered to detect, segment, and quantify foliar crop diseases from in-field RGB imagery.

The project is structured into progressive engineering phases:
* **Phase 1 (Completed)**: Dataset Discovery, Empirical Audit, Integrity Validation, Data-Leakage Analysis & Class Feasibility Scoping.
* **Phase 2 (Upcoming)**: Baseline Architecture Selection, Leakage-Free Stratified Benchmarking & Model Training.
* **Phase 3**: Multi-Task Segmentation, Lesion Area Quantification & Severity Analytics.
* **Phase 4**: Edge Optimization, ONNX/TensorRT Export & Production API / Dashboard.

---

## 🔬 Phase 1: Empirical Audit & Dataset Evidence Report

Phase 1 performed an automated, non-destructive audit of the **PlantSeg** benchmark dataset (Zenodo/official distribution).

### 📊 Key Dataset Statistics & Findings
| Metric | Audit Value | Description |
|---|---|---|
| **Total Images Discovered** | **7,774** | 100% verified decodable RGB images |
| **Total Ground Truth Masks** | **7,774** | 7,766 valid masks (8 zero-pixel/edge anomalies flagged) |
| **Plant Host Species** | **34 Hosts** | Apple, Tomato, Potato, Grape, Corn, Wheat, Banana, Citrus, etc. |
| **Pathology / Disease Classes**| **115 Categories**| Multi-host fungal, bacterial, viral, and healthy conditions |
| **Official Partition Split** | **Train: 5,367 \| Val: 1,180 \| Test: 1,227** | Preserved official Zenodo partitioning |
| **Annotation Formats** | **Dual Convention** | Per-sample binary masks (`.png`) + COCO instance JSONs |
| **Median Lesion Area Ratio** | **5.42%** | Skewed distribution (range: 0.01% to 84.1%) |
| **Exact Duplicate Image Pairs**| **290 Pairs** | Byte-exact MD5 collisions across dataset |
| **Near-Duplicate Pairs (dHash)**| **431 Pairs** | Perceptual 256-bit hash matches (Hamming dist $\le 6$) |
| **Cross-Split Data Leakage** | **340 Candidate Pairs** | Matches across Train $\leftrightarrow$ Val/Test partitions |

---

## 🎯 Class Feasibility Scoping Summary

Based on sample volume, mask integrity, and partition representation, classes are categorized into evidence tiers:

* **Strong Candidates (21 Classes)**: High sample count ($\ge 100$), robust mask density ($\ge 80$), well-balanced splits. (e.g. *Tomato Early Blight*, *Potato Late Blight*, *Apple Black Rot*, *Grape Downy Mildew*, *Corn Gray Leaf Spot*).
* **Usable (50 Classes)**: Adequate samples ($\ge 40$) and masks ($\ge 30$) suitable for baseline training and transfer learning.
* **Limited (37 Classes)**: Modest volume ($15\text{--}39$) requiring heavy data augmentation or few-shot techniques.
* **Insufficient (7 Classes)**: Very low volume ($< 15$ samples); reserved for future data expansion.

Full breakdown available in [`outputs/phase1/reports/class_feasibility.csv`](outputs/phase1/reports/class_feasibility.csv).

---

## 📁 Repository Structure

```
LeafSentinel/
├── configs/
│   └── phase1.yaml               # Configurable thresholds, paths, perceptual hash parameters
├── data/
│   ├── raw/                      # Raw dataset directory (.gitignored)
│   │   └── plantseg/             # Unmodified PlantSeg dataset
│   ├── interim/                  # Intermediate staging caches (.gitignored)
│   └── processed/                # Production model-ready tensors (.gitignored)
├── outputs/
│   └── phase1/
│       ├── reports/              # JSON summary, statistical CSVs, feasibility & duplicate reports
│       │   ├── dataset_summary.json
│       │   ├── dataset_statistics.csv
│       │   ├── class_feasibility.csv
│       │   ├── duplicate_candidates.csv
│       │   ├── validation_errors.csv
│       │   ├── host_distribution.csv
│       │   └── disease_distribution.csv
│       ├── figures/              # Publication-grade Matplotlib distribution plots (300 DPI)
│       │   ├── host_distribution.png
│       │   ├── disease_distribution.png
│       │   ├── split_distribution.png
│       │   ├── disease_by_split.png
│       │   ├── mask_ratio_distribution.png
│       │   └── resolution_distribution.png
│       └── samples/              # 3-Panel qualitative sample cards (Image | Mask | Overlay)
├── src/
│   ├── __init__.py
│   └── dataset/
│       ├── __init__.py
│       ├── inspect.py            # Dynamic dataset auto-discovery & metadata parsing
│       ├── validate.py           # Incremental, memory-safe stream validation (images & masks)
│       ├── duplicates.py         # Exact (MD5) & Perceptual (dHash) duplicate & leakage detection
│       ├── statistics.py         # Statistical distributions & summary aggregation
│       ├── feasibility.py        # Class feasibility heuristic tier scoring
│       └── visualize.py          # Pure Matplotlib chart & 3-panel sample generator
├── scripts/
│   └── run_phase1.py             # Single entrypoint CLI pipeline runner
├── tests/
│   ├── __init__.py
│   └── test_phase1.py            # Automated unit and integration test suite
├── .gitignore                    # Production gitignore (excludes weights, venv, raw data)
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation and Phase 1 evidence report
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python tests/test_phase1.py
```

### 3. Execute Phase 1 Audit Pipeline
```bash
python scripts/run_phase1.py --data data/raw/plantseg
```

---

## 📋 Phase 2 Roadmap
With Phase 1 complete, the dataset evidence will be reviewed to:
1. Define the initial multi-class disease scope (prioritizing the 21 Strong and top Usable classes).
2. Construct a **leakage-free split** that removes cross-partition duplicates flagged in `duplicate_candidates.csv`.
3. Establish baseline benchmarks for disease classification (ResNet/ConvNeXt/ViT) and semantic/instance segmentation (UNet++/DeepLabV3+/Mask2Former).
