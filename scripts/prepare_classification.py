"""Phase 3 Dataset Preparation for Disease Classification.

Performs:
1. Load Phase 2 manifest (authoritative source of splits and duplicate groups).
2. Filter out healthy controls (due to low support).
3. Assign deterministic integer class indices for the 10 disease classes.
4. Create portable relative paths.
5. Verify leakage invariants.
6. Generate classification manifest and class distribution figures.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.leakage import validate_zero_leakage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LeafSentinel.Phase3.Prepare")

def get_relative_path(abs_path: str, root_dir: Path) -> str:
    """Convert Windows or POSIX Phase 2 paths into PlantSeg-relative paths."""
    normalized = str(abs_path).replace(chr(92), "/")

    for marker in ("images/", "annotations/"):
        idx = normalized.find(marker)
        if idx != -1:
            return normalized[idx:]

    raise ValueError(
        f"Could not derive PlantSeg-relative path from: {abs_path}"
    )

def run_preparation(phase2_manifest_path: Path, output_dir: Path, dataset_root: Path):
    logger.info("================================================================================")
    logger.info(" LeafSentinel Phase 3 — Classification Dataset Preparation")
    logger.info("================================================================================")
    
    if not phase2_manifest_path.exists():
        logger.error(f"Phase 2 manifest not found at {phase2_manifest_path}")
        sys.exit(1)

    df = pd.read_csv(phase2_manifest_path)
    logger.info(f"Loaded Phase 2 manifest with {len(df)} total samples.")

    # 1. Separate healthy controls
    healthy_mask = df["is_healthy"] == True
    healthy_df = df[healthy_mask].copy()
    diseased_df = df[~healthy_mask].copy()

    # Save healthy controls (exclusion log)
    excluded_dir = output_dir / "dataset"
    excluded_dir.mkdir(parents=True, exist_ok=True)
    
    healthy_df["exclusion_reason"] = "insufficient healthy-class support for 11-class benchmark"
    healthy_df.to_csv(excluded_dir / "excluded_controls.csv", index=False)
    logger.info(f"Excluded {len(healthy_df)} healthy controls (saved to excluded_controls.csv).")
    
    # 2. Validate exactly 10 classes
    classes = sorted(diseased_df["display_class"].unique())
    if len(classes) != 10:
        logger.error(f"Expected 10 disease classes, found {len(classes)}.")
        sys.exit(1)
        
    logger.info("Validated 10 target disease classes.")

    # 3. Create deterministic class mapping
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    with open(excluded_dir / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=4)
        
    diseased_df["class_index"] = diseased_df["display_class"].map(class_to_idx)

    # 4. Generate portable relative paths
    # (Fixes the Phase 2 absolute path portability issue)
    diseased_df["image_relpath"] = diseased_df["image_path"].apply(lambda p: get_relative_path(p, dataset_root))
    diseased_df["mask_relpath"] = diseased_df["mask_path"].apply(lambda p: get_relative_path(p, dataset_root))

    # Keep only necessary columns for Phase 3
    keep_cols = [
        "image_id", "image_relpath", "mask_relpath", "host", "disease", 
        "display_class", "class_index", "original_split", "benchmark_split", 
        "duplicate_group_id", "mask_area_ratio"
    ]
    clf_df = diseased_df[keep_cols].copy()

    # 5. Verify leakage invariants
    # Invariant A: No duplicate group crosses benchmark splits
    leaked_groups = []
    for gid, group in clf_df.groupby("duplicate_group_id"):
        splits = group["benchmark_split"].unique()
        if len(splits) > 1:
            leaked_groups.append(gid)
            
    if leaked_groups:
        logger.error(f"LEAKAGE DETECTED! {len(leaked_groups)} groups span multiple splits.")
        sys.exit(1)
        
    # Invariant B: No image appears in multiple splits
    image_counts = clf_df.groupby("image_id")["benchmark_split"].nunique()
    bad_images = image_counts[image_counts > 1]
    if not bad_images.empty:
        logger.error(f"LEAKAGE DETECTED! {len(bad_images)} images span multiple splits.")
        sys.exit(1)
        
    logger.info("Verified zero leakage invariants.")

    # 6. Save Classification Manifest
    manifest_path = excluded_dir / "classification_manifest.csv"
    clf_df.to_csv(manifest_path, index=False)
    logger.info(f"Saved classification manifest to {manifest_path}")

    # 7. Generate class distribution statistics & class weights
    # Class weights must be computed strictly on the TRAIN split
    train_df = clf_df[clf_df["benchmark_split"] == "train"]
    train_counts = train_df["class_index"].value_counts().sort_index()
    
    total_train = len(train_df)
    n_classes = len(classes)
    
    # Sklearn balanced weight formula: n_samples / (n_classes * np.bincount(y))
    class_weights = []
    for idx in range(n_classes):
        count = train_counts.get(idx, 0)
        weight = total_train / (n_classes * count) if count > 0 else 0.0
        class_weights.append(weight)
        
    # Cap weights for numerical stability (e.g., max weight of 10.0)
    class_weights = np.clip(class_weights, 0.0, 10.0).tolist()
    
    with open(excluded_dir / "dataset_summary.json", "w") as f:
        json.dump({
            "total_samples": len(clf_df),
            "train_samples": len(clf_df[clf_df["benchmark_split"] == "train"]),
            "val_samples": len(clf_df[clf_df["benchmark_split"] == "val"]),
            "test_samples": len(clf_df[clf_df["benchmark_split"] == "test"]),
            "n_classes": 10,
            "train_class_weights": class_weights
        }, f, indent=4)
        
    # Save split distribution CSV
    dist_df = clf_df.groupby(["display_class", "benchmark_split"]).size().unstack(fill_value=0)
    dist_df["total"] = dist_df.sum(axis=1)
    dist_df.to_csv(excluded_dir / "class_distribution.csv")

    # 8. Generate Figures
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Figure 1: Class Distribution by Split
    fig, ax = plt.subplots(figsize=(12, 6))
    splits = ["train", "val", "test"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    dist_sorted = dist_df.sort_values("total", ascending=True)
    bottom = np.zeros(len(dist_sorted))
    
    for split, color in zip(splits, colors):
        if split in dist_sorted.columns:
            vals = dist_sorted[split].values
            ax.barh(dist_sorted.index, vals, left=bottom, label=split.capitalize(), color=color)
            bottom += vals
            
    ax.set_title("Phase 3 Classification Dataset - Class Distribution by Split")
    ax.set_xlabel("Number of Samples")
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "classification_class_distribution.png")
    plt.close(fig)
    
    # Figure 2: Train Class Weights
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(classes, class_weights, color="#d62728")
    ax.set_title("Computed Class Weights (Balanced Cross-Entropy) from Train Split")
    ax.set_ylabel("Weight")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(fig_dir / "classification_class_weights.png")
    plt.close(fig)
    
    logger.info("Generated classification dataset figures.")
    logger.info("Classification Dataset Preparation Complete.")

def main():
    parser = argparse.ArgumentParser(description="LeafSentinel Phase 3 Dataset Preparation")
    parser.add_argument("--phase2-manifest", type=str, default="data/outputs/dataset/manifest.csv", help="Path to Phase 2 manifest")
    parser.add_argument("--output-dir", type=str, default="outputs/classification", help="Output directory for Phase 3")
    parser.add_argument("--data-root", type=str, default="data/raw/plantseg", help="Root data directory")
    args = parser.parse_args()

    run_preparation(
        phase2_manifest_path=Path(args.phase2_manifest),
        output_dir=Path(args.output_dir),
        dataset_root=Path(args.data_root)
    )

if __name__ == "__main__":
    main()
