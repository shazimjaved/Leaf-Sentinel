"""Phase 4 Severity Target Feasibility Audit.

This script performs a rigorous data-driven audit of the existing PlantSeg dataset
to determine if true disease severity is scientifically calculable.
"""

import json
import logging
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("Phase4.SeverityAudit")

def run_audit():
    root = Path("data/raw/plantseg")
    out_dir = Path("outputs/severity/audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report = {}

    # 1. Metadata Inspection
    meta_path = root / "Metadata.csv"
    meta_df = pd.read_csv(meta_path)
    report["metadata_columns"] = list(meta_df.columns)
    report["metadata_dtypes"] = {k: str(v) for k, v in meta_df.dtypes.items()}
    
    # Check for specific fields
    severity_keywords = [
        "severity", "grade", "stage", "lesion percentage", "infection percentage",
        "affected area", "leaf area", "healthy area", "score"
    ]
    found_keywords = [c for c in meta_df.columns if any(k in c.lower() for k in severity_keywords)]
    report["metadata_severity_fields_found"] = found_keywords

    # 2. COCO Annotations Inspection
    coco_path = root / "annotation_train.json"
    with open(coco_path, "r") as f:
        coco = json.load(f)
        
    report["coco_categories"] = [c["name"] for c in coco.get("categories", [])]
    report["coco_num_categories"] = len(coco.get("categories", []))
    
    sample_ann = coco["annotations"][0] if coco.get("annotations") else {}
    report["coco_sample_annotation_keys"] = list(sample_ann.keys())
    report["coco_has_severity_attributes"] = "attributes" in sample_ann and any(k in str(sample_ann["attributes"]).lower() for k in severity_keywords)
    
    # 3. Mask Files Inspection
    mask_sample_path = root / "annotations/train/apple_black_rot_1.png" # known from phase 3
    if mask_sample_path.exists():
        mask = cv2.imread(str(mask_sample_path), cv2.IMREAD_GRAYSCALE)
        unique_vals = np.unique(mask)
        report["mask_pixel_values"] = [int(v) for v in unique_vals]
        report["mask_is_binary"] = len(unique_vals) <= 2
        
    # 4. Verify mask_area_ratio against true calculation
    manifest_path = Path("outputs/classification/dataset/classification_manifest.csv")
    if not manifest_path.exists():
        logger.error("Classification manifest not found.")
        sys.exit(1)
        
    clf_df = pd.read_csv(manifest_path)
    
    sample_df = clf_df.sample(n=50, random_state=42)
    diffs = []
    for _, row in sample_df.iterrows():
        m_path = root.parent.parent / row["mask_relpath"]
        mask = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            lesion_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            calculated_ratio = lesion_pixels / total_pixels
            reported_ratio = row["mask_area_ratio"]
            diffs.append(abs(calculated_ratio - reported_ratio))
            
    report["mask_area_ratio_verification"] = {
        "mean_absolute_difference": float(np.mean(diffs)) if diffs else None,
        "max_absolute_difference": float(np.max(diffs)) if diffs else None
    }

    # 5. Data Distribution Analysis (Using benchmark split)
    report["total_samples"] = len(clf_df)
    report["split_counts"] = clf_df["benchmark_split"].value_counts().to_dict()
    
    target = clf_df["mask_area_ratio"]
    report["target_stats"] = {
        "min": float(target.min()),
        "max": float(target.max()),
        "mean": float(target.mean()),
        "median": float(target.median()),
        "std": float(target.std()),
        "q25": float(target.quantile(0.25)),
        "q75": float(target.quantile(0.75))
    }
    
    # Plotting target distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(target, bins=50, color='skyblue', edgecolor='black')
    ax.set_title("Distribution of Mask Area Ratio (Lesion Burden Proxy)")
    ax.set_xlabel("Lesion Pixels / Total Image Pixels")
    ax.set_ylabel("Frequency")
    fig.savefig(out_dir / "target_distribution.png")
    plt.close(fig)
    
    # Plotting target distribution by class
    fig, ax = plt.subplots(figsize=(12, 8))
    clf_df.boxplot(column="mask_area_ratio", by="display_class", ax=ax, vert=False)
    ax.set_title("Lesion Burden Distribution by Disease Class")
    plt.suptitle("")
    ax.set_xlabel("Mask Area Ratio")
    plt.tight_layout()
    fig.savefig(out_dir / "target_distribution_by_class.png")
    plt.close(fig)

    # Save outputs
    with open(out_dir / "severity_feasibility.json", "w") as f:
        json.dump(report, f, indent=4)
        
    # Save detailed stats
    class_stats = clf_df.groupby("display_class")["mask_area_ratio"].describe()
    class_stats.to_csv(out_dir / "per_class_target_statistics.csv")
    
    split_stats = clf_df.groupby("benchmark_split")["mask_area_ratio"].describe()
    split_stats.to_csv(out_dir / "target_statistics_by_split.csv")
    
    logger.info("Audit complete. Outputs saved to outputs/severity/audit/")

if __name__ == "__main__":
    run_audit()
