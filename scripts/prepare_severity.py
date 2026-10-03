"""Phase 4 Dataset Preparation for Image-Relative Lesion Burden Regression."""

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
logger = logging.getLogger("LeafSentinel.Phase4.Prepare")

def get_relative_path(abs_path: str, root_dir: Path) -> str:
    normalized = str(abs_path).replace(chr(92), "/")
    if "images/" in normalized:
        return "images/" + normalized.split("images/")[-1]
    elif "annotations/" in normalized:
        return "annotations/" + normalized.split("annotations/")[-1]
    else:
        raise ValueError(f"Neither 'images/' nor 'annotations/' found in path: {abs_path}")

def run_preparation(phase2_manifest_path: Path, output_dir: Path, dataset_root: Path):
    logger.info("================================================================================")
    logger.info(" LeafSentinel Phase 4 — Severity Dataset Preparation")
    logger.info("================================================================================")
    
    if not phase2_manifest_path.exists():
        logger.error(f"Phase 2 manifest not found at {phase2_manifest_path}")
        sys.exit(1)

    df = pd.read_csv(phase2_manifest_path)
    logger.info(f"Loaded Phase 2 manifest with {len(df)} total samples.")

    # We MUST KEEP healthy samples for Phase 4 since mask_area_ratio = 0.0 is a valid target.
    # Convert paths to relative
    df["image_relpath"] = df["image_path"].apply(lambda p: get_relative_path(p, dataset_root))
    
    # Handle NaNs in mask_path for healthy samples if they don't have masks
    def safe_mask_path(p):
        if pd.isna(p) or str(p) == "nan":
            return ""
        return get_relative_path(p, dataset_root)
        
    df["mask_relpath"] = df["mask_path"].apply(safe_mask_path)
    
    # Ensure mask_area_ratio is bounded [0, 1]
    assert df["mask_area_ratio"].min() >= 0.0, "mask_area_ratio must be >= 0"
    assert df["mask_area_ratio"].max() <= 1.0, "mask_area_ratio must be <= 1"

    keep_cols = [
        "image_id", "image_relpath", "mask_relpath", "host", "disease", 
        "display_class", "original_split", "benchmark_split", 
        "duplicate_group_id", "mask_area_ratio", "is_healthy"
    ]
    sev_df = df[keep_cols].copy()

    # Verify leakage invariants
    leaked_groups = []
    for gid, group in sev_df.groupby("duplicate_group_id"):
        splits = group["benchmark_split"].unique()
        if len(splits) > 1:
            leaked_groups.append(gid)
            
    if leaked_groups:
        logger.error(f"LEAKAGE DETECTED! {len(leaked_groups)} groups span multiple splits.")
        sys.exit(1)
        
    logger.info("Verified zero leakage invariants.")

    # Save Severity Manifest
    dataset_out_dir = output_dir / "dataset"
    dataset_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_out_dir / "severity_manifest.csv"
    sev_df.to_csv(manifest_path, index=False)
    logger.info(f"Saved severity manifest to {manifest_path}")

    # Generate Stats
    train_df = sev_df[sev_df["benchmark_split"] == "train"]
    val_df = sev_df[sev_df["benchmark_split"] == "val"]
    test_df = sev_df[sev_df["benchmark_split"] == "test"]
    
    train_target = train_df["mask_area_ratio"]
    
    stats_dict = {
        "total_samples": len(sev_df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "healthy_samples": int(sev_df["is_healthy"].sum()),
        "target_train_mean": float(train_target.mean()),
        "target_train_median": float(train_target.median()),
        "target_train_std": float(train_target.std()),
        "target_train_min": float(train_target.min()),
        "target_train_max": float(train_target.max()),
        "target_train_q25": float(train_target.quantile(0.25)),
        "target_train_q75": float(train_target.quantile(0.75)),
    }
    
    with open(dataset_out_dir / "dataset_summary.json", "w") as f:
        json.dump(stats_dict, f, indent=4)
        
    logger.info(f"Train samples: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    logger.info(f"Healthy samples retained: {stats_dict['healthy_samples']}")
    
    # Save per-class stats
    class_stats = sev_df.groupby("display_class")["mask_area_ratio"].describe()
    class_stats.to_csv(dataset_out_dir / "per_class_target_statistics.csv")
    
    split_stats = sev_df.groupby("benchmark_split")["mask_area_ratio"].describe()
    split_stats.to_csv(dataset_out_dir / "target_statistics_by_split.csv")
    
    logger.info("Severity Dataset Preparation Complete.")

def main():
    parser = argparse.ArgumentParser(description="LeafSentinel Phase 4 Severity Dataset Preparation")
    parser.add_argument("--phase2-manifest", type=str, default="data/outputs/dataset/manifest.csv", help="Path to Phase 2 manifest")
    parser.add_argument("--output-dir", type=str, default="outputs/severity", help="Output directory for Phase 4")
    parser.add_argument("--data-root", type=str, default="data/raw/plantseg", help="Root data directory")
    args = parser.parse_args()

    run_preparation(
        phase2_manifest_path=Path(args.phase2_manifest),
        output_dir=Path(args.output_dir),
        dataset_root=Path(args.data_root)
    )

if __name__ == "__main__":
    main()
