"""Dataset Preparation & Leakage-Free Manifest Generator for LeafSentinel.

Performs:
1. Class selection & healthy control policy enforcement.
2. Two-stage duplicate verification (MD5 exact + SSIM-verified dHash).
3. Leakage-free stratified train/val/test split generation.
4. Mandatory zero-leakage split assertion.
5. Manifest (`manifest.csv`), exclusions audit (`exclusions.csv`), and summary generation.
6. Dataset publication figure generation.
"""

import argparse
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.inspect import discover_dataset
from src.dataset.validate import validate_dataset
from src.dataset.leakage import build_leakage_groups, create_leakage_free_splits, validate_zero_leakage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LeafSentinel.PrepareDataset")

SPLIT_COLORS = {
    "train": "#2e7d32",
    "val": "#1565c0",
    "test": "#c62828"
}


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_preparation(config_path: Path, data_dir_override: Optional[str] = None):
    logger.info("================================================================================")
    logger.info(" LeafSentinel — Leakage-Free Benchmark Dataset Preparation & Manifest Generator  ")
    logger.info("================================================================================")

    config = load_yaml_config(config_path)
    data_dir_str = data_dir_override or config.get("dataset", {}).get("root", "data/raw/plantseg")
    data_dir = Path(data_dir_str).resolve()
    
    paths_cfg = config.get("paths", {})
    output_dir = Path(paths_cfg.get("output_dir", "outputs")).resolve()
    dataset_dir = Path(paths_cfg.get("dataset_dir", output_dir / "dataset")).resolve()
    figures_dir = Path(paths_cfg.get("figures_dir", output_dir / "figures")).resolve()

    dataset_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Discovery & Validation
    logger.info("\n>>> [Step 1/6] Discovering raw dataset & loading verified records...")
    discovery = discover_dataset(data_dir)
    val_report = validate_dataset(discovery.records_df)
    validated_df = val_report.validated_df.copy()

    # Load duplicate candidates if available
    dup_candidates_paths = [
        Path("outputs/audit/reports/duplicate_candidates.csv"),
        Path("outputs/reports/duplicate_candidates.csv"),
        Path("outputs/phase1/reports/duplicate_candidates.csv")
    ]
    dup_candidates_df = pd.DataFrame()
    for dp in dup_candidates_paths:
        if dp.exists():
            dup_candidates_df = pd.read_csv(dp)
            logger.info(f"Loaded {len(dup_candidates_df)} duplicate candidate pairs from {dp}.")
            break

    # 2. Class Selection & Healthy Sample Policy
    logger.info("\n>>> [Step 2/6] Applying class selection and exclusion policies...")
    selected_classes_cfg = config.get("dataset", {}).get("selected_classes", [])
    
    target_disease_map = {}
    for item in selected_classes_cfg:
        d_key = str(item.get("disease")).strip().lower()
        target_disease_map[d_key] = item.get("display_name", item.get("disease"))

    exclusions: List[Dict[str, Any]] = []
    included_records: List[Dict[str, Any]] = []

    anom_policy = config.get("dataset", {}).get("anomalous_masks", {}).get("action", "exclude")
    healthy_cfg = config.get("dataset", {}).get("healthy", {})
    healthy_enabled = healthy_cfg.get("enabled", True)
    healthy_max_ratio = float(healthy_cfg.get("max_ratio", 0.20))

    healthy_pool = []
    
    for idx, row in validated_df.iterrows():
        sid = row["sample_id"]
        img_p = row["image_path"]
        orig_split = row["split"]
        is_val_img = row["is_valid_image"]
        is_val_mask = row["is_valid_mask"]
        disease_raw = str(row["disease"]).strip().lower()
        aff_px = row.get("affected_pixels", 0)

        if not is_val_img:
            exclusions.append({
                "image_id": sid,
                "original_path": img_p,
                "reason": f"Corrupt or unreadable image: {row.get('image_error')}",
                "source": row.get("source", "PlantSeg"),
                "original_split": orig_split
            })
            continue

        is_healthy = "healthy" in disease_raw or aff_px == 0
        
        if is_healthy:
            if healthy_enabled:
                healthy_pool.append(row)
            else:
                exclusions.append({
                    "image_id": sid,
                    "original_path": img_p,
                    "reason": "Healthy control excluded by configuration policy",
                    "source": row.get("source", "PlantSeg"),
                    "original_split": orig_split
                })
            continue

        if disease_raw not in target_disease_map:
            exclusions.append({
                "image_id": sid,
                "original_path": img_p,
                "reason": f"Class '{row['disease']}' not in selected benchmark subset",
                "source": row.get("source", "PlantSeg"),
                "original_split": orig_split
            })
            continue

        if not is_val_mask or aff_px == 0:
            if anom_policy == "exclude":
                exclusions.append({
                    "image_id": sid,
                    "original_path": img_p,
                    "reason": f"Anomalous/empty mask for diseased sample: {row.get('mask_error') or 'Zero lesion pixels'}",
                    "source": row.get("source", "PlantSeg"),
                    "original_split": orig_split
                })
                continue

        row_dict = row.to_dict()
        row_dict["is_healthy"] = False
        row_dict["display_class"] = target_disease_map.get(disease_raw, row["disease"])
        included_records.append(row_dict)

    n_diseased = len(included_records)
    if healthy_pool and healthy_enabled:
        max_healthy_allowed = int(round(n_diseased * healthy_max_ratio / (1.0 - healthy_max_ratio))) if healthy_max_ratio < 1.0 else len(healthy_pool)
        rng = random.Random(config.get("split", {}).get("seed", 42))
        shuffled_healthy = list(healthy_pool)
        rng.shuffle(shuffled_healthy)
        
        for h_row in shuffled_healthy[:max_healthy_allowed]:
            h_dict = h_row.to_dict()
            h_dict["is_healthy"] = True
            h_dict["display_class"] = f"{h_row['host']} — Healthy"
            included_records.append(h_dict)

        for h_row in shuffled_healthy[max_healthy_allowed:]:
            exclusions.append({
                "image_id": h_row["sample_id"],
                "original_path": h_row["image_path"],
                "reason": f"Healthy sample capped by max_ratio ({healthy_max_ratio:.2f}) threshold",
                "source": h_row.get("source", "PlantSeg"),
                "original_split": h_row["split"]
            })

    active_df = pd.DataFrame(included_records)
    exclusions_df = pd.DataFrame(exclusions)
    
    logger.info(f"Class filtering complete: {len(active_df)} samples included ({n_diseased} diseased, {len(active_df) - n_diseased} healthy).")
    logger.info(f"Total exclusions logged: {len(exclusions_df)} records.")

    # 3. Two-Stage Duplicate Verification & Union-Find Grouping
    logger.info("\n>>> [Step 3/6] Running two-stage duplicate verification and connected grouping...")
    leak_cfg = config.get("leakage", {})
    dhash_thresh = int(leak_cfg.get("near_duplicates", {}).get("dhash_threshold", 6))
    ssim_thresh = float(leak_cfg.get("near_duplicates", {}).get("ssim_threshold", 0.85))

    sample_to_group, group_report = build_leakage_groups(
        df=active_df,
        duplicate_candidates_df=dup_candidates_df,
        dhash_threshold=dhash_thresh,
        ssim_threshold=ssim_thresh
    )
    active_df["duplicate_group_id"] = active_df["sample_id"].map(sample_to_group)

    # 4. Generate Stratified Leakage-Free Splits
    logger.info("\n>>> [Step 4/6] Generating stratified leakage-free splits...")
    split_cfg = config.get("split", {})
    train_r = float(split_cfg.get("train_ratio", 0.70))
    val_r = float(split_cfg.get("val_ratio", 0.15))
    test_r = float(split_cfg.get("test_ratio", 0.15))
    seed = int(split_cfg.get("seed", 42))

    active_df["benchmark_split"] = create_leakage_free_splits(
        df=active_df,
        sample_to_group=sample_to_group,
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        seed=seed
    )
    active_df["phase2_split"] = active_df["benchmark_split"]  # Backward compatibility alias

    # 5. Mandatory Zero-Leakage Assertion
    logger.info("\n>>> [Step 5/6] Enforcing mandatory zero-leakage split assertion...")
    validate_zero_leakage(active_df)

    manifest_df = pd.DataFrame({
        "image_id": active_df["sample_id"],
        "image_path": active_df["image_path"],
        "mask_path": active_df["mask_path"],
        "host": active_df["host"],
        "disease": active_df["disease"],
        "display_class": active_df["display_class"],
        "original_split": active_df["split"],
        "benchmark_split": active_df["benchmark_split"],
        "phase2_split": active_df["benchmark_split"],
        "duplicate_group_id": active_df["duplicate_group_id"],
        "is_healthy": active_df["is_healthy"],
        "image_width": active_df["width"],
        "image_height": active_df["height"],
        "mask_pixels": active_df["affected_pixels"],
        "mask_area_ratio": active_df["affected_area_ratio"]
    })

    manifest_path = dataset_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Saved authoritative dataset manifest ({len(manifest_df)} records) to {manifest_path}")

    exclusions_path = dataset_dir / "exclusions.csv"
    exclusions_df.to_csv(exclusions_path, index=False)
    logger.info(f"Saved exclusions audit ({len(exclusions_df)} records) to {exclusions_path}")

    split_counts = manifest_df["benchmark_split"].value_counts().to_dict()
    class_dist_df = manifest_df.groupby(["display_class", "benchmark_split"]).size().unstack(fill_value=0)
    class_dist_df["Total"] = class_dist_df.sum(axis=1)
    class_dist_path = dataset_dir / "class_distribution.csv"
    class_dist_df.to_csv(class_dist_path)
    logger.info(f"Saved class distribution breakdown to {class_dist_path}")

    summary_dict = {
        "project": "LeafSentinel",
        "total_manifest_samples": len(manifest_df),
        "split_counts": split_counts,
        "split_percentages": {k: round(v / len(manifest_df) * 100, 2) for k, v in split_counts.items()},
        "diseased_samples": int((~manifest_df["is_healthy"]).sum()),
        "healthy_samples": int(manifest_df["is_healthy"].sum()),
        "selected_classes_count": int(manifest_df["display_class"].nunique()),
        "duplicate_groups_count": group_report.total_duplicate_groups,
        "multi_sample_duplicate_groups": group_report.multi_sample_groups,
        "exact_duplicate_pairs_grouped": group_report.exact_duplicate_pairs_grouped,
        "near_duplicate_pairs_confirmed": group_report.near_pairs_confirmed,
        "near_duplicate_candidates_rejected": group_report.near_candidates_rejected,
        "cross_original_split_groups": group_report.cross_original_split_groups,
        "total_exclusions": len(exclusions_df),
        "mask_area_ratio_median": float(manifest_df[~manifest_df["is_healthy"]]["mask_area_ratio"].median()),
        "mask_area_ratio_mean": float(manifest_df[~manifest_df["is_healthy"]]["mask_area_ratio"].mean())
    }

    summary_json_path = dataset_dir / "dataset_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    logger.info(f"Saved dataset summary JSON to {summary_json_path}")

    # 6. Generate Figures
    logger.info("\n>>> [Step 6/6] Generating publication figures...")
    _plot_dataset_visuals(manifest_df, figures_dir)

    logger.info("\n================================================================================")
    logger.info(" Benchmark Dataset Preparation Complete & Verified!                            ")
    logger.info("================================================================================")
    logger.info(f" • Total Manifest Samples   : {len(manifest_df):,}")
    logger.info(f" • Train Split              : {split_counts.get('train', 0):,} ({split_counts.get('train', 0)/len(manifest_df)*100:.1f}%)")
    logger.info(f" • Val Split                : {split_counts.get('val', 0):,} ({split_counts.get('val', 0)/len(manifest_df)*100:.1f}%)")
    logger.info(f" • Test Split               : {split_counts.get('test', 0):,} ({split_counts.get('test', 0)/len(manifest_df)*100:.1f}%)")
    logger.info(f" • Duplicate Groups Confined: {group_report.total_duplicate_groups} groups (ZERO LEAKAGE)")
    logger.info(f" • Manifest File            : {manifest_path}")
    logger.info("================================================================================")


def _plot_dataset_visuals(manifest_df: pd.DataFrame, figures_dir: Path):
    dpi = 300
    
    # 1. Final Class Distribution
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    c_counts = manifest_df["display_class"].value_counts(ascending=True)
    ax.barh(c_counts.index, c_counts.values, color="#1b4d3e", height=0.65)
    for i, (name, val) in enumerate(c_counts.items()):
        ax.text(val + 3, i, f"{val:,}", va="center", fontsize=9, fontweight="bold", color="#263238")
    ax.set_xlim(0, max(c_counts.values) * 1.15)
    ax.set_title("Selected Benchmark Class Distribution", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Sample Count", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "class_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Split Distribution
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=dpi)
    s_counts = manifest_df["benchmark_split"].value_counts()
    colors = [SPLIT_COLORS.get(s, "#78909c") for s in s_counts.index]
    bars = ax.bar(s_counts.index.str.capitalize(), s_counts.values, color=colors, width=0.55)
    for bar, val in zip(bars, s_counts.values):
        pct = val / len(manifest_df) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{val:,}\n({pct:.1f}%)", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, max(s_counts.values) * 1.2)
    ax.set_title("Leakage-Free Benchmark Split Distribution", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Sample Count", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "split_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Class by Split
    fig, ax = plt.subplots(figsize=(11, 7), dpi=dpi)
    ct = manifest_df.groupby(["display_class", "benchmark_split"]).size().unstack(fill_value=0)
    bottom = np.zeros(len(ct))
    for col in ["train", "val", "test"]:
        if col in ct.columns:
            vals = ct[col].values
            ax.barh(ct.index, vals, left=bottom, label=col.capitalize(), color=SPLIT_COLORS.get(col), height=0.65)
            bottom += vals
    ax.set_title("Class Distribution by Leakage-Free Split", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Sample Count", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "class_by_split.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Mask Area Ratio Histogram
    fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
    diseased_df = manifest_df[~manifest_df["is_healthy"]]
    ratios = diseased_df["mask_area_ratio"].dropna().values * 100
    if len(ratios) > 0:
        ax.hist(ratios, bins=35, color="#e65100", edgecolor="white", alpha=0.85)
        med = float(np.median(ratios))
        ax.axvline(med, color="#b71c1c", linestyle="--", linewidth=1.5, label=f"Median: {med:.2f}%")
        ax.legend(frameon=True)
    ax.set_title("Lesion Area Ratio Distribution (% of Image)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Lesion Area (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Sample Count", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "mask_ratio_distribution.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="LeafSentinel Dataset Preparation CLI")
    parser.add_argument("--config", type=str, default="configs/segmentation.yaml", help="Path to YAML config")
    parser.add_argument("--data", type=str, default=None, help="Optional override for raw data directory")
    args = parser.parse_args()

    run_preparation(config_path=Path(args.config), data_dir_override=args.data)


if __name__ == "__main__":
    main()
