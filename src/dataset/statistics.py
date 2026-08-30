"""Statistical aggregation and distribution reporting for Leaf Sentinel (Phase 1).

Computes comprehensive dataset statistics, cross-tabulations, mask ratio distributions,
resolution profiles, and serializes summary reports to JSON and CSV.
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetStatistics:
    """Encapsulates all computed statistical distributions and metrics."""
    summary_dict: Dict[str, Any]
    dataset_statistics_df: pd.DataFrame
    host_dist_df: pd.DataFrame
    disease_dist_df: pd.DataFrame
    host_disease_df: pd.DataFrame
    split_dist_df: pd.DataFrame
    disease_split_df: pd.DataFrame


def _compute_numeric_stats(series: pd.Series) -> Dict[str, Optional[float]]:
    """Compute summary statistics for a numeric series safely."""
    s = series.dropna()
    if len(s) == 0:
        return {
            "count": 0, "min": None, "max": None, "mean": None,
            "median": None, "std": None, "p25": None, "p75": None
        }
    return {
        "count": int(len(s)),
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(round(s.mean(), 4)),
        "median": float(round(s.median(), 4)),
        "std": float(round(s.std(), 4)) if len(s) > 1 else 0.0,
        "p25": float(round(s.quantile(0.25), 4)),
        "p75": float(round(s.quantile(0.75), 4))
    }


def compute_dataset_statistics(
    validated_df: pd.DataFrame,
    dataset_root: Path | str,
    structure_type: str = "auto_discovered",
    validation_summary: Optional[Dict[str, Any]] = None,
    duplicate_summary: Optional[Dict[str, Any]] = None
) -> DatasetStatistics:
    """Compute rich aggregate and per-class statistics from validated DataFrame.
    
    Args:
        validated_df: DataFrame with image and mask validation columns.
        dataset_root: Absolute or relative path to dataset root.
        structure_type: Discovered layout structure.
        validation_summary: Summary dictionary from ValidationReport.
        duplicate_summary: Summary dictionary from DuplicateReport.
        
    Returns:
        DatasetStatistics container with summary dictionary and distribution DataFrames.
    """
    logger.info("Computing dataset statistics and distributions...")
    df = validated_df.copy()
    now_iso = datetime.now().isoformat()

    total_records = len(df)
    valid_images_df = df[df["is_valid_image"] == True] if "is_valid_image" in df.columns else df
    valid_masks_df = df[df["is_valid_mask"] == True] if "is_valid_mask" in df.columns else pd.DataFrame()

    # 1. Host distribution
    host_counts = df["host"].value_counts(dropna=False).reset_index()
    host_counts.columns = ["host", "count"]
    host_counts["percentage"] = (host_counts["count"] / total_records * 100).round(2) if total_records > 0 else 0

    # 2. Disease distribution
    disease_counts = df["disease"].value_counts(dropna=False).reset_index()
    disease_counts.columns = ["disease", "count"]
    disease_counts["percentage"] = (disease_counts["count"] / total_records * 100).round(2) if total_records > 0 else 0

    # Healthy vs Diseased categorization
    def _is_healthy(d: str) -> bool:
        return "healthy" in str(d).lower() or "normal" in str(d).lower()

    df["is_healthy"] = df["disease"].apply(_is_healthy)
    healthy_count = int(df["is_healthy"].sum())
    diseased_count = int((~df["is_healthy"]).sum())

    # 3. Host x Disease cross-tabulation
    host_disease_ct = pd.crosstab(df["host"], df["disease"], margins=True, margins_name="Total")

    # 4. Split distribution
    split_counts = df["split"].value_counts(dropna=False).reset_index()
    split_counts.columns = ["split", "count"]
    split_counts["percentage"] = (split_counts["count"] / total_records * 100).round(2) if total_records > 0 else 0

    # 5. Disease x Split cross-tabulation
    disease_split_ct = pd.crosstab(df["disease"], df["split"], margins=True, margins_name="Total")

    # 6. Resolution statistics
    width_stats = _compute_numeric_stats(valid_images_df["width"]) if "width" in valid_images_df.columns else {}
    height_stats = _compute_numeric_stats(valid_images_df["height"]) if "height" in valid_images_df.columns else {}
    
    if "width" in valid_images_df.columns and "height" in valid_images_df.columns:
        aspect_series = valid_images_df["width"] / valid_images_df["height"].replace(0, np.nan)
        aspect_stats = _compute_numeric_stats(aspect_series)
    else:
        aspect_stats = {}

    channel_dist = valid_images_df["channels"].value_counts().to_dict() if "channels" in valid_images_df.columns else {}

    # Common resolutions
    if "width" in valid_images_df.columns and "height" in valid_images_df.columns:
        res_series = valid_images_df["width"].astype(str) + "x" + valid_images_df["height"].astype(str)
        top_resolutions = res_series.value_counts().head(10).to_dict()
    else:
        top_resolutions = {}

    # 7. Mask area / ratio statistics
    mask_ratio_stats = _compute_numeric_stats(valid_masks_df["affected_area_ratio"]) if "affected_area_ratio" in valid_masks_df.columns else {}
    lesion_comp_stats = _compute_numeric_stats(valid_masks_df["num_lesion_components"]) if "num_lesion_components" in valid_masks_df.columns else {}
    mask_conventions = valid_masks_df["mask_convention"].value_counts().to_dict() if "mask_convention" in valid_masks_df.columns else {}

    # 8. Per-class summary
    per_class_list = []
    if "host" in df.columns and "disease" in df.columns:
        grouped = df.groupby(["host", "disease"])
        for (h, d), group in grouped:
            g_valid_img = group[group["is_valid_image"] == True] if "is_valid_image" in group.columns else group
            g_valid_mask = group[group["is_valid_mask"] == True] if "is_valid_mask" in group.columns else pd.DataFrame()
            
            ratios = g_valid_mask["affected_area_ratio"].dropna() if "affected_area_ratio" in g_valid_mask.columns else pd.Series()

            split_c = group["split"].value_counts().to_dict() if "split" in group.columns else {}

            per_class_list.append({
                "host": h,
                "disease": d,
                "total_samples": len(group),
                "valid_images": len(g_valid_img),
                "valid_masks": len(g_valid_mask),
                "train_samples": split_c.get("train", 0),
                "val_samples": split_c.get("val", split_c.get("validation", 0)),
                "test_samples": split_c.get("test", 0),
                "unassigned_samples": split_c.get("unassigned", 0),
                "median_mask_ratio": float(round(ratios.median(), 4)) if len(ratios) > 0 else 0.0,
                "mean_mask_ratio": float(round(ratios.mean(), 4)) if len(ratios) > 0 else 0.0,
                "min_mask_ratio": float(round(ratios.min(), 4)) if len(ratios) > 0 else 0.0,
                "max_mask_ratio": float(round(ratios.max(), 4)) if len(ratios) > 0 else 0.0,
            })

    # High-level summary dictionary (for dataset_summary.json)
    summary_dict = {
        "dataset_name": "PlantSeg",
        "dataset_root": str(dataset_root),
        "structure_type": structure_type,
        "inspection_timestamp": now_iso,
        "overview": {
            "total_records": total_records,
            "valid_images": len(valid_images_df),
            "invalid_images": total_records - len(valid_images_df),
            "total_masks_referenced": int(df["mask_path"].notna().sum()) if "mask_path" in df.columns else 0,
            "valid_masks": len(valid_masks_df),
            "missing_masks": int(validation_summary.get("missing_masks", 0)) if validation_summary else 0,
            "corrupt_masks": int(validation_summary.get("corrupt_masks", 0)) if validation_summary else 0,
            "dimension_mismatches": int(validation_summary.get("dimension_mismatch_masks", 0)) if validation_summary else 0,
            "empty_masks": int(validation_summary.get("empty_masks", 0)) if validation_summary else 0,
        },
        "taxonomic_scope": {
            "total_hosts": int(df["host"].nunique()),
            "total_diseases": int(df["disease"].nunique()),
            "total_host_disease_combinations": len(per_class_list),
            "healthy_samples": healthy_count,
            "diseased_samples": diseased_count,
            "hosts": sorted(list(df["host"].dropna().unique())),
            "diseases": sorted(list(df["disease"].dropna().unique()))
        },
        "split_counts": split_counts.set_index("split")["count"].to_dict(),
        "resolution_statistics": {
            "width": width_stats,
            "height": height_stats,
            "aspect_ratio": aspect_stats,
            "channel_distribution": channel_dist,
            "top_resolutions": top_resolutions
        },
        "segmentation_mask_statistics": {
            "mask_ratio_distribution": mask_ratio_stats,
            "lesion_components_distribution": lesion_comp_stats,
            "detected_mask_conventions": mask_conventions
        },
        "data_integrity_and_duplicates": duplicate_summary or {},
        "per_class_statistics": per_class_list
    }

    # Flat table for dataset_statistics.csv
    stat_rows = [
        {"metric_category": "Overview", "metric_name": "Total Records", "value": total_records},
        {"metric_category": "Overview", "metric_name": "Valid Images", "value": len(valid_images_df)},
        {"metric_category": "Overview", "metric_name": "Valid Masks", "value": len(valid_masks_df)},
        {"metric_category": "Taxonomy", "metric_name": "Unique Hosts", "value": df["host"].nunique()},
        {"metric_category": "Taxonomy", "metric_name": "Unique Diseases", "value": df["disease"].nunique()},
        {"metric_category": "Taxonomy", "metric_name": "Healthy Samples", "value": healthy_count},
        {"metric_category": "Taxonomy", "metric_name": "Diseased Samples", "value": diseased_count},
        {"metric_category": "Resolution", "metric_name": "Mean Width", "value": width_stats.get("mean")},
        {"metric_category": "Resolution", "metric_name": "Mean Height", "value": height_stats.get("mean")},
        {"metric_category": "Resolution", "metric_name": "Median Resolution", "value": f"{width_stats.get('median')}x{height_stats.get('median')}"},
        {"metric_category": "Mask Area", "metric_name": "Median Mask Ratio", "value": mask_ratio_stats.get("median")},
        {"metric_category": "Mask Area", "metric_name": "Mean Mask Ratio", "value": mask_ratio_stats.get("mean")},
        {"metric_category": "Mask Area", "metric_name": "Min Mask Ratio", "value": mask_ratio_stats.get("min")},
        {"metric_category": "Mask Area", "metric_name": "Max Mask Ratio", "value": mask_ratio_stats.get("max")},
    ]
    for split_name, cnt in split_counts.set_index("split")["count"].to_dict().items():
        stat_rows.append({"metric_category": "Splits", "metric_name": f"Split [{split_name}] Count", "value": cnt})

    if duplicate_summary:
        stat_rows.append({"metric_category": "Duplicates", "metric_name": "Exact Duplicate Pairs", "value": duplicate_summary.get("exact_duplicate_pairs", 0)})
        stat_rows.append({"metric_category": "Duplicates", "metric_name": "Near Duplicate Pairs", "value": duplicate_summary.get("near_duplicate_pairs", 0)})
        stat_rows.append({"metric_category": "Duplicates", "metric_name": "Cross-Split Leakage Pairs", "value": duplicate_summary.get("cross_split_leakage_pairs", 0)})

    dataset_statistics_df = pd.DataFrame(stat_rows)

    return DatasetStatistics(
        summary_dict=summary_dict,
        dataset_statistics_df=dataset_statistics_df,
        host_dist_df=host_counts,
        disease_dist_df=disease_counts,
        host_disease_df=host_disease_ct,
        split_dist_df=split_counts,
        disease_split_df=disease_split_ct
    )
