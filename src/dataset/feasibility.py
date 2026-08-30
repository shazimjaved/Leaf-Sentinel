"""Class feasibility and candidate scoping analysis for Leaf Sentinel (Phase 1).

Evaluates each host-disease combination across sample volume, mask integrity,
split availability, and lesion area distributions to assign evidence-based
feasibility tiers for future model training phases.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _assign_feasibility_tier(
    total_samples: int,
    valid_masks: int,
    train_samples: int,
    val_samples: int,
    test_samples: int,
    strong_min_samples: int = 100,
    strong_min_masks: int = 80,
    usable_min_samples: int = 40,
    usable_min_masks: int = 30,
    limited_min_samples: int = 15,
    limited_min_masks: int = 10
) -> Tuple[str, str]:
    """Assign feasibility tier and provide rationale."""
    has_splits = (train_samples > 0 and (val_samples > 0 or test_samples > 0)) or (train_samples == 0 and total_samples >= strong_min_samples)
    
    if total_samples >= strong_min_samples and valid_masks >= strong_min_masks:
        return "Strong candidate", "High sample volume, rich mask coverage, suitable for robust segmentation & classification."
    elif total_samples >= usable_min_samples and valid_masks >= usable_min_masks:
        return "Usable", "Adequate sample count and mask annotations for initial baseline training and transfer learning."
    elif total_samples >= limited_min_samples and valid_masks >= limited_min_masks:
        return "Limited", "Modest sample volume; may require heavy augmentation or few-shot techniques."
    else:
        reasons = []
        if total_samples < limited_min_samples:
            reasons.append(f"low sample count ({total_samples})")
        if valid_masks < limited_min_masks:
            reasons.append(f"low valid masks ({valid_masks})")
        reason_str = ", ".join(reasons) if reasons else "insufficient data"
        return "Insufficient", f"Restricted data volume ({reason_str}); prioritize for subsequent collection phases."


def analyze_class_feasibility(
    validated_df: pd.DataFrame,
    strong_min_samples: int = 100,
    strong_min_masks: int = 80,
    usable_min_samples: int = 40,
    usable_min_masks: int = 30,
    limited_min_samples: int = 15,
    limited_min_masks: int = 10
) -> pd.DataFrame:
    """Analyze feasibility of host-disease classes and generate ranked report.
    
    Args:
        validated_df: DataFrame with validation metrics.
        strong_min_samples: Sample threshold for 'Strong candidate'.
        strong_min_masks: Mask threshold for 'Strong candidate'.
        usable_min_samples: Sample threshold for 'Usable'.
        usable_min_masks: Mask threshold for 'Usable'.
        limited_min_samples: Sample threshold for 'Limited'.
        limited_min_masks: Mask threshold for 'Limited'.
        
    Returns:
        pd.DataFrame sorted by feasibility tier and sample volume.
    """
    logger.info("Computing class feasibility assessment across host x disease pairs...")

    if validated_df.empty or "host" not in validated_df.columns or "disease" not in validated_df.columns:
        logger.warning("No data available for class feasibility analysis.")
        return pd.DataFrame(columns=[
            "host", "disease", "feasibility_tier", "total_samples", "valid_images",
            "valid_masks", "mask_validity_rate_pct", "train_samples", "val_samples",
            "test_samples", "unassigned_samples", "median_mask_ratio", "min_mask_ratio",
            "max_mask_ratio", "rationale"
        ])

    rows: List[Dict[str, Any]] = []
    grouped = validated_df.groupby(["host", "disease"])

    for (host, disease), group in grouped:
        total_samples = len(group)
        val_img_df = group[group["is_valid_image"] == True] if "is_valid_image" in group.columns else group
        val_mask_df = group[group["is_valid_mask"] == True] if "is_valid_mask" in group.columns else pd.DataFrame()
        
        valid_img_cnt = len(val_img_df)
        valid_mask_cnt = len(val_mask_df)
        mask_rate = round((valid_mask_cnt / total_samples * 100), 2) if total_samples > 0 else 0.0

        split_counts = group["split"].value_counts().to_dict() if "split" in group.columns else {}
        train_cnt = split_counts.get("train", 0)
        val_cnt = split_counts.get("val", split_counts.get("validation", 0))
        test_cnt = split_counts.get("test", 0)
        unassigned_cnt = split_counts.get("unassigned", 0)

        mask_ratios = val_mask_df["affected_area_ratio"].dropna() if "affected_area_ratio" in val_mask_df.columns else pd.Series()
        med_ratio = float(round(mask_ratios.median(), 4)) if len(mask_ratios) > 0 else 0.0
        min_ratio = float(round(mask_ratios.min(), 4)) if len(mask_ratios) > 0 else 0.0
        max_ratio = float(round(mask_ratios.max(), 4)) if len(mask_ratios) > 0 else 0.0

        tier, rationale = _assign_feasibility_tier(
            total_samples=total_samples,
            valid_masks=valid_mask_cnt,
            train_samples=train_cnt,
            val_samples=val_cnt,
            test_samples=test_cnt,
            strong_min_samples=strong_min_samples,
            strong_min_masks=strong_min_masks,
            usable_min_samples=usable_min_samples,
            usable_min_masks=usable_min_masks,
            limited_min_samples=limited_min_samples,
            limited_min_masks=limited_min_masks
        )

        rows.append({
            "host": host,
            "disease": disease,
            "feasibility_tier": tier,
            "total_samples": total_samples,
            "valid_images": valid_img_cnt,
            "valid_masks": valid_mask_cnt,
            "mask_validity_rate_pct": mask_rate,
            "train_samples": train_cnt,
            "val_samples": val_cnt,
            "test_samples": test_cnt,
            "unassigned_samples": unassigned_cnt,
            "median_mask_ratio": med_ratio,
            "min_mask_ratio": min_ratio,
            "max_mask_ratio": max_ratio,
            "rationale": rationale
        })

    feasibility_df = pd.DataFrame(rows)

    # Sort by tier priority and sample count
    tier_order = {"Strong candidate": 0, "Usable": 1, "Limited": 2, "Insufficient": 3}
    feasibility_df["tier_rank"] = feasibility_df["feasibility_tier"].map(tier_order)
    feasibility_df = feasibility_df.sort_values(
        by=["tier_rank", "total_samples", "valid_masks"],
        ascending=[True, False, False]
    ).drop(columns=["tier_rank"]).reset_index(drop=True)

    logger.info(
        f"Feasibility evaluation: "
        f"{(feasibility_df['feasibility_tier'] == 'Strong candidate').sum()} Strong, "
        f"{(feasibility_df['feasibility_tier'] == 'Usable').sum()} Usable, "
        f"{(feasibility_df['feasibility_tier'] == 'Limited').sum()} Limited, "
        f"{(feasibility_df['feasibility_tier'] == 'Insufficient').sum()} Insufficient."
    )

    return feasibility_df
