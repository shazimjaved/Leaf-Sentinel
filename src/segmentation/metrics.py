"""Numerically stable segmentation evaluation metrics for LeafSentinel (Phase 2).

Implements:
- Dice Coefficient, IoU (Jaccard), Precision, Recall, Pixel Accuracy.
- Specialized evaluation for diseased vs. healthy control samples.
- False-Positive Area Ratio for healthy control audit.
- Per-class and aggregate metric accumulation.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

EPSILON = 1e-7


@dataclass
class SampleMetricResult:
    """Stores evaluation metrics for an individual test sample."""
    image_id: str
    host: str
    disease: str
    display_class: str
    is_healthy: bool
    dice: float
    iou: float
    precision: float
    recall: float
    pixel_accuracy: float
    gt_lesion_ratio: float
    pred_lesion_ratio: float
    false_positive_area_ratio: float  # Only meaningful when is_healthy is True or GT is empty


@dataclass
class SegmentationMetrics:
    """Accumulates and summarizes segmentation evaluation metrics across a dataset."""
    sample_results: List[SampleMetricResult] = field(default_factory=list)

    def add_sample(
        self,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
        metadata: Dict[str, Any],
        threshold: float = 0.5
    ):
        """Compute and record metrics for a single sample."""
        pred_bin = (pred_mask >= threshold).astype(np.uint8)
        gt_bin = (gt_mask >= 0.5).astype(np.uint8)
        total_px = pred_bin.size

        # Pixel counts
        intersection = np.sum(pred_bin * gt_bin)
        pred_sum = np.sum(pred_bin)
        gt_sum = np.sum(gt_bin)
        union = pred_sum + gt_sum - intersection

        tp = intersection
        fp = pred_sum - intersection
        fn = gt_sum - intersection
        tn = total_px - (tp + fp + fn)

        is_healthy = bool(metadata.get("is_healthy", False)) or (gt_sum == 0)

        # Handling metrics
        if gt_sum == 0:
            # Healthy or empty ground truth
            # Perfect prediction is pred_sum == 0
            if pred_sum == 0:
                dice = 1.0
                iou = 1.0
                precision = 1.0
                recall = 1.0
            else:
                dice = 0.0
                iou = 0.0
                precision = 0.0
                recall = 1.0  # All (zero) true lesions retrieved
            fp_area_ratio = float(pred_sum / total_px)
        else:
            dice = float((2.0 * intersection + EPSILON) / (pred_sum + gt_sum + EPSILON))
            iou = float((intersection + EPSILON) / (union + EPSILON))
            precision = float((intersection + EPSILON) / (pred_sum + EPSILON))
            recall = float((intersection + EPSILON) / (gt_sum + EPSILON))
            fp_area_ratio = float(fp / total_px)

        pixel_acc = float((tp + tn) / total_px) if total_px > 0 else 0.0
        gt_ratio = float(gt_sum / total_px) if total_px > 0 else 0.0
        pred_ratio = float(pred_sum / total_px) if total_px > 0 else 0.0

        res = SampleMetricResult(
            image_id=str(metadata.get("image_id", "unknown")),
            host=str(metadata.get("host", "unknown")),
            disease=str(metadata.get("disease", "unknown")),
            display_class=str(metadata.get("display_class", metadata.get("disease", "unknown"))),
            is_healthy=is_healthy,
            dice=dice,
            iou=iou,
            precision=precision,
            recall=recall,
            pixel_accuracy=pixel_acc,
            gt_lesion_ratio=gt_ratio,
            pred_lesion_ratio=pred_ratio,
            false_positive_area_ratio=fp_area_ratio
        )
        self.sample_results.append(res)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert sample results to a DataFrame."""
        return pd.DataFrame([r.__dict__ for r in self.sample_results])

    def get_summary(self) -> Dict[str, Any]:
        """Compute aggregate, diseased-only, healthy-only, and per-class metrics."""
        df = self.to_dataframe()
        if df.empty:
            return {"total_samples": 0}

        diseased_df = df[~df["is_healthy"]]
        healthy_df = df[df["is_healthy"]]

        summary = {
            "total_samples": len(df),
            "diseased_samples": len(diseased_df),
            "healthy_samples": len(healthy_df),
            "overall": {
                "mean_dice": float(df["dice"].mean()),
                "median_dice": float(df["dice"].median()),
                "mean_iou": float(df["iou"].mean()),
                "median_iou": float(df["iou"].median()),
                "mean_precision": float(df["precision"].mean()),
                "mean_recall": float(df["recall"].mean()),
                "mean_pixel_accuracy": float(df["pixel_accuracy"].mean()),
            },
            "diseased_only": {
                "mean_dice": float(diseased_df["dice"].mean()) if not diseased_df.empty else 0.0,
                "median_dice": float(diseased_df["dice"].median()) if not diseased_df.empty else 0.0,
                "mean_iou": float(diseased_df["iou"].mean()) if not diseased_df.empty else 0.0,
                "median_iou": float(diseased_df["iou"].median()) if not diseased_df.empty else 0.0,
                "mean_precision": float(diseased_df["precision"].mean()) if not diseased_df.empty else 0.0,
                "mean_recall": float(diseased_df["recall"].mean()) if not diseased_df.empty else 0.0,
                "mean_pixel_accuracy": float(diseased_df["pixel_accuracy"].mean()) if not diseased_df.empty else 0.0,
            },
            "healthy_only": {
                "mean_false_positive_area_ratio": float(healthy_df["false_positive_area_ratio"].mean()) if not healthy_df.empty else 0.0,
                "max_false_positive_area_ratio": float(healthy_df["false_positive_area_ratio"].max()) if not healthy_df.empty else 0.0,
                "perfect_clean_leaf_rate": float((healthy_df["pred_lesion_ratio"] == 0).mean()) if not healthy_df.empty else 1.0,
            }
        }
        return summary

    def get_per_class_dataframe(self) -> pd.DataFrame:
        """Compute per-class summary metrics sorted by sample count."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        records = []
        for display_name, grp in df.groupby("display_class"):
            host = grp["host"].iloc[0]
            disease = grp["disease"].iloc[0]
            is_h = bool(grp["is_healthy"].all())
            
            records.append({
                "display_class": display_name,
                "host": host,
                "disease": disease,
                "is_healthy": is_h,
                "n_samples": len(grp),
                "mean_dice": float(round(grp["dice"].mean(), 4)),
                "median_dice": float(round(grp["dice"].median(), 4)),
                "mean_iou": float(round(grp["iou"].mean(), 4)),
                "median_iou": float(round(grp["iou"].median(), 4)),
                "mean_precision": float(round(grp["precision"].mean(), 4)),
                "mean_recall": float(round(grp["recall"].mean(), 4)),
                "mean_gt_lesion_area": float(round(grp["gt_lesion_ratio"].mean(), 4)),
                "mean_pred_lesion_area": float(round(grp["pred_lesion_ratio"].mean(), 4))
            })

        res_df = pd.DataFrame(records).sort_values(by=["mean_dice", "n_samples"], ascending=[False, False]).reset_index(drop=True)
        return res_df


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute fast batch-averaged Dice and IoU for monitoring during training."""
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets >= 0.5).float()

    # Flatten spatial dims: (B, -1)
    preds_flat = preds.view(preds.shape[0], -1)
    targets_flat = targets.view(targets.shape[0], -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    pred_sum = preds_flat.sum(dim=1)
    target_sum = targets_flat.sum(dim=1)
    union = pred_sum + target_sum - intersection

    dice_batch = (2.0 * intersection + EPSILON) / (pred_sum + target_sum + EPSILON)
    iou_batch = (intersection + EPSILON) / (union + EPSILON)

    return {
        "dice": float(dice_batch.mean().item()),
        "iou": float(iou_batch.mean().item())
    }
