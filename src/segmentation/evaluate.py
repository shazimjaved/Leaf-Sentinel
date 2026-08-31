"""Test set evaluation and 5-panel qualitative prediction auditing for LeafSentinel (Phase 2).

Evaluates the trained lesion segmentation model exclusively on the unseen Phase 2 test split,
computing aggregate, per-class, and healthy control metrics, and rendering comprehensive
5-panel diagnostic prediction cards.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.segmentation.model import ResNetUNet, build_segmentation_model
from src.segmentation.metrics import SegmentationMetrics
from src.segmentation.train import get_device

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized PyTorch image tensor (3, H, W) to RGB uint8 (H, W, 3)."""
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    arr = (arr * IMAGENET_STD) + IMAGENET_MEAN
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def create_prediction_card(
    img_rgb: np.ndarray,
    gt_mask: np.ndarray,
    pred_prob: np.ndarray,
    output_path: Path,
    title: str,
    subtitle: str,
    threshold: float = 0.5,
    gt_color: Tuple[int, int, int] = (46, 125, 50),     # Forest Green
    pred_color: Tuple[int, int, int] = (230, 40, 40),   # Vibrant Coral Red
    alpha: float = 0.45,
    dpi: int = 200
):
    """Render a 5-panel diagnostic card: Original | Ground Truth | Prediction | GT Overlay | Pred Overlay."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), dpi=dpi)
    
    pred_bin = (pred_prob >= threshold).astype(np.uint8)
    gt_bin = (gt_mask >= 0.5).astype(np.uint8)

    # 1. Original Image
    axes[0].imshow(img_rgb)
    axes[0].set_title("1. Original Image", fontsize=10, fontweight="bold", pad=6)
    axes[0].axis("off")

    # 2. Ground Truth Mask
    axes[1].imshow(gt_bin, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("2. Ground Truth Mask", fontsize=10, fontweight="bold", pad=6)
    axes[1].axis("off")

    # 3. Predicted Probability Map
    im3 = axes[2].imshow(pred_prob, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("3. Predicted Heatmap", fontsize=10, fontweight="bold", pad=6)
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Ground Truth Overlay
    gt_overlay = img_rgb.copy()
    gt_pos = gt_bin > 0
    gt_overlay[gt_pos] = ((1 - alpha) * img_rgb[gt_pos] + alpha * np.array(gt_color)).astype(np.uint8)
    axes[3].imshow(gt_overlay)
    axes[3].set_title("4. Ground Truth Overlay", fontsize=10, fontweight="bold", pad=6)
    axes[3].axis("off")

    # 5. Prediction Binary Overlay
    pred_overlay = img_rgb.copy()
    pred_pos = pred_bin > 0
    pred_overlay[pred_pos] = ((1 - alpha) * img_rgb[pred_pos] + alpha * np.array(pred_color)).astype(np.uint8)
    axes[4].imshow(pred_overlay)
    axes[4].set_title(f"5. Prediction Overlay (τ={threshold:.2f})", fontsize=10, fontweight="bold", pad=6)
    axes[4].axis("off")

    fig.suptitle(f"{title} | {subtitle}", fontsize=11, fontweight="bold", y=0.98, color="#1b4d3e")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_segmentation_model(
    config: Dict[str, Any],
    checkpoint_path: Path | str,
    test_loader: DataLoader,
    output_dir: Optional[Path | str] = None,
    num_sample_cards: int = 24
) -> Dict[str, Any]:
    """Run full test split evaluation, metric accumulation, and qualitative failure auditing."""
    if output_dir is None:
        eval_root = Path(config.get("paths", {}).get("evaluation_dir", "outputs/phase2/evaluation"))
    else:
        eval_root = Path(output_dir)

    eval_root.mkdir(parents=True, exist_ok=True)
    preds_dir = eval_root / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    checkpoint_p = Path(checkpoint_path)
    
    if not checkpoint_p.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_p}")

    logger.info(f"Loading model checkpoint from {checkpoint_p}...")
    checkpoint = torch.load(checkpoint_p, map_location=device)

    model = build_segmentation_model(config).to(device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    threshold = float(config.get("evaluation", {}).get("threshold", 0.5))
    metrics_tracker = SegmentationMetrics()
    cached_predictions: List[Dict[str, Any]] = []

    logger.info(f"Evaluating {len(test_loader.dataset)} test samples (threshold={threshold})...")

    with torch.no_grad():
        for images, masks, metadata_batch in tqdm(test_loader, desc="Evaluating Test Set"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B, 1, H, W)
            masks_np = masks.numpy()                      # (B, 1, H, W)

            batch_size = images.size(0)
            for b in range(batch_size):
                # Extract per-sample metadata
                sample_meta = {
                    key: metadata_batch[key][b] if isinstance(metadata_batch[key], list) or isinstance(metadata_batch[key], torch.Tensor) else metadata_batch[key]
                    for key in metadata_batch
                }

                p_prob = probs[b, 0]
                g_mask = masks_np[b, 0]
                img_raw = denormalize_image(images[b])

                metrics_tracker.add_sample(
                    pred_mask=p_prob,
                    gt_mask=g_mask,
                    metadata=sample_meta,
                    threshold=threshold
                )

                cached_predictions.append({
                    "image_rgb": img_raw,
                    "gt_mask": g_mask,
                    "pred_prob": p_prob,
                    "metadata": sample_meta,
                    "metric_res": metrics_tracker.sample_results[-1]
                })

    # Summary Metrics
    summary = metrics_tracker.get_summary()
    per_class_df = metrics_tracker.get_per_class_dataframe()
    samples_df = metrics_tracker.to_dataframe()

    # Save CSVs & JSONs
    metrics_json_path = eval_root / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    per_class_csv_path = eval_root / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv_path, index=False)

    samples_csv_path = eval_root / "metrics.csv"
    samples_df.to_csv(samples_csv_path, index=False)

    logger.info(f"Saved evaluation metrics to {metrics_json_path} and {per_class_csv_path}")

    # Generate 5-Panel Qualitative Prediction Cards
    logger.info(f"Rendering {num_sample_cards} qualitative diagnostic cards across performance categories...")
    
    # Sort samples by Dice to pick best, moderate, worst, and healthy
    diseased_preds = [cp for cp in cached_predictions if not cp["metadata"].get("is_healthy", False)]
    healthy_preds = [cp for cp in cached_predictions if cp["metadata"].get("is_healthy", False)]
    
    diseased_preds_sorted = sorted(diseased_preds, key=lambda cp: cp["metric_res"].dice)

    selected_cards = []
    # Best cases (top 6)
    if len(diseased_preds_sorted) >= 6:
        selected_cards.extend([("best", cp) for cp in diseased_preds_sorted[-6:]])
    # Moderate cases (middle 6)
    if len(diseased_preds_sorted) >= 12:
        mid_start = len(diseased_preds_sorted) // 2 - 3
        selected_cards.extend([("moderate", cp) for cp in diseased_preds_sorted[mid_start:mid_start+6]])
    # Worst / Failure cases (bottom 6)
    if len(diseased_preds_sorted) >= 6:
        selected_cards.extend([("failure", cp) for cp in diseased_preds_sorted[:6]])
    # Healthy cases (up to 6)
    if healthy_preds:
        selected_cards.extend([("healthy", cp) for cp in healthy_preds[:min(6, len(healthy_preds))]])

    for card_idx, (cat_label, cp) in enumerate(selected_cards, start=1):
        m = cp["metadata"]
        res = cp["metric_res"]
        host_str = m.get("host", "Unknown").replace(" ", "_")
        disease_str = m.get("disease", "Unknown").replace(" ", "_")
        
        card_name = f"pred_{card_idx:02d}_{cat_label}_{host_str}_{disease_str}.png"
        card_out = preds_dir / card_name

        sub_info = f"Dice: {res.dice:.4f} | IoU: {res.iou:.4f} | Prec: {res.precision:.4f} | Rec: {res.recall:.4f}"
        if res.is_healthy:
            sub_info = f"Healthy Control | FP Area Ratio: {res.false_positive_area_ratio*100:.2f}%"

        create_prediction_card(
            img_rgb=cp["image_rgb"],
            gt_mask=cp["gt_mask"],
            pred_prob=cp["pred_prob"],
            output_path=card_out,
            title=f"Sample #{card_idx:02d} [{cat_label.upper()}]: {m.get('display_class')}",
            subtitle=sub_info,
            threshold=threshold
        )

    logger.info(f"Generated {len(selected_cards)} qualitative prediction cards in {preds_dir}")

    logger.info("================================================================================")
    logger.info(" LeafSentinel — Phase 2 Test Evaluation Results                                 ")
    logger.info("================================================================================")
    logger.info(f" • Test Samples Evaluated   : {summary['total_samples']} (Diseased: {summary['diseased_samples']}, Healthy: {summary['healthy_samples']})")
    logger.info(f" • Overall Mean Dice        : {summary['overall']['mean_dice']:.4f} (Median: {summary['overall']['median_dice']:.4f})")
    logger.info(f" • Overall Mean IoU         : {summary['overall']['mean_iou']:.4f} (Median: {summary['overall']['median_iou']:.4f})")
    logger.info(f" • Overall Mean Precision   : {summary['overall']['mean_precision']:.4f}")
    logger.info(f" • Overall Mean Recall      : {summary['overall']['mean_recall']:.4f}")
    if summary['healthy_samples'] > 0:
        logger.info(f" • Healthy FP Area Ratio    : {summary['healthy_only']['mean_false_positive_area_ratio']*100:.2f}% (Clean Leaf Rate: {summary['healthy_only']['perfect_clean_leaf_rate']*100:.1f}%)")
    logger.info("================================================================================")

    return summary
