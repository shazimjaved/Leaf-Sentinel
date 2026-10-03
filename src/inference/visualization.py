"""Phase 5 — Diagnostic visualization for LeafSentinel inference results.

Generates a multi-panel diagnostic card with:
  1. Original image
  2. Predicted lesion mask
  3. Lesion overlay on original image
  4. Text annotations with classification and burden results
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

from src.inference.schemas import InferenceResult

logger = logging.getLogger("LeafSentinel.Phase5.Visualization")


def _create_overlay(
    image: np.ndarray, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.4
) -> np.ndarray:
    """Overlay a binary mask on an image with transparency.

    Args:
        image: RGB image as numpy array (H, W, 3).
        mask: Binary mask as numpy array (H, W) with values 0 or 255.
        color: RGB color for the overlay.
        alpha: Transparency of the overlay.

    Returns:
        Overlaid image as numpy array (H, W, 3).
    """
    overlay = image.copy()
    mask_bool = mask > 127
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            (1 - alpha) * image[:, :, c] + alpha * color[c],
            image[:, :, c],
        )
    return overlay.astype(np.uint8)


def save_mask(mask: np.ndarray, output_path: str) -> None:
    """Save binary mask as PNG at original image dimensions."""
    mask_img = Image.fromarray(mask)
    mask_img.save(output_path)
    logger.info(f"Saved mask to {output_path}")


def save_overlay(
    image: Image.Image, mask: np.ndarray, output_path: str
) -> None:
    """Save lesion overlay image at original image dimensions."""
    img_array = np.array(image)
    overlay = _create_overlay(img_array, mask)
    overlay_img = Image.fromarray(overlay)
    overlay_img.save(output_path)
    logger.info(f"Saved overlay to {output_path}")


def generate_diagnostic_card(
    image: Image.Image,
    mask: np.ndarray,
    result: InferenceResult,
    output_path: str,
    probability_map: Optional[np.ndarray] = None,
) -> None:
    """Generate a multi-panel diagnostic visualization card.

    Panels:
      1. Original image
      2. Predicted lesion mask
      3. Lesion overlay
    Plus text annotations with classification and burden results.

    Args:
        image: Original PIL Image (RGB).
        mask: Binary mask at original image dimensions (H, W), values 0/255.
        result: InferenceResult from the pipeline.
        output_path: Path to save the diagnostic card PNG.
        probability_map: Optional probability map for a 4th panel.
    """
    img_array = np.array(image)
    overlay = _create_overlay(img_array, mask)

    n_panels = 4 if probability_map is not None else 3

    fig = plt.figure(figsize=(6 * n_panels + 4, 6), facecolor="white")
    gs = GridSpec(1, n_panels + 1, width_ratios=[1] * n_panels + [1.2], figure=fig)

    # Panel 1: Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_array)
    ax1.set_title("Original Image", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # Panel 2: Mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask, cmap="gray", vmin=0, vmax=255)
    ax2.set_title("Predicted Lesion Mask", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # Panel 3: Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay)
    ax3.set_title("Lesion Overlay", fontsize=12, fontweight="bold")
    ax3.axis("off")

    # Panel 4 (optional): Probability map
    if probability_map is not None:
        ax4 = fig.add_subplot(gs[0, 3])
        im = ax4.imshow(probability_map, cmap="hot", vmin=0, vmax=1)
        ax4.set_title("Lesion Probability", fontsize=12, fontweight="bold")
        ax4.axis("off")
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # Text panel
    ax_text = fig.add_subplot(gs[0, n_panels])
    ax_text.axis("off")

    clf = result.classification
    seg = result.segmentation

    lines = [
        ("LeafSentinel Diagnostic Report", 16, "bold"),
        ("", 10, "normal"),
        ("— Classification —", 13, "bold"),
        (f"Predicted disease: {clf.predicted_class}", 11, "normal"),
        (f"Confidence: {clf.confidence:.4f}", 11, "normal"),
        (f"2nd candidate: {clf.top2[1].class_name}", 11, "normal"),
        (f"2nd confidence: {clf.top2[1].confidence:.4f}", 11, "normal"),
        (f"Top1–Top2 margin: {clf.top1_top2_margin:.4f}", 11, "normal"),
        (f"Normalized entropy: {clf.normalized_entropy:.4f}", 11, "normal"),
        ("", 10, "normal"),
        ("— Segmentation —", 13, "bold"),
        (f"Image-relative lesion burden: {seg.burden_percent:.2f}%", 11, "normal"),
        (f"Lesion pixels: {seg.lesion_pixels:,} / {seg.total_pixels:,}", 11, "normal"),
        (f"Threshold: {seg.threshold}", 11, "normal"),
        (f"Grid: {seg.measurement_grid[0]}×{seg.measurement_grid[1]}", 11, "normal"),
        ("", 8, "normal"),
        ("Burden is lesion area relative to image area,", 9, "italic"),
        ("not total leaf area.", 9, "italic"),
    ]

    # Add warnings
    if result.system.warnings:
        lines.append(("", 8, "normal"))
        lines.append(("— Warnings —", 13, "bold"))
        for w in result.system.warnings:
            lines.append((f"⚠ {w}", 10, "normal"))

    # Add direct regression diagnostic if present
    if result.direct_regression is not None:
        dr = result.direct_regression
        lines.append(("", 8, "normal"))
        lines.append(("— Direct Regression (diagnostic) —", 13, "bold"))
        lines.append((f"Direct regression burden: {dr.direct_regression_burden:.4f}", 11, "normal"))
        lines.append((f"Segmentation burden: {dr.segmentation_burden:.4f}", 11, "normal"))
        lines.append((f"|Disagreement|: {dr.absolute_disagreement:.4f}", 11, "normal"))

    y = 0.98
    for text, fontsize, weight in lines:
        fontstyle = "italic" if weight == "italic" else "normal"
        actual_weight = "normal" if weight == "italic" else weight
        
        ax_text.text(
            0.05, y, text,
            transform=ax_text.transAxes,
            fontsize=fontsize,
            fontweight=actual_weight,
            fontstyle=fontstyle,
            verticalalignment="top",
            fontfamily="sans-serif",
        )
        y -= 0.05 if fontsize >= 11 else 0.035

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved diagnostic card to {output_path}")
