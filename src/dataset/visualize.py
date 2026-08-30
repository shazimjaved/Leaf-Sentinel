"""Publication-quality matplotlib visualizations and qualitative sample rendering for Leaf Sentinel (Phase 1).

Generates distribution charts, split cross-tabulations, mask ratio histograms,
resolution scatter plots, and 3-panel (Image | Mask | Overlay) qualitative samples.
Strictly uses Matplotlib without Seaborn.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as ticker

logger = logging.getLogger(__name__)

# Premium color palette tokens (Modern Emerald & Tech Slate)
COLOR_PRIMARY = "#1b4d3e"      # Deep Forest Green
COLOR_SECONDARY = "#2e8b57"    # Sea Green
COLOR_ACCENT = "#e65100"       # Vibrant Amber/Orange
COLOR_SLATE = "#37474f"        # Dark Slate
COLOR_LIGHT_BG = "#f8f9fa"     # Off-white background
SPLIT_COLORS = {
    "train": "#2e7d32",        # Green
    "val": "#1565c0",          # Blue
    "validation": "#1565c0",
    "test": "#c62828",         # Crimson
    "unassigned": "#78909c"    # Neutral Grey
}


def _apply_plot_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str):
    """Apply unified, professional styling to a matplotlib axes."""
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cfd8dc")
    ax.spines["bottom"].set_color("#cfd8dc")
    ax.grid(axis="x", linestyle="--", alpha=0.3, color="#90a4ae")
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#90a4ae")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#263238")
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold", labelpad=8, color="#37474f")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold", labelpad=8, color="#37474f")
    ax.tick_params(axis="both", which="major", labelsize=9, colors="#37474f")


def plot_host_distribution(host_df: pd.DataFrame, output_path: Path, dpi: int = 300):
    """Plot host plant species distribution as a clean horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(10, max(5, len(host_df) * 0.4)), dpi=dpi)
    
    df_sorted = host_df.sort_values(by="count", ascending=True)
    hosts = df_sorted["host"].tolist()
    counts = df_sorted["count"].tolist()
    percentages = df_sorted["percentage"].tolist()

    bars = ax.barh(hosts, counts, color=COLOR_SECONDARY, edgecolor="none", height=0.65)
    
    # Add data labels
    max_c = max(counts) if counts else 1
    for bar, c, p in zip(bars, counts, percentages):
        ax.text(
            bar.get_width() + (max_c * 0.015),
            bar.get_y() + bar.get_height() / 2,
            f"{c:,} ({p:.1f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#263238",
            fontweight="semibold"
        )

    ax.set_xlim(0, max_c * 1.18)
    _apply_plot_style(ax, "Plant Host Species Distribution", "Number of Samples", "Host Plant")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved host distribution figure to {output_path}")


def plot_disease_distribution(disease_df: pd.DataFrame, output_path: Path, dpi: int = 300, max_classes: int = 35):
    """Plot disease category distribution as a sorted horizontal bar chart."""
    df_sorted = disease_df.sort_values(by="count", ascending=True)
    if len(df_sorted) > max_classes:
        df_sorted = df_sorted.tail(max_classes)
        title_suffix = f" (Top {max_classes})"
    else:
        title_suffix = ""

    fig, ax = plt.subplots(figsize=(11, max(6, len(df_sorted) * 0.35)), dpi=dpi)
    diseases = df_sorted["disease"].tolist()
    counts = df_sorted["count"].tolist()
    percentages = df_sorted["percentage"].tolist()

    colors = [COLOR_PRIMARY if "healthy" not in d.lower() else "#81c784" for d in diseases]
    bars = ax.barh(diseases, counts, color=colors, height=0.65)

    max_c = max(counts) if counts else 1
    for bar, c, p in zip(bars, counts, percentages):
        ax.text(
            bar.get_width() + (max_c * 0.015),
            bar.get_y() + bar.get_height() / 2,
            f"{c:,} ({p:.1f}%)",
            va="center",
            ha="left",
            fontsize=8,
            color="#263238",
            fontweight="semibold"
        )

    ax.set_xlim(0, max_c * 1.18)
    _apply_plot_style(ax, f"Disease Pathology Distribution{title_suffix}", "Number of Samples", "Disease / Pathology")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved disease distribution figure to {output_path}")


def plot_split_distribution(split_df: pd.DataFrame, output_path: Path, dpi: int = 300):
    """Plot train/val/test split partition sizes."""
    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    
    splits = split_df["split"].tolist()
    counts = split_df["count"].tolist()
    colors = [SPLIT_COLORS.get(s.lower(), "#78909c") for s in splits]

    # Bar chart
    bars = ax_bar.bar(splits, counts, color=colors, width=0.55)
    max_c = max(counts) if counts else 1
    for bar, c in zip(bars, counts):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max_c * 0.015),
            f"{c:,}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold"
        )
    ax_bar.set_ylim(0, max_c * 1.15)
    _apply_plot_style(ax_bar, "Dataset Splits Breakdown", "Split / Partition", "Number of Images")

    # Donut / Pie chart
    wedges, texts, autotexts = ax_pie.pie(
        counts,
        labels=splits,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        pctdistance=0.75
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax_pie.set_title("Split Proportions", fontsize=12, fontweight="bold", pad=12, color="#263238")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved split distribution figure to {output_path}")


def plot_disease_by_split(disease_split_df: pd.DataFrame, output_path: Path, dpi: int = 300, max_diseases: int = 25):
    """Plot disease distribution stratified by split partitions."""
    # Filter out margins
    ct = disease_split_df.drop(index=["Total"], errors="ignore").drop(columns=["Total"], errors="ignore")
    
    if len(ct) > max_diseases:
        top_indices = ct.sum(axis=1).sort_values(ascending=False).head(max_diseases).index
        ct = ct.loc[top_indices]

    ct = ct.sort_values(by=list(ct.columns), ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(6, len(ct) * 0.38)), dpi=dpi)
    
    bottom = np.zeros(len(ct))
    for col in ct.columns:
        c_val = ct[col].values
        color = SPLIT_COLORS.get(str(col).lower(), "#90a4ae")
        ax.barh(ct.index, c_val, left=bottom, label=str(col).capitalize(), color=color, height=0.65)
        bottom += c_val

    _apply_plot_style(ax, "Disease Distribution by Partition Split", "Sample Count", "Disease")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cfd8dc", loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved disease-by-split figure to {output_path}")


def plot_mask_ratio_distribution(validated_df: pd.DataFrame, output_path: Path, dpi: int = 300):
    """Plot lesion/mask affected-area ratio distribution histogram and box plot."""
    valid_masks = validated_df[validated_df["is_valid_mask"] == True]
    ratios = valid_masks["affected_area_ratio"].dropna().values if not valid_masks.empty else np.array([])

    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(10, 7), dpi=dpi, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    if len(ratios) > 0:
        ax_hist.hist(ratios * 100, bins=40, color=COLOR_ACCENT, edgecolor="white", alpha=0.85)
        med = np.median(ratios) * 100
        mean = np.mean(ratios) * 100
        ax_hist.axvline(med, color="#b71c1c", linestyle="--", linewidth=1.5, label=f"Median: {med:.2f}%")
        ax_hist.axvline(mean, color="#0d47a1", linestyle=":", linewidth=1.5, label=f"Mean: {mean:.2f}%")
        ax_hist.legend(frameon=True, facecolor="white")
        
        # Box plot (horizontal)
        try:
            ax_box.boxplot(
                ratios * 100, orientation="horizontal", patch_artist=True,
                boxprops=dict(facecolor="#ffe0b2", color=COLOR_ACCENT),
                medianprops=dict(color="#b71c1c", linewidth=2)
            )
        except TypeError:
            ax_box.boxplot(
                ratios * 100, vert=False, patch_artist=True,
                boxprops=dict(facecolor="#ffe0b2", color=COLOR_ACCENT),
                medianprops=dict(color="#b71c1c", linewidth=2)
            )
    else:
        ax_hist.text(0.5, 0.5, "No valid masks available", ha="center", va="center", transform=ax_hist.transAxes)

    _apply_plot_style(ax_hist, "Lesion Affected-Area Ratio Distribution", "", "Image Count")
    _apply_plot_style(ax_box, "", "Affected Area (% of Total Image Pixels)", "")
    ax_box.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved mask ratio distribution figure to {output_path}")


def plot_resolution_distribution(validated_df: pd.DataFrame, output_path: Path, dpi: int = 300):
    """Plot scatter and histograms of image resolutions."""
    valid_imgs = validated_df[validated_df["is_valid_image"] == True]
    w = valid_imgs["width"].dropna().values if not valid_imgs.empty else np.array([])
    h = valid_imgs["height"].dropna().values if not valid_imgs.empty else np.array([])

    fig, ax = plt.subplots(figsize=(9, 7), dpi=dpi)

    if len(w) > 0 and len(h) > 0:
        # Add subtle jitter for visualization if discrete resolutions
        jitter_w = w + np.random.normal(0, 1.5, size=len(w))
        jitter_h = h + np.random.normal(0, 1.5, size=len(h))
        
        scatter = ax.scatter(jitter_w, jitter_h, alpha=0.45, color=COLOR_PRIMARY, edgecolors="none", s=25)
        
        # Diagonal line for 1:1 aspect ratio
        max_dim = max(w.max(), h.max())
        min_dim = min(w.min(), h.min())
        ax.plot([min_dim, max_dim], [min_dim, max_dim], linestyle="--", color="#90a4ae", alpha=0.7, label="1:1 Aspect Ratio")
        ax.legend(frameon=True)
    else:
        ax.text(0.5, 0.5, "No valid images available", ha="center", va="center", transform=ax.transAxes)

    _apply_plot_style(ax, "Dataset Image Resolution Distribution", "Width (pixels)", "Height (pixels)")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved resolution distribution figure to {output_path}")


def create_sample_card(
    image_path: Path,
    mask_path: Optional[Path],
    output_path: Path,
    title: str,
    subtitle: str,
    overlay_color: Tuple[int, int, int] = (230, 40, 40),
    alpha: float = 0.45,
    dpi: int = 200
):
    """Render a 3-panel qualitative sample card (Original Image | Mask | Alpha Overlay)."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.5), dpi=dpi)
    
    # 1. Original Image
    try:
        with Image.open(image_path) as img:
            img_rgb = np.array(img.convert("RGB"))
    except Exception:
        img_rgb = np.zeros((256, 256, 3), dtype=np.uint8)

    ax1.imshow(img_rgb)
    ax1.set_title("Original Image", fontsize=11, fontweight="bold", pad=8, color="#263238")
    ax1.axis("off")

    # 2. Ground Truth Mask
    mask_binary = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    if mask_path and Path(mask_path).exists():
        try:
            m_arr = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if m_arr is not None:
                if m_arr.ndim == 3:
                    m_arr = cv2.cvtColor(m_arr, cv2.COLOR_BGR2GRAY)
                mask_binary = (m_arr > 0).astype(np.uint8)
                # Resize if minor discrepancy
                if mask_binary.shape[:2] != img_rgb.shape[:2]:
                    mask_binary = cv2.resize(
                        mask_binary, (img_rgb.shape[1], img_rgb.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
        except Exception:
            pass

    ax2.imshow(mask_binary, cmap="gray")
    ax2.set_title("Ground Truth Mask", fontsize=11, fontweight="bold", pad=8, color="#263238")
    ax2.axis("off")

    # 3. Alpha Overlay
    overlay = img_rgb.copy()
    color_mask = np.zeros_like(img_rgb)
    color_mask[mask_binary > 0] = overlay_color
    
    # Blend where mask is positive
    pos_idx = mask_binary > 0
    overlay[pos_idx] = (
        (1 - alpha) * img_rgb[pos_idx] + alpha * np.array(overlay_color)
    ).astype(np.uint8)

    # Optional boundary contour
    try:
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    except Exception:
        pass

    ax3.imshow(overlay)
    ax3.set_title("Disease Overlay", fontsize=11, fontweight="bold", pad=8, color="#263238")
    ax3.axis("off")

    fig.suptitle(f"{title} — {subtitle}", fontsize=12, fontweight="bold", y=0.98, color="#1b4d3e")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_representative_samples(
    validated_df: pd.DataFrame,
    output_dir: Path,
    sample_count: int = 24,
    overlay_color: Tuple[int, int, int] = (230, 40, 40),
    alpha: float = 0.45
) -> List[Path]:
    """Intelligently sample across hosts, diseases, and mask sizes to generate sample cards.
    
    Returns:
        List of generated sample image paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_records = validated_df[
        (validated_df["is_valid_image"] == True) & 
        (validated_df["is_valid_mask"] == True) &
        (validated_df["affected_pixels"] > 0)
    ].copy()

    if valid_records.empty:
        # Fallback to any valid images
        valid_records = validated_df[validated_df["is_valid_image"] == True].copy()

    if valid_records.empty:
        logger.warning("No valid samples found to generate representative sample cards.")
        return []

    # Stratified selection: pick diverse host x disease x mask size
    sampled_indices = []
    
    # 1. Group by host and pick representatives
    hosts = valid_records["host"].unique()
    samples_per_host = max(1, sample_count // len(hosts)) if len(hosts) > 0 else sample_count

    for h in hosts:
        h_group = valid_records[valid_records["host"] == h]
        # Try to sample small, medium, large mask ratio if available
        if "affected_area_ratio" in h_group.columns and len(h_group) >= 3:
            h_sorted = h_group.sort_values(by="affected_area_ratio")
            idx_small = h_sorted.index[0]
            idx_med = h_sorted.index[len(h_sorted) // 2]
            idx_large = h_sorted.index[-1]
            sampled_indices.extend([idx_small, idx_med, idx_large])
        else:
            sampled_indices.extend(h_group.head(samples_per_host).index.tolist())

    # De-duplicate and trim/pad to sample_count
    sampled_indices = list(dict.fromkeys(sampled_indices))
    if len(sampled_indices) > sample_count:
        sampled_indices = sampled_indices[:sample_count]
    elif len(sampled_indices) < sample_count and len(valid_records) > len(sampled_indices):
        remaining = [i for i in valid_records.index if i not in sampled_indices]
        sampled_indices.extend(remaining[:sample_count - len(sampled_indices)])

    generated_paths = []
    for card_idx, idx in enumerate(sampled_indices):
        row = valid_records.loc[idx]
        img_p = Path(row["image_path"])
        mask_p = Path(row["mask_path"]) if pd.notna(row.get("mask_path")) else None
        
        host_name = str(row.get("host", "Unknown")).replace(" ", "_")
        disease_name = str(row.get("disease", "Unknown")).replace(" ", "_")
        ratio = float(row.get("affected_area_ratio", 0.0)) * 100

        out_name = f"sample_{card_idx+1:02d}_{host_name}_{disease_name}.png"
        out_file = output_dir / out_name

        create_sample_card(
            image_path=img_p,
            mask_path=mask_p,
            output_path=out_file,
            title=f"Sample #{card_idx+1}: {row.get('host')} — {row.get('disease')}",
            subtitle=f"Split: {row.get('split', 'unassigned')} | Lesion Area: {ratio:.1f}%",
            overlay_color=overlay_color,
            alpha=alpha
        )
        generated_paths.append(out_file)

    logger.info(f"Generated {len(generated_paths)} qualitative sample cards in {output_dir}")
    return generated_paths


def generate_all_visualizations(
    stats: Any,
    validated_df: pd.DataFrame,
    figures_dir: Path,
    samples_dir: Path,
    sample_count: int = 24,
    dpi: int = 300,
    overlay_color: Tuple[int, int, int] = (230, 40, 40),
    alpha: float = 0.45
):
    """Generate the full suite of Phase 1 publication figures and qualitative sample cards."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating publication figures...")

    # 1. Host distribution
    if not stats.host_dist_df.empty:
        plot_host_distribution(stats.host_dist_df, figures_dir / "host_distribution.png", dpi=dpi)

    # 2. Disease distribution
    if not stats.disease_dist_df.empty:
        plot_disease_distribution(stats.disease_dist_df, figures_dir / "disease_distribution.png", dpi=dpi)

    # 3. Split distribution
    if not stats.split_dist_df.empty:
        plot_split_distribution(stats.split_dist_df, figures_dir / "split_distribution.png", dpi=dpi)

    # 4. Disease by split
    if not stats.disease_split_df.empty:
        plot_disease_by_split(stats.disease_split_df, figures_dir / "disease_by_split.png", dpi=dpi)

    # 5. Mask ratio distribution
    plot_mask_ratio_distribution(validated_df, figures_dir / "mask_ratio_distribution.png", dpi=dpi)

    # 6. Resolution distribution
    plot_resolution_distribution(validated_df, figures_dir / "resolution_distribution.png", dpi=dpi)

    # 7. Qualitative sample cards
    generate_representative_samples(
        validated_df=validated_df,
        output_dir=samples_dir,
        sample_count=sample_count,
        overlay_color=overlay_color,
        alpha=alpha
    )
    logger.info("All visual assets successfully generated.")
