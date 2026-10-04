"""Generate portfolio-ready LeafSentinel figures from tracked benchmark results.

Run from the repository root:
    python scripts/generate_project_figures.py

Outputs are written to:
    docs/assets/

Inputs:
    results/phase5/metrics.json
    results/phase5/predictions.csv
    results/phase5/per_class_metrics.csv

No model checkpoints or raw PlantSeg images are required.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase5"
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

METRICS_PATH = RESULTS / "metrics.json"
PREDICTIONS_PATH = RESULTS / "predictions.csv"
PER_CLASS_PATH = RESULTS / "per_class_metrics.csv"

for path in (METRICS_PATH, PREDICTIONS_PATH, PER_CLASS_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")

with METRICS_PATH.open("r", encoding="utf-8") as f:
    metrics = json.load(f)

pred = pd.read_csv(PREDICTIONS_PATH)
per_class = pd.read_csv(PER_CLASS_PATH)

PRIMARY = "#1F6F5F"
SECONDARY = "#446A78"
ACCENT = "#C98243"
DARK = "#1F2933"
MUTED = "#667085"
LIGHT = "#F4F6F8"
GRID = "#D0D5DD"


def save(fig, name):
    path = ASSETS / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path.relative_to(ROOT)}")


def add_box(ax, xy, width, height, title, subtitle="", facecolor=LIGHT, edgecolor=PRIMARY):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.8, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height * 0.63, title,
            ha="center", va="center", fontsize=12, fontweight="bold", color=DARK)
    if subtitle:
        ax.text(x + width / 2, y + height * 0.32, subtitle,
                ha="center", va="center", fontsize=9.5, color=MUTED, wrap=True)
    return box


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=16,
        linewidth=1.6, color=MUTED, connectionstyle="arc3,rad=0.0"
    ))


# 1. Pipeline overview
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.5, 0.95, "LeafSentinel — Phase 5 Integrated Inference",
        ha="center", va="center", fontsize=20, fontweight="bold", color=DARK)
ax.text(0.5, 0.90,
        "What disease?  •  Where are the lesions?  •  How much of the image is affected?",
        ha="center", va="center", fontsize=11, color=MUTED)

add_box(ax, (0.05, 0.40), 0.17, 0.20, "RGB Leaf Image", "Single input image", "#FFFFFF", DARK)
add_box(ax, (0.31, 0.62), 0.24, 0.20, "EfficientNet-B0",
        "Phase 3 disease classifier\n224×224 • PadToSquare • ImageNet normalization",
        "#F3F8F6", PRIMARY)
add_box(ax, (0.31, 0.23), 0.24, 0.20, "ResNet18-U-Net",
        "Phase 2 lesion segmenter\n512×512 • threshold = 0.5",
        "#F6F8FA", SECONDARY)
add_box(ax, (0.65, 0.62), 0.29, 0.20, "Disease Prediction",
        "Top-1 / Top-2 • confidence • margin • entropy", "#F3F8F6", PRIMARY)
add_box(ax, (0.65, 0.23), 0.29, 0.20, "Lesion Mask + Burden",
        "Predicted lesion pixels / total 512×512 pixels", "#FFF8F1", ACCENT)
arrow(ax, (0.22, 0.53), (0.31, 0.72))
arrow(ax, (0.22, 0.47), (0.31, 0.33))
arrow(ax, (0.55, 0.72), (0.65, 0.72))
arrow(ax, (0.55, 0.33), (0.65, 0.33))
ax.text(0.5, 0.08,
        "Healthy-vs-diseased classification is not validated. Burden is image-relative, not true whole-leaf disease severity.",
        ha="center", va="center", fontsize=10, color=MUTED)
save(fig, "pipeline_overview.png")


# 2. Phase 5 benchmark summary
classification = metrics["classification"]
segmentation = metrics["segmentation"]
burden = metrics["image_relative_lesion_burden"]

cards = [
    ("Accuracy", classification["accuracy"] * 100, "%"),
    ("Macro F1", classification["macro_f1"] * 100, "%"),
    ("Top-2 Accuracy", classification["top2_accuracy"] * 100, "%"),
    ("Mean Dice", segmentation["mean_dice"], ""),
    ("Burden MAE", burden["mae_percentage_points"], " pp"),
    ("Burden R²", burden["r2"], ""),
]

fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
fig.suptitle("Phase 5 Benchmark — 194 Diseased Held-Out Images",
             fontsize=18, fontweight="bold", color=DARK, y=0.98)
fig.text(0.5, 0.925,
         "10 disease classes • final recovered Phase 3 classifier + Phase 2 segmenter",
         ha="center", fontsize=10.5, color=MUTED)

for ax, (label, value, suffix) in zip(axes.flat, cards):
    ax.set_facecolor(LIGHT)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    if suffix == "%": value_text = f"{value:.2f}%"
    elif suffix == " pp": value_text = f"{value:.2f} pp"
    else: value_text = f"{value:.4f}"
    ax.text(0.5, 0.62, value_text, ha="center", va="center",
            fontsize=28, fontweight="bold", color=PRIMARY, transform=ax.transAxes)
    ax.text(0.5, 0.30, label, ha="center", va="center",
            fontsize=12, color=DARK, transform=ax.transAxes)

fig.text(0.5, 0.035,
         "Burden = predicted lesion area relative to the 512×512 image grid, not total leaf area.",
         ha="center", fontsize=9.5, color=MUTED)
fig.tight_layout(rect=[0.03, 0.07, 0.97, 0.90])
save(fig, "phase5_benchmark_summary.png")


# 3. Confusion matrix
class_order = list(per_class["true_class"])
cm = pd.crosstab(
    pd.Categorical(pred["true_class"], categories=class_order, ordered=True),
    pd.Categorical(pred["predicted_class"], categories=class_order, ordered=True),
    dropna=False,
).to_numpy()

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Disease Classification Confusion Matrix", fontsize=17, fontweight="bold", color=DARK, pad=18)
ax.set_xlabel("Predicted class", fontsize=11); ax.set_ylabel("True class", fontsize=11)
short_labels = [x.replace(" — ", "\n") for x in class_order]
ax.set_xticks(np.arange(len(class_order))); ax.set_yticks(np.arange(len(class_order)))
ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8.5)
ax.set_yticklabels(short_labels, fontsize=8.5)
threshold = cm.max() / 2 if cm.size else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8.5,
                color="white" if cm[i, j] > threshold else DARK)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Images")
fig.tight_layout()
save(fig, "classification_confusion_matrix.png")


# 4. Predicted vs true burden
x = pred["true_burden"].to_numpy() * 100
y = pred["predicted_burden"].to_numpy() * 100
limit = max(float(np.max(x)), float(np.max(y))) * 1.05
fig, ax = plt.subplots(figsize=(8.5, 7.2))
ax.scatter(x, y, s=30, alpha=0.72, edgecolors="none")
ax.plot([0, limit], [0, limit], linestyle="--", linewidth=1.5, color=MUTED)
ax.set_xlim(0, limit); ax.set_ylim(0, limit)
ax.set_xlabel("Ground-truth image-relative lesion burden (%)")
ax.set_ylabel("Predicted image-relative lesion burden (%)")
ax.set_title("Lesion Burden — Predicted vs Ground Truth", fontsize=17, fontweight="bold", color=DARK, pad=14)
ax.grid(alpha=0.22)
ax.text(0.04, 0.94,
        f"MAE = {burden['mae_percentage_points']:.2f} pp\nR² = {burden['r2']:.3f}\nPearson r = {burden['pearson']:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=GRID))
fig.tight_layout()
save(fig, "burden_scatter.png")


# 5. Per-class integrated performance
plot_df = per_class.copy()
plot_df["burden_mae_pp"] = plot_df["mean_burden_absolute_error"] * 100
plot_df = plot_df.sort_values("classification_accuracy", ascending=True)
labels = list(plot_df["true_class"])
fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
axes[0].barh(labels, plot_df["classification_accuracy"] * 100)
axes[0].set_xlabel("Accuracy (%)"); axes[0].set_title("Classification", fontweight="bold", color=DARK); axes[0].set_xlim(0, 105)
axes[1].barh(labels, plot_df["mean_segmentation_dice"])
axes[1].set_xlabel("Mean Dice"); axes[1].set_title("Segmentation", fontweight="bold", color=DARK); axes[1].set_xlim(0, 1.02)
axes[2].barh(labels, plot_df["burden_mae_pp"])
axes[2].set_xlabel("MAE (percentage points)"); axes[2].set_title("Lesion Burden", fontweight="bold", color=DARK)
for ax in axes:
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.suptitle("Per-Class Phase 5 Performance", fontsize=18, fontweight="bold", color=DARK, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
save(fig, "per_class_performance.png")


# 6. Integrated diagnostics
integrated = metrics["integrated"]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
groups = ["Class correct", "Class wrong"]

burden_values = [
    integrated["mean_burden_ae_when_classification_correct"] * 100,
    integrated["mean_burden_ae_when_classification_wrong"] * 100,
]
axes[0].bar(groups, burden_values)
axes[0].set_ylabel("Burden MAE (percentage points)")
axes[0].set_title("Burden error by classification outcome", fontweight="bold")
axes[0].grid(axis="y", alpha=0.2)

dice_values = [
    integrated["mean_segmentation_dice_when_classification_correct"],
    integrated["mean_segmentation_dice_when_classification_wrong"],
]
axes[1].bar(groups, dice_values)
axes[1].set_ylabel("Mean segmentation Dice")
axes[1].set_title("Segmentation quality by classification outcome", fontweight="bold")
axes[1].set_ylim(0, 1); axes[1].grid(axis="y", alpha=0.2)

for ax in axes:
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Integrated Failure Analysis", fontsize=17, fontweight="bold", color=DARK, y=0.99)
fig.text(0.5, 0.015,
         "Descriptive association only; these plots do not imply a causal relationship between the tasks.",
         ha="center", fontsize=9.5, color=MUTED)
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
save(fig, "integrated_diagnostics.png")

print("\nDone. Generated portfolio figures:")
for p in sorted(ASSETS.glob("*.png")):
    print(" -", p.relative_to(ROOT))
