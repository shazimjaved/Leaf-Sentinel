"""Generate portfolio-ready LeafSentinel figures from tracked benchmark results.

Run from the repository root:
    python scripts/generate_project_figures.py

Outputs:
    docs/assets/

Inputs:
    results/phase5/metrics.json
    results/phase5/predictions.csv
    results/phase5/per_class_metrics.csv

The result-folder names reflect the project's internal development history.
Public-facing figures intentionally avoid internal phase labels.
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
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.63,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=DARK,
    )
    if subtitle:
        ax.text(
            x + width / 2,
            y + height * 0.32,
            subtitle,
            ha="center",
            va="center",
            fontsize=9.5,
            color=MUTED,
            wrap=True,
        )
