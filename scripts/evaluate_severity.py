import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.severity.model import BurdenRegressor
from src.severity.metrics import get_regression_metrics
from src.severity.dataset import ImageRelativeBurdenDataset
from src.severity.evaluate import evaluate_model
from src.severity.baselines import evaluate_trivial_baseline
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LeafSentinel.Phase4.EvaluateSeverity")

def plot_scatter(predictions_df, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(predictions_df["true_burden"], predictions_df["clipped_predicted_burden"], alpha=0.5)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel("True Image-Relative Lesion Burden")
    ax.set_ylabel("Predicted Image-Relative Lesion Burden")
    ax.set_title("Predicted vs True Burden")
    fig.savefig(out_path)
    plt.close(fig)

def plot_residuals(predictions_df, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(predictions_df["true_burden"], predictions_df["signed_error"], alpha=0.5)
    ax.axhline(0, color='r', linestyle='--')
    ax.set_xlabel("True Image-Relative Lesion Burden")
    ax.set_ylabel("Signed Error (Predicted - True)")
    ax.set_title("Residuals")
    fig.savefig(out_path)
    plt.close(fig)

def plot_per_class_mae(per_class_df, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    per_class_df.set_index("display_class")["mae"].sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Mean Absolute Error (MAE)")
    ax.set_title("MAE by Disease Class")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_target_distribution(predictions_df, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(predictions_df["true_burden"], bins=30, alpha=0.7, label="True")
    ax.hist(predictions_df["clipped_predicted_burden"], bins=30, alpha=0.7, label="Predicted")
    ax.set_xlabel("Image-Relative Lesion Burden")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of True vs Predicted Burden")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 Severity Regression Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the training run directory containing best_model.pth")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "best_model.pth"
    
    if not ckpt_path.exists():
        logger.error(f"Model checkpoint not found at {ckpt_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BurdenRegressor()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load dataset
    manifest_path = Path(config["dataset"]["severity_manifest"])
    dataset_root = Path(config["dataset"]["root"])
    image_size = ckpt.get("image_size", 224)
    
    ds = ImageRelativeBurdenDataset(manifest_path, dataset_root, args.split, image_size)
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"])

    # Define outputs
    out_dir = Path(f"outputs/severity/evaluation/{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_fn = get_regression_metrics(device=device)

    logger.info(f"Evaluating neural network on {len(ds)} {args.split} samples...")
    metrics_res, predictions = evaluate_model(model, loader, metrics_fn, device=device)

    # Convert predictions to DF and merge metadata
    pred_df = pd.DataFrame(predictions)
    pred_df["image_id"] = ds.df["image_id"]
    pred_df["host"] = ds.df["host"]
    pred_df["disease"] = ds.df["disease"]
    pred_df["display_class"] = ds.df["display_class"]
    pred_df["benchmark_split"] = ds.df["benchmark_split"]

    # Calculate per-class metrics
    class_metrics = []
    for cls, group in pred_df.groupby("display_class"):
        mae = group["absolute_error"].mean()
        bias = group["signed_error"].mean()
        rmse = (group["absolute_error"] ** 2).mean() ** 0.5
        class_metrics.append({
            "display_class": cls,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "support": len(group)
        })
    class_df = pd.DataFrame(class_metrics)

    # Evaluate trivial baselines using stored metadata
    train_mean = ckpt.get("train_target_mean", ds.df["mask_area_ratio"].mean()) # fallback if missing
    train_median = ckpt.get("train_target_median", ds.df["mask_area_ratio"].median())
    
    logger.info("Evaluating trivial baselines...")
    mean_metrics = evaluate_trivial_baseline(train_mean, loader, metrics_fn, device=device)
    median_metrics = evaluate_trivial_baseline(train_median, loader, metrics_fn, device=device)

    # Format global metrics
    global_metrics = {k: float(v) for k, v in metrics_res.items()}
    global_metrics["mae_percentage_points"] = global_metrics["mae"] * 100
    
    out_json = {
        "nn_metrics": global_metrics,
        "mean_baseline_metrics": {k: float(v) for k, v in mean_metrics.items()},
        "median_baseline_metrics": {k: float(v) for k, v in median_metrics.items()}
    }

    # Save outputs
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out_json, f, indent=4)

    metrics_flat = pd.DataFrame([global_metrics])
    metrics_flat.to_csv(out_dir / "metrics.csv", index=False)
    class_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    
    cols_order = ["image_id", "true_burden", "raw_predicted_burden", "clipped_predicted_burden", 
                  "absolute_error", "signed_error", "host", "disease", "display_class", "benchmark_split"]
    pred_df[cols_order].to_csv(out_dir / "predictions.csv", index=False)

    # Generate figures
    plot_scatter(pred_df, fig_dir / "scatter_predicted_vs_true.png")
    plot_residuals(pred_df, fig_dir / "residuals.png")
    plot_per_class_mae(class_df, fig_dir / "per_class_mae.png")
    plot_target_distribution(pred_df, fig_dir / "target_distribution.png")

    logger.info(f"Evaluation complete. NN MAE: {global_metrics['mae']:.4f}")
    logger.info(f"Outputs saved to {out_dir}")

if __name__ == "__main__":
    main()
