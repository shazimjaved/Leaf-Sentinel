"""Training engine for lesion segmentation with BCE+Dice loss, checkpointing, and early stopping.

LeafSentinel Phase 2 baseline training module.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.segmentation.model import ResNetUNet, build_segmentation_model
from src.segmentation.metrics import compute_batch_metrics

logger = logging.getLogger(__name__)

EPSILON = 1e-7


class DiceLoss(nn.Module):
    """Soft Dice Loss computed directly from raw logits."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


class CombinedBCEDiceLoss(nn.Module):
    """Weighted sum of BCEWithLogitsLoss and DiceLoss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        total_loss = (self.bce_weight * loss_bce) + (self.dice_weight * loss_dice)
        return total_loss, loss_bce, loss_dice


def get_device() -> torch.device:
    """Determine best available compute device and log details."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Compute Device: CUDA | GPU: {gpu_name}")
        return torch.device("cuda:0")
    else:
        logger.info("Compute Device: CPU")
        return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedBCEDiceLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[Any] = None
) -> Tuple[float, float, float]:
    """Execute one training epoch."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = len(loader)

    for images, masks, _ in tqdm(loader, desc="Training Batch", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss, _, _ = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss, _, _ = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        # Batch metrics
        batch_metrics = compute_batch_metrics(logits.detach(), masks.detach())
        total_loss += loss.item()
        total_dice += batch_metrics["dice"]
        total_iou += batch_metrics["iou"]

    avg_loss = total_loss / max(1, n_batches)
    avg_dice = total_dice / max(1, n_batches)
    avg_iou = total_iou / max(1, n_batches)
    return avg_loss, avg_dice, avg_iou


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedBCEDiceLoss,
    device: torch.device
) -> Tuple[float, float, float]:
    """Execute one validation epoch."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n_batches = len(loader)

    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Validation Batch", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss, _, _ = criterion(logits, masks)

            batch_metrics = compute_batch_metrics(logits, masks)
            total_loss += loss.item()
            total_dice += batch_metrics["dice"]
            total_iou += batch_metrics["iou"]

    avg_loss = total_loss / max(1, n_batches)
    avg_dice = total_dice / max(1, n_batches)
    avg_iou = total_iou / max(1, n_batches)
    return avg_loss, avg_dice, avg_iou


def plot_training_curves(history_df: pd.DataFrame, output_path: Path):
    """Plot and save training/validation loss, Dice, and IoU curves."""
    fig, (ax_loss, ax_metrics) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    epochs = history_df["epoch"]

    # 1. Loss Curve
    ax_loss.plot(epochs, history_df["train_loss"], label="Train Loss", color="#1b4d3e", linewidth=2)
    ax_loss.plot(epochs, history_df["val_loss"], label="Val Loss", color="#c62828", linewidth=2, linestyle="--")
    ax_loss.set_title("Training & Validation Loss", fontsize=12, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax_loss.set_ylabel("Combined BCE+Dice Loss", fontsize=10, fontweight="bold")
    ax_loss.legend(frameon=True)
    ax_loss.grid(True, linestyle="--", alpha=0.3)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)

    # 2. Metric Curves (Dice & IoU)
    ax_metrics.plot(epochs, history_df["val_dice"], label="Val Dice", color="#2e7d32", linewidth=2)
    ax_metrics.plot(epochs, history_df["val_iou"], label="Val IoU", color="#1565c0", linewidth=2)
    ax_metrics.plot(epochs, history_df["train_dice"], label="Train Dice", color="#81c784", linewidth=1.5, linestyle=":")
    ax_metrics.set_title("Validation Segmentation Metrics", fontsize=12, fontweight="bold", pad=10)
    ax_metrics.set_xlabel("Epoch", fontsize=10, fontweight="bold")
    ax_metrics.set_ylabel("Score (0.0 to 1.0)", fontsize=10, fontweight="bold")
    ax_metrics.set_ylim(0.0, 1.02)
    ax_metrics.legend(frameon=True)
    ax_metrics.grid(True, linestyle="--", alpha=0.3)
    ax_metrics.spines["top"].set_visible(False)
    ax_metrics.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved training curves figure to {output_path}")


def train_segmentation_model(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_dir: Path,
    is_smoke_test: bool = False
) -> Dict[str, Any]:
    """Execute end-to-end training loop, checkpointing, and evaluation tracking."""
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    seed = int(config.get("split", {}).get("seed", 42))

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device()
    model = build_segmentation_model(config).to(device)

    # Loss & Optimizer
    bce_w = float(loss_cfg.get("bce_weight", 0.5))
    dice_w = float(loss_cfg.get("dice_weight", 0.5))
    criterion = CombinedBCEDiceLoss(bce_weight=bce_w, dice_weight=dice_w)

    lr = float(train_cfg.get("learning_rate", 0.0001))
    weight_decay = float(train_cfg.get("weight_decay", 0.0001))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 1 if is_smoke_test else int(train_cfg.get("epochs", 30))
    patience = int(train_cfg.get("early_stopping_patience", 7))
    use_amp = bool(train_cfg.get("mixed_precision", True)) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    logger.info(f"Starting Training: {epochs} epochs, initial LR={lr}, Device={device.type.upper()}, SmokeTest={is_smoke_test}.")

    history: List[Dict[str, Any]] = []
    best_val_dice = -1.0
    best_epoch = -1
    no_improve_epochs = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_dice, train_iou = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler
        )

        val_loss, val_dice, val_iou = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )

        epoch_duration = time.time() - epoch_start

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_iou": train_iou,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
            "epoch_duration_sec": epoch_duration
        })

        # Check for best model
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            best_epoch = epoch
            no_improve_epochs = 0
            
            # Save Best Model Checkpoint
            best_checkpoint_path = run_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "best_val_iou": val_iou,
                "config": config,
                "seed": seed
            }, best_checkpoint_path)
            logger.info(f"  ★ New best validation Dice ({best_val_dice:.4f}) at epoch {epoch}! Saved {best_checkpoint_path.name}")
        else:
            no_improve_epochs += 1

        # Always save last model checkpoint
        last_checkpoint_path = run_dir / "last_model.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_dice,
            "config": config,
            "seed": seed
        }, last_checkpoint_path)

        # Early stopping check
        if not is_smoke_test and no_improve_epochs >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {total_training_time/60:.2f} minutes. Best Epoch: {best_epoch} (Val Dice: {best_val_dice:.4f}).")

    # Save training history CSV
    history_df = pd.DataFrame(history)
    history_csv_path = run_dir / "training_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    logger.info(f"Saved training history to {history_csv_path}")

    # Plot training curves
    plot_training_curves(history_df, figures_dir / "training_curves.png")

    # Save run metadata JSON
    run_summary = {
        "project": "LeafSentinel",
        "phase": "phase2",
        "timestamp": datetime.now().isoformat(),
        "is_smoke_test": is_smoke_test,
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "total_training_time_sec": round(total_training_time, 2),
        "best_checkpoint": str(run_dir / "best_model.pth"),
        "history_csv": str(history_csv_path)
    }

    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    return run_summary
