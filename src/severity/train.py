import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.severity.dataset import ImageRelativeBurdenDataset
from src.severity.model import BurdenRegressor
from src.severity.metrics import get_regression_metrics

logger = logging.getLogger("LeafSentinel.Phase4.Train")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataloaders(config: dict, manifest_path: Path, dataset_root: Path):
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    image_size = config["dataset"]["image_size"]
    
    train_ds = ImageRelativeBurdenDataset(manifest_path, dataset_root, "train", image_size)
    val_ds = ImageRelativeBurdenDataset(manifest_path, dataset_root, "val", image_size)
    test_ds = ImageRelativeBurdenDataset(manifest_path, dataset_root, "test", image_size)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

def train_one_epoch(model, loader, criterion, optimizer, scaler, metrics, device):
    model.train()
    total_loss = 0.0
    metrics.reset()
    
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            preds = model(images)
            loss = criterion(preds, targets)
            
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        metrics.update(preds, targets)
        
    epoch_metrics = metrics.compute()
    return total_loss / len(loader), epoch_metrics

@torch.no_grad()
def validate(model, loader, criterion, metrics, device):
    model.eval()
    total_loss = 0.0
    metrics.reset()
    
    for images, targets in tqdm(loader, desc="Validate", leave=False):
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
        loss = criterion(preds, targets)
        
        total_loss += loss.item()
        metrics.update(preds, targets)
        
    epoch_metrics = metrics.compute()
    return total_loss / len(loader), epoch_metrics

def run_training(config: dict, smoke_test: bool = False):
    set_seed(config["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if not smoke_test and device.type == "cpu":
        logger.warning("Full Phase 4 training requested without CUDA. This experiment is intended for a GPU environment.")
    
    manifest_path = Path(config["dataset"]["severity_manifest"])
    dataset_root = Path(config["dataset"]["root"])
    
    # Load summary stats for baseline info
    summary_path = manifest_path.parent / "dataset_summary.json"
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    train_mean = summary["target_train_mean"]
    train_median = summary["target_train_median"]
    
    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config, manifest_path, dataset_root)
    
    if smoke_test:
        logger.info("SMOKE TEST MODE: Truncating loaders.")
        train_loader.dataset.df = train_loader.dataset.df.head(64)
        val_loader.dataset.df = val_loader.dataset.df.head(32)
        config["training"]["epochs"] = 1
        
    # Model
    model = BurdenRegressor().to(device)
    tot, train = model.get_parameter_counts()
    logger.info(f"Model initialized: {tot:,} total params, {train:,} trainable.")
    
    # Configurable Loss
    loss_name = config["loss"].get("name", "l1").lower()
    if loss_name in ["mse", "l2"]:
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
        
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["training"]["learning_rate"], 
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Validation MAE used for model selection (Lower is better)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    train_metrics = get_regression_metrics(device=device)
    val_metrics = get_regression_metrics(device=device)
    
    scaler = torch.amp.GradScaler('cuda') if config["training"].get("mixed_precision") and device.type == "cuda" else None
    
    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]
    best_val_mae = float("inf")
    epochs_no_improve = 0
    
    out_dir = Path(config["paths"]["training_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    history = []
    
    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, tr_m = train_one_epoch(model, train_loader, criterion, optimizer, scaler, train_metrics, device)
        val_loss, val_m = validate(model, val_loader, criterion, val_metrics, device)
        
        val_mae = val_m["mae"].item()
        scheduler.step(val_mae)
        
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"Train - Loss: {train_loss:.4f}, MAE: {tr_m['mae']:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.4f} ({(val_mae*100):.1f} pp), RMSE: {val_m['rmse']:.4f}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae": tr_m["mae"].item(),
            "val_mae": val_mae,
            "val_rmse": val_m["rmse"].item(),
            "val_r2": val_m["r2"].item()
        })
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_mae": best_val_mae,
            "config": config,
            "input_mode": "full_rgb",
            "image_size": config["dataset"]["image_size"],
            "loss": loss_name,
            "seed": config["training"].get("seed", 42),
            "target_definition": "mask_area_ratio",
            "train_target_mean": train_mean,
            "train_target_median": train_median
        }
        
        torch.save(checkpoint, out_dir / "last_model.pth")
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            epochs_no_improve = 0
            checkpoint["best_val_mae"] = best_val_mae
            torch.save(checkpoint, out_dir / "best_model.pth")
            logger.info("Saved new best model.")
            
            # Save metadata dict
            meta_copy = checkpoint.copy()
            del meta_copy["model_state_dict"]
            del meta_copy["optimizer_state_dict"]
            del meta_copy["scheduler_state_dict"]
            with open(out_dir / "run_metadata.json", "w") as f:
                json.dump(meta_copy, f, indent=4)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without MAE improvement.")
                break
                
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    logger.info("Training complete.")
