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

from src.classification.dataset import ClassificationDataset
from src.classification.model import DiseaseClassifier
from src.classification.metrics import get_classification_metrics

logger = logging.getLogger("LeafSentinel.Phase3.Train")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_dataloaders(config: dict, manifest_path: Path, dataset_root: Path):
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    input_mode = config["dataset"]["input_mode"]
    image_size = config["dataset"]["image_size"]
    
    train_ds = ClassificationDataset(manifest_path, dataset_root, "train", input_mode, image_size)
    val_ds = ClassificationDataset(manifest_path, dataset_root, "val", input_mode, image_size)
    test_ds = ClassificationDataset(manifest_path, dataset_root, "test", input_mode, image_size)
    
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
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets)
            
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        metrics.update(logits, targets)
        
    epoch_metrics = metrics.compute()
    return total_loss / len(loader), epoch_metrics

@torch.no_grad()
def validate(model, loader, criterion, metrics, device):
    model.eval()
    total_loss = 0.0
    metrics.reset()
    
    for images, targets in tqdm(loader, desc="Validate", leave=False):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        
        total_loss += loss.item()
        metrics.update(logits, targets)
        
    epoch_metrics = metrics.compute()
    return total_loss / len(loader), epoch_metrics

def run_training(config: dict, smoke_test: bool = False):
    set_seed(config["training"].get("seed", 42))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if not smoke_test and device.type == "cpu":
        logger.warning("Full Phase 3 training requested without CUDA. This experiment is intended for a GPU environment.")
        logger.warning("Use --smoke-test locally or explicitly confirm CPU training.")
    
    manifest_path = Path(config["dataset"]["classification_manifest"])
    dataset_root = Path(config["dataset"]["root"])
    
    # Load class weights
    summary_path = manifest_path.parent / "dataset_summary.json"
    with open(summary_path, "r") as f:
        summary = json.load(f)
    class_weights = torch.tensor(summary["train_class_weights"], dtype=torch.float32).to(device)
    logger.info(f"Loaded class weights from train split: {class_weights.tolist()}")
    
    # Dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(config, manifest_path, dataset_root)
    
    if smoke_test:
        logger.info("SMOKE TEST MODE: Truncating loaders.")
        train_loader.dataset.df = train_loader.dataset.df.head(64)
        val_loader.dataset.df = val_loader.dataset.df.head(32)
        config["training"]["epochs"] = 1
        
    # Model
    model = DiseaseClassifier(num_classes=10).to(device)
    tot, train = model.get_parameter_counts()
    logger.info(f"Model initialized: {tot:,} total params, {train:,} trainable.")
    
    # Loss, Optimizer, Scheduler
    label_smoothing = config["loss"].get("label_smoothing", 0.0)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["training"]["learning_rate"], 
        weight_decay=config["training"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Metrics
    train_metrics = get_classification_metrics(num_classes=10, device=device)
    val_metrics = get_classification_metrics(num_classes=10, device=device)
    
    scaler = torch.cuda.amp.GradScaler() if config["training"].get("mixed_precision") and device.type == "cuda" else None
    
    epochs = config["training"]["epochs"]
    patience = config["training"]["early_stopping_patience"]
    best_val_f1 = 0.0
    epochs_no_improve = 0
    
    out_dir = Path(config["paths"]["training_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    history = []
    
    logger.info(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss, tr_m = train_one_epoch(model, train_loader, criterion, optimizer, scaler, train_metrics, device)
        val_loss, val_m = validate(model, val_loader, criterion, val_metrics, device)
        
        # Primary metric is Macro F1
        val_macro_f1 = val_m["macro_f1"].item()
        
        scheduler.step(val_macro_f1)
        
        logger.info(f"Epoch {epoch}/{epochs}")
        logger.info(f"Train - Loss: {train_loss:.4f}, Macro F1: {tr_m['macro_f1']:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Macro F1: {val_macro_f1:.4f}, Balanced Acc: {val_m['balanced_accuracy']:.4f}")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_macro_f1": tr_m["macro_f1"].item(),
            "val_macro_f1": val_macro_f1,
            "val_balanced_acc": val_m["balanced_accuracy"].item()
        })
        
        # Checkpointing
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_macro_f1": best_val_f1,
            "class_weights": class_weights.tolist(),
            "config": config,
            "input_mode": config["dataset"]["input_mode"]
        }
        
        torch.save(checkpoint, out_dir / "last_model.pth")
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            epochs_no_improve = 0
            checkpoint["best_val_macro_f1"] = best_val_f1
            torch.save(checkpoint, out_dir / "best_model.pth")
            logger.info("Saved new best model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without Macro F1 improvement.")
                break
                
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    logger.info("Training complete.")
