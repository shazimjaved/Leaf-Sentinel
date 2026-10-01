import json
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.severity.dataset import ImageRelativeBurdenDataset
from src.severity.model import BurdenRegressor
from src.severity.metrics import get_regression_metrics
from src.severity.baselines import evaluate_trivial_baseline

logger = logging.getLogger("LeafSentinel.Phase4.Evaluate")

@torch.no_grad()
def evaluate_model(model, loader, metrics, device="cpu"):
    model.eval()
    metrics.reset()
    
    predictions = []
    
    for images, targets in tqdm(loader, desc="Evaluating"):
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
        metrics.update(preds, targets)
        
        # Save raw predictions and clipped predictions
        preds_clipped = torch.clamp(preds, 0.0, 1.0)
        
        for i in range(targets.size(0)):
            predictions.append({
                "true_burden": targets[i].item(),
                "raw_predicted_burden": preds[i].item(),
                "clipped_predicted_burden": preds_clipped[i].item(),
                "absolute_error": abs(targets[i].item() - preds_clipped[i].item()),
                "signed_error": preds_clipped[i].item() - targets[i].item()
            })
            
    return metrics.compute(), predictions
