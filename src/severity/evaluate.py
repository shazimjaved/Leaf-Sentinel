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
def evaluate_model(model, loader, raw_metrics, clipped_metrics, device="cpu"):
    model.eval()
    raw_metrics.reset()
    clipped_metrics.reset()
    
    predictions = []
    
    for images, targets in tqdm(loader, desc="Evaluating"):
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
        preds_clipped = torch.clamp(preds, 0.0, 1.0)
        
        raw_metrics.update(preds, targets)
        clipped_metrics.update(preds_clipped, targets)
        
        for i in range(targets.size(0)):
            raw_p = preds[i].item()
            clip_p = preds_clipped[i].item()
            t = targets[i].item()
            predictions.append({
                "true_burden": t,
                "raw_predicted_burden": raw_p,
                "clipped_predicted_burden": clip_p,
                "raw_absolute_error": abs(t - raw_p),
                "raw_signed_error": raw_p - t,
                "clipped_absolute_error": abs(t - clip_p),
                "clipped_signed_error": clip_p - t
            })
            
    return raw_metrics.compute(), clipped_metrics.compute(), predictions
