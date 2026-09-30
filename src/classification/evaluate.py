import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.classification.dataset import ClassificationDataset
from src.classification.model import DiseaseClassifier
from src.classification.metrics import get_classification_metrics, get_per_class_metrics, get_confusion_matrix

logger = logging.getLogger("LeafSentinel.Phase3.Evaluate")

@torch.no_grad()
def run_evaluation(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on device: {device}")
    
    out_dir = Path(config["paths"]["evaluation_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    train_dir = Path(config["paths"]["training_dir"])
    chk_path = train_dir / "best_model.pth"
    if not chk_path.exists():
        logger.error(f"Checkpoint not found at {chk_path}")
        return
        
    checkpoint = torch.load(chk_path, map_location=device)
    
    manifest_path = Path(config["dataset"]["classification_manifest"])
    dataset_root = Path(config["dataset"]["root"])
    
    # Load class mapping
    summary_path = manifest_path.parent / "class_to_idx.json"
    with open(summary_path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    test_ds = ClassificationDataset(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        split="test",
        input_mode=checkpoint.get("input_mode", config["dataset"]["input_mode"]),
        image_size=config["dataset"]["image_size"]
    )
    
    if len(test_ds) == 0:
        logger.error("No test samples found.")
        return
        
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=config["training"]["num_workers"])
    
    model = DiseaseClassifier(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    metrics = get_classification_metrics(num_classes=10, device=device)
    per_class_metrics = get_per_class_metrics(num_classes=10, device=device)
    conf_matrix = get_confusion_matrix(num_classes=10, device=device)
    
    all_preds = []
    all_targets = []
    all_confs = []
    all_top2_classes = []
    all_top2_probs = []
    
    for images, targets in tqdm(test_loader, desc="Evaluate Test Set"):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        
        metrics.update(logits, targets)
        per_class_metrics.update(logits, targets)
        conf_matrix.update(logits, targets)
        
        probs = torch.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)
        
        top2_p, top2_c = torch.topk(probs, k=2, dim=1)
        
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
        all_confs.extend(confs.cpu().tolist())
        all_top2_classes.extend(top2_c.cpu().tolist())
        all_top2_probs.extend(top2_p.cpu().tolist())
        
    final_metrics = {k: v.item() for k, v in metrics.compute().items()}
    
    # Save scalar metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    logger.info("--- Test Results ---")
    logger.info(f"Macro F1: {final_metrics['macro_f1']:.4f}")
    logger.info(f"Balanced Acc: {final_metrics['balanced_accuracy']:.4f}")
    logger.info(f"Top-2 Acc: {final_metrics['top2_accuracy']:.4f}")
    
    # Per-class metrics
    pc = {k: v.cpu().numpy() for k, v in per_class_metrics.compute().items()}
    pc_df = pd.DataFrame({
        "class_index": list(range(10)),
        "class_name": [idx_to_class[i] for i in range(10)],
        "precision": pc["precision"],
        "recall": pc["recall"],
        "f1": pc["f1"]
    })
    pc_df.to_csv(out_dir / "per_class_metrics.csv", index=False)
    
    # Confusion Matrix
    cm = conf_matrix.compute().cpu().numpy()
    
    # Ensure no seaborn used as per instructions (matplotlib only)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=[idx_to_class[i] for i in range(10)],
           yticklabels=[idx_to_class[i] for i in range(10)],
           title="Confusion Matrix",
           ylabel='True label',
           xlabel='Predicted label')
           
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2. else "black")
                    
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png")
    plt.close(fig)
    
    # Per-class F1 Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(pc_df["class_name"], pc_df["f1"], color="#2ca02c")
    ax.set_title("Test F1 Score per Class")
    ax.set_xlabel("F1 Score")
    plt.tight_layout()
    fig.savefig(fig_dir / "per_class_f1.png")
    plt.close(fig)

    # Predictions CSV
    preds_df = test_ds.df.copy()
    preds_df["pred_class_idx"] = all_preds
    preds_df["pred_class_name"] = [idx_to_class[i] for i in all_preds]
    preds_df["confidence"] = all_confs
    preds_df["top2_classes"] = [str([idx_to_class[i] for i in c]) for c in all_top2_classes]
    preds_df["top2_probs"] = [str([round(p, 4) for p in probs]) for probs in all_top2_probs]
    preds_df["correct"] = preds_df["class_index"] == preds_df["pred_class_idx"]
    
    preds_df.to_csv(out_dir / "predictions.csv", index=False)
    logger.info("Evaluation complete. Results saved.")
