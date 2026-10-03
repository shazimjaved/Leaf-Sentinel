import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.model import ResNetUNet
from src.severity.metrics import get_regression_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LeafSentinel.Phase4.EvalSegBurden")

@torch.no_grad()
def evaluate_segmentation_baseline(model, df, data_root, device, metrics_fn):
    model.eval()
    metrics_fn.reset()
    predictions = []
    
    # Phase 2 transforms
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Seg Baseline"):
        img_path = data_root / row["image_relpath"]
        target = row["mask_area_ratio"]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open {img_path}: {e}")
            continue
            
        tensor_img = transform(image).unsqueeze(0).to(device)
        
        # Predict mask
        logits = model(tensor_img)
        probs = torch.sigmoid(logits)
        
        # Phase 2 evaluation threshold was 0.5
        pred_mask = (probs > 0.5).float()
        
        # Calculate burden on the 512x512 mask (since aspect-ratio-distorting resize preserves exact area fraction!)
        predicted_pixels = pred_mask.sum().item()
        total_pixels = 512 * 512
        predicted_burden = predicted_pixels / total_pixels
        
        pred_tensor = torch.tensor([[predicted_burden]], dtype=torch.float32, device=device)
        target_tensor = torch.tensor([[target]], dtype=torch.float32, device=device)
        
        metrics_fn.update(pred_tensor, target_tensor)
        
        predictions.append({
            "image_id": row["image_id"],
            "true_burden": target,
            "predicted_burden": predicted_burden,
            "absolute_error": abs(target - predicted_burden),
            "signed_error": predicted_burden - target,
            "host": row["host"],
            "disease": row["disease"],
            "display_class": row["display_class"]
        })
        
    return metrics_fn.compute(), predictions

def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 Segmentation Baseline for Phase 4")
    parser.add_argument("--config", type=str, required=True, help="Path to Phase 4 severity config YAML")
    parser.add_argument("--seg-ckpt", type=str, required=True, help="Path to Phase 2 segmentation best_model.pth")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate")
    args = parser.parse_args()
    
    ckpt_path = Path(args.seg_ckpt)
    if not ckpt_path.exists():
        logger.error(f"Phase 2 segmentation checkpoint NOT FOUND at {ckpt_path}")
        logger.error("The segmentation-derived burden baseline cannot be evaluated without the Phase 2 model.")
        sys.exit(1)
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load Phase 2 model
    logger.info(f"Loading Phase 2 segmentation model from {ckpt_path}")
    model = ResNetUNet(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    
    # Load dataset
    manifest_path = Path(config["dataset"]["severity_manifest"])
    data_root = Path(config["dataset"]["root"])
    df = pd.read_csv(manifest_path)
    df = df[df["benchmark_split"] == args.split].reset_index(drop=True)
    
    metrics_fn = get_regression_metrics(device=device)
    
    logger.info(f"Evaluating {len(df)} samples...")
    metrics_res, predictions = evaluate_segmentation_baseline(model, df, data_root, device, metrics_fn)
    
    out_dir = Path(f"outputs/severity/evaluation/segmentation_baseline_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_metrics = {k: float(v) for k, v in metrics_res.items()}
    out_metrics["mae_percentage_points"] = out_metrics["mae"] * 100
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(out_metrics, f, indent=4)
        
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    
    logger.info(f"Evaluation complete. Segmentation Baseline MAE: {out_metrics['mae']:.4f}")
    logger.info(f"Outputs saved to {out_dir}")

if __name__ == "__main__":
    main()
