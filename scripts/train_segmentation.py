"""CLI runner for LeafSentinel lesion segmentation baseline training.

Usage:
    # Run rapid CPU smoke test:
    python scripts/train_segmentation.py --config configs/phase2.yaml --smoke-test

    # Run full baseline training:
    python scripts/train_segmentation.py --config configs/phase2.yaml
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.segmentation.dataset import get_dataloaders
from src.segmentation.train import train_segmentation_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LeafSentinel.Phase2.TrainCLI")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="LeafSentinel Lesion Segmentation Training CLI")
    parser.add_argument("--config", type=str, default="configs/phase2.yaml", help="Path to Phase 2 YAML config")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid CPU smoke test (1 epoch, 32 samples)")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--manifest", type=str, default=None, help="Override manifest path")

    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    # Apply overrides
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("training", {})["learning_rate"] = args.lr

    training_root = Path(config.get("paths", {}).get("training_dir", "outputs/phase2/training"))
    
    if args.smoke_test:
        smoke_cfg = config.get("smoke_test", {})
        run_name = "smoke_test_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = training_root / run_name
        
        logger.info(">>> Running Smoke Test Mode: 1 epoch, subset=32, image_size=256...")
        train_loader, val_loader, _, _ = get_dataloaders(
            config=config,
            manifest_path=args.manifest,
            image_size_override=smoke_cfg.get("image_size", 256),
            batch_size_override=smoke_cfg.get("batch_size", 4),
            max_samples_override=smoke_cfg.get("max_samples", 32)
        )
    else:
        run_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = training_root / run_name
        
        train_loader, val_loader, _, _ = get_dataloaders(
            config=config,
            manifest_path=args.manifest
        )

    # Execute training
    summary = train_segmentation_model(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        run_dir=run_dir,
        is_smoke_test=args.smoke_test
    )

    logger.info(f"Training run completed! Artifacts saved to: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
