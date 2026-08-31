"""CLI runner for LeafSentinel lesion segmentation baseline test evaluation.

Usage:
    python scripts/evaluate_segmentation.py \
        --config configs/phase2.yaml \
        --checkpoint outputs/phase2/training/<run_name>/best_model.pth
"""

import argparse
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
from src.segmentation.evaluate import evaluate_segmentation_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LeafSentinel.Phase2.EvalCLI")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="LeafSentinel Lesion Segmentation Evaluation CLI")
    parser.add_argument("--config", type=str, default="configs/phase2.yaml", help="Path to Phase 2 YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--manifest", type=str, default=None, help="Optional override for manifest.csv")
    parser.add_argument("--output", type=str, default=None, help="Optional override for evaluation output directory")

    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    # Get Test DataLoader
    _, _, test_loader, _ = get_dataloaders(
        config=config,
        manifest_path=args.manifest
    )

    # Execute evaluation
    summary = evaluate_segmentation_model(
        config=config,
        checkpoint_path=args.checkpoint,
        test_loader=test_loader,
        output_dir=args.output
    )

    print("\n--- Summary Metrics JSON ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
