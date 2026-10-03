"""LeafSentinel Phase 5 — Single-image inference CLI.

Usage:
    python scripts/run_inference.py \
        --config configs/inference.yaml \
        --image path/to/image.jpg \
        --classifier-checkpoint path/to/classifier.pth \
        --segmentation-checkpoint path/to/segmenter.pth \
        --class-map path/to/class_to_idx.json \
        --output-dir outputs/inference/sample

Outputs:
    result.json
    mask.png
    overlay.png
    diagnostic_card.png
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import torch
import yaml
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.loaders import (
    load_classifier,
    load_segmenter,
    load_regressor,
    load_class_map,
)
from src.inference.pipeline import LeafSentinelPredictor
from src.inference.visualization import (
    generate_diagnostic_card,
    save_mask,
    save_overlay,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LeafSentinel.Phase5.Inference")


def _resolve_arg(cli_value, config_value, default=None):
    """CLI argument > config value > default."""
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def main():
    parser = argparse.ArgumentParser(
        description="LeafSentinel Phase 5 — Single Image Inference"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to inference config YAML")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--classifier-checkpoint", type=str, default=None)
    parser.add_argument("--segmentation-checkpoint", type=str, default=None)
    parser.add_argument("--regressor-checkpoint", type=str, default=None, help="Optional Phase 4 regressor")
    parser.add_argument("--class-map", type=str, default=None, help="Path to class_to_idx.json")
    parser.add_argument("--output-dir", type=str, default="outputs/inference/sample")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    args = parser.parse_args()

    # Load config if provided
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    # Resolve paths: CLI > config > None
    ckpt_cfg = config.get("checkpoints", {})
    clf_ckpt = _resolve_arg(args.classifier_checkpoint, ckpt_cfg.get("classifier"))
    seg_ckpt = _resolve_arg(args.segmentation_checkpoint, ckpt_cfg.get("segmenter"))
    reg_ckpt = _resolve_arg(args.regressor_checkpoint, ckpt_cfg.get("regressor"))
    class_map_path = _resolve_arg(args.class_map, config.get("class_map"))

    # Validate required paths
    if clf_ckpt is None:
        logger.error("Classifier checkpoint not provided. Use --classifier-checkpoint or set in config.")
        sys.exit(1)
    if seg_ckpt is None:
        logger.error("Segmentation checkpoint not provided. Use --segmentation-checkpoint or set in config.")
        sys.exit(1)
    if class_map_path is None:
        logger.error("Class mapping not provided. Use --class-map or set in config.")
        sys.exit(1)

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    elif config.get("device"):
        device = torch.device(config["device"])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load models (ONCE)
    class_map = load_class_map(class_map_path)
    classifier, clf_meta = load_classifier(clf_ckpt, device)
    segmenter, seg_meta = load_segmenter(seg_ckpt, device)

    regressor = None
    if reg_ckpt:
        regressor, _ = load_regressor(reg_ckpt, device)

    # Build predictor
    abstain_threshold = config.get("classification_abstain_threshold", None)
    predictor = LeafSentinelPredictor(
        classifier=classifier,
        segmenter=segmenter,
        class_map=class_map,
        device=device,
        regressor=regressor,
        classification_abstain_threshold=abstain_threshold,
    )

    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)
    image = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded image: {image_path} ({image.size[0]}x{image.size[1]})")

    # Run inference
    result = predictor.predict(image, image_path=str(image_path))

    # Prepare output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save result.json
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        f.write(result.to_json())
    logger.info(f"Saved result.json")

    # Save mask and overlay at original image dimensions
    mask_display = predictor.get_segmentation_mask_for_display(image)
    save_mask(mask_display, str(out_dir / "mask.png"))
    save_overlay(image, mask_display, str(out_dir / "overlay.png"))

    # Generate diagnostic card
    generate_diagnostic_card(
        image=image,
        mask=mask_display,
        result=result,
        output_path=str(out_dir / "diagnostic_card.png"),
    )

    # Summary
    clf = result.classification
    seg = result.segmentation
    logger.info(
        f"Prediction: {clf.predicted_class} "
        f"(conf={clf.confidence:.4f}, "
        f"burden={seg.burden_percent:.2f}%)"
    )
    logger.info(f"All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
