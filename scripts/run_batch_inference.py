"""LeafSentinel Phase 5 — Batch inference CLI.

Usage:
    python scripts/run_batch_inference.py \
        --config configs/inference.yaml \
        --input-dir path/to/images \
        --classifier-checkpoint path/to/classifier.pth \
        --segmentation-checkpoint path/to/segmenter.pth \
        --class-map path/to/class_to_idx.json \
        --output-dir outputs/inference/batch

Outputs:
    predictions.csv
    predictions.json
    visualizations/  (per-image diagnostic cards)
"""

import argparse
import csv
import json
import logging
import traceback
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
logger = logging.getLogger("LeafSentinel.Phase5.BatchInference")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _resolve_arg(cli_value, config_value, default=None):
    """CLI argument > config value > default."""
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def main():
    parser = argparse.ArgumentParser(
        description="LeafSentinel Phase 5 — Batch Inference"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--input-dir", type=str, required=True, help="Directory of images")
    parser.add_argument("--classifier-checkpoint", type=str, default=None)
    parser.add_argument("--segmentation-checkpoint", type=str, default=None)
    parser.add_argument("--regressor-checkpoint", type=str, default=None)
    parser.add_argument("--class-map", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/inference/batch")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-visualizations", action="store_true",
                        help="Skip generating per-image visualizations")
    args = parser.parse_args()

    # Load config
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    ckpt_cfg = config.get("checkpoints", {})
    clf_ckpt = _resolve_arg(args.classifier_checkpoint, ckpt_cfg.get("classifier"))
    seg_ckpt = _resolve_arg(args.segmentation_checkpoint, ckpt_cfg.get("segmenter"))
    reg_ckpt = _resolve_arg(args.regressor_checkpoint, ckpt_cfg.get("regressor"))
    class_map_path = _resolve_arg(args.class_map, config.get("class_map"))

    if clf_ckpt is None:
        logger.error("Classifier checkpoint not provided.")
        sys.exit(1)
    if seg_ckpt is None:
        logger.error("Segmentation checkpoint not provided.")
        sys.exit(1)
    if class_map_path is None:
        logger.error("Class mapping not provided.")
        sys.exit(1)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif config.get("device"):
        device = torch.device(config["device"])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load models ONCE
    class_map = load_class_map(class_map_path)
    classifier, _ = load_classifier(clf_ckpt, device)
    segmenter, _ = load_segmenter(seg_ckpt, device)

    regressor = None
    if reg_ckpt:
        regressor, _ = load_regressor(reg_ckpt, device)

    abstain_threshold = config.get("classification_abstain_threshold", None)
    predictor = LeafSentinelPredictor(
        classifier=classifier,
        segmenter=segmenter,
        class_map=class_map,
        device=device,
        regressor=regressor,
        classification_abstain_threshold=abstain_threshold,
    )

    # Discover images
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    logger.info(f"Found {len(image_files)} images in {input_dir}")

    if not image_files:
        logger.warning("No supported images found. Exiting.")
        sys.exit(0)

    # Prepare output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations"
    if not args.skip_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    results_json = []
    csv_rows = []

    for idx, img_path in enumerate(image_files, 1):
        logger.info(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
        try:
            image = Image.open(img_path).convert("RGB")
            result = predictor.predict(image, image_path=str(img_path))

            clf = result.classification
            seg = result.segmentation

            row = {
                "image": img_path.name,
                "status": "ok",
                "error": "",
                "predicted_class": clf.predicted_class,
                "confidence": round(clf.confidence, 6),
                "top2_class": clf.top2[1].class_name,
                "top2_confidence": round(clf.top2[1].confidence, 6),
                "confidence_margin": round(clf.top1_top2_margin, 6),
                "normalized_entropy": round(clf.normalized_entropy, 6),
                "lesion_burden": round(seg.image_relative_lesion_burden, 6),
                "burden_percent": round(seg.burden_percent, 4),
                "warning_count": len(result.system.warnings),
            }
            csv_rows.append(row)
            results_json.append(result.to_dict())

            # Generate visualizations
            if not args.skip_visualizations:
                stem = img_path.stem
                mask = predictor.get_segmentation_mask_for_display(image)
                generate_diagnostic_card(
                    image=image,
                    mask=mask,
                    result=result,
                    output_path=str(vis_dir / f"{stem}_diagnostic.png"),
                )

        except Exception as e:
            logger.error(f"Failed on {img_path.name}: {e}")
            logger.debug(traceback.format_exc())
            row = {
                "image": img_path.name,
                "status": "error",
                "error": str(e),
                "predicted_class": "",
                "confidence": "",
                "top2_class": "",
                "top2_confidence": "",
                "confidence_margin": "",
                "normalized_entropy": "",
                "lesion_burden": "",
                "burden_percent": "",
                "warning_count": "",
            }
            csv_rows.append(row)

    # Write predictions.csv
    csv_path = out_dir / "predictions.csv"
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    logger.info(f"Saved predictions.csv ({len(csv_rows)} rows)")

    # Write predictions.json
    with open(out_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved predictions.json ({len(results_json)} successful results)")

    # Summary
    ok_count = sum(1 for r in csv_rows if r["status"] == "ok")
    err_count = sum(1 for r in csv_rows if r["status"] == "error")
    logger.info(f"Batch complete: {ok_count} succeeded, {err_count} failed.")


if __name__ == "__main__":
    main()
