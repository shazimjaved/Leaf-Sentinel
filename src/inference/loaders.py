"""Phase 5 — Robust model checkpoint loaders.

Each loader:
  - Instantiates the architecture with pretrained=False (no ImageNet download)
  - Loads the checkpoint with map_location
  - Validates state_dict compatibility
  - Sets model.eval() and model.requires_grad_(False)
  - Fails clearly on missing or incompatible checkpoints
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from src.classification.model import DiseaseClassifier
from src.segmentation.model import ResNetUNet
from src.severity.model import BurdenRegressor

logger = logging.getLogger("LeafSentinel.Phase5.Loaders")


def _validate_state_dict_keys(
    model: torch.nn.Module,
    state_dict: dict,
    model_name: str,
) -> None:
    """Validate that checkpoint state_dict keys match the model architecture."""
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        raise RuntimeError(
            f"{model_name}: checkpoint is missing {len(missing)} keys. "
            f"First 5: {list(missing)[:5]}"
        )
    if unexpected:
        logger.warning(
            f"{model_name}: checkpoint has {len(unexpected)} unexpected keys "
            f"(will be ignored). First 5: {list(unexpected)[:5]}"
        )


def load_classifier(
    ckpt_path: str,
    device: torch.device,
    num_classes: int = 10,
) -> Tuple[DiseaseClassifier, dict]:
    """Load Phase 3 Disease Classifier from checkpoint.

    Args:
        ckpt_path: Path to the classifier checkpoint (.pth).
        device: Target device.
        num_classes: Number of output classes.

    Returns:
        Tuple of (model, checkpoint_metadata).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If state_dict is incompatible.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {path}")

    logger.info(f"Loading classifier from {path}")
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model = DiseaseClassifier(num_classes=num_classes, pretrained=False)
    _validate_state_dict_keys(model, state_dict, "DiseaseClassifier")
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    model.requires_grad_(False)

    logger.info("Classifier loaded successfully.")
    return model, ckpt


def load_segmenter(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[ResNetUNet, dict]:
    """Load Phase 2 ResNet18-U-Net Segmentation model from checkpoint.

    Args:
        ckpt_path: Path to the segmentation checkpoint (.pth).
        device: Target device.

    Returns:
        Tuple of (model, checkpoint_metadata).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If state_dict is incompatible.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {path}")

    logger.info(f"Loading segmenter from {path}")
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model = ResNetUNet(pretrained=False)
    _validate_state_dict_keys(model, state_dict, "ResNetUNet")
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    model.requires_grad_(False)

    logger.info("Segmenter loaded successfully.")
    return model, ckpt


def load_regressor(
    ckpt_path: str,
    device: torch.device,
) -> Tuple[BurdenRegressor, dict]:
    """Load Phase 4 Direct Burden Regressor from checkpoint (optional).

    Args:
        ckpt_path: Path to the regressor checkpoint (.pth).
        device: Target device.

    Returns:
        Tuple of (model, checkpoint_metadata).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If state_dict is incompatible.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Regressor checkpoint not found: {path}")

    logger.info(f"Loading direct regressor from {path}")
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    model = BurdenRegressor(pretrained=False)
    _validate_state_dict_keys(model, state_dict, "BurdenRegressor")
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()
    model.requires_grad_(False)

    logger.info("Direct regressor loaded successfully.")
    return model, ckpt


def load_class_map(path: str) -> Dict[str, int]:
    """Load and validate the Phase 3 class-to-index mapping.

    Args:
        path: Path to class_to_idx.json.

    Returns:
        Dictionary mapping class names to integer indices.

    Raises:
        FileNotFoundError: If mapping file does not exist.
        ValueError: If mapping is invalid (wrong count, duplicate indices, etc.).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Class mapping file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    # Validate exactly 10 classes
    if len(class_to_idx) != 10:
        raise ValueError(
            f"Expected exactly 10 classes, got {len(class_to_idx)}"
        )

    # Validate indices are 0..9 with no duplicates
    indices = sorted(class_to_idx.values())
    if indices != list(range(10)):
        raise ValueError(
            f"Class indices must be exactly 0..9, got {indices}"
        )

    logger.info(f"Loaded class mapping with {len(class_to_idx)} classes from {p}")
    return class_to_idx
