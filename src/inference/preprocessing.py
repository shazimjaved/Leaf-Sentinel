"""Phase 5 — Preprocessing pipelines for classification and segmentation.

Each model uses its own preprocessing matching the original training pipeline.
Do NOT reuse one transformed tensor for both models.
"""

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


class PadToSquare:
    """Pad an image to make it square, preserving aspect ratio.

    Replicates the Phase 3 training transform exactly.
    Fill=0 (black) matches the Phase 3 training pipeline.
    """

    def __init__(self, fill: int = 0):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        max_wh = max(w, h)
        pad_left = (max_wh - w) // 2
        pad_top = (max_wh - h) // 2
        pad_right = max_wh - w - pad_left
        pad_bottom = max_wh - h - pad_top
        return TF.pad(
            img,
            (pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
            padding_mode="constant",
        )


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_classification_transform(image_size: int = 224) -> transforms.Compose:
    """Phase 3 classifier preprocessing: PadToSquare -> Resize(224) -> Normalize.

    Matches the Phase 3 validation/test pipeline exactly.
    """
    return transforms.Compose([
        PadToSquare(fill=0),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_segmentation_transform(image_size: int = 512) -> transforms.Compose:
    """Phase 2 segmenter preprocessing: Resize(512) -> Normalize.

    Matches the Phase 2 validation/test pipeline exactly.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_regression_transform(image_size: int = 224) -> transforms.Compose:
    """Phase 4 direct regression preprocessing: Resize(224) -> Normalize.

    Matches the Phase 4 validation/test pipeline exactly.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
