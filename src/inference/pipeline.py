"""Phase 5 — Integrated multi-model inference pipeline.

Combines Phase 2 (segmentation), Phase 3 (classification), and optionally
Phase 4 (direct regression) into a single prediction call per image.

Models are loaded ONCE and reused across all images. No model is reloaded
per image.
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from src.inference.preprocessing import (
    get_classification_transform,
    get_segmentation_transform,
    get_regression_transform,
)
from src.inference.schemas import (
    ClassificationResult,
    DirectRegressionDiagnostic,
    ImageInfo,
    InferenceResult,
    SegmentationResult,
    SystemInfo,
    TopKEntry,
)

logger = logging.getLogger("LeafSentinel.Phase5.Pipeline")

# Segmentation grid matches Phase 4 validated protocol
SEGMENTATION_GRID_SIZE = 512
SEGMENTATION_THRESHOLD = 0.5


def _compute_normalized_entropy(
    probs: torch.Tensor, num_classes: int, eps: float = 1e-10
) -> float:
    """Compute H(p) / log(K) for a probability distribution.

    Args:
        probs: 1-D tensor of class probabilities summing to ~1.
        num_classes: K, the number of classes.
        eps: Small constant for numerical stability.

    Returns:
        Normalized entropy in [0, 1].
    """
    clamped = torch.clamp(probs, min=eps)
    entropy = -(clamped * torch.log(clamped)).sum().item()
    max_entropy = math.log(num_classes)
    if max_entropy == 0:
        return 0.0
    return entropy / max_entropy


class LeafSentinelPredictor:
    """Unified inference engine combining all frozen LeafSentinel models.

    Models are loaded once at construction and reused for every image.

    Attributes:
        classifier: Phase 3 DiseaseClassifier in eval mode.
        segmenter: Phase 2 ResNetUNet in eval mode.
        regressor: Optional Phase 4 BurdenRegressor in eval mode.
        class_map: Mapping of class name -> integer index.
        idx_to_class: Reverse mapping of integer index -> class name.
        device: Computation device (cpu or cuda).
    """

    def __init__(
        self,
        classifier: torch.nn.Module,
        segmenter: torch.nn.Module,
        class_map: Dict[str, int],
        device: torch.device,
        regressor: Optional[torch.nn.Module] = None,
        classification_abstain_threshold: Optional[float] = None,
    ):
        self.classifier = classifier
        self.segmenter = segmenter
        self.regressor = regressor
        self.class_map = class_map
        self.idx_to_class = {v: k for k, v in class_map.items()}
        self.device = device
        self.classification_abstain_threshold = classification_abstain_threshold

        # Build transforms — each model uses its own pipeline
        self.clf_transform = get_classification_transform(image_size=224)
        self.seg_transform = get_segmentation_transform(image_size=SEGMENTATION_GRID_SIZE)
        if self.regressor is not None:
            self.reg_transform = get_regression_transform(image_size=224)

        self.supported_classes = sorted(class_map.keys(), key=lambda c: class_map[c])
        self.num_classes = len(class_map)

    @torch.inference_mode()
    def predict(self, image: Image.Image, image_path: str = "") -> InferenceResult:
        """Run the full inference pipeline on a single RGB image.

        Args:
            image: PIL Image in RGB mode.
            image_path: Original file path (for metadata only).

        Returns:
            InferenceResult containing classification, segmentation, and
            system diagnostics.
        """
        width, height = image.size

        # ---- Classification (Phase 3) ----
        clf_tensor = self.clf_transform(image).unsqueeze(0).to(self.device)
        logits = self.classifier(clf_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        # Top-2 extraction
        top2_values, top2_indices = torch.topk(probs, k=2)

        top1_idx = top2_indices[0].item()
        top1_conf = top2_values[0].item()
        top2_idx = top2_indices[1].item()
        top2_conf = top2_values[1].item()
        margin = top1_conf - top2_conf

        normalized_entropy = _compute_normalized_entropy(probs, self.num_classes)

        classification = ClassificationResult(
            predicted_class=self.idx_to_class[top1_idx],
            class_index=top1_idx,
            confidence=round(top1_conf, 6),
            top2=[
                TopKEntry(
                    class_name=self.idx_to_class[top1_idx],
                    class_index=top1_idx,
                    confidence=round(top1_conf, 6),
                ),
                TopKEntry(
                    class_name=self.idx_to_class[top2_idx],
                    class_index=top2_idx,
                    confidence=round(top2_conf, 6),
                ),
            ],
            top1_top2_margin=round(margin, 6),
            normalized_entropy=round(normalized_entropy, 6),
        )

        # ---- Segmentation (Phase 2) ----
        seg_tensor = self.seg_transform(image).unsqueeze(0).to(self.device)
        seg_logits = self.segmenter(seg_tensor)
        seg_probs = torch.sigmoid(seg_logits).squeeze(0).squeeze(0)  # (512, 512)

        binary_mask = (seg_probs >= SEGMENTATION_THRESHOLD).float()
        lesion_pixels = int(binary_mask.sum().item())
        total_pixels = SEGMENTATION_GRID_SIZE * SEGMENTATION_GRID_SIZE
        burden = lesion_pixels / total_pixels

        segmentation = SegmentationResult(
            threshold=SEGMENTATION_THRESHOLD,
            lesion_pixels=lesion_pixels,
            total_pixels=total_pixels,
            image_relative_lesion_burden=round(burden, 6),
            burden_percent=round(burden * 100, 4),
            mean_probability=round(seg_probs.mean().item(), 6),
            max_probability=round(seg_probs.max().item(), 6),
            measurement_grid=[SEGMENTATION_GRID_SIZE, SEGMENTATION_GRID_SIZE],
        )

        # ---- System warnings ----
        warnings = []
        if top1_conf < 0.5:
            warnings.append(
                "Low classification confidence (<0.5). "
                "Input may be outside supported disease set."
            )
        if normalized_entropy > 0.8:
            warnings.append(
                "High predictive entropy (>0.8). "
                "Model is uncertain across multiple classes."
            )

        system = SystemInfo(
            healthy_detection_supported=False,
            supported_classes=self.supported_classes,
            warnings=warnings,
        )

        # ---- Optional direct regression diagnostic (Phase 4) ----
        direct_regression = None
        if self.regressor is not None:
            reg_tensor = self.reg_transform(image).unsqueeze(0).to(self.device)
            reg_output = self.regressor(reg_tensor)
            reg_burden = torch.clamp(reg_output, 0.0, 1.0).squeeze().item()

            direct_regression = DirectRegressionDiagnostic(
                direct_regression_burden=round(reg_burden, 6),
                segmentation_burden=round(burden, 6),
                absolute_disagreement=round(abs(burden - reg_burden), 6),
            )

        return InferenceResult(
            image=ImageInfo(path=image_path, width=width, height=height),
            classification=classification,
            segmentation=segmentation,
            system=system,
            direct_regression=direct_regression,
        )

    def predict_from_path(self, image_path: str) -> InferenceResult:
        """Load an image from disk and run full inference.

        Args:
            image_path: Path to an image file.

        Returns:
            InferenceResult.

        Raises:
            FileNotFoundError: If the image does not exist.
        """
        from pathlib import Path

        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        image = Image.open(p).convert("RGB")
        return self.predict(image, image_path=str(p))

    def get_segmentation_mask_for_display(
        self, image: Image.Image
    ) -> np.ndarray:
        """Get the binary segmentation mask resized to original image dimensions.

        Used for visualization only. The numeric burden is always computed
        on the validated 512x512 grid.

        Args:
            image: Original PIL Image.

        Returns:
            Binary mask as numpy array of shape (H, W) with values 0 or 255.
        """
        with torch.inference_mode():
            seg_tensor = self.seg_transform(image).unsqueeze(0).to(self.device)
            seg_logits = self.segmenter(seg_tensor)
            seg_probs = torch.sigmoid(seg_logits).squeeze(0).squeeze(0)
            binary_mask_512 = (seg_probs >= SEGMENTATION_THRESHOLD).cpu().numpy()

        # Resize mask back to original dimensions for display
        w, h = image.size
        mask_pil = Image.fromarray((binary_mask_512 * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((w, h), Image.NEAREST)
        return np.array(mask_pil)

    def get_probability_map_for_display(
        self, image: Image.Image
    ) -> np.ndarray:
        """Get the segmentation probability map resized to original dimensions.

        Args:
            image: Original PIL Image.

        Returns:
            Probability map as float32 numpy array of shape (H, W) in [0, 1].
        """
        with torch.inference_mode():
            seg_tensor = self.seg_transform(image).unsqueeze(0).to(self.device)
            seg_logits = self.segmenter(seg_tensor)
            seg_probs = torch.sigmoid(seg_logits).squeeze(0).squeeze(0).cpu().numpy()

        w, h = image.size
        prob_pil = Image.fromarray(seg_probs.astype(np.float32))
        prob_pil = prob_pil.resize((w, h), Image.BILINEAR)
        return np.array(prob_pil)
