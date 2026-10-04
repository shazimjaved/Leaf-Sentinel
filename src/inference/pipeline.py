"""Phase 5 — Integrated multi-model inference pipeline.

Combines Phase 2 (segmentation), Phase 3 (classification), and optionally
Phase 4 (direct regression) into a single prediction call per image.

Models are loaded ONCE and reused across all images. No model is reloaded
per image.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

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
        classification_abstain_threshold: If not None, a top-1 confidence
            value below which a warning is emitted. Must be derived from
            validation data. Defaults to None (no automatic warning).
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

    def _run_segmentation(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single segmentation forward pass.

        Returns:
            Tuple of (binary_mask_512, seg_probs_512) — both CPU tensors,
            shape (512, 512).
        """
        seg_tensor = self.seg_transform(image).unsqueeze(0).to(self.device)
        seg_logits = self.segmenter(seg_tensor)
        seg_probs = torch.sigmoid(seg_logits).squeeze(0).squeeze(0).cpu()
        binary_mask = (seg_probs >= SEGMENTATION_THRESHOLD)
        return binary_mask, seg_probs

    def _mask_to_display(
        self, binary_mask_512: torch.Tensor, orig_w: int, orig_h: int
    ) -> np.ndarray:
        """Resize 512x512 binary mask to original image dimensions for display.

        The numeric burden is NOT recomputed here — this is purely for
        visualization. Values in the returned array are 0 or 255.
        """
        mask_np = binary_mask_512.numpy().astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_np)
        mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
        return np.array(mask_pil)

    @torch.inference_mode()
    def predict(self, image: Image.Image, image_path: str = "") -> InferenceResult:
        """Run the full inference pipeline on a single RGB image.

        Performs exactly one forward pass through each model.

        Args:
            image: PIL Image in RGB mode.
            image_path: Original file path (for metadata only).

        Returns:
            InferenceResult. Does not include the display mask array;
            use predict_with_artifacts() when visualization is also needed.
        """
        result, _ = self.predict_with_artifacts(image, image_path=image_path)
        return result

    @torch.inference_mode()
    def predict_with_artifacts(
        self, image: Image.Image, image_path: str = ""
    ) -> Tuple[InferenceResult, np.ndarray]:
        """Run the full inference pipeline and return result + display mask.

        Performs exactly one segmentation forward pass. The display mask
        (resized to original image dimensions) is returned alongside the
        InferenceResult so callers never need a second segmentation call.

        Args:
            image: PIL Image in RGB mode.
            image_path: Original file path (for metadata only).

        Returns:
            Tuple of:
                - InferenceResult (JSON-serializable, does not embed mask array)
                - display_mask: numpy uint8 array (H, W), values 0 or 255,
                  at original image dimensions. For visualization only.
        """
        width, height = image.size

        # ---- Classification (Phase 3) ----
        clf_tensor = self.clf_transform(image).unsqueeze(0).to(self.device)
        logits = self.classifier(clf_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

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

        # ---- Segmentation (Phase 2) — exactly ONE forward pass ----
        binary_mask_512, seg_probs_512 = self._run_segmentation(image)

        lesion_pixels = int(binary_mask_512.sum().item())
        total_pixels = SEGMENTATION_GRID_SIZE * SEGMENTATION_GRID_SIZE
        burden = lesion_pixels / total_pixels

        segmentation = SegmentationResult(
            threshold=SEGMENTATION_THRESHOLD,
            lesion_pixels=lesion_pixels,
            total_pixels=total_pixels,
            image_relative_lesion_burden=round(burden, 6),
            burden_percent=round(burden * 100, 4),
            mean_probability=round(seg_probs_512.mean().item(), 6),
            max_probability=round(seg_probs_512.max().item(), 6),
            measurement_grid=[SEGMENTATION_GRID_SIZE, SEGMENTATION_GRID_SIZE],
        )

        # ---- System warnings ----
        # Warnings are ONLY generated when classification_abstain_threshold
        # is explicitly configured (not None). No hard-coded numeric cutoffs
        # are applied. No entropy threshold exists. input_within_supported_scope
        # remains null — the pipeline cannot verify domain membership.
        warnings: List[str] = []
        if (
            self.classification_abstain_threshold is not None
            and top1_conf < self.classification_abstain_threshold
        ):
            warnings.append(
                "Classification confidence is below the configured "
                f"abstention threshold ({self.classification_abstain_threshold:.4f})."
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

        result = InferenceResult(
            image=ImageInfo(path=image_path, width=width, height=height),
            classification=classification,
            segmentation=segmentation,
            system=system,
            direct_regression=direct_regression,
        )

        # Build display mask at original dimensions — visualization only
        display_mask = self._mask_to_display(binary_mask_512, width, height)

        return result, display_mask

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

    def predict_from_path_with_artifacts(
        self, image_path: str
    ) -> Tuple[InferenceResult, np.ndarray]:
        """Load an image from disk and run inference, returning result + mask.

        Args:
            image_path: Path to an image file.

        Returns:
            Tuple of (InferenceResult, display_mask).

        Raises:
            FileNotFoundError: If the image does not exist.
        """
        from pathlib import Path

        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        image = Image.open(p).convert("RGB")
        return self.predict_with_artifacts(image, image_path=str(p))
