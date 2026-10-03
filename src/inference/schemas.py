"""Phase 5 — Structured output schemas for LeafSentinel inference results."""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json


@dataclass
class ImageInfo:
    path: str
    width: int
    height: int


@dataclass
class TopKEntry:
    class_name: str
    class_index: int
    confidence: float


@dataclass
class ClassificationResult:
    predicted_class: str
    class_index: int
    confidence: float
    top2: List[TopKEntry]
    top1_top2_margin: float
    normalized_entropy: float


@dataclass
class SegmentationResult:
    threshold: float
    lesion_pixels: int
    total_pixels: int
    image_relative_lesion_burden: float
    burden_percent: float
    mean_probability: float
    max_probability: float
    measurement_grid: List[int]


@dataclass
class SystemInfo:
    healthy_detection_supported: bool = False
    input_scope_assumption: str = (
        "symptomatic leaf image from one of the 10 supported disease classes"
    )
    input_within_supported_scope: Optional[bool] = None
    supported_classes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DirectRegressionDiagnostic:
    direct_regression_burden: float
    segmentation_burden: float
    absolute_disagreement: float


@dataclass
class InferenceResult:
    image: ImageInfo
    classification: ClassificationResult
    segmentation: SegmentationResult
    system: SystemInfo
    direct_regression: Optional[DirectRegressionDiagnostic] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        d = asdict(self)
        # Remove direct_regression if None
        if d["direct_regression"] is None:
            del d["direct_regression"]
        return d

    def to_json(self, indent: int = 4) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
