"""Phase 5 — Unit tests for LeafSentinel inference pipeline.

All tests are CPU-only and internet-independent.
Tests use synthetic/dummy model weights — no real checkpoints required.
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.preprocessing import (
    PadToSquare,
    get_classification_transform,
    get_segmentation_transform,
    get_regression_transform,
)
from src.inference.schemas import (
    ClassificationResult,
    ImageInfo,
    InferenceResult,
    SegmentationResult,
    SystemInfo,
    TopKEntry,
)
from src.inference.loaders import load_class_map, _validate_state_dict_keys
from src.inference.pipeline import (
    LeafSentinelPredictor,
    _compute_normalized_entropy,
    SEGMENTATION_GRID_SIZE,
    SEGMENTATION_THRESHOLD,
)

# ============================================================
# Fixtures
# ============================================================

CLASS_MAP = {
    "Apple \u2014 Black Rot": 0,
    "Banana \u2014 Black Sigatoka": 1,
    "Bell Pepper \u2014 Bacterial Spot": 2,
    "Citrus \u2014 Citrus Canker": 3,
    "Corn \u2014 Gray Leaf Spot": 4,
    "Grape \u2014 Downy Mildew": 5,
    "Potato \u2014 Late Blight": 6,
    "Soybean \u2014 Frogeye Leaf Spot": 7,
    "Tomato \u2014 Early Blight": 8,
    "Wheat \u2014 Leaf Rust": 9,
}


def _make_dummy_classifier(num_classes=10):
    """Create a tiny dummy classifier for testing."""
    import torch.nn as nn

    class DummyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 224 * 224, num_classes)

        def forward(self, x):
            return self.fc(x.flatten(1))

    model = DummyClassifier()
    model.eval()
    model.requires_grad_(False)
    return model


def _make_dummy_segmenter():
    """Create a tiny dummy segmenter for testing."""
    import torch.nn as nn

    class DummySegmenter(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 1, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    model = DummySegmenter()
    model.eval()
    model.requires_grad_(False)
    return model


def _make_dummy_image(w=300, h=200):
    """Create a synthetic RGB PIL Image."""
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ============================================================
# Preprocessing Tests
# ============================================================


class TestPreprocessing:
    def test_classification_output_shape(self):
        """Classifier preprocessing produces (3, 224, 224) tensor."""
        transform = get_classification_transform(image_size=224)
        img = _make_dummy_image(300, 200)
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_segmentation_output_shape(self):
        """Segmentation preprocessing produces (3, 512, 512) tensor."""
        transform = get_segmentation_transform(image_size=512)
        img = _make_dummy_image(300, 200)
        tensor = transform(img)
        assert tensor.shape == (3, 512, 512)

    def test_regression_output_shape(self):
        """Regression preprocessing produces (3, 224, 224) tensor."""
        transform = get_regression_transform(image_size=224)
        img = _make_dummy_image(300, 200)
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_pad_to_square_landscape(self):
        """PadToSquare on landscape image produces square."""
        pad = PadToSquare(fill=0)
        img = _make_dummy_image(400, 200)
        result = pad(img)
        assert result.size == (400, 400)

    def test_pad_to_square_portrait(self):
        """PadToSquare on portrait image produces square."""
        pad = PadToSquare(fill=0)
        img = _make_dummy_image(200, 400)
        result = pad(img)
        assert result.size == (400, 400)

    def test_pad_to_square_already_square(self):
        """PadToSquare on square image is identity."""
        pad = PadToSquare(fill=0)
        img = _make_dummy_image(300, 300)
        result = pad(img)
        assert result.size == (300, 300)


# ============================================================
# Class Map Tests
# ============================================================


class TestClassMap:
    def test_valid_class_map(self):
        """Valid class_to_idx.json loads and validates."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(CLASS_MAP, f, ensure_ascii=True)
            f.flush()
            path = f.name

        try:
            loaded = load_class_map(path)
            assert len(loaded) == 10
            assert sorted(loaded.values()) == list(range(10))
        finally:
            os.unlink(path)

    def test_missing_class_map_raises(self):
        """Missing class map file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_class_map("/nonexistent/path/class_to_idx.json")

    def test_wrong_class_count_raises(self):
        """Class map with != 10 entries raises ValueError."""
        bad_map = {"A": 0, "B": 1}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(bad_map, f)
            path = f.name

        try:
            with pytest.raises(ValueError, match="exactly 10 classes"):
                load_class_map(path)
        finally:
            os.unlink(path)

    def test_duplicate_indices_raises(self):
        """Class map with duplicate indices raises ValueError."""
        bad_map = {f"Class{i}": 0 for i in range(10)}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(bad_map, f)
            path = f.name

        try:
            with pytest.raises(ValueError, match="0..9"):
                load_class_map(path)
        finally:
            os.unlink(path)


# ============================================================
# Confidence & Entropy Tests
# ============================================================


class TestConfidenceMetrics:
    def test_top2_extraction(self):
        """Top-2 extraction returns correct indices and values."""
        probs = torch.zeros(10)
        probs[3] = 0.7
        probs[7] = 0.2
        top2_vals, top2_idx = torch.topk(probs, k=2)
        assert top2_idx[0].item() == 3
        assert top2_idx[1].item() == 7
        assert abs(top2_vals[0].item() - 0.7) < 1e-5

    def test_confidence_margin(self):
        """Top1-top2 margin is computed correctly."""
        probs = torch.zeros(10)
        probs[3] = 0.7
        probs[7] = 0.2
        top2_vals, _ = torch.topk(probs, k=2)
        margin = top2_vals[0].item() - top2_vals[1].item()
        assert abs(margin - 0.5) < 1e-5

    def test_entropy_uniform(self):
        """Uniform distribution has entropy ~1.0."""
        probs = torch.ones(10) / 10
        ent = _compute_normalized_entropy(probs, 10)
        assert abs(ent - 1.0) < 0.01

    def test_entropy_certain(self):
        """One-hot distribution has entropy ~0.0."""
        probs = torch.zeros(10)
        probs[0] = 1.0
        ent = _compute_normalized_entropy(probs, 10)
        assert ent < 0.01

    def test_entropy_in_range(self):
        """Entropy is always in [0, 1]."""
        for _ in range(20):
            probs = torch.softmax(torch.randn(10), dim=0)
            ent = _compute_normalized_entropy(probs, 10)
            assert 0.0 <= ent <= 1.0 + 1e-6


# ============================================================
# Burden Calculation Tests
# ============================================================


class TestBurdenCalculation:
    def test_zero_burden_empty_mask(self):
        """All-zero mask -> burden exactly 0."""
        mask = torch.zeros(SEGMENTATION_GRID_SIZE, SEGMENTATION_GRID_SIZE)
        lesion_pixels = int(mask.sum().item())
        total_pixels = SEGMENTATION_GRID_SIZE * SEGMENTATION_GRID_SIZE
        burden = lesion_pixels / total_pixels
        assert burden == 0.0

    def test_full_burden_full_mask(self):
        """All-one mask -> burden exactly 1."""
        mask = torch.ones(SEGMENTATION_GRID_SIZE, SEGMENTATION_GRID_SIZE)
        lesion_pixels = int(mask.sum().item())
        total_pixels = SEGMENTATION_GRID_SIZE * SEGMENTATION_GRID_SIZE
        burden = lesion_pixels / total_pixels
        assert burden == 1.0

    def test_half_burden(self):
        """Half-filled mask -> burden ~0.5."""
        mask = torch.zeros(SEGMENTATION_GRID_SIZE, SEGMENTATION_GRID_SIZE)
        mask[:SEGMENTATION_GRID_SIZE // 2, :] = 1.0
        lesion_pixels = int(mask.sum().item())
        total_pixels = SEGMENTATION_GRID_SIZE * SEGMENTATION_GRID_SIZE
        burden = lesion_pixels / total_pixels
        assert abs(burden - 0.5) < 0.01

    def test_measurement_grid_is_512(self):
        """Burden calculation uses the validated 512x512 grid."""
        assert SEGMENTATION_GRID_SIZE == 512


# ============================================================
# Pipeline Integration Tests
# ============================================================


class TestPipeline:
    def test_end_to_end_with_dummy_models(self):
        """Full pipeline runs with dummy models and produces valid result."""
        device = torch.device("cpu")
        clf = _make_dummy_classifier()
        seg = _make_dummy_segmenter()
        predictor = LeafSentinelPredictor(
            classifier=clf,
            segmenter=seg,
            class_map=CLASS_MAP,
            device=device,
        )
        image = _make_dummy_image(300, 200)
        result = predictor.predict(image, image_path="test.jpg")

        # Validate structure
        assert isinstance(result, InferenceResult)
        assert result.image.width == 300
        assert result.image.height == 200
        assert result.classification.class_index in range(10)
        assert 0 <= result.classification.confidence <= 1
        assert len(result.classification.top2) == 2
        assert 0 <= result.segmentation.image_relative_lesion_burden <= 1
        assert result.segmentation.measurement_grid == [512, 512]
        assert result.segmentation.threshold == 0.5

    def test_healthy_detection_always_false(self):
        """System always reports healthy_detection_supported=False."""
        device = torch.device("cpu")
        clf = _make_dummy_classifier()
        seg = _make_dummy_segmenter()
        predictor = LeafSentinelPredictor(
            classifier=clf,
            segmenter=seg,
            class_map=CLASS_MAP,
            device=device,
        )
        result = predictor.predict(_make_dummy_image())
        assert result.system.healthy_detection_supported is False

    def test_input_within_supported_scope_is_null(self):
        """System does not claim automatic scope detection."""
        device = torch.device("cpu")
        clf = _make_dummy_classifier()
        seg = _make_dummy_segmenter()
        predictor = LeafSentinelPredictor(
            classifier=clf,
            segmenter=seg,
            class_map=CLASS_MAP,
            device=device,
        )
        result = predictor.predict(_make_dummy_image())
        assert result.system.input_within_supported_scope is None

    def test_supported_classes_populated(self):
        """System reports the correct 10 supported classes."""
        device = torch.device("cpu")
        clf = _make_dummy_classifier()
        seg = _make_dummy_segmenter()
        predictor = LeafSentinelPredictor(
            classifier=clf,
            segmenter=seg,
            class_map=CLASS_MAP,
            device=device,
        )
        result = predictor.predict(_make_dummy_image())
        assert len(result.system.supported_classes) == 10

    def test_direct_regression_disabled_by_default(self):
        """Direct regression is None when regressor is not provided."""
        device = torch.device("cpu")
        clf = _make_dummy_classifier()
        seg = _make_dummy_segmenter()
        predictor = LeafSentinelPredictor(
            classifier=clf,
            segmenter=seg,
            class_map=CLASS_MAP,
            device=device,
        )
        result = predictor.predict(_make_dummy_image())
        assert result.direct_regression is None

    def test_result_json_serializable(self):
        """InferenceResult can be serialized to valid JSON."""
        device = torch.device("cpu")
        clf = _make_dummy_classifier()
        seg = _make_dummy_segmenter()
        predictor = LeafSentinelPredictor(
            classifier=clf,
            segmenter=seg,
            class_map=CLASS_MAP,
            device=device,
        )
        result = predictor.predict(_make_dummy_image())
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert "image" in parsed
        assert "classification" in parsed
        assert "segmentation" in parsed
        assert "system" in parsed


# ============================================================
# Visualization Tests
# ============================================================


class TestVisualization:
    def test_diagnostic_card_generation(self):
        """Diagnostic card saves a PNG file."""
        from src.inference.visualization import generate_diagnostic_card

        device = torch.device("cpu")
        clf = _make_dummy_classifier()
        seg = _make_dummy_segmenter()
        predictor = LeafSentinelPredictor(
            classifier=clf,
            segmenter=seg,
            class_map=CLASS_MAP,
            device=device,
        )
        image = _make_dummy_image(300, 200)
        result = predictor.predict(image)
        mask = predictor.get_segmentation_mask_for_display(image)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            generate_diagnostic_card(image, mask, result, output_path)
            assert Path(output_path).exists()
            assert Path(output_path).stat().st_size > 0
        finally:
            os.unlink(output_path)

    def test_mask_display_dimensions(self):
        """Display mask matches original image dimensions."""
        device = torch.device("cpu")
        clf = _make_dummy_classifier()
        seg = _make_dummy_segmenter()
        predictor = LeafSentinelPredictor(
            classifier=clf,
            segmenter=seg,
            class_map=CLASS_MAP,
            device=device,
        )
        image = _make_dummy_image(300, 200)
        mask = predictor.get_segmentation_mask_for_display(image)
        assert mask.shape == (200, 300)


# ============================================================
# Loader Validation Tests
# ============================================================


class TestLoaders:
    def test_state_dict_validation_missing_keys(self):
        """Missing keys in state_dict raises RuntimeError."""
        import torch.nn as nn

        model = nn.Linear(10, 5)
        bad_dict = {"nonexistent.weight": torch.randn(5, 10)}

        with pytest.raises(RuntimeError, match="missing"):
            _validate_state_dict_keys(model, bad_dict, "TestModel")

    def test_classifier_checkpoint_not_found(self):
        """Missing classifier checkpoint raises FileNotFoundError."""
        from src.inference.loaders import load_classifier

        with pytest.raises(FileNotFoundError):
            load_classifier("/nonexistent/model.pth", torch.device("cpu"))

    def test_segmenter_checkpoint_not_found(self):
        """Missing segmenter checkpoint raises FileNotFoundError."""
        from src.inference.loaders import load_segmenter

        with pytest.raises(FileNotFoundError):
            load_segmenter("/nonexistent/model.pth", torch.device("cpu"))


# ============================================================
# Batch Error Isolation Test
# ============================================================


class TestBatchErrorIsolation:
    def test_corrupt_image_does_not_stop_batch(self):
        """Simulates batch: a bad path raises but doesn't crash processing."""
        device = torch.device("cpu")
        clf = _make_dummy_classifier()
        seg = _make_dummy_segmenter()
        predictor = LeafSentinelPredictor(
            classifier=clf,
            segmenter=seg,
            class_map=CLASS_MAP,
            device=device,
        )

        results = []
        errors = []

        # Simulate batch with one good image and one bad path
        good_image = _make_dummy_image()
        bad_path = "/nonexistent/corrupt.jpg"

        # Process good image
        try:
            result = predictor.predict(good_image, "good.jpg")
            results.append(result)
        except Exception as e:
            errors.append(str(e))

        # Process bad image (should raise, but batch continues)
        try:
            predictor.predict_from_path(bad_path)
            results.append(None)
        except FileNotFoundError:
            errors.append(f"Failed: {bad_path}")

        assert len(results) == 1
        assert len(errors) == 1
