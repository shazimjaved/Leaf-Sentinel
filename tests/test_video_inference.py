"""Tests for LeafSentinel Video Inference Engine."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pytest
from PIL import Image
import torch

from src.inference.schemas import (
    ClassificationResult,
    ImageInfo,
    InferenceResult,
    SegmentationResult,
    SystemInfo,
    TopKEntry,
)
from scripts.run_video_inference import (
    render_dashboard_frame,
    _get_font,
)


def _create_mock_result() -> InferenceResult:
    """Create a mock InferenceResult matching Phase 5 schema."""
    return InferenceResult(
        image=ImageInfo(path="test.png", width=1280, height=720),
        classification=ClassificationResult(
            predicted_class="Tomato — Early Blight",
            class_index=8,
            confidence=0.9421,
            top2=[
                TopKEntry(class_name="Tomato — Early Blight", class_index=8, confidence=0.9421),
                TopKEntry(class_name="Potato — Late Blight", class_index=6, confidence=0.0382),
            ],
            top1_top2_margin=0.9039,
            normalized_entropy=0.1420,
        ),
        segmentation=SegmentationResult(
            threshold=0.5,
            lesion_pixels=12634,
            total_pixels=262144,
            image_relative_lesion_burden=0.048195,
            burden_percent=4.8195,
            mean_probability=0.7842,
            max_probability=0.9912,
            measurement_grid=[512, 512],
        ),
        system=SystemInfo(
            healthy_detection_supported=False,
            supported_classes=["Tomato — Early Blight", "Potato — Late Blight"],
            warnings=[],
        ),
    )


def test_font_loading():
    """Verify system font loader returns a usable font."""
    font = _get_font(14, bold=False)
    assert font is not None


def test_render_dashboard_frame():
    """Verify render_dashboard_frame produces correct canvas shape and format."""
    H, W = 720, 1280
    frame_rgb = np.full((H, W, 3), 50, dtype=np.uint8)
    # Add green leaf pattern
    frame_rgb[200:500, 300:900] = [34, 139, 34]
    mask_display = np.zeros((H, W), dtype=np.uint8)
    mask_display[300:400, 400:600] = 255

    result = _create_mock_result()
    canvas = render_dashboard_frame(
        frame_rgb=frame_rgb,
        mask_display=mask_display,
        result=result,
        frame_idx=1,
        total_frames=10,
        timestamp_sec=0.2,
        inference_time_ms=145.2,
        device_name="cpu",
        canvas_w=1920,
        canvas_h=1080,
    )

    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (1080, 1920, 3)
    assert canvas.dtype == np.uint8
    # Canvas should not be all black or blank
    assert canvas.mean() > 5.0


def test_video_generation_with_synthetic_clip():
    """Verify that OpenCV can write and read synthetic frames for video pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_video_path = Path(tmpdir) / "test_synth.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(test_video_path), fourcc, 10.0, (640, 360))

        for i in range(10):
            frame = np.full((360, 640, 3), i * 20, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        assert test_video_path.exists()
        assert test_video_path.stat().st_size > 0

        cap = cv2.VideoCapture(str(test_video_path))
        assert cap.isOpened()
        ret, frame_read = cap.read()
        assert ret is True
        assert frame_read.shape == (360, 640, 3)
        cap.release()
