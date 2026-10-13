"""LeafSentinel — Professional Video Inference Engine.

Runs dual-model inference (disease classification + lesion segmentation)
over video footage using frozen checkpoints and the existing LeafSentinelPredictor.

Features:
- Dual-model loading once; single forward pass per model per sampled frame.
- Image-relative lesion burden measured on the native 512x512 grid.
- Clean, restrained, professional dark dashboard visual layout (suitable for GitHub/portfolio).
- H.264 video encoding via FFmpeg for high compression and universal playback.
- Comprehensive runtime metrics: per-frame CSV, execution summary JSON, checkpoint verification.
"""

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.loaders import (
    load_classifier,
    load_segmenter,
    load_class_map,
)
from src.inference.pipeline import LeafSentinelPredictor
from src.inference.visualization import _create_overlay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LeafSentinel.VideoInference")


def _get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load high-quality system font with fallbacks."""
    candidates = []
    if sys.platform == "win32":
        candidates = [
            "C:\\Windows\\Fonts\\segoeuib.ttf" if bold else "C:\\Windows\\Fonts\\segoeui.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\calibrib.ttf" if bold else "C:\\Windows\\Fonts\\calibri.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]

    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def render_dashboard_frame(
    frame_rgb: np.ndarray,
    mask_display: np.ndarray,
    result,
    frame_idx: int,
    total_frames: int,
    timestamp_sec: float,
    inference_time_ms: float,
    device_name: str,
    canvas_w: int = 1920,
    canvas_h: int = 1080,
) -> np.ndarray:
    """Compose video frame + telemetry into a professional dark dashboard canvas.

    Layout (1920x1080):
    - Top header: branding, task title, device badge
    - Left area: 16:9 video viewport with red lesion overlay and HUD badges
    - Right area: clean telemetry cards (Model Prediction, Lesion Analysis, Notice)
    - Bottom bar: model provenance and grid specifications
    """
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(11, 15, 23)) # #0B0F17
    draw = ImageDraw.Draw(canvas)

    # Fonts
    font_brand = _get_font(26, bold=True)
    font_subbrand = _get_font(13, bold=False)
    font_badge = _get_font(12, bold=True)
    font_card_title = _get_font(13, bold=True)
    font_pred_large = _get_font(22, bold=True)
    font_metric_large = _get_font(26, bold=True)
    font_body = _get_font(13, bold=False)
    font_body_bold = _get_font(13, bold=True)
    font_small = _get_font(11, bold=False)
    font_small_italic = _get_font(11, bold=False)

    # 1. Header Bar
    # Brand: LeafSentinel
    draw.text((40, 28), "LEAFSENTINEL", fill=(241, 245, 249), font=font_brand)
    draw.text((250, 36), "PLANT DISEASE INTELLIGENCE", fill=(16, 185, 129), font=font_subbrand)

    # Device & Live Badge
    badge_text = f"● INFERENCE STREAM  |  ENGINE: {device_name.upper()}"
    badge_w = 340
    badge_x = canvas_w - 40 - badge_w
    draw.rounded_rectangle(
        [badge_x, 26, badge_x + badge_w, 60],
        radius=6,
        fill=(19, 27, 38),
        outline=(35, 48, 68),
        width=1,
    )
    draw.text((badge_x + 16, 35), badge_text, fill=(52, 211, 153), font=font_badge)

    # Header separator rule
    draw.line([(40, 76), (canvas_w - 40, 76)], fill=(30, 41, 59), width=1)

    # 2. Left Video Viewport
    # Video container dimensions
    # Let video be 1280x720 placed at (40, 100)
    vid_x, vid_y = 40, 100
    vid_w, vid_h = 1280, 720

    # Create overlay on frame_rgb (red semi-transparent overlay: rgb(239, 68, 68), alpha 0.40)
    overlay_rgb = _create_overlay(frame_rgb, mask_display, color=(239, 68, 68), alpha=0.40)
    overlay_pil = Image.fromarray(overlay_rgb)

    # Resize preserving aspect ratio to fit inside vid_w, vid_h
    orig_h, orig_w = frame_rgb.shape[:2]
    scale = min(vid_w / orig_w, vid_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    overlay_resized = overlay_pil.resize((new_w, new_h), Image.BILINEAR)

    # Compute offset inside container if any
    offset_x = vid_x + (vid_w - new_w) // 2
    offset_y = vid_y + (vid_h - new_h) // 2

    # Draw container background & border
    draw.rounded_rectangle(
        [vid_x - 1, vid_y - 1, vid_x + vid_w + 1, vid_y + vid_h + 1],
        radius=8,
        fill=(15, 23, 34),
        outline=(30, 41, 59),
        width=1,
    )
    canvas.paste(overlay_resized, (offset_x, offset_y))

    # Inner HUD overlay badges on top of video
    # Top-left HUD badge: "512x512 RESNET-UNET LESION OVERLAY"
    hud_tl_text = "512×512 SEGMENTATION OVERLAY  (THRESHOLD: 0.50)"
    draw.rounded_rectangle(
        [offset_x + 16, offset_y + 16, offset_x + 390, offset_y + 44],
        radius=4,
        fill=(11, 15, 23),
        outline=(51, 65, 85),
        width=1,
    )
    draw.text((offset_x + 28, offset_y + 23), hud_tl_text, fill=(203, 213, 225), font=font_small)

    # Bottom-left HUD badge: timestamp and frame counter
    hud_bl_text = f"FRAME: {frame_idx:03d}/{total_frames:03d}   |   TIME: {timestamp_sec:04.1f}s   |   LATENCY: {inference_time_ms:.1f}ms"
    draw.rounded_rectangle(
        [offset_x + 16, offset_y + new_h - 44, offset_x + 440, offset_y + new_h - 16],
        radius=4,
        fill=(11, 15, 23),
        outline=(51, 65, 85),
        width=1,
    )
    draw.text((offset_x + 26, offset_y + new_h - 37), hud_bl_text, fill=(148, 163, 184), font=font_small)

    # 3. Right Telemetry Sidebar
    side_x = 1350
    side_w = 530

    # Extract classification & segmentation results
    clf = result.classification
    seg = result.segmentation

    # --- Card 1: Model Prediction ---
    c1_y = 100
    c1_h = 240
    draw.rounded_rectangle(
        [side_x, c1_y, side_x + side_w, c1_y + c1_h],
        radius=8,
        fill=(18, 24, 38),
        outline=(35, 48, 68),
        width=1,
    )
    # Title
    draw.text((side_x + 20, c1_y + 18), "MODEL PREDICTION", fill=(148, 163, 184), font=font_card_title)
    draw.line([(side_x + 20, c1_y + 42), (side_x + side_w - 20, c1_y + 42)], fill=(30, 41, 59), width=1)

    # Top prediction disease name
    pred_name = clf.predicted_class
    draw.text((side_x + 20, c1_y + 54), pred_name, fill=(248, 250, 252), font=font_pred_large)

    # Confidence bar
    conf_pct = clf.confidence * 100.0
    conf_label = f"Confidence: {conf_pct:.1f}%"
    draw.text((side_x + 20, c1_y + 92), conf_label, fill=(16, 185, 129), font=font_body_bold)

    # Bar track
    bar_x = side_x + 20
    bar_y = c1_y + 116
    bar_w = side_w - 40
    bar_h = 8
    draw.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=4, fill=(30, 41, 59))
    fill_w = max(4, int(bar_w * clf.confidence))
    draw.rounded_rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], radius=4, fill=(16, 185, 129))

    # Second candidate
    top2_entry = clf.top2[1]
    top2_pct = top2_entry.confidence * 100.0
    draw.text(
        (side_x + 20, c1_y + 140),
        f"2nd Candidate:  {top2_entry.class_name} ({top2_pct:.1f}%)",
        fill=(148, 163, 184),
        font=font_body,
    )

    # Top1-Top2 margin & Entropy
    metrics_str = f"Margin: +{clf.top1_top2_margin*100:.1f}%    |    Entropy: {clf.normalized_entropy:.3f}"
    draw.text((side_x + 20, c1_y + 168), metrics_str, fill=(100, 116, 139), font=font_small)

    # Architecture note
    draw.text((side_x + 20, c1_y + 198), "Architecture: EfficientNet-B0 (Frozen, Epoch 19)", fill=(71, 85, 105), font=font_small)

    # --- Card 2: Lesion Analysis ---
    c2_y = 360
    c2_h = 240
    draw.rounded_rectangle(
        [side_x, c2_y, side_x + side_w, c2_y + c2_h],
        radius=8,
        fill=(18, 24, 38),
        outline=(35, 48, 68),
        width=1,
    )
    draw.text((side_x + 20, c2_y + 18), "LESION BURDEN ANALYSIS", fill=(148, 163, 184), font=font_card_title)
    draw.line([(side_x + 20, c2_y + 42), (side_x + side_w - 20, c2_y + 42)], fill=(30, 41, 59), width=1)

    # Large metric: Image-relative lesion burden %
    burden_str = f"{seg.burden_percent:.2f}%"
    draw.text((side_x + 20, c2_y + 54), burden_str, fill=(248, 250, 252), font=font_metric_large)
    draw.text((side_x + 160, c2_y + 64), "Image-Relative Lesion Area", fill=(148, 163, 184), font=font_body)

    # Burden bar (scaled relative to 25% max for visual nuance, or clamped)
    bar2_y = c2_y + 106
    draw.rounded_rectangle([bar_x, bar2_y, bar_x + bar_w, bar2_y + bar_h], radius=4, fill=(30, 41, 59))
    fill2_w = max(4, min(bar_w, int(bar_w * (seg.image_relative_lesion_burden / 0.25))))
    draw.rounded_rectangle([bar_x, bar2_y, bar_x + fill2_w, bar2_y + bar_h], radius=4, fill=(239, 68, 68))

    # Details
    draw.text(
        (side_x + 20, c2_y + 128),
        f"Active Lesion Pixels:  {seg.lesion_pixels:,} / {seg.total_pixels:,}",
        fill=(203, 213, 225),
        font=font_body,
    )
    draw.text(
        (side_x + 20, c2_y + 154),
        f"Mean Seg. Probability: {seg.mean_probability:.4f}    |    Max: {seg.max_probability:.4f}",
        fill=(148, 163, 184),
        font=font_small,
    )
    draw.text(
        (side_x + 20, c2_y + 180),
        "Native Measurement Grid: 512×512 (Frozen, Epoch 29)",
        fill=(100, 116, 139),
        font=font_small,
    )
    draw.text(
        (side_x + 20, c2_y + 204),
        "Burden is relative to total image area, not individual leaf boundaries.",
        fill=(71, 85, 105),
        font=font_small,
    )

    # --- Card 3: Scientific Notice & Limitations ---
    c3_y = 620
    c3_h = 200
    draw.rounded_rectangle(
        [side_x, c3_y, side_x + side_w, c3_y + c3_h],
        radius=8,
        fill=(15, 20, 30),
        outline=(30, 41, 59),
        width=1,
    )
    draw.text((side_x + 20, c3_y + 16), "DEMONSTRATION NOTICE", fill=(245, 158, 11), font=font_card_title)
    draw.line([(side_x + 20, c3_y + 38), (side_x + side_w - 20, c3_y + 38)], fill=(30, 41, 59), width=1)

    notice_lines = [
        "External video footage (Pexels ID 3664502). Ground-truth",
        "labels are unverified and footage may contain unsupported",
        "pathologies. Model predictions visualize pipeline behavior",
        "and lesion quantification on unconstrained foliage, and",
        "should not be interpreted as confirmed clinical diagnoses.",
    ]
    ny = c3_y + 48
    for nline in notice_lines:
        draw.text((side_x + 20, ny), nline, fill=(148, 163, 184), font=font_small)
        ny += 20

    # 4. Footer Bar
    draw.line([(40, 846), (canvas_w - 40, 846)], fill=(30, 41, 59), width=1)
    footer_left = "LeafSentinel: Leakage-Aware Plant Pathology AI  |  EfficientNet-B0 + ResNet18-U-Net  |  Independent Forward Passes"
    footer_right = "shazimjaved/Leaf-Sentinel"
    draw.text((40, 860), footer_left, fill=(100, 116, 139), font=font_small)
    draw.text((canvas_w - 220, 860), footer_right, fill=(100, 116, 139), font=font_small)

    return np.array(canvas)


def _resolve_arg(cli_value, config_value, default=None):
    """CLI argument > config value > default."""
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def main():
    parser = argparse.ArgumentParser(
        description="LeafSentinel — Professional Video Inference Engine"
    )
    parser.add_argument("--video", type=str, required=True, help="Input video path (MP4/MOV/AVI)")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Inference config path")
    parser.add_argument("--classifier-checkpoint", type=str, default=None, help="Phase 3 classifier checkpoint")
    parser.add_argument("--segmentation-checkpoint", type=str, default=None, help="Phase 2 segmenter checkpoint")
    parser.add_argument("--class-map", type=str, default=None, help="Phase 3 class_to_idx.json")
    parser.add_argument("--output-video", type=str, default="docs/assets/demo/leafsentinel_demo.mp4", help="Output video path")
    parser.add_argument("--output-dir", type=str, default="outputs/video_inference", help="Diagnostics output dir")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--sample-fps", type=float, default=5.0, help="Inference sampling frame rate (default: 5.0)")
    parser.add_argument("--duration", type=float, default=None, help="Max duration in seconds to process (default: all)")
    parser.add_argument("--frame-mode", type=str, default="sampled", choices=["sampled", "hold"],
                        help="sampled: output strictly independently inferred frames (default); hold: reuse latest mask between samples")
    args = parser.parse_args()

    # Load config if exists
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    ckpt_cfg = config.get("checkpoints", {})
    clf_ckpt = _resolve_arg(args.classifier_checkpoint, ckpt_cfg.get("classifier"))
    seg_ckpt = _resolve_arg(args.segmentation_checkpoint, ckpt_cfg.get("segmenter"))
    class_map_path = _resolve_arg(args.class_map, config.get("class_map"))

    if not clf_ckpt or not Path(clf_ckpt).exists():
        logger.error(f"Classifier checkpoint not found: {clf_ckpt}")
        sys.exit(1)
    if not seg_ckpt or not Path(seg_ckpt).exists():
        logger.error(f"Segmentation checkpoint not found: {seg_ckpt}")
        sys.exit(1)
    if not class_map_path or not Path(class_map_path).exists():
        logger.error(f"Class map not found: {class_map_path}")
        sys.exit(1)

    # Validate checkpoints metadata
    logger.info("Validating checkpoint metadata...")
    clf_raw = torch.load(clf_ckpt, map_location="cpu", weights_only=False)
    clf_epoch = clf_raw.get("epoch", None)
    clf_macro_f1 = clf_raw.get("best_val_macro_f1", None)
    logger.info(f"Classifier checkpoint metadata: epoch={clf_epoch}, best_val_macro_f1={clf_macro_f1}")

    if clf_epoch != 19:
        logger.error(
            f"CRITICAL: Classifier checkpoint reported epoch {clf_epoch}, expected epoch 19! "
            "Do not run inference with smoke test or wrong epoch checkpoints."
        )
        sys.exit(1)

    seg_raw = torch.load(seg_ckpt, map_location="cpu", weights_only=False)
    seg_epoch = seg_raw.get("epoch", None)
    seg_dice = seg_raw.get("best_val_dice", None)
    seg_iou = seg_raw.get("best_val_iou", None)
    logger.info(f"Segmenter checkpoint metadata: epoch={seg_epoch}, best_val_dice={seg_dice}, best_val_iou={seg_iou}")

    if seg_epoch != 29:
        logger.warning(f"Segmenter checkpoint reported epoch {seg_epoch}, expected epoch 29.")

    # Device selection
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

    predictor = LeafSentinelPredictor(
        classifier=classifier,
        segmenter=segmenter,
        class_map=class_map,
        device=device,
        classification_abstain_threshold=config.get("classification_abstain_threshold", None),
    )
    logger.info("LeafSentinelPredictor initialized successfully.")

    # Open video
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Input video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        sys.exit(1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = orig_total_frames / orig_fps if orig_fps > 0 else 0.0

    logger.info(
        f"Video opened: {orig_w}x{orig_h} @ {orig_fps:.2f} FPS, "
        f"{orig_total_frames} frames ({video_duration:.2f}s)"
    )

    # Determine frames to process
    max_duration = args.duration if args.duration is not None else video_duration
    max_frames_to_read = min(orig_total_frames, int(max_duration * orig_fps))

    # Frame sampling interval
    sample_step = max(1, int(round(orig_fps / args.sample_fps)))
    sampled_indices = list(range(0, max_frames_to_read, sample_step))
    total_sampled = len(sampled_indices)

    logger.info(
        f"Processing plan: duration={max_duration:.2f}s, sample_fps={args.sample_fps}, "
        f"total_samples={total_sampled}, frame_mode={args.frame_mode}"
    )

    # Set up output directories
    out_video_path = Path(args.output_video)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine canvas resolution & output fps
    canvas_w, canvas_h = 1920, 1080
    out_fps = args.sample_fps if args.frame_mode == "sampled" else orig_fps

    # Temp raw video output path
    temp_video_path = out_dir / "temp_raw_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video_path), fourcc, out_fps, (canvas_w, canvas_h))

    # Per-frame telemetry records
    frame_records = []
    latencies = []
    rendered_frames = []

    sampled_set = set(sampled_indices)
    current_read_idx = 0
    latest_canvas_bgr = None
    sample_count = 0

    start_total_time = time.time()

    while current_read_idx < max_frames_to_read:
        ret, frame_bgr = cap.read()
        if not ret:
            logger.warning(f"Frame read stopped at index {current_read_idx}")
            break

        if current_read_idx in sampled_set:
            sample_count += 1
            timestamp = current_read_idx / orig_fps

            # Convert BGR to RGB PIL Image
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # Predict with artifacts (single forward pass per model)
            t0 = time.time()
            result, mask_display = predictor.predict_with_artifacts(pil_img)
            t_infer = (time.time() - t0) * 1000.0
            latencies.append(t_infer)

            # Render dashboard canvas
            canvas_rgb = render_dashboard_frame(
                frame_rgb=frame_rgb,
                mask_display=mask_display,
                result=result,
                frame_idx=sample_count,
                total_frames=total_sampled,
                timestamp_sec=timestamp,
                inference_time_ms=t_infer,
                device_name=str(device),
                canvas_w=canvas_w,
                canvas_h=canvas_h,
            )
            latest_canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)

            # Record metrics
            clf = result.classification
            seg = result.segmentation
            frame_records.append({
                "sample_idx": sample_count,
                "source_frame_idx": current_read_idx,
                "timestamp_sec": round(timestamp, 3),
                "predicted_class": clf.predicted_class,
                "confidence": round(clf.confidence, 4),
                "top2_class": clf.top2[1].class_name,
                "top2_confidence": round(clf.top2[1].confidence, 4),
                "top1_top2_margin": round(clf.top1_top2_margin, 4),
                "normalized_entropy": round(clf.normalized_entropy, 4),
                "burden_percent": round(seg.burden_percent, 4),
                "lesion_pixels": seg.lesion_pixels,
                "total_pixels": seg.total_pixels,
                "inference_ms": round(t_infer, 2),
            })

            # Store for deliverable thumbnails
            rendered_frames.append((sample_count, timestamp, canvas_rgb))

            if args.frame_mode == "sampled":
                writer.write(latest_canvas_bgr)

            if sample_count % 5 == 0 or sample_count == total_sampled:
                logger.info(
                    f"Sample [{sample_count}/{total_sampled}] @ {timestamp:.2f}s: "
                    f"{clf.predicted_class} ({clf.confidence*100:.1f}%), "
                    f"burden={seg.burden_percent:.2f}%, latency={t_infer:.1f}ms"
                )

        elif args.frame_mode == "hold" and latest_canvas_bgr is not None:
            writer.write(latest_canvas_bgr)

        current_read_idx += 1

    cap.release()
    writer.release()
    total_processing_time = time.time() - start_total_time

    logger.info(
        f"Inference complete: processed {sample_count} sampled frames in {total_processing_time:.2f}s "
        f"({sample_count / total_processing_time:.2f} samples/sec)"
    )

    # Encode with FFmpeg (H.264)
    ffmpeg_available = shutil.which("ffmpeg") is not None
    if ffmpeg_available:
        logger.info(f"Transcoding {temp_video_path} to H.264 {out_video_path} via FFmpeg...")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_video_path),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_video_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and out_video_path.exists():
            logger.info(f"H.264 encoding successful: {out_video_path} ({out_video_path.stat().st_size / (1024*1024):.2f} MB)")
            try:
                temp_video_path.unlink()
            except Exception:
                pass
        else:
            logger.warning(f"FFmpeg transcode failed (exit {res.returncode}), using raw mp4v fallback.")
            shutil.copy(str(temp_video_path), str(out_video_path))
    else:
        logger.warning("FFmpeg not found in PATH, using raw MP4 output.")
        shutil.copy(str(temp_video_path), str(out_video_path))

    # Save deliverables (thumbnails and key frames)
    # Target deliverables:
    # docs/assets/demo/demo_thumbnail.png
    # docs/assets/demo/demo_frame_01.png
    # docs/assets/demo/demo_frame_02.png
    # docs/assets/demo/demo_frame_03.png
    demo_assets_dir = out_video_path.parent
    demo_assets_dir.mkdir(parents=True, exist_ok=True)

    if rendered_frames:
        # Thumbnail: choose middle or representative frame
        mid_idx = len(rendered_frames) // 2
        thumb_rgb = rendered_frames[mid_idx][2]
        Image.fromarray(thumb_rgb).save(demo_assets_dir / "demo_thumbnail.png")
        logger.info(f"Saved {demo_assets_dir / 'demo_thumbnail.png'}")

        # Frame 1: ~20% point
        idx_01 = max(0, int(len(rendered_frames) * 0.20))
        Image.fromarray(rendered_frames[idx_01][2]).save(demo_assets_dir / "demo_frame_01.png")

        # Frame 2: ~50% point
        idx_02 = max(0, int(len(rendered_frames) * 0.50))
        Image.fromarray(rendered_frames[idx_02][2]).save(demo_assets_dir / "demo_frame_02.png")

        # Frame 3: ~80% point
        idx_03 = max(0, min(len(rendered_frames) - 1, int(len(rendered_frames) * 0.80)))
        Image.fromarray(rendered_frames[idx_03][2]).save(demo_assets_dir / "demo_frame_03.png")

        logger.info(f"Saved keyframes: demo_frame_01.png, demo_frame_02.png, demo_frame_03.png")

    # Save CSV of per-frame predictions
    csv_path = out_dir / "per_sampled_frame_predictions.csv"
    if frame_records:
        keys = list(frame_records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(frame_records)
        logger.info(f"Saved per-frame predictions to {csv_path}")

    # Compute execution statistics
    disease_counts = {}
    confs = [r["confidence"] for r in frame_records]
    burdens = [r["burden_percent"] for r in frame_records]
    for r in frame_records:
        d = r["predicted_class"]
        disease_counts[d] = disease_counts.get(d, 0) + 1

    summary = {
        "execution_summary": {
            "source_video": str(video_path),
            "video_duration_seconds": round(video_duration, 2),
            "processed_duration_seconds": round(max_duration, 2),
            "original_resolution": [orig_w, orig_h],
            "original_fps": round(orig_fps, 2),
            "sampled_frames_count": sample_count,
            "sample_fps": args.sample_fps,
            "frame_mode": args.frame_mode,
            "device": str(device),
            "total_processing_seconds": round(total_processing_time, 2),
            "effective_sampling_fps": round(sample_count / total_processing_time, 2),
            "latency_ms": {
                "mean": round(float(np.mean(latencies)), 2) if latencies else 0,
                "median": round(float(np.median(latencies)), 2) if latencies else 0,
                "p95": round(float(np.percentile(latencies, 95)), 2) if latencies else 0,
                "min": round(float(np.min(latencies)), 2) if latencies else 0,
                "max": round(float(np.max(latencies)), 2) if latencies else 0,
            },
        },
        "model_checkpoints": {
            "classifier": {
                "path": str(clf_ckpt),
                "architecture": "EfficientNet-B0",
                "verified_epoch": clf_epoch,
                "best_val_macro_f1": clf_macro_f1,
            },
            "segmenter": {
                "path": str(seg_ckpt),
                "architecture": "ResNet18-U-Net",
                "verified_epoch": seg_epoch,
                "best_val_dice": seg_dice,
                "best_val_iou": seg_iou,
                "grid_size": [512, 512],
                "binary_threshold": 0.50,
            },
        },
        "predictions_distribution": {
            "disease_frequency": disease_counts,
            "confidence_stats": {
                "mean": round(float(np.mean(confs)), 4) if confs else 0,
                "std": round(float(np.std(confs)), 4) if confs else 0,
                "min": round(float(np.min(confs)), 4) if confs else 0,
                "max": round(float(np.max(confs)), 4) if confs else 0,
            },
            "burden_percent_stats": {
                "mean": round(float(np.mean(burdens)), 4) if burdens else 0,
                "std": round(float(np.std(burdens)), 4) if burdens else 0,
                "min": round(float(np.min(burdens)), 4) if burdens else 0,
                "max": round(float(np.max(burdens)), 4) if burdens else 0,
            },
        },
        "scientific_disclaimer": (
            "Predictions and lesion burden estimates on this external demonstration footage "
            "are illustrative visualizations of model inference and do NOT constitute a "
            "scientifically validated disease-severity measurement or verified medical diagnosis."
        ),
    }

    summary_path = out_dir / "inference_execution_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved execution summary to {summary_path}")

    # Checkpoint metadata standalone JSON
    ckpt_meta_path = out_dir / "model_checkpoint_metadata.json"
    with open(ckpt_meta_path, "w", encoding="utf-8") as f:
        json.dump(summary["model_checkpoints"], f, indent=2)
    logger.info(f"Saved checkpoint metadata to {ckpt_meta_path}")

    print("\n" + "=" * 60)
    print("LeafSentinel Video Inference Complete")
    print(f"Output Video: {out_video_path} ({out_video_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Sampled Frames: {sample_count} @ {args.sample_fps} FPS")
    print(f"Mean Latency: {summary['execution_summary']['latency_ms']['mean']} ms / frame")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
