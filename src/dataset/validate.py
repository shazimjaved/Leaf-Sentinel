"""Incremental image and segmentation mask validation for Leaf Sentinel (Phase 1).

Performs robust integrity checks on images and masks incrementally without
overloading system RAM, recording corruptions, dimension mismatches,
mask conventions, and affected-area ratios.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Stores full validation metrics and error records."""
    total_records: int = 0
    valid_images: int = 0
    missing_images: int = 0
    corrupt_images: int = 0
    unusual_images: int = 0
    total_masks_referenced: int = 0
    valid_masks: int = 0
    missing_masks: int = 0
    corrupt_masks: int = 0
    dimension_mismatch_masks: int = 0
    empty_masks: int = 0
    ratio_discrepancies: int = 0
    validated_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation summary to serializable dictionary."""
        return {
            "total_records": self.total_records,
            "valid_images": self.valid_images,
            "missing_images": self.missing_images,
            "corrupt_images": self.corrupt_images,
            "unusual_images": self.unusual_images,
            "total_masks_referenced": self.total_masks_referenced,
            "valid_masks": self.valid_masks,
            "missing_masks": self.missing_masks,
            "corrupt_masks": self.corrupt_masks,
            "dimension_mismatch_masks": self.dimension_mismatch_masks,
            "empty_masks": self.empty_masks,
            "ratio_discrepancies": self.ratio_discrepancies,
            "summary": self.summary
        }


def _validate_single_image(img_path_str: Optional[str], max_aspect_ratio: float = 5.0) -> Tuple[bool, Optional[int], Optional[int], Optional[int], Optional[str], Optional[str]]:
    """Validate a single image file.
    
    Returns:
        (is_valid, width, height, channels, unusual_flag, error_msg)
    """
    if not img_path_str:
        return False, None, None, None, None, "Missing image path in record"

    p = Path(img_path_str)
    if not p.exists():
        return False, None, None, None, None, f"File does not exist: {p}"

    if p.stat().st_size == 0:
        return False, None, None, None, None, "Zero-byte file"

    try:
        with Image.open(p) as img:
            img.verify()  # Fast structural verification
        
        # Re-open to read dimensions & mode (verify closes the file)
        with Image.open(p) as img:
            w, h = img.size
            mode = img.mode
            channels = len(img.getbands())

            # Check aspect ratio
            aspect = max(w / h, h / w) if h > 0 and w > 0 else float('inf')
            unusual = None
            if aspect > max_aspect_ratio:
                unusual = f"Extreme aspect ratio ({aspect:.2f})"
            elif min(w, h) < 32:
                unusual = f"Very low resolution ({w}x{h})"

            return True, w, h, channels, unusual, None

    except Exception as e:
        return False, None, None, None, None, f"Image decoding error: {str(e)}"


def _validate_single_mask(
    mask_path_str: Optional[str],
    expected_w: Optional[int],
    expected_h: Optional[int],
    meta_ratio: Optional[float] = None,
    ratio_tolerance: float = 0.05,
    check_components: bool = True
) -> Tuple[bool, Optional[str], int, float, int, Optional[str], Optional[str]]:
    """Validate a segmentation mask and compute area metrics.
    
    Returns:
        (is_valid_mask, convention, affected_pixels, affected_ratio, num_components, discrepancy_note, error_msg)
    """
    if not mask_path_str:
        return False, None, 0, 0.0, 0, None, "Missing mask path"

    p = Path(mask_path_str)
    if not p.exists():
        return False, None, 0, 0.0, 0, None, f"Mask file does not exist: {p}"

    if p.stat().st_size == 0:
        return False, None, 0, 0.0, 0, None, "Zero-byte mask file"

    try:
        # Load mask via OpenCV / PIL for pixel analysis
        mask_arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if mask_arr is None:
            # Fallback to PIL
            with Image.open(p) as img:
                mask_arr = np.array(img)

        if mask_arr is None or mask_arr.size == 0:
            return False, None, 0, 0.0, 0, None, "Failed to decode mask array"

        # Check dimensions
        if mask_arr.ndim == 3:
            mh, mw = mask_arr.shape[:2]
            # Convert multi-channel mask to single channel
            if mask_arr.shape[2] == 4:
                # RGBA
                mask_2d = mask_arr[:, :, 0]
            else:
                mask_2d = cv2.cvtColor(mask_arr, cv2.COLOR_BGR2GRAY)
        else:
            mh, mw = mask_arr.shape
            mask_2d = mask_arr

        if expected_w is not None and expected_h is not None:
            if mw != expected_w or mh != expected_h:
                return False, "dimension_mismatch", 0, 0.0, 0, None, (
                    f"Dimension mismatch: mask is {mw}x{mh}, image is {expected_w}x{expected_h}"
                )

        unique_vals = np.unique(mask_2d)
        total_pixels = mw * mh

        # Identify convention
        if set(unique_vals).issubset({0, 1}):
            convention = "binary_0_1"
            binary_mask = (mask_2d > 0).astype(np.uint8)
        elif set(unique_vals).issubset({0, 255}):
            convention = "binary_0_255"
            binary_mask = (mask_2d > 0).astype(np.uint8)
        else:
            convention = f"multivalue_max_{int(unique_vals.max())}"
            binary_mask = (mask_2d > 0).astype(np.uint8)

        affected_pixels = int(np.sum(binary_mask))
        affected_ratio = float(affected_pixels / total_pixels) if total_pixels > 0 else 0.0

        # Connected components (lesion clusters)
        num_components = 0
        if check_components and affected_pixels > 0:
            try:
                num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                num_components = max(0, num_labels - 1)  # subtract background
            except Exception:
                num_components = 1

        # Check metadata ratio discrepancy
        discrepancy = None
        if meta_ratio is not None and not np.isnan(meta_ratio):
            diff = abs(affected_ratio - meta_ratio)
            if diff > ratio_tolerance:
                discrepancy = f"Calculated ratio ({affected_ratio:.4f}) differs from metadata ({meta_ratio:.4f}) by {diff:.4f}"

        is_valid = True
        return is_valid, convention, affected_pixels, affected_ratio, num_components, discrepancy, None

    except Exception as e:
        return False, None, 0, 0.0, 0, None, f"Mask processing exception: {str(e)}"


def validate_dataset(
    records_df: pd.DataFrame,
    max_aspect_ratio: float = 5.0,
    ratio_tolerance: float = 0.05,
    check_components: bool = True,
    max_samples: Optional[int] = None
) -> ValidationReport:
    """Incrementally validate all images and masks in records DataFrame.
    
    Args:
        records_df: Standardized DataFrame from dataset discovery.
        max_aspect_ratio: Threshold to flag unusual aspect ratios.
        ratio_tolerance: Allowed difference between calculated and metadata mask ratio.
        check_components: Whether to compute connected lesion components.
        max_samples: Optional limit for testing/dry runs.
        
    Returns:
        ValidationReport containing validated DataFrame, errors DataFrame, and summary.
    """
    logger.info(f"Starting incremental validation of {len(records_df)} records...")
    
    df = records_df.copy()
    if max_samples is not None and max_samples > 0:
        df = df.iloc[:max_samples].copy()

    total_records = len(df)
    if total_records == 0:
        logger.warning("Empty records DataFrame provided for validation.")
        return ValidationReport()

    validated_rows = []
    error_rows = []

    valid_images = 0
    missing_images = 0
    corrupt_images = 0
    unusual_images = 0

    total_masks_ref = 0
    valid_masks = 0
    missing_masks = 0
    corrupt_masks = 0
    dim_mismatch_masks = 0
    empty_masks = 0
    ratio_discrepancies = 0

    for idx, row in tqdm(df.iterrows(), total=total_records, desc="Validating dataset samples"):
        sample_id = row.get("sample_id", f"sample_{idx:06d}")
        img_path = row.get("image_path")
        mask_path = row.get("mask_path")
        host = row.get("host", "Unknown")
        disease = row.get("disease", "Unknown")
        split = row.get("split", "unassigned")
        meta_ratio = row.get("meta_mask_ratio")

        # Validate image
        is_val_img, w, h, ch, unusual_flag, img_err = _validate_single_image(
            img_path, max_aspect_ratio=max_aspect_ratio
        )

        if is_val_img:
            valid_images += 1
            if unusual_flag:
                unusual_images += 1
        else:
            if img_err and "does not exist" in img_err:
                missing_images += 1
            else:
                corrupt_images += 1

        # Validate mask
        is_val_mask = False
        convention = None
        affected_px = 0
        affected_ratio = 0.0
        num_components = 0
        discrepancy_note = None
        mask_err = None

        if mask_path and pd.notna(mask_path):
            total_masks_ref += 1
            is_val_mask, convention, affected_px, affected_ratio, num_components, discrepancy_note, mask_err = _validate_single_mask(
                mask_path,
                expected_w=w,
                expected_h=h,
                meta_ratio=meta_ratio,
                ratio_tolerance=ratio_tolerance,
                check_components=check_components
            )

            if is_val_mask:
                valid_masks += 1
                if affected_px == 0:
                    empty_masks += 1
                if discrepancy_note:
                    ratio_discrepancies += 1
            else:
                if mask_err and "does not exist" in mask_err:
                    missing_masks += 1
                elif mask_err and "Dimension mismatch" in mask_err:
                    dim_mismatch_masks += 1
                else:
                    corrupt_masks += 1

        # Record validated info
        record = {
            "sample_id": sample_id,
            "image_path": img_path,
            "mask_path": mask_path,
            "host": host,
            "disease": disease,
            "split": split,
            "is_valid_image": is_val_img,
            "width": w,
            "height": h,
            "channels": ch,
            "unusual_image_flag": unusual_flag,
            "is_valid_mask": is_val_mask,
            "mask_convention": convention,
            "affected_pixels": affected_px,
            "affected_area_ratio": affected_ratio,
            "num_lesion_components": num_components,
            "meta_mask_ratio": meta_ratio,
            "ratio_discrepancy_note": discrepancy_note,
            "image_error": img_err,
            "mask_error": mask_err
        }
        validated_rows.append(record)

        if img_err or mask_err or unusual_flag or discrepancy_note:
            error_rows.append({
                "sample_id": sample_id,
                "image_path": img_path,
                "mask_path": mask_path,
                "image_error": img_err,
                "mask_error": mask_err,
                "unusual_image_flag": unusual_flag,
                "ratio_discrepancy_note": discrepancy_note
            })

    validated_df = pd.DataFrame(validated_rows)
    errors_df = pd.DataFrame(error_rows) if error_rows else pd.DataFrame(columns=[
        "sample_id", "image_path", "mask_path", "image_error", "mask_error", "unusual_image_flag", "ratio_discrepancy_note"
    ])

    summary = {
        "total_records_processed": total_records,
        "valid_images": valid_images,
        "missing_images": missing_images,
        "corrupt_images": corrupt_images,
        "unusual_images": unusual_images,
        "total_masks_referenced": total_masks_ref,
        "valid_masks": valid_masks,
        "missing_masks": missing_masks,
        "corrupt_masks": corrupt_masks,
        "dimension_mismatch_masks": dim_mismatch_masks,
        "empty_masks": empty_masks,
        "ratio_discrepancies": ratio_discrepancies,
        "error_records_count": len(errors_df)
    }

    logger.info(
        f"Validation complete: {valid_images}/{total_records} valid images, "
        f"{valid_masks}/{total_masks_ref} valid masks, {len(errors_df)} issues flagged."
    )

    return ValidationReport(
        total_records=total_records,
        valid_images=valid_images,
        missing_images=missing_images,
        corrupt_images=corrupt_images,
        unusual_images=unusual_images,
        total_masks_referenced=total_masks_ref,
        valid_masks=valid_masks,
        missing_masks=missing_masks,
        corrupt_masks=corrupt_masks,
        dimension_mismatch_masks=dim_mismatch_masks,
        empty_masks=empty_masks,
        ratio_discrepancies=ratio_discrepancies,
        validated_df=validated_df,
        errors_df=errors_df,
        summary=summary
    )
