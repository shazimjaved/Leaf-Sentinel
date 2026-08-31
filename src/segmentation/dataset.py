"""PyTorch Dataset and DataLoader for PlantSeg lesion segmentation in LeafSentinel.

Features:
- Manifest-driven loading preserving split provenance and metadata.
- Clean binary mask conversion (0 = background, 1 = lesion).
- Strict separation of synchronized spatial transforms (image + mask) and photometric transforms (image only).
- ImageNet normalization.
"""

import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PlantLesionDataset(Dataset):
    """PyTorch Dataset for plant leaf image and lesion segmentation mask loading."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split: str = "train",
        image_size: int = 512,
        is_training: bool = True,
        max_samples: Optional[int] = None
    ):
        super().__init__()
        self.split = split.lower()
        self.image_size = image_size
        self.is_training = is_training

        # Filter by split
        if "phase2_split" in manifest_df.columns:
            self.df = manifest_df[manifest_df["phase2_split"] == self.split].copy().reset_index(drop=True)
        else:
            self.df = manifest_df.copy().reset_index(drop=True)

        if max_samples is not None and max_samples > 0:
            self.df = self.df.iloc[:max_samples].copy().reset_index(drop=True)

        logger.info(f"Loaded PlantLesionDataset ({self.split}): {len(self.df)} samples, image_size={image_size}, is_training={is_training}.")

    def __len__(self) -> int:
        return len(self.df)

    def _apply_augmentations(self, img_pil: Image.Image, mask_pil: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """Apply synchronized spatial transforms to (image, mask) and photometric transforms to (image only)."""
        # 1. Synchronized Spatial Transforms
        # Random Horizontal Flip
        if random.random() > 0.5:
            img_pil = TF.hflip(img_pil)
            mask_pil = TF.hflip(mask_pil)

        # Random Vertical Flip
        if random.random() > 0.5:
            img_pil = TF.vflip(img_pil)
            mask_pil = TF.vflip(mask_pil)

        # Random Rotation (-15 to +15 degrees)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            img_pil = TF.rotate(img_pil, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask_pil = TF.rotate(mask_pil, angle, interpolation=TF.InterpolationMode.NEAREST)

        # 2. Photometric Transforms (Image ONLY, NEVER on mask)
        # Random Brightness
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            enhancer = ImageEnhance.Brightness(img_pil)
            img_pil = enhancer.enhance(factor)

        # Random Contrast
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            enhancer = ImageEnhance.Contrast(img_pil)
            img_pil = enhancer.enhance(factor)

        # Random Color Saturation
        if random.random() > 0.5:
            factor = random.uniform(0.9, 1.1)
            enhancer = ImageEnhance.Color(img_pil)
            img_pil = enhancer.enhance(factor)

        return img_pil, mask_pil

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"]) if pd.notna(row.get("mask_path")) else None
        is_healthy = bool(row.get("is_healthy", False))

        # 1. Load RGB Image
        try:
            with Image.open(img_path) as img:
                img_pil = img.convert("RGB")
                orig_w, orig_h = img_pil.size
        except Exception as e:
            logger.error(f"Error reading image {img_path}: {e}")
            img_pil = Image.new("RGB", (self.image_size, self.image_size), color=(0, 0, 0))
            orig_w, orig_h = self.image_size, self.image_size

        # 2. Load Mask (or create zero mask for healthy)
        if is_healthy or mask_path is None or not mask_path.exists():
            mask_arr = np.zeros((orig_h, orig_w), dtype=np.uint8)
        else:
            try:
                m_arr = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if m_arr is None:
                    with Image.open(mask_path) as m_img:
                        m_arr = np.array(m_img)

                if m_arr.ndim == 3:
                    if m_arr.shape[2] == 4:
                        m_arr = m_arr[:, :, 0]
                    else:
                        m_arr = cv2.cvtColor(m_arr, cv2.COLOR_BGR2GRAY)

                # Binary conversion: 0 = background, 1 = lesion
                mask_arr = (m_arr > 0).astype(np.uint8)
            except Exception as e:
                logger.error(f"Error loading mask {mask_path}: {e}")
                mask_arr = np.zeros((orig_h, orig_w), dtype=np.uint8)

        mask_pil = Image.fromarray(mask_arr)

        # 3. Resize to target resolution
        img_pil = img_pil.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask_pil = mask_pil.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        # 4. Apply synchronized augmentations during training
        if self.is_training:
            img_pil, mask_pil = self._apply_augmentations(img_pil, mask_pil)

        # 5. Convert to PyTorch tensors
        # Image: (3, H, W), normalized
        img_tensor = TF.to_tensor(img_pil)
        img_tensor = TF.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Mask: (1, H, W), float32 {0.0, 1.0}
        mask_np = np.array(mask_pil, dtype=np.float32)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # (1, H, W)
        mask_tensor = (mask_tensor > 0.5).float()

        metadata = {
            "image_id": str(row.get("image_id", f"sample_{idx:05d}")),
            "host": str(row.get("host", "Unknown")),
            "disease": str(row.get("disease", "Unknown")),
            "display_class": str(row.get("display_class", row.get("disease", "Unknown"))),
            "is_healthy": is_healthy,
            "original_split": str(row.get("original_split", "unassigned")),
            "phase2_split": str(row.get("phase2_split", self.split)),
            "duplicate_group_id": str(row.get("duplicate_group_id", "grp_00000")),
            "original_width": orig_w,
            "original_height": orig_h,
            "image_path": str(img_path),
            "mask_path": str(mask_path) if mask_path else ""
        }

        return img_tensor, mask_tensor, metadata


def get_dataloaders(
    config: Dict[str, Any],
    manifest_path: Optional[Path | str] = None,
    image_size_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    max_samples_override: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    """Build train, val, test PyTorch DataLoaders from manifest.csv."""
    if manifest_path is None:
        dataset_dir = Path(config.get("paths", {}).get("dataset_dir", "outputs/phase2/dataset"))
        manifest_path = dataset_dir / "manifest.csv"

    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(
            f"Phase 2 manifest not found at {manifest_p}. "
            "Please run 'python scripts/prepare_phase2.py' before training or evaluation."
        )

    manifest_df = pd.read_csv(manifest_p)
    logger.info(f"Loaded manifest from {manifest_p}: {len(manifest_df)} total records.")

    train_cfg = config.get("training", {})
    ds_cfg = config.get("dataset", {})

    image_size = image_size_override or int(ds_cfg.get("image_size", 512))
    batch_size = batch_size_override or int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 0))

    train_ds = PlantLesionDataset(
        manifest_df=manifest_df,
        split="train",
        image_size=image_size,
        is_training=True,
        max_samples=max_samples_override
    )
    val_ds = PlantLesionDataset(
        manifest_df=manifest_df,
        split="val",
        image_size=image_size,
        is_training=False,
        max_samples=max_samples_override
    )
    test_ds = PlantLesionDataset(
        manifest_df=manifest_df,
        split="test",
        image_size=image_size,
        is_training=False,
        max_samples=max_samples_override
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, manifest_df
