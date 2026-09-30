import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
from pathlib import Path
import json
import numpy as np
import logging

logger = logging.getLogger("LeafSentinel.Phase3.Dataset")

class PadToSquare:
    """Pad an image to make it square, preserving aspect ratio."""
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        pad_left = (max_wh - w) // 2
        pad_top = (max_wh - h) // 2
        pad_right = max_wh - w - pad_left
        pad_bottom = max_wh - h - pad_top
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill, padding_mode='constant')

class ClassificationDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        split: str,
        input_mode: str = "full_rgb",
        image_size: int = 224,
        lesion_margin: float = 0.15
    ):
        """
        LeafSentinel Phase 3 Classification Dataset.
        
        Args:
            manifest_path: Path to the classification_manifest.csv
            dataset_root: Root directory of PlantSeg dataset
            split: 'train', 'val', or 'test' (must match benchmark_split)
            input_mode: 'full_rgb' (primary) or 'lesion_crop_gt' (oracle experiment)
            image_size: Target image size (default 224)
            lesion_margin: Margin ratio for lesion crop mode
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.input_mode = input_mode
        self.image_size = image_size
        self.lesion_margin = lesion_margin
        
        if self.input_mode not in ["full_rgb", "lesion_crop_gt"]:
            raise ValueError(f"Unknown input_mode {self.input_mode}")
            
        df = pd.read_csv(manifest_path)
        
        # Determine train/val/test membership strictly by benchmark_split
        self.df = df[df["benchmark_split"] == split].reset_index(drop=True)
        
        if len(self.df) == 0:
            logger.warning(f"No samples found for split '{split}' in manifest {manifest_path}")

        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        # Base normalizations for ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if self.split == 'train':
            # Agricultural conservative augmentations requested by user
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                PadToSquare(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            return transforms.Compose([
                PadToSquare(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalize
            ])

    def _get_lesion_crop(self, image: Image.Image, mask_path: Path) -> Image.Image:
        """Oracle lesion-aware crop (ablation protocol)."""
        if not mask_path.exists():
            # Fallback to full image if mask missing
            return image
            
        mask = Image.open(mask_path).convert("L")
        mask_np = np.array(mask)
        
        # Find bounding box of lesion
        coords = np.argwhere(mask_np > 0)
        if len(coords) == 0:
            # Safe handling for empty mask
            return image
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add margin
        h, w = mask_np.shape
        crop_h, crop_w = y_max - y_min, x_max - x_min
        
        margin_y = int(crop_h * self.lesion_margin)
        margin_x = int(crop_w * self.lesion_margin)
        
        y_min = max(0, y_min - margin_y)
        y_max = min(h, y_max + margin_y)
        x_min = max(0, x_min - margin_x)
        x_max = min(w, x_max + margin_x)
        
        return image.crop((x_min, y_min, x_max, y_max))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Resolve paths dynamically relative to dataset_root
        img_path = self.dataset_root / row["image_relpath"]
        
        if not img_path.exists():
            raise FileNotFoundError(
                f"Classification image not found: {img_path}"
            )

        image = Image.open(img_path).convert("RGB")
            
        if self.input_mode == "lesion_crop_gt":
            mask_path = self.dataset_root / row["mask_relpath"]
            image = self._get_lesion_crop(image, mask_path)
            
        if self.transform:
            image = self.transform(image)
            
        class_idx = int(row["class_index"])
        
        return image, class_idx
