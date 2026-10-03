import logging
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

logger = logging.getLogger("LeafSentinel.Phase4.Dataset")

class ImageRelativeBurdenDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        split: str,
        image_size: int = 224,
    ):
        """
        LeafSentinel Phase 4 Severity Dataset.
        
        Target: Image-Relative Lesion Burden (mask_area_ratio)
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        
        df = pd.read_csv(manifest_path)
        self.df = df[df["benchmark_split"] == split].reset_index(drop=True)
        
        if len(self.df) == 0:
            logger.warning(f"No samples found for split '{split}' in manifest {manifest_path}")

        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        # Base normalizations for ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if self.split == 'train':
            # Note on Phase 4 vs Phase 3 augmentations:
            # Phase 4 target is a pixel-area fraction. Arbitrary rotations (e.g. 15 deg)
            # would require padding (changing the background area) or cropping (changing
            # the lesion fraction). Therefore, we only use flips and photometric jitter.
            # We use deterministic resize which distorts aspect ratio but preserves the mathematically exact 
            # area fraction of lesion-to-image.
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
                transforms.ToTensor(),
                normalize
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = self.dataset_root / row["image_relpath"]
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image {img_path}: {e}")
            
        if self.transform:
            image = self.transform(image)
            
        # Target: Image-Relative Lesion Burden [0, 1]
        target = float(row["mask_area_ratio"])
        
        # Returns shape (1,) tensor to match model output shape (B, 1)
        return image, torch.tensor([target], dtype=torch.float32)
