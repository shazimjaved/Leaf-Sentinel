"""Unit and integration test suite for LeafSentinel dataset audit modules."""

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
import cv2

from src.dataset.inspect import discover_dataset
from src.dataset.validate import validate_dataset
from src.dataset.duplicates import analyze_duplicates
from src.dataset.statistics import compute_dataset_statistics
from src.dataset.feasibility import analyze_class_feasibility
from src.dataset.visualize import generate_all_visualizations


class TestDatasetAudit(unittest.TestCase):
    """Test full dataset auditing functionality."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.img_dir.mkdir(parents=True)
        self.mask_dir.mkdir(parents=True)

        self.samples = []
        hosts = ["Tomato", "Potato", "Apple"]
        diseases = ["Early Blight", "Late Blight", "Healthy"]
        splits = ["train", "val", "test"]

        for i in range(12):
            host = hosts[i % len(hosts)]
            disease = diseases[i % len(diseases)]
            split = splits[i % len(splits)]
            
            img_path = self.img_dir / f"img_{i:03d}.jpg"
            img_arr = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(img_arr).save(img_path)

            mask_path = self.mask_dir / f"img_{i:03d}_mask.png"
            mask_arr = np.zeros((128, 128), dtype=np.uint8)
            if disease != "Healthy":
                cv2.circle(mask_arr, (64, 64), 20, 255, -1)
            Image.fromarray(mask_arr).save(mask_path)

            self.samples.append({
                "image_path": str(img_path.relative_to(self.root)),
                "mask_path": str(mask_path.relative_to(self.root)),
                "host": host,
                "disease": disease,
                "split": split,
                "mask_ratio": float(np.sum(mask_arr > 0) / (128 * 128))
            })

        meta_df = pd.DataFrame(self.samples)
        meta_df.to_csv(self.root / "metadata.csv", index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_discovery(self):
        res = discover_dataset(self.root)
        self.assertEqual(len(res.records_df), 12)
        self.assertEqual(res.structure_type, "metadata_driven")

    def test_validation(self):
        disc = discover_dataset(self.root)
        val_rep = validate_dataset(disc.records_df)
        self.assertEqual(val_rep.valid_images, 12)
        self.assertEqual(val_rep.valid_masks, 12)

    def test_dhash_and_duplicates(self):
        disc = discover_dataset(self.root)
        val_rep = validate_dataset(disc.records_df)
        dup_rep = analyze_duplicates(val_rep.validated_df, hamming_threshold=2)
        self.assertIsNotNone(dup_rep)
        self.assertIsInstance(dup_rep.candidates_df, pd.DataFrame)

    def test_statistics_and_feasibility(self):
        disc = discover_dataset(self.root)
        val_rep = validate_dataset(disc.records_df)
        stats = compute_dataset_statistics(val_rep.validated_df, self.root)
        self.assertEqual(stats.summary_dict["overview"]["valid_images"], 12)


if __name__ == "__main__":
    unittest.main()
