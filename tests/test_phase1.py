"""Unit and integration test suite for Leaf Sentinel Phase 1 pipeline."""

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
from src.dataset.duplicates import analyze_duplicates, compute_dhash, hamming_distance
from src.dataset.statistics import compute_dataset_statistics
from src.dataset.feasibility import analyze_class_feasibility
from src.dataset.visualize import generate_all_visualizations


class TestPhase1Pipeline(unittest.TestCase):
    """Test full Phase 1 dataset auditing functionality."""

    def setUp(self):
        """Create a mock synthetic dataset in a temporary directory."""
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
            
            # Create synthetic image
            img_path = self.img_dir / f"img_{i:03d}.jpg"
            img_arr = np.random.randint(50, 200, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(img_arr).save(img_path)

            # Create synthetic mask
            mask_path = self.mask_dir / f"img_{i:03d}_mask.png"
            mask_arr = np.zeros((128, 128), dtype=np.uint8)
            if disease != "Healthy":
                # Add a circular lesion in center
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

        # Save metadata CSV
        meta_df = pd.DataFrame(self.samples)
        meta_df.to_csv(self.root / "metadata.csv", index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_discovery(self):
        """Test dataset discovery parses metadata correctly."""
        res = discover_dataset(self.root)
        self.assertEqual(len(res.records_df), 12)
        self.assertEqual(res.structure_type, "metadata_driven")
        self.assertIn("train", res.discovered_splits)

    def test_validation(self):
        """Test incremental image and mask validation."""
        disc = discover_dataset(self.root)
        val_rep = validate_dataset(disc.records_df)
        self.assertEqual(val_rep.valid_images, 12)
        self.assertEqual(val_rep.valid_masks, 12)
        self.assertEqual(val_rep.corrupt_images, 0)
        self.assertEqual(val_rep.corrupt_masks, 0)

    def test_dhash_and_duplicates(self):
        """Test perceptual hashing and duplicate detection."""
        disc = discover_dataset(self.root)
        val_rep = validate_dataset(disc.records_df)
        dup_rep = analyze_duplicates(val_rep.validated_df, hamming_threshold=2)
        self.assertIsNotNone(dup_rep)
        self.assertIsInstance(dup_rep.candidates_df, pd.DataFrame)

    def test_statistics_and_feasibility(self):
        """Test statistical aggregation and feasibility assignment."""
        disc = discover_dataset(self.root)
        val_rep = validate_dataset(disc.records_df)
        dup_rep = analyze_duplicates(val_rep.validated_df)
        stats = compute_dataset_statistics(val_rep.validated_df, self.root)
        
        self.assertEqual(stats.summary_dict["overview"]["valid_images"], 12)
        
        feas_df = analyze_class_feasibility(
            val_rep.validated_df,
            strong_min_samples=10,
            usable_min_samples=3,
            limited_min_samples=1
        )
        self.assertGreater(len(feas_df), 0)

    def test_visualization_generation(self):
        """Test figure and qualitative sample rendering."""
        disc = discover_dataset(self.root)
        val_rep = validate_dataset(disc.records_df)
        stats = compute_dataset_statistics(val_rep.validated_df, self.root)
        
        out_figs = self.root / "outputs" / "figures"
        out_samples = self.root / "outputs" / "samples"
        
        generate_all_visualizations(
            stats=stats,
            validated_df=val_rep.validated_df,
            figures_dir=out_figs,
            samples_dir=out_samples,
            sample_count=4,
            dpi=100
        )
        
        self.assertTrue((out_figs / "host_distribution.png").exists())
        self.assertTrue((out_figs / "disease_distribution.png").exists())
        self.assertTrue((out_figs / "split_distribution.png").exists())
        self.assertTrue((out_figs / "disease_by_split.png").exists())
        self.assertTrue((out_figs / "mask_ratio_distribution.png").exists())
        self.assertTrue((out_figs / "resolution_distribution.png").exists())


if __name__ == "__main__":
    unittest.main()
