"""Unit and integration test suite for LeafSentinel segmentation models, leakage prevention, and metrics."""

from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.leakage import UnionFind, validate_zero_leakage
from src.segmentation.model import ResNetUNet
from src.segmentation.metrics import SegmentationMetrics, compute_batch_metrics
from src.segmentation.train import CombinedBCEDiceLoss


class TestSegmentationPipeline(unittest.TestCase):
    """Test suite for leakage prevention, U-Net model mechanics, metrics, and loss."""

    def test_union_find(self):
        """Test Disjoint Set Union operations."""
        uf = UnionFind()
        items = ["img_1", "img_2", "img_3", "img_4"]
        for it in items:
            uf.find(it)

        uf.union("img_1", "img_2")
        uf.union("img_2", "img_3")

        comps = uf.get_components(items)
        self.assertEqual(len(comps), 2)
        self.assertEqual(uf.find("img_1"), uf.find("img_3"))

    def test_zero_leakage_validator(self):
        """Test the strict zero-leakage validator on valid and invalid split assignments."""
        valid_df = pd.DataFrame({
            "duplicate_group_id": ["grp_1", "grp_1", "grp_2", "grp_3", "grp_3"],
            "phase2_split": ["train", "train", "val", "test", "test"]
        })
        try:
            validate_zero_leakage(valid_df)
        except ValueError:
            self.fail("validate_zero_leakage raised ValueError on a valid split!")

        leaked_df = pd.DataFrame({
            "duplicate_group_id": ["grp_1", "grp_1", "grp_2"],
            "phase2_split": ["train", "val", "test"]
        })
        with self.assertRaises(ValueError):
            validate_zero_leakage(leaked_df)

    def test_model_forward_pass(self):
        """Test ResNetUNet model initialization and forward pass output shape."""
        model = ResNetUNet(pretrained=False, in_channels=3, num_classes=1)
        model.eval()
        dummy_input = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, (2, 1, 128, 128))
        params = model.count_parameters()
        self.assertGreater(params["total_parameters"], 10_000_000)

    def test_metrics_known_cases(self):
        """Test Dice, IoU, Precision, Recall, and FP Area Ratio on known ground truths."""
        tracker = SegmentationMetrics()

        # Perfect diseased prediction
        gt_a = np.zeros((100, 100), dtype=np.float32)
        gt_a[:10, :10] = 1.0
        pred_a = gt_a.copy()
        tracker.add_sample(pred_a, gt_a, {"image_id": "a", "is_healthy": False})
        res_a = tracker.sample_results[-1]
        self.assertAlmostEqual(res_a.dice, 1.0, places=4)
        self.assertAlmostEqual(res_a.iou, 1.0, places=4)

        # Disjoint prediction
        gt_b = np.zeros((100, 100), dtype=np.float32)
        gt_b[:10, :10] = 1.0
        pred_b = np.zeros((100, 100), dtype=np.float32)
        pred_b[50:60, 50:60] = 1.0
        tracker.add_sample(pred_b, gt_b, {"image_id": "b", "is_healthy": False})
        res_b = tracker.sample_results[-1]
        self.assertAlmostEqual(res_b.dice, 0.0, places=3)
        self.assertAlmostEqual(res_b.iou, 0.0, places=3)

        # Clean healthy leaf
        gt_c = np.zeros((100, 100), dtype=np.float32)
        pred_c = np.zeros((100, 100), dtype=np.float32)
        tracker.add_sample(pred_c, gt_c, {"image_id": "c", "is_healthy": True})
        res_c = tracker.sample_results[-1]
        self.assertEqual(res_c.dice, 1.0)
        self.assertEqual(res_c.false_positive_area_ratio, 0.0)

        # False positive on healthy leaf
        gt_d = np.zeros((100, 100), dtype=np.float32)
        pred_d = np.zeros((100, 100), dtype=np.float32)
        pred_d[:5, :5] = 1.0
        tracker.add_sample(pred_d, gt_d, {"image_id": "d", "is_healthy": True})
        res_d = tracker.sample_results[-1]
        self.assertEqual(res_d.dice, 0.0)
        self.assertAlmostEqual(res_d.false_positive_area_ratio, 25 / 10000, places=4)

    def test_loss_and_backward_step(self):
        """Test BCE+Dice loss calculation and backward gradient propagation."""
        criterion = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        model = ResNetUNet(pretrained=False, in_channels=3, num_classes=1)
        model.train()

        dummy_img = torch.randn(2, 3, 64, 64, requires_grad=True)
        dummy_mask = torch.randint(0, 2, (2, 1, 64, 64)).float()

        logits = model(dummy_img)
        total_loss, loss_bce, loss_dice = criterion(logits, dummy_mask)

        self.assertTrue(torch.isfinite(total_loss))
        total_loss.backward()

        has_grads = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        self.assertTrue(has_grads)


if __name__ == "__main__":
    unittest.main()
