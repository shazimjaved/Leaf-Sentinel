import os
from pathlib import Path
import pandas as pd
import torch
import pytest

from src.classification.dataset import ClassificationDataset
from src.classification.model import DiseaseClassifier

# Dummy data generator for tests
@pytest.fixture
def dummy_manifest(tmp_path):
    manifest_path = tmp_path / "classification_manifest.csv"
    data = {
        "image_id": ["img1", "img2", "img3", "img4"],
        "image_relpath": ["images/train/1.jpg", "images/val/2.jpg", "images/test/3.jpg", "images/train/4.jpg"],
        "mask_relpath": ["annotations/train/1.png", "annotations/val/2.png", "annotations/test/3.png", "annotations/train/4.png"],
        "display_class": ["classA", "classB", "classA", "classB"],
        "class_index": [0, 1, 0, 1],
        "original_split": ["train", "val", "test", "train"],
        # Crucial part: benchmark_split overrides original filesystem split!
        "benchmark_split": ["train", "train", "val", "test"] 
    }
    df = pd.DataFrame(data)
    df.to_csv(manifest_path, index=False)
    return manifest_path, tmp_path

def test_benchmark_split_override(dummy_manifest):
    """
    Tests that dataset loading strictly honors benchmark_split
    and ignores the original filesystem split.
    """
    manifest_path, dataset_root = dummy_manifest
    
    # img2 is originally 'val', but benchmark_split is 'train'
    # img4 is originally 'train', but benchmark_split is 'test'
    train_ds = ClassificationDataset(manifest_path, dataset_root, "train")
    
    assert len(train_ds) == 2
    ids = train_ds.df["image_id"].tolist()
    assert "img1" in ids
    assert "img2" in ids
    
    val_ds = ClassificationDataset(manifest_path, dataset_root, "val")
    assert len(val_ds) == 1
    assert val_ds.df.iloc[0]["image_id"] == "img3"
    
    test_ds = ClassificationDataset(manifest_path, dataset_root, "test")
    assert len(test_ds) == 1
    assert test_ds.df.iloc[0]["image_id"] == "img4"

def test_model_shape():
    """Verify EfficientNet-B0 shape."""
    model = DiseaseClassifier(num_classes=10)
    
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    
    assert logits.shape == (2, 10)

def test_mock_training_step():
    """Verify that a single train step runs backward successfully."""
    model = DiseaseClassifier(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([1, 9], dtype=torch.long)
    
    logits = model(x)
    loss = criterion(logits, y)
    
    loss.backward()
    optimizer.step()
    
    # If it reaches here without error, it works.
    assert True
