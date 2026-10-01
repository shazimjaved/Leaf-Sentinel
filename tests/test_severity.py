import json
import pytest
import torch
import pandas as pd
from pathlib import Path
from src.severity.model import BurdenRegressor
from src.severity.dataset import ImageRelativeBurdenDataset
from src.severity.metrics import get_regression_metrics

MANIFEST_PATH = Path("outputs/severity/dataset/severity_manifest.csv")
DATA_ROOT = Path("data/raw/plantseg")
SUMMARY_PATH = Path("outputs/severity/dataset/dataset_summary.json")

@pytest.fixture
def manifest():
    if not MANIFEST_PATH.exists():
        pytest.skip("Manifest not generated yet.")
    return pd.read_csv(MANIFEST_PATH)

def test_dataset_size_and_healthies(manifest):
    # Total samples
    assert len(manifest) == 1304, f"Expected 1304 total samples, got {len(manifest)}"
    
    # Split counts
    train_c = len(manifest[manifest["benchmark_split"] == "train"])
    val_c = len(manifest[manifest["benchmark_split"] == "val"])
    test_c = len(manifest[manifest["benchmark_split"] == "test"])
    
    assert train_c == 913, f"Expected 913 train, got {train_c}"
    assert val_c == 196, f"Expected 196 val, got {val_c}"
    assert test_c == 195, f"Expected 195 test, got {test_c}"
    
    # Healthy samples
    healthies = manifest[manifest["is_healthy"] == True]
    assert len(healthies) == 8, f"Expected 8 healthy samples, got {len(healthies)}"
    assert (healthies["mask_area_ratio"] == 0.0).all(), "Healthy samples must have 0.0 target"

def test_leakage_invariants(manifest):
    for gid, group in manifest.groupby("duplicate_group_id"):
        splits = group["benchmark_split"].unique()
        assert len(splits) == 1, f"Group {gid} spans multiple splits: {splits}"

def test_target_type_and_bounds():
    if not MANIFEST_PATH.exists():
        pytest.skip()
    ds = ImageRelativeBurdenDataset(MANIFEST_PATH, DATA_ROOT, "train", 224)
    img, target = ds[0]
    
    assert isinstance(target, torch.Tensor)
    assert target.dtype == torch.float32
    assert target.shape == (1,)
    assert 0.0 <= target.item() <= 1.0
    
def test_model_forward():
    model = BurdenRegressor()
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 1)
    
def test_cpu_forward_backward():
    model = BurdenRegressor()
    model.train()
    x = torch.randn(2, 3, 224, 224)
    y = torch.tensor([[0.5], [0.1]], dtype=torch.float32)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    
    assert loss.item() >= 0

def test_metrics():
    metrics = get_regression_metrics(device="cpu")
    # Exact MAE test
    preds = torch.tensor([[0.1], [0.2], [0.3]])
    targets = torch.tensor([[0.1], [0.1], [0.1]])
    # Errors: 0.0, 0.1, 0.2 -> MAE = 0.1
    metrics.update(preds, targets)
    res = metrics.compute()
    assert torch.isclose(res["mae"], torch.tensor(0.1))
