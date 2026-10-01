import torch

class TrivialBaseline:
    def __init__(self, constant_value: float):
        self.constant_value = constant_value
        
    def predict(self, num_samples: int) -> torch.Tensor:
        """Returns predictions of shape (B, 1) all containing the constant."""
        return torch.full((num_samples, 1), self.constant_value, dtype=torch.float32)

def evaluate_trivial_baseline(constant_value: float, loader, metrics, device="cpu"):
    """Evaluates a trivial constant baseline on the given loader."""
    metrics.reset()
    for _, targets in loader:
        targets = targets.to(device)
        preds = TrivialBaseline(constant_value).predict(targets.size(0)).to(device)
        metrics.update(preds, targets)
    return metrics.compute()
