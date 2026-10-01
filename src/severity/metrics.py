import torch
from torchmetrics import MetricCollection, Metric
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
    PearsonCorrCoef,
    SpearmanCorrCoef
)

class MeanSignedError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.sum_error += torch.sum(preds - target)
        self.total += target.numel()

    def compute(self):
        return self.sum_error / self.total if self.total > 0 else torch.tensor(0.0)
        
class MedianAbsoluteError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("errors", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.errors.append(torch.abs(preds - target))

    def compute(self):
        if len(self.errors) == 0:
            return torch.tensor(0.0)
        all_errors = torch.cat(self.errors)
        return torch.median(all_errors)

def get_regression_metrics(device: str = "cpu"):
    """
    Instantiate MetricCollection for Phase 4 Lesion Burden Regression.
    """
    metrics = MetricCollection({
        "mae": MeanAbsoluteError(),
        "mse": MeanSquaredError(), # will take sqrt manually for RMSE, or torchmetrics has it? Yes, we can just use MSE and sqrt.
        "rmse": MeanSquaredError(squared=False),
        "r2": R2Score(),
        "pearson": PearsonCorrCoef(),
        "spearman": SpearmanCorrCoef(),
        "bias": MeanSignedError(),
        "median_ae": MedianAbsoluteError(),
    })
    return metrics.to(device)
