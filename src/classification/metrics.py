import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix
)

def get_classification_metrics(num_classes: int = 10, device: str = "cpu"):
    """
    Instantiate and return a MetricCollection for Phase 3 Disease Classification.
    """
    metrics = MetricCollection({
        # Macro averages (Primary)
        "macro_f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "macro_precision": MulticlassPrecision(num_classes=num_classes, average="macro"),
        "macro_recall": MulticlassRecall(num_classes=num_classes, average="macro"),
        
        # Weighted averages
        "weighted_f1": MulticlassF1Score(num_classes=num_classes, average="weighted"),
        
        # Accuracies
        "accuracy": MulticlassAccuracy(num_classes=num_classes, average="micro"),
        "balanced_accuracy": MulticlassAccuracy(num_classes=num_classes, average="macro"),
        "top2_accuracy": MulticlassAccuracy(num_classes=num_classes, top_k=2, average="micro"),
    })
    return metrics.to(device)

def get_per_class_metrics(num_classes: int = 10, device: str = "cpu"):
    """
    Instantiate per-class metrics for detailed evaluation.
    """
    metrics = MetricCollection({
        "precision": MulticlassPrecision(num_classes=num_classes, average="none"),
        "recall": MulticlassRecall(num_classes=num_classes, average="none"),
        "f1": MulticlassF1Score(num_classes=num_classes, average="none"),
    })
    return metrics.to(device)

def get_confusion_matrix(num_classes: int = 10, device: str = "cpu"):
    return MulticlassConfusionMatrix(num_classes=num_classes).to(device)
