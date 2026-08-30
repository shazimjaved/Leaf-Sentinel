"""Dataset inspection, validation, analysis, and visualization modules for Leaf Sentinel."""

from src.dataset.inspect import discover_dataset, DatasetDiscoveryResult
from src.dataset.validate import validate_dataset, ValidationReport
from src.dataset.duplicates import analyze_duplicates, DuplicateReport
from src.dataset.statistics import compute_dataset_statistics, DatasetStatistics
from src.dataset.feasibility import analyze_class_feasibility
from src.dataset.visualize import generate_all_visualizations

__all__ = [
    "discover_dataset",
    "DatasetDiscoveryResult",
    "validate_dataset",
    "ValidationReport",
    "analyze_duplicates",
    "DuplicateReport",
    "compute_dataset_statistics",
    "DatasetStatistics",
    "analyze_class_feasibility",
    "generate_all_visualizations",
]
