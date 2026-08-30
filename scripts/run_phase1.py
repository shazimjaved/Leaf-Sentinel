"""Phase 1 Pipeline Runner for Leaf Sentinel (AgriVision AI).

Executes the complete dataset discovery, validation, statistical profiling,
duplicate/leakage analysis, class feasibility heuristic scoring, and visual reporting.

Usage:
    python scripts/run_phase1.py --data data/raw/plantseg
    python scripts/run_phase1.py --config configs/phase1.yaml
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.inspect import discover_dataset
from src.dataset.validate import validate_dataset
from src.dataset.duplicates import analyze_duplicates
from src.dataset.statistics import compute_dataset_statistics
from src.dataset.feasibility import analyze_class_feasibility
from src.dataset.visualize import generate_all_visualizations

# Setup clean console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LeafSentinel.Phase1")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file if available."""
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
    return {}


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    max_samples: int | None = None,
    quick_mode: bool = False
):
    """Execute the end-to-end Phase 1 dataset discovery and validation audit."""
    logger.info("================================================================================")
    logger.info(" Leaf Sentinel — Phase 1: Dataset Discovery, Validation & Baseline Preparation ")
    logger.info("================================================================================")
    logger.info(f"Target Dataset Root : {data_dir}")
    logger.info(f"Output Directory    : {output_dir}")

    # Output subdirectories
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    samples_dir = output_dir / "samples"
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Discovery
    logger.info("\n>>> [Step 1/7] Discovering dataset structure and metadata...")
    discovery = discover_dataset(data_dir)
    logger.info(f"Discovery Summary: {discovery.summary}")

    if discovery.records_df.empty:
        logger.error(
            f"No image records or metadata found in {data_dir}. "
            "Ensure the raw PlantSeg dataset is downloaded and extracted into the data directory."
        )
        # Write minimal placeholder summary and exit gracefully
        with open(reports_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(discovery.to_dict(), f, indent=2)
        return

    # 2. Incremental Validation
    logger.info("\n>>> [Step 2/7] Running incremental image and mask validation...")
    val_cfg = config.get("validation", {})
    val_report = validate_dataset(
        records_df=discovery.records_df,
        max_aspect_ratio=val_cfg.get("max_aspect_ratio", 5.0),
        ratio_tolerance=val_cfg.get("mask_ratio_tolerance", 0.05),
        check_components=val_cfg.get("check_connected_components", True),
        max_samples=max_samples
    )
    
    # Save validation errors if any
    if not val_report.errors_df.empty:
        val_errors_path = reports_dir / "validation_errors.csv"
        val_report.errors_df.to_csv(val_errors_path, index=False)
        logger.info(f"Saved {len(val_report.errors_df)} validation issues to {val_errors_path}")

    # 3. Duplicate and Cross-Split Data Leakage Analysis
    logger.info("\n>>> [Step 3/7] Performing duplicate and cross-split leakage analysis...")
    dup_cfg = config.get("duplicates", {})
    dup_report = analyze_duplicates(
        validated_df=val_report.validated_df,
        enable_perceptual_hash=not quick_mode and dup_cfg.get("enable_perceptual_hash", True),
        dhash_size=dup_cfg.get("dhash_size", 16),
        hamming_threshold=dup_cfg.get("hamming_distance_threshold", 6)
    )
    
    # Save duplicate candidates report
    dup_candidates_path = reports_dir / "duplicate_candidates.csv"
    dup_report.candidates_df.to_csv(dup_candidates_path, index=False)
    logger.info(f"Saved {len(dup_report.candidates_df)} duplicate candidate pairs to {dup_candidates_path}")

    # 4. Statistical Profiling & Aggregation
    logger.info("\n>>> [Step 4/7] Computing dataset distributions and summary statistics...")
    stats = compute_dataset_statistics(
        validated_df=val_report.validated_df,
        dataset_root=data_dir,
        structure_type=discovery.structure_type,
        validation_summary=val_report.summary,
        duplicate_summary=dup_report.summary
    )

    # Save distribution CSVs
    stats.dataset_statistics_df.to_csv(reports_dir / "dataset_statistics.csv", index=False)
    stats.host_dist_df.to_csv(reports_dir / "host_distribution.csv", index=False)
    stats.disease_dist_df.to_csv(reports_dir / "disease_distribution.csv", index=False)
    stats.host_disease_df.to_csv(reports_dir / "host_x_disease_distribution.csv")
    stats.split_dist_df.to_csv(reports_dir / "split_distribution.csv", index=False)
    stats.disease_split_df.to_csv(reports_dir / "disease_x_split_distribution.csv")

    # Save main dataset_summary.json
    summary_json_path = reports_dir / "dataset_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(stats.summary_dict, f, indent=2)
    logger.info(f"Saved comprehensive summary JSON to {summary_json_path}")

    # 5. Class Feasibility Heuristic Analysis
    logger.info("\n>>> [Step 5/7] Analyzing class feasibility tiers...")
    feas_cfg = config.get("feasibility", {})
    feasibility_df = analyze_class_feasibility(
        validated_df=val_report.validated_df,
        strong_min_samples=feas_cfg.get("strong_candidate_min_samples", 100),
        strong_min_masks=feas_cfg.get("strong_candidate_min_masks", 80),
        usable_min_samples=feas_cfg.get("usable_min_samples", 40),
        usable_min_masks=feas_cfg.get("usable_min_masks", 30),
        limited_min_samples=feas_cfg.get("limited_min_samples", 15),
        limited_min_masks=feas_cfg.get("limited_min_masks", 10)
    )
    feasibility_path = reports_dir / "class_feasibility.csv"
    feasibility_df.to_csv(feasibility_path, index=False)
    logger.info(f"Saved class feasibility assessment to {feasibility_path}")

    # 6. Visual Analysis & Qualitative Samples
    logger.info("\n>>> [Step 6/7] Generating publication-quality figures and qualitative samples...")
    vis_cfg = config.get("visualization", {})
    generate_all_visualizations(
        stats=stats,
        validated_df=val_report.validated_df,
        figures_dir=figures_dir,
        samples_dir=samples_dir,
        sample_count=vis_cfg.get("sample_count", 24),
        dpi=vis_cfg.get("dpi", 300),
        overlay_color=tuple(vis_cfg.get("mask_overlay_color", [230, 40, 40])),
        alpha=vis_cfg.get("mask_overlay_alpha", 0.45)
    )

    # 7. Final Completion Report
    logger.info("\n>>> [Step 7/7] Phase 1 Pipeline Successfully Executed!")
    logger.info("================================================================================")
    logger.info(" Key Phase 1 Audit Findings:")
    logger.info(f" • Total Records Discovered : {stats.summary_dict['overview']['total_records']:,}")
    logger.info(f" • Valid Images Validated   : {stats.summary_dict['overview']['valid_images']:,}")
    logger.info(f" • Valid Masks Validated    : {stats.summary_dict['overview']['valid_masks']:,}")
    logger.info(f" • Plant Host Species       : {stats.summary_dict['taxonomic_scope']['total_hosts']}")
    logger.info(f" • Disease Categories       : {stats.summary_dict['taxonomic_scope']['total_diseases']}")
    logger.info(f" • Cross-Split Leakages     : {dup_report.cross_split_leakage_pairs} candidate pairs")
    logger.info(f" • Reports Saved to         : {reports_dir}")
    logger.info(f" • Figures Saved to         : {figures_dir}")
    logger.info(f" • Sample Cards Saved to    : {samples_dir}")
    logger.info("================================================================================")


def main():
    parser = argparse.ArgumentParser(
        description="Leaf Sentinel — Phase 1: Dataset Discovery, Validation & Baseline Preparation"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to raw dataset directory (default: read from config or data/raw/plantseg)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase1.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output directory (default: read from config or outputs/phase1)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to process (useful for rapid dry-runs)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in quick mode (skips heavy pairwise perceptual hashing)"
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    data_dir_str = args.data or config.get("paths", {}).get("raw_data_dir", "data/raw/plantseg")
    output_dir_str = args.output or config.get("paths", {}).get("output_dir", "outputs/phase1")

    data_dir = Path(data_dir_str)
    output_dir = Path(output_dir_str)

    run_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        config=config,
        max_samples=args.max_samples,
        quick_mode=args.quick
    )


if __name__ == "__main__":
    main()
