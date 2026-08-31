"""Leakage-free dataset preparation and two-stage duplicate verification for LeafSentinel (Phase 2).

Implements:
1. Two-stage duplicate verification (Exact MD5 + SSIM-verified perceptual dHash).
2. Disjoint Set Union (Union-Find) connected-component grouping.
3. Stratified group-aware train/val/test split generation ensuring zero duplicate leakage.
4. Duplicate group split integrity validation.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class UnionFind:
    """Disjoint Set Union (Union-Find) data structure with path compression and rank union."""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  # Path compression
        return self.parent[item]

    def union(self, item1: str, item2: str):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 == root2:
            return
        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1

    def get_components(self, all_items: List[str]) -> Dict[str, List[str]]:
        """Return mapping of component root -> list of member items."""
        components: Dict[str, List[str]] = {}
        for item in all_items:
            root = self.find(item)
            components.setdefault(root, []).append(item)
        return components


def compute_image_ssim(path1: Path, path2: Path, target_size: Tuple[int, int] = (256, 256)) -> float:
    """Compute Structural Similarity Index (SSIM) between two images.
    
    Loads both images, converts to grayscale, resizes to target_size,
    and computes standard SSIM using OpenCV/NumPy.
    """
    try:
        with Image.open(path1) as img1, Image.open(path2) as img2:
            gray1 = np.array(img1.convert("L").resize(target_size, Image.Resampling.BILINEAR), dtype=np.float32)
            gray2 = np.array(img2.convert("L").resize(target_size, Image.Resampling.BILINEAR), dtype=np.float32)

        # Compute mean, variance, covariance
        mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(gray1 * gray1, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray2 * gray2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return float(np.clip(ssim_map.mean(), -1.0, 1.0))
    except Exception as e:
        logger.warning(f"Failed to compute SSIM between {path1} and {path2}: {e}")
        return 0.0


@dataclass
class LeakageGroupingReport:
    """Detailed results from two-stage duplicate verification and Union-Find grouping."""
    exact_duplicate_pairs_grouped: int = 0
    near_candidates_examined: int = 0
    near_pairs_confirmed: int = 0
    near_candidates_rejected: int = 0
    total_duplicate_groups: int = 0
    multi_sample_groups: int = 0
    singleton_groups: int = 0
    cross_original_split_groups: int = 0
    sample_to_group_map: Dict[str, str] = field(default_factory=dict)
    group_details: List[Dict[str, Any]] = field(default_factory=list)


def build_leakage_groups(
    df: pd.DataFrame,
    duplicate_candidates_df: Optional[pd.DataFrame] = None,
    dhash_threshold: int = 6,
    ssim_threshold: float = 0.85
) -> Tuple[Dict[str, str], LeakageGroupingReport]:
    """Perform two-stage duplicate verification and build connected duplicate groups.
    
    Args:
        df: Filtered dataset DataFrame with columns `sample_id`, `image_path`, `split`.
        duplicate_candidates_df: Candidate pairs from Phase 1 duplicate analysis.
        dhash_threshold: Max dHash Hamming distance.
        ssim_threshold: Minimum SSIM score to confirm near-duplicate identity.
        
    Returns:
        (sample_to_group_id_map, LeakageGroupingReport)
    """
    logger.info("Initializing two-stage duplicate grouping...")
    all_sample_ids = df["sample_id"].tolist()
    sample_path_map = dict(zip(df["sample_id"], df["image_path"]))
    sample_split_map = dict(zip(df["sample_id"], df["split"]))
    
    valid_ids_set = set(all_sample_ids)
    uf = UnionFind()
    for sid in all_sample_ids:
        uf.find(sid)

    exact_pairs_grouped = 0
    near_candidates_examined = 0
    near_pairs_confirmed = 0
    near_candidates_rejected = 0

    if duplicate_candidates_df is not None and not duplicate_candidates_df.empty:
        for _, row in duplicate_candidates_df.iterrows():
            s1 = row.get("sample_id_1")
            s2 = row.get("sample_id_2")
            
            # Check if both samples belong to our active filtered dataset
            if s1 not in valid_ids_set or s2 not in valid_ids_set:
                continue

            dup_type = row.get("duplicate_type")
            p1 = Path(sample_path_map[s1]) if s1 in sample_path_map else None
            p2 = Path(sample_path_map[s2]) if s2 in sample_path_map else None

            if dup_type == "exact_md5":
                # Stage 1: Exact MD5 duplicate -> Automatically union
                uf.union(s1, s2)
                exact_pairs_grouped += 1
            elif dup_type == "near_perceptual":
                # Stage 2: Near-duplicate candidate -> Secondary SSIM verification
                near_candidates_examined += 1
                metric_val = float(row.get("metric_value", 99.0))
                
                if metric_val <= dhash_threshold and p1 and p2 and p1.exists() and p2.exists():
                    ssim_val = compute_image_ssim(p1, p2)
                    if ssim_val >= ssim_threshold:
                        uf.union(s1, s2)
                        near_pairs_confirmed += 1
                    else:
                        near_candidates_rejected += 1
                else:
                    near_candidates_rejected += 1

    # Extract connected components
    components = uf.get_components(all_sample_ids)
    
    sample_to_group: Dict[str, str] = {}
    group_details: List[Dict[str, Any]] = []
    
    multi_sample_count = 0
    singleton_count = 0
    cross_split_groups = 0

    # Sort components deterministically by size (descending) and first sample ID
    sorted_roots = sorted(components.keys(), key=lambda r: (-len(components[r]), sorted(components[r])[0]))

    for grp_idx, root in enumerate(sorted_roots, start=1):
        group_id = f"grp_{grp_idx:05d}"
        members = sorted(components[root])
        
        orig_splits = {sample_split_map.get(m, "unassigned") for m in members}
        is_cross_split = len(orig_splits) > 1

        if len(members) > 1:
            multi_sample_count += 1
            if is_cross_split:
                cross_split_groups += 1
        else:
            singleton_count += 1

        for m in members:
            sample_to_group[m] = group_id

        group_details.append({
            "group_id": group_id,
            "size": len(members),
            "members": members,
            "original_splits": list(orig_splits),
            "crosses_original_splits": is_cross_split
        })

    report = LeakageGroupingReport(
        exact_duplicate_pairs_grouped=exact_pairs_grouped,
        near_candidates_examined=near_candidates_examined,
        near_pairs_confirmed=near_pairs_confirmed,
        near_candidates_rejected=near_candidates_rejected,
        total_duplicate_groups=len(components),
        multi_sample_groups=multi_sample_count,
        singleton_groups=singleton_count,
        cross_original_split_groups=cross_split_groups,
        sample_to_group_map=sample_to_group,
        group_details=group_details
    )

    logger.info(
        f"Duplicate grouping complete: {exact_pairs_grouped} exact pairs, "
        f"{near_pairs_confirmed}/{near_candidates_examined} near-duplicate pairs confirmed (SSIM >= {ssim_threshold}), "
        f"{near_candidates_rejected} rejected. Total groups: {len(components)} "
        f"({multi_sample_count} multi-sample, {cross_split_groups} crossing original splits)."
    )

    return sample_to_group, report


def create_leakage_free_splits(
    df: pd.DataFrame,
    sample_to_group: Dict[str, str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> pd.Series:
    """Generate stratified group-aware train/val/test splits.
    
    Guarantees that 100% of samples belonging to the same `duplicate_group_id`
    are assigned to the exact same split, while preserving class balance.
    """
    logger.info(f"Generating deterministic leakage-free splits (seed={seed}, ratios={train_ratio}/{val_ratio}/{test_ratio})...")
    np.random.seed(seed)
    
    df_work = df.copy()
    df_work["duplicate_group_id"] = df_work["sample_id"].map(sample_to_group)
    
    # Aggregate group-level metadata
    group_records = []
    for group_id, grp in df_work.groupby("duplicate_group_id"):
        # Most frequent class in group
        top_disease = grp["disease"].mode().iloc[0] if "disease" in grp.columns else "unknown"
        top_host = grp["host"].mode().iloc[0] if "host" in grp.columns else "unknown"
        is_healthy = bool(grp["is_healthy"].all()) if "is_healthy" in grp.columns else False
        
        group_records.append({
            "duplicate_group_id": group_id,
            "sample_count": len(grp),
            "stratum": f"{top_host}___{top_disease}" if not is_healthy else "healthy_control"
        })

    groups_df = pd.DataFrame(group_records)
    
    # Shuffle groups deterministically
    groups_df = groups_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Stratified greedy allocation by stratum
    train_groups: Set[str] = set()
    val_groups: Set[str] = set()
    test_groups: Set[str] = set()

    train_target = train_ratio
    val_target = val_ratio
    test_target = test_ratio

    total_samples = len(df_work)
    target_train_n = int(round(total_samples * train_target))
    target_val_n = int(round(total_samples * val_target))
    target_test_n = total_samples - target_train_n - target_val_n

    current_train_n = 0
    current_val_n = 0
    current_test_n = 0

    # Group by stratum for balanced distribution
    for stratum, s_df in groups_df.groupby("stratum"):
        s_groups = s_df.to_dict("records")
        for g in s_groups:
            gid = g["duplicate_group_id"]
            cnt = g["sample_count"]
            
            # Compute deficits
            deficits = {
                "train": (target_train_n - current_train_n) / max(1, target_train_n),
                "val": (target_val_n - current_val_n) / max(1, target_val_n),
                "test": (target_test_n - current_test_n) / max(1, target_test_n),
            }
            # Pick split with largest deficit
            best_split = max(deficits, key=deficits.get)

            if best_split == "train":
                train_groups.add(gid)
                current_train_n += cnt
            elif best_split == "val":
                val_groups.add(gid)
                current_val_n += cnt
            else:
                test_groups.add(gid)
                current_test_n += cnt

    # Assign split to each sample
    split_assignments = []
    for _, row in df_work.iterrows():
        gid = row["duplicate_group_id"]
        if gid in train_groups:
            split_assignments.append("train")
        elif gid in val_groups:
            split_assignments.append("val")
        elif gid in test_groups:
            split_assignments.append("test")
        else:
            split_assignments.append("train")  # fallback

    df_work["phase2_split"] = split_assignments

    # CRITICAL VALIDATION: Verify zero leakage
    validate_zero_leakage(df_work)

    pct_train = (current_train_n / total_samples * 100) if total_samples > 0 else 0
    pct_val = (current_val_n / total_samples * 100) if total_samples > 0 else 0
    pct_test = (current_test_n / total_samples * 100) if total_samples > 0 else 0

    logger.info(
        f"Split generation successful: Train={current_train_n} ({pct_train:.1f}%), "
        f"Val={current_val_n} ({pct_val:.1f}%), Test={current_test_n} ({pct_test:.1f}%)."
    )

    return df_work["phase2_split"]


def validate_zero_leakage(df: pd.DataFrame):
    """Explicitly assert that no duplicate_group_id spans across multiple phase2_split values.
    
    Raises:
        ValueError: If any duplicate group is found in more than one split.
    """
    if "duplicate_group_id" not in df.columns or "phase2_split" not in df.columns:
        raise ValueError("DataFrame must contain 'duplicate_group_id' and 'phase2_split' columns.")

    leaked_groups = []
    for gid, group in df.groupby("duplicate_group_id"):
        splits = group["phase2_split"].unique()
        if len(splits) > 1:
            leaked_groups.append((gid, list(splits), len(group)))

    if leaked_groups:
        err_msg = (
            f"DATA LEAKAGE DETECTED! {len(leaked_groups)} duplicate groups span across multiple splits: "
            f"{leaked_groups[:5]}"
        )
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info("ZERO DATA LEAKAGE VERIFIED: All duplicate groups are strictly confined to a single split partition.")
