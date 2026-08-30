"""Duplicate image and cross-split data leakage detection for Leaf Sentinel (Phase 1).

Implements exact MD5 hashing and fast difference hashing (dHash) in pure NumPy/PIL
to detect exact duplicates, near duplicates, and potential train/val/test data leakage.
"""

from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DuplicateCandidate:
    """Represents a potential duplicate pair."""
    sample_id_1: str
    sample_id_2: str
    path_1: str
    path_2: str
    split_1: str
    split_2: str
    host_1: str
    host_2: str
    disease_1: str
    disease_2: str
    duplicate_type: str  # 'exact_md5', 'near_perceptual'
    similarity_metric: str
    metric_value: float  # 0.0 for exact, hamming distance for dHash
    leakage_flag: bool   # True if across different splits (e.g. train vs test)


@dataclass
class DuplicateReport:
    """Stores duplicate analysis results and candidates."""
    total_samples_analyzed: int = 0
    exact_duplicate_pairs: int = 0
    near_duplicate_pairs: int = 0
    cross_split_leakage_pairs: int = 0
    leakage_train_val: int = 0
    leakage_train_test: int = 0
    leakage_val_test: int = 0
    candidates_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert duplicate summary to dictionary."""
        return {
            "total_samples_analyzed": self.total_samples_analyzed,
            "exact_duplicate_pairs": self.exact_duplicate_pairs,
            "near_duplicate_pairs": self.near_duplicate_pairs,
            "cross_split_leakage_pairs": self.cross_split_leakage_pairs,
            "leakage_train_val": self.leakage_train_val,
            "leakage_train_test": self.leakage_train_test,
            "leakage_val_test": self.leakage_val_test,
            "summary": self.summary
        }


def compute_md5(file_path: Path) -> Optional[str]:
    """Compute MD5 hash of a file incrementally."""
    try:
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def compute_dhash(image_path: Path, hash_size: int = 16) -> Optional[int]:
    """Compute fast Difference Hash (dHash) as a Python integer.
    
    Args:
        image_path: Path to image file.
        hash_size: Hash grid size (hash_size * hash_size bits).
        
    Returns:
        Integer bit representation of dHash.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize to (hash_size + 1, hash_size)
            gray = img.convert("L").resize(
                (hash_size + 1, hash_size),
                Image.Resampling.BILINEAR
            )
            pixels = np.array(gray, dtype=np.int16)
            
            # Compute horizontal difference: True if pixel[x+1] > pixel[x]
            diff = pixels[:, 1:] > pixels[:, :-1]
            
            # Convert boolean array to compact int
            flat_bits = diff.flatten()
            hash_int = 0
            for bit in flat_bits:
                hash_int = (hash_int << 1) | int(bit)
            return hash_int
    except Exception:
        return None


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance between two integer hashes."""
    return bin(h1 ^ h2).count("1")


def analyze_duplicates(
    validated_df: pd.DataFrame,
    enable_perceptual_hash: bool = True,
    dhash_size: int = 16,
    hamming_threshold: int = 6,
    max_samples_for_pairwise: int = 5000
) -> DuplicateReport:
    """Perform exact and perceptual duplicate detection and cross-split leakage checks.
    
    Args:
        validated_df: DataFrame from validation step with valid images.
        enable_perceptual_hash: Whether to calculate dHash.
        dhash_size: Grid size for dHash.
        hamming_threshold: Max Hamming distance to flag near-duplicate.
        max_samples_for_pairwise: Cap for full pairwise perceptual matching to prevent O(N^2) slowdown.
        
    Returns:
        DuplicateReport containing candidate pairs and split leakage flags.
    """
    logger.info("Starting duplicate and data-leakage analysis...")
    
    valid_records = validated_df[validated_df["is_valid_image"] == True].copy()
    total_valid = len(valid_records)

    if total_valid < 2:
        logger.info("Not enough valid images to perform duplicate analysis.")
        return DuplicateReport(total_samples_analyzed=total_valid)

    # 1. Compute MD5 hashes
    logger.info(f"Computing MD5 hashes for {total_valid} images...")
    md5_dict: Dict[str, List[int]] = {}  # hash -> list of row indices
    md5_list: List[Optional[str]] = []
    
    for idx, row in tqdm(valid_records.iterrows(), total=total_valid, desc="Computing MD5 hashes"):
        p = Path(row["image_path"])
        h = compute_md5(p)
        md5_list.append(h)
        if h:
            md5_dict.setdefault(h, []).append(idx)

    valid_records["md5"] = md5_list

    candidates: List[DuplicateCandidate] = []
    processed_pairs = set()

    # Exact duplicates via MD5
    exact_count = 0
    for h, indices in md5_dict.items():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    pair_key = tuple(sorted([idx1, idx2]))
                    if pair_key in processed_pairs:
                        continue
                    processed_pairs.add(pair_key)

                    r1 = valid_records.loc[idx1]
                    r2 = valid_records.loc[idx2]

                    s1, s2 = str(r1["split"]).lower(), str(r2["split"]).lower()
                    is_leakage = (s1 != s2 and s1 != "unassigned" and s2 != "unassigned")

                    candidates.append(DuplicateCandidate(
                        sample_id_1=r1["sample_id"],
                        sample_id_2=r2["sample_id"],
                        path_1=str(r1["image_path"]),
                        path_2=str(r2["image_path"]),
                        split_1=r1["split"],
                        split_2=r2["split"],
                        host_1=r1["host"],
                        host_2=r2["host"],
                        disease_1=r1["disease"],
                        disease_2=r2["disease"],
                        duplicate_type="exact_md5",
                        similarity_metric="byte_exact",
                        metric_value=0.0,
                        leakage_flag=is_leakage
                    ))
                    exact_count += 1

    # 2. Perceptual hashing (dHash)
    near_count = 0
    if enable_perceptual_hash:
        logger.info(f"Computing dHash (size={dhash_size}) for {total_valid} images...")
        dhashes: List[Optional[int]] = []
        
        for idx, row in tqdm(valid_records.iterrows(), total=total_valid, desc="Computing perceptual hashes"):
            p = Path(row["image_path"])
            dh = compute_dhash(p, hash_size=dhash_size)
            dhashes.append(dh)

        valid_records["dhash"] = dhashes

        # Group by buckets (top 32 bits) for efficient near-match candidate generation
        records_with_hash = valid_records[valid_records["dhash"].notna()].copy()
        
        # If total records <= max_samples_for_pairwise, do pairwise or bucketed search
        rows = list(records_with_hash.iterrows())
        n_rows = len(rows)

        if n_rows <= max_samples_for_pairwise:
            logger.info(f"Running pairwise perceptual hash comparison on {n_rows} samples...")
            for i in range(n_rows):
                idx1, r1 = rows[i]
                h1 = r1["dhash"]
                for j in range(i + 1, n_rows):
                    idx2, r2 = rows[j]
                    pair_key = tuple(sorted([idx1, idx2]))
                    if pair_key in processed_pairs:
                        continue

                    h2 = r2["dhash"]
                    dist = hamming_distance(h1, h2)
                    
                    if dist <= hamming_threshold:
                        processed_pairs.add(pair_key)
                        s1, s2 = str(r1["split"]).lower(), str(r2["split"]).lower()
                        is_leakage = (s1 != s2 and s1 != "unassigned" and s2 != "unassigned")

                        candidates.append(DuplicateCandidate(
                            sample_id_1=r1["sample_id"],
                            sample_id_2=r2["sample_id"],
                            path_1=str(r1["image_path"]),
                            path_2=str(r2["image_path"]),
                            split_1=r1["split"],
                            split_2=r2["split"],
                            host_1=r1["host"],
                            host_2=r2["host"],
                            disease_1=r1["disease"],
                            disease_2=r2["disease"],
                            duplicate_type="near_perceptual",
                            similarity_metric="hamming_distance",
                            metric_value=float(dist),
                            leakage_flag=is_leakage
                        ))
                        near_count += 1
        else:
            # Scalable bucketing for large datasets: split hash into 4 chunks (64-bit each)
            logger.info(f"Large dataset ({n_rows} samples): Using multi-index hashing for dHash...")
            buckets: Dict[Tuple[int, int], List[int]] = {}
            for i, (idx, r) in enumerate(rows):
                h = r["dhash"]
                # 4 sub-chunks of 64 bits each (for 256-bit hash)
                for chunk_idx in range(4):
                    chunk_val = (h >> (chunk_idx * 64)) & 0xFFFFFFFFFFFFFFFF
                    buckets.setdefault((chunk_idx, chunk_val), []).append(i)

            candidate_indices = set()
            for b_list in buckets.values():
                if len(b_list) > 1 and len(b_list) < 100:  # avoid ultra-dense collision
                    for i in range(len(b_list)):
                        for j in range(i + 1, len(b_list)):
                            candidate_indices.add(tuple(sorted([b_list[i], b_list[j]])))

            for i1, i2 in candidate_indices:
                idx1, r1 = rows[i1]
                idx2, r2 = rows[i2]
                pair_key = tuple(sorted([idx1, idx2]))
                if pair_key in processed_pairs:
                    continue

                dist = hamming_distance(r1["dhash"], r2["dhash"])
                if dist <= hamming_threshold:
                    processed_pairs.add(pair_key)
                    s1, s2 = str(r1["split"]).lower(), str(r2["split"]).lower()
                    is_leakage = (s1 != s2 and s1 != "unassigned" and s2 != "unassigned")

                    candidates.append(DuplicateCandidate(
                        sample_id_1=r1["sample_id"],
                        sample_id_2=r2["sample_id"],
                        path_1=str(r1["image_path"]),
                        path_2=str(r2["image_path"]),
                        split_1=r1["split"],
                        split_2=r2["split"],
                        host_1=r1["host"],
                        host_2=r2["host"],
                        disease_1=r1["disease"],
                        disease_2=r2["disease"],
                        duplicate_type="near_perceptual",
                        similarity_metric="hamming_distance",
                        metric_value=float(dist),
                        leakage_flag=is_leakage
                    ))
                    near_count += 1

    # Convert candidates to DataFrame
    cand_dicts = [c.__dict__ for c in candidates]
    candidates_df = pd.DataFrame(cand_dicts) if cand_dicts else pd.DataFrame(columns=[
        "sample_id_1", "sample_id_2", "path_1", "path_2", "split_1", "split_2",
        "host_1", "host_2", "disease_1", "disease_2", "duplicate_type",
        "similarity_metric", "metric_value", "leakage_flag"
    ])

    leakage_train_val = 0
    leakage_train_test = 0
    leakage_val_test = 0
    cross_split_total = 0

    if not candidates_df.empty:
        leakage_rows = candidates_df[candidates_df["leakage_flag"] == True]
        cross_split_total = len(leakage_rows)
        for _, row in leakage_rows.iterrows():
            splits = {str(row["split_1"]).lower(), str(row["split_2"]).lower()}
            if "train" in splits and ("val" in splits or "validation" in splits):
                leakage_train_val += 1
            elif "train" in splits and "test" in splits:
                leakage_train_test += 1
            elif ("val" in splits or "validation" in splits) and "test" in splits:
                leakage_val_test += 1

    summary = {
        "total_samples_analyzed": total_valid,
        "exact_duplicate_pairs": exact_count,
        "near_duplicate_pairs": near_count,
        "total_duplicate_candidates": len(candidates_df),
        "cross_split_leakage_pairs": cross_split_total,
        "leakage_train_val": leakage_train_val,
        "leakage_train_test": leakage_train_test,
        "leakage_val_test": leakage_val_test
    }

    logger.info(
        f"Duplicate analysis complete: {exact_count} exact pairs, {near_count} near-duplicate pairs, "
        f"{cross_split_total} cross-split leakage candidate pairs."
    )

    return DuplicateReport(
        total_samples_analyzed=total_valid,
        exact_duplicate_pairs=exact_count,
        near_duplicate_pairs=near_count,
        cross_split_leakage_pairs=cross_split_total,
        leakage_train_val=leakage_train_val,
        leakage_train_test=leakage_train_test,
        leakage_val_test=leakage_val_test,
        candidates_df=candidates_df,
        summary=summary
    )
