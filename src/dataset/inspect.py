"""Dataset discovery and metadata parsing for Leaf Sentinel (Phase 1).

Dynamically scans the dataset root directory to identify images, masks,
metadata files, annotation formats, and data splits without hard-coding assumptions.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
METADATA_EXTENSIONS = {".csv", ".tsv", ".json", ".txt"}


@dataclass
class DatasetDiscoveryResult:
    """Stores structured discovery information about the dataset."""
    dataset_root: Path
    records_df: pd.DataFrame
    metadata_files: List[Path] = field(default_factory=list)
    split_files: List[Path] = field(default_factory=list)
    annotation_files: List[Path] = field(default_factory=list)
    doc_files: List[Path] = field(default_factory=list)
    structure_type: str = "unknown"
    image_dir: Optional[Path] = None
    mask_dir: Optional[Path] = None
    discovered_splits: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert discovery summary to serializable dictionary."""
        return {
            "dataset_root": str(self.dataset_root),
            "structure_type": self.structure_type,
            "total_discovered_records": len(self.records_df),
            "metadata_files": [str(p) for p in self.metadata_files],
            "split_files": [str(p) for p in self.split_files],
            "annotation_files": [str(p) for p in self.annotation_files],
            "doc_files": [str(p) for p in self.doc_files],
            "image_dir": str(self.image_dir) if self.image_dir else None,
            "mask_dir": str(self.mask_dir) if self.mask_dir else None,
            "discovered_splits": self.discovered_splits,
            "summary": self.summary,
        }


def _find_all_files(root: Path) -> Tuple[List[Path], List[Path], List[Path], List[Path], List[Path]]:
    """Scan all files under root and partition them by type."""
    images: List[Path] = []
    masks: List[Path] = []
    metadata_files: List[Path] = []
    annotation_files: List[Path] = []
    doc_files: List[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        name_lower = path.name.lower()
        parent_lower = str(path.parent).lower()

        if "readme" in name_lower or "license" in name_lower or "citation" in name_lower:
            doc_files.append(path)
        elif ext == ".csv" or ext == ".tsv":
            metadata_files.append(path)
        elif ext == ".json":
            if "annotation" in name_lower or "coco" in name_lower:
                annotation_files.append(path)
            else:
                metadata_files.append(path)
        elif "split" in name_lower and ext == ".txt":
            metadata_files.append(path)
        elif ext in IMAGE_EXTENSIONS:
            parent_parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
            is_mask = (
                any(p in {"mask", "masks", "annotation", "annotations", "label", "labels", "ground_truth", "gt", "segmentation", "segmentations"} for p in parent_parts)
                or "mask" in name_lower
                or "_seg" in name_lower
            )
            if is_mask:
                masks.append(path)
            else:
                images.append(path)

    return images, masks, metadata_files, annotation_files, doc_files


def _normalize_column_name(col: str) -> str:
    """Normalize metadata column name for robust mapping."""
    return col.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")


def _find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Find the first matching candidate among DataFrame columns."""
    norm_map = {_normalize_column_name(c): c for c in columns}
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    return None


def _parse_metadata_file(meta_path: Path, root: Path) -> Optional[pd.DataFrame]:
    """Attempt to parse a CSV/TSV/JSON metadata file into a DataFrame."""
    try:
        if meta_path.suffix.lower() in {".csv", ".tsv", ".txt"}:
            sep = "\t" if meta_path.suffix.lower() == ".tsv" else ","
            try:
                df = pd.read_csv(meta_path, sep=sep)
            except Exception:
                df = pd.read_csv(meta_path, sep=None, engine="python")
        elif meta_path.suffix.lower() == ".json":
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Check for common nested keys like 'annotations', 'images', 'samples'
                for key in ["images", "annotations", "samples", "data", "records"]:
                    if key in data and isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        break
                else:
                    df = pd.DataFrame([data])
            else:
                return None
        else:
            return None

        logger.info(f"Loaded metadata from {meta_path.name}: {len(df)} rows, columns={list(df.columns)}")
        return df
    except Exception as e:
        logger.warning(f"Could not parse metadata file {meta_path}: {e}")
        return None


def _standardize_metadata_df(
    df: pd.DataFrame,
    root: Path,
    image_lookup: Dict[str, Path],
    mask_lookup: Dict[str, Path]
) -> pd.DataFrame:
    """Map arbitrary metadata columns to a standard schema with indexed path resolution."""
    cols = list(df.columns)
    
    img_col = _find_column(cols, [
        "name", "image_path", "img_path", "image", "img", "image_name", "filename", "file_name", "filepath", "path"
    ])
    mask_col = _find_column(cols, [
        "label_file", "mask_path", "mask", "mask_name", "segmentation_path", "seg_path", "ground_truth", "gt_path", "label_path"
    ])
    id_col = _find_column(cols, [
        "sample_id", "image_id", "img_id", "id", "index", "sample"
    ])
    host_col = _find_column(cols, [
        "plant", "host", "species", "crop", "host_plant", "plant_species", "crop_type"
    ])
    disease_col = _find_column(cols, [
        "disease", "disease_case", "label", "category", "class", "diagnosis", "pathology", "disease_name"
    ])
    split_col = _find_column(cols, [
        "split", "partition", "dataset_split", "subset", "set", "fold"
    ])
    source_col = _find_column(cols, [
        "url", "source", "dataset", "origin", "database", "source_dataset"
    ])
    ratio_col = _find_column(cols, [
        "mask_ratio", "affected_ratio", "severity", "ratio", "affected_area_ratio", "mask_area_ratio"
    ])
    res_col = _find_column(cols, ["resolution", "dimensions", "size", "image_resolution"])
    width_col = _find_column(cols, ["width", "image_width", "w"])
    height_col = _find_column(cols, ["height", "image_height", "h"])
    license_col = _find_column(cols, ["license", "licence"])

    records = []
    for idx, row in df.iterrows():
        img_val = str(row[img_col]).strip() if img_col and pd.notna(row[img_col]) else None
        mask_val = str(row[mask_col]).strip() if mask_col and pd.notna(row[mask_col]) else None
        
        # Fast indexed lookup
        img_path = None
        if img_val:
            img_path = image_lookup.get(img_val) or image_lookup.get(Path(img_val).name)
            if not img_path:
                cand = root / img_val
                img_path = cand if cand.exists() else None

        mask_path = None
        if mask_val:
            mask_path = mask_lookup.get(mask_val) or mask_lookup.get(Path(mask_val).name)
            if not mask_path:
                cand = root / mask_val
                mask_path = cand if cand.exists() else None

        # Sample ID
        sample_id = f"sample_{idx:06d}"
        if id_col and pd.notna(row[id_col]):
            sample_id = f"{row[id_col]}_{Path(img_val).stem if img_val else idx}"
        elif img_val:
            sample_id = Path(img_val).stem

        host = str(row[host_col]).strip() if host_col and pd.notna(row[host_col]) else "Unknown"
        disease = str(row[disease_col]).strip() if disease_col and pd.notna(row[disease_col]) else "Unknown"
        
        # Normalize split name (e.g. Training -> train, Validation -> val, Test -> test)
        raw_split = str(row[split_col]).strip().lower() if split_col and pd.notna(row[split_col]) else "unassigned"
        if "train" in raw_split:
            split = "train"
        elif "val" in raw_split:
            split = "val"
        elif "test" in raw_split:
            split = "test"
        else:
            split = raw_split

        source = str(row[source_col]).strip() if source_col and pd.notna(row[source_col]) else "PlantSeg"
        license_val = str(row[license_col]).strip() if license_col and pd.notna(row[license_col]) else "Unknown"
        
        meta_ratio = float(row[ratio_col]) if ratio_col and pd.notna(row[ratio_col]) else None
        
        # Parse resolution (e.g. "640x480" or separate width/height)
        meta_w = int(row[width_col]) if width_col and pd.notna(row[width_col]) else None
        meta_h = int(row[height_col]) if height_col and pd.notna(row[height_col]) else None
        if (meta_w is None or meta_h is None) and res_col and pd.notna(row[res_col]):
            try:
                res_str = str(row[res_col]).lower().replace(" ", "")
                if "x" in res_str:
                    w_str, h_str = res_str.split("x", 1)
                    meta_w = int(w_str)
                    meta_h = int(h_str)
            except Exception:
                pass

        records.append({
            "sample_id": sample_id,
            "image_path": str(img_path) if img_path else None,
            "mask_path": str(mask_path) if mask_path else None,
            "host": host,
            "disease": disease,
            "split": split,
            "source": source,
            "license": license_val,
            "meta_mask_ratio": meta_ratio,
            "meta_width": meta_w,
            "meta_height": meta_h,
            "original_index": idx
        })

    return pd.DataFrame(records)


def _infer_from_filesystem(
    root: Path,
    image_paths: List[Path],
    mask_paths: List[Path]
) -> pd.DataFrame:
    """Infer records, pairings, and labels directly from directory layout."""
    mask_lookup: Dict[str, Path] = {}
    for mp in mask_paths:
        stem = mp.stem.lower()
        mask_lookup[stem] = mp
        if stem.endswith("_mask") or stem.endswith("_seg"):
            base = stem.rsplit("_", 1)[0]
            mask_lookup[base] = mp

    records = []
    for idx, ip in enumerate(image_paths):
        stem = ip.stem.lower()
        mask_path = mask_lookup.get(stem) or mask_lookup.get(stem.replace("image", "mask"))
        
        rel_parts = ip.relative_to(root).parts
        
        split = "unassigned"
        host = "Unknown"
        disease = "Unknown"

        for part in rel_parts[:-1]:
            p_low = part.lower()
            if p_low in {"train", "training", "val", "validation", "valid", "test", "testing"}:
                split = "train" if "train" in p_low else ("val" if "val" in p_low else "test")
            elif "___" in part:
                h, d = part.split("___", 1)
                host = h.replace("_", " ").strip()
                disease = d.replace("_", " ").strip()
            elif "_" in part and host == "Unknown":
                pieces = part.split("_")
                host = pieces[0].capitalize()
                disease = " ".join(pieces[1:]).capitalize()

        records.append({
            "sample_id": f"sample_{idx:06d}",
            "image_path": str(ip),
            "mask_path": str(mask_path) if mask_path else None,
            "host": host,
            "disease": disease,
            "split": split,
            "source": "PlantSeg",
            "license": "Unknown",
            "meta_mask_ratio": None,
            "meta_width": None,
            "meta_height": None,
            "original_index": idx
        })

    return pd.DataFrame(records)


def discover_dataset(data_dir: Path | str) -> DatasetDiscoveryResult:
    """Perform end-to-end dynamic dataset discovery.
    
    Args:
        data_dir: Path to dataset root directory.
        
    Returns:
        DatasetDiscoveryResult with standardized records DataFrame and metadata summary.
    """
    root = Path(data_dir).resolve()
    logger.info(f"Starting dataset discovery in: {root}")

    if not root.exists():
        logger.warning(f"Dataset root directory does not exist: {root}")
        return DatasetDiscoveryResult(
            dataset_root=root,
            records_df=pd.DataFrame(columns=[
                "sample_id", "image_path", "mask_path", "host", "disease",
                "split", "source", "license", "meta_mask_ratio", "meta_width", "meta_height"
            ]),
            structure_type="missing_directory",
            summary={"status": "Directory does not exist yet"}
        )

    images, masks, meta_files, annot_files, doc_files = _find_all_files(root)
    logger.info(f"Discovered: {len(images)} raw images, {len(masks)} raw masks, "
                f"{len(meta_files)} metadata tables, {len(annot_files)} JSON annotation files, {len(doc_files)} doc files.")

    # Build fast lookup indexes by filename
    image_lookup: Dict[str, Path] = {p.name: p for p in images}
    mask_lookup: Dict[str, Path] = {p.name: p for p in masks}

    # Check for metadata CSV/TSV tables first
    primary_meta_df: Optional[pd.DataFrame] = None
    used_meta_file: Optional[Path] = None

    # Priority sort: CSV with 'meta' or 'dataset' first
    sorted_meta = sorted(meta_files, key=lambda p: (
        0 if p.suffix.lower() in {".csv", ".tsv"} and any(k in p.name.lower() for k in ["meta", "dataset", "labels", "plantseg"]) else (
            1 if p.suffix.lower() in {".csv", ".tsv"} else 2
        ),
        -p.stat().st_size
    ))

    for mf in sorted_meta:
        df = _parse_metadata_file(mf, root)
        if df is not None and len(df) > 0:
            primary_meta_df = df
            used_meta_file = mf
            break

    if primary_meta_df is not None:
        logger.info(f"Standardizing metadata from {used_meta_file.name}")
        records_df = _standardize_metadata_df(primary_meta_df, root, image_lookup, mask_lookup)
        structure_type = "metadata_driven"
    elif len(images) > 0:
        logger.info("No parseable metadata table found. Inferring from filesystem layout.")
        records_df = _infer_from_filesystem(root, images, masks)
        structure_type = "directory_hierarchy"
    else:
        logger.warning("No images or metadata files found in dataset root.")
        records_df = pd.DataFrame(columns=[
            "sample_id", "image_path", "mask_path", "host", "disease",
            "split", "source", "license", "meta_mask_ratio", "meta_width", "meta_height"
        ])
        structure_type = "empty_or_unrecognized"

    discovered_splits = sorted(list(set(records_df["split"].unique()))) if len(records_df) > 0 else []

    summary = {
        "dataset_root": str(root),
        "structure_type": structure_type,
        "total_records": len(records_df),
        "metadata_file_used": str(used_meta_file) if used_meta_file else None,
        "raw_image_files_found": len(images),
        "raw_mask_files_found": len(masks),
        "annotation_json_files": [p.name for p in annot_files],
        "discovered_splits": discovered_splits,
        "unique_hosts": int(records_df["host"].nunique()) if len(records_df) > 0 else 0,
        "unique_diseases": int(records_df["disease"].nunique()) if len(records_df) > 0 else 0,
        "records_with_image_path": int(records_df["image_path"].notna().sum()) if len(records_df) > 0 else 0,
        "records_with_mask_path": int(records_df["mask_path"].notna().sum()) if len(records_df) > 0 else 0
    }

    return DatasetDiscoveryResult(
        dataset_root=root,
        records_df=records_df,
        metadata_files=meta_files,
        split_files=[p for p in meta_files if "split" in p.name.lower()],
        annotation_files=annot_files,
        doc_files=doc_files,
        structure_type=structure_type,
        discovered_splits=discovered_splits,
        summary=summary
    )
