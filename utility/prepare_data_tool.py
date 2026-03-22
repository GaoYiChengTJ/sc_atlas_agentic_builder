"""
PrepareDataTool — load, validate, and unify input datasets into a single AnnData.

Handles multiple input formats (h5ad, 10X h5, 10X MTX directory, CSV/TSV, loom).
For multiple files: detects format, harmonizes gene names (Ensembl → symbol
conversion, species prefix stripping), intersects genes, and concatenates
with batch labels.

Does NOT preprocess (no QC, normalization, HVG, or PCA) — that is left to
run_preprocessing so the agent can control parameters.
"""

import logging
import os
import time
from typing import Optional

from ..base import ToolWrapper

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "batch_key": "batch",
    "gene_id_type": "auto",
    "gene_intersection_mode": "intersection",
    "obs_key_mapping": None,
}

# Common var columns that hold gene symbols in 10X / scanpy outputs.
_SYMBOL_COLUMNS = ("gene_symbols", "gene_name", "gene_names", "Symbol", "symbol")
# Common var columns that hold Ensembl IDs.
_ENSEMBL_COLUMNS = ("gene_ids", "gene_id", "ensembl_id", "Accession")


def _detect_format(path: str) -> str:
    """Detect input file/directory format."""
    if os.path.isdir(path):
        contents = os.listdir(path)
        mtx_files = [f for f in contents if f.endswith(".mtx") or f.endswith(".mtx.gz")]
        if mtx_files:
            return "10x_mtx"
        raise ValueError(
            f"Directory '{path}' does not look like a 10X MTX directory "
            f"(no .mtx or .mtx.gz file found). Contents: {contents[:10]}"
        )

    ext = path.lower()
    if ext.endswith(".h5ad"):
        return "h5ad"
    if ext.endswith(".h5"):
        return "10x_h5"
    if ext.endswith(".csv") or ext.endswith(".csv.gz"):
        return "csv"
    if ext.endswith(".tsv") or ext.endswith(".tsv.gz"):
        return "tsv"
    if ext.endswith(".loom"):
        return "loom"
    if ext.endswith(".mtx") or ext.endswith(".mtx.gz"):
        return "mtx"

    raise ValueError(
        f"Cannot detect format for '{path}'. "
        f"Supported: .h5ad, .h5 (10X), 10X MTX directory, .csv, .tsv, .loom"
    )


def _detect_transposed(adata) -> bool:
    """Heuristic: detect if a CSV/TSV was loaded genes×cells (transposed).

    Signs of transposition:
      - Many more obs (rows) than vars (columns), AND
      - obs_names look like gene symbols (contain known marker genes).
    """
    if adata.n_obs <= adata.n_vars:
        return False

    # Check if obs_names contain common gene markers.
    known_genes = {
        "GAPDH", "ACTB", "CD3D", "CD14", "CD79A", "TP53", "MALAT1",
        "MT-CO1", "PTPRC", "EPCAM", "Gapdh", "Actb",
    }
    obs_set = set(adata.obs_names[:500].tolist())
    matches = obs_set & known_genes
    if len(matches) >= 2:
        return True

    # Also flag if obs >> vars by a large ratio and obs_names look gene-like
    # (short strings with uppercase letters).
    if adata.n_obs > adata.n_vars * 5:
        sample = adata.obs_names[:100].tolist()
        gene_like = sum(1 for s in sample if len(s) < 20 and any(c.isupper() for c in s))
        if gene_like > 80:
            return True

    return False


def _load_single(path: str, fmt: str = None):
    """Load a single dataset into AnnData, auto-detecting format."""
    import anndata as ad

    if fmt is None:
        fmt = _detect_format(path)

    logger.info(f"  Loading '{os.path.basename(path)}' as {fmt}")

    if fmt == "h5ad":
        adata = ad.read_h5ad(path)
    elif fmt == "10x_h5":
        import scanpy as sc
        adata = sc.read_10x_h5(path)
    elif fmt == "10x_mtx":
        import scanpy as sc
        adata = sc.read_10x_mtx(path)
    elif fmt == "csv":
        adata = ad.read_csv(path)
    elif fmt == "tsv":
        adata = ad.read_csv(path, delimiter="\t")
    elif fmt == "loom":
        adata = ad.read_loom(path)
    elif fmt == "mtx":
        import scanpy as sc
        parent = os.path.dirname(path) or "."
        parent_contents = os.listdir(parent)
        has_barcodes = any(f.startswith("barcodes") for f in parent_contents)
        has_genes = any(
            f.startswith("genes") or f.startswith("features")
            for f in parent_contents
        )
        if not (has_barcodes and has_genes):
            missing = []
            if not has_barcodes:
                missing.append("barcodes.tsv(.gz)")
            if not has_genes:
                missing.append("genes.tsv(.gz) or features.tsv(.gz)")
            raise ValueError(
                f"Bare .mtx file '{path}' requires 10X sidecar files in the "
                f"same directory. Missing: {', '.join(missing)}"
            )
        adata = sc.read_10x_mtx(parent)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    # Ensure var_names and obs_names are strings.
    adata.var_names = adata.var_names.astype(str)
    adata.obs_names = adata.obs_names.astype(str)

    # Fix transposed CSV/TSV (genes as rows, cells as columns).
    if fmt in ("csv", "tsv") and _detect_transposed(adata):
        logger.info("  Detected transposed matrix (genes×cells) — transposing")
        adata = adata.T

    # Make names unique if needed.
    if not adata.var_names.is_unique:
        n_dup = adata.var_names.duplicated().sum()
        logger.warning(f"  {n_dup} duplicate gene names found — making unique")
        adata.var_names_make_unique()
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()

    return adata


def _detect_gene_id_type(var_names) -> str:
    """Heuristic: are gene names Ensembl IDs or symbols?"""
    sample = var_names[:100].tolist()
    ensembl_count = sum(
        1 for g in sample
        if g.startswith("ENSG") or g.startswith("ENSMUSG") or g.startswith("ENS")
    )
    if ensembl_count > len(sample) * 0.5:
        return "ensembl"
    return "symbol"


def _resolve_gene_names(adata) -> tuple:
    """Ensure var_names are gene symbols; store Ensembl IDs in var column.

    Handles two cases:
      1. var_names are Ensembl IDs with symbols available in a var column
         → swap: symbols become var_names, Ensembl IDs stored in var['ensembl_id'].
      2. var_names are Ensembl IDs with NO symbol column available
         → keep Ensembl IDs as var_names (cannot convert without mapping table).

    Returns (adata, action_taken) where action_taken is a string describing
    what was done, or None if no change.
    """
    import pandas as pd

    id_type = _detect_gene_id_type(adata.var_names)

    if id_type != "ensembl":
        return adata, None

    # var_names are Ensembl IDs — look for a symbol column.
    symbol_col = None
    for col in _SYMBOL_COLUMNS:
        if col in adata.var.columns:
            symbol_col = col
            break

    if symbol_col is None:
        logger.info("  Gene names are Ensembl IDs but no symbol column found — keeping as-is")
        return adata, "ensembl_no_symbol_column"

    symbols = adata.var[symbol_col].astype(str)

    # Check quality: skip if most symbols are empty or 'nan'.
    valid = symbols.str.len() > 0
    valid &= symbols != "nan"
    valid &= symbols != "None"
    valid &= symbols != ""
    n_valid = int(valid.sum())

    if n_valid < adata.n_vars * 0.5:
        logger.warning(
            f"  Symbol column '{symbol_col}' has only {n_valid}/{adata.n_vars} "
            "valid entries — keeping Ensembl IDs as var_names"
        )
        return adata, "ensembl_low_quality_symbols"

    # Store original Ensembl IDs.
    adata.var["ensembl_id"] = adata.var_names.tolist()

    # Use symbols as var_names; fall back to Ensembl ID where symbol is missing.
    new_names = symbols.where(valid, adata.var_names)
    adata.var_names = pd.Index(new_names.values)

    # Make unique after swap (symbols can have duplicates).
    if not adata.var_names.is_unique:
        n_dup = int(adata.var_names.duplicated().sum())
        logger.info(f"  {n_dup} duplicate symbols after Ensembl→symbol swap — making unique")
        adata.var_names_make_unique()

    logger.info(
        f"  Converted var_names from Ensembl IDs to symbols "
        f"(via '{symbol_col}', {n_valid} mapped)"
    )
    return adata, "ensembl_to_symbol"


def _strip_species_prefix(var_names):
    """Remove common species prefixes like 'GRCh38_', 'mm10_', 'hg19_'.

    After stripping, makes names unique if collisions are introduced.
    """
    import pandas as pd

    prefixes = ("GRCh38_", "GRCh37_", "hg19_", "hg38_", "mm10_", "mm9_")
    stripped = var_names.astype(str)
    n_stripped = 0
    for prefix in prefixes:
        mask = stripped.str.startswith(prefix)
        if mask.any():
            stripped = stripped.where(~mask, stripped.str[len(prefix):])
            n_stripped += int(mask.sum())

    if n_stripped > 0:
        logger.info(f"  Stripped species prefix from {n_stripped} gene names")

    result = pd.Index(stripped)
    if not result.is_unique:
        n_dup = int(result.duplicated().sum())
        logger.warning(
            f"  {n_dup} duplicate gene names after prefix stripping — making unique"
        )
        import anndata as ad
        result = pd.Index(ad.utils.make_index_unique(result))

    return result


def _validate_counts(adata) -> dict:
    """Check whether .X looks like raw counts vs. normalized data."""
    import numpy as np
    from scipy import sparse

    X = adata.X

    # For sparse matrices, X.data only contains non-zero entries.
    # We need to include zeros in the sample for accurate heuristics.
    if sparse.issparse(X):
        n_elements = X.shape[0] * X.shape[1]
        n_nonzero = X.nnz
        n_zero = n_elements - n_nonzero
        nonzero_vals = X.data
    else:
        flat = X.ravel()
        n_elements = len(flat)
        nonzero_vals = flat[flat != 0] if n_elements > 0 else flat
        n_nonzero = len(nonzero_vals)
        n_zero = n_elements - n_nonzero

    if n_elements == 0 or n_nonzero == 0:
        return {
            "likely_type": "unknown",
            "is_integer": True,
            "has_negative": False,
            "min_val": 0.0,
            "max_val": 0.0,
            "sparsity": 1.0,
        }

    # Sample non-zero values, then mix in the right proportion of zeros.
    sample_size = min(10_000, n_elements)
    rng = np.random.default_rng(42)

    zero_frac = n_zero / n_elements
    n_sample_zeros = int(sample_size * zero_frac)
    n_sample_nonzero = sample_size - n_sample_zeros

    if n_sample_nonzero > len(nonzero_vals):
        sampled_nz = np.asarray(nonzero_vals, dtype=np.float64)
    else:
        sampled_nz = rng.choice(
            np.asarray(nonzero_vals, dtype=np.float64),
            size=n_sample_nonzero, replace=False,
        )

    data = np.concatenate([sampled_nz, np.zeros(n_sample_zeros)])

    has_negative = bool(np.any(data < 0))
    is_integer = bool(np.allclose(data, np.round(data), atol=1e-6))
    max_val = float(np.max(data))
    min_val = float(np.min(data))
    sparsity = round(zero_frac, 3)

    if has_negative:
        likely_type = "scaled_or_residuals"
    elif is_integer:
        likely_type = "raw_counts"
    elif max_val < 20:
        likely_type = "log_normalized"
    else:
        likely_type = "normalized"

    return {
        "likely_type": likely_type,
        "is_integer": is_integer,
        "has_negative": has_negative,
        "min_val": round(min_val, 3),
        "max_val": round(max_val, 3),
        "sparsity": sparsity,
    }


def _harmonize_obs_columns(adatas: list, obs_key_mapping: dict = None) -> tuple:
    """Rename obs columns using an explicit mapping provided by the caller
    (typically the LLM agent after inspecting per-file obs_columns).

    Parameters
    ----------
    adatas : list of AnnData
    obs_key_mapping : dict, optional
        {canonical_name: [synonym1, synonym2, ...]} or
        {canonical_name: synonym_string}.
        Each synonym found in a dataset's obs columns is renamed to the
        canonical name.

    Returns (adatas, rename_log) where rename_log records what was renamed.
    """
    if not obs_key_mapping:
        return adatas, {}

    # Normalize mapping: ensure values are lists.
    mapping = {}
    for canonical, synonyms in obs_key_mapping.items():
        if isinstance(synonyms, str):
            synonyms = [synonyms]
        mapping[canonical] = synonyms

    rename_log = {}  # {file_index: {old_name: new_name}}

    for i, adata_i in enumerate(adatas):
        renames = {}
        obs_cols = set(adata_i.obs.columns)

        for canonical, synonyms in mapping.items():
            # Skip if the canonical name already exists.
            if canonical in obs_cols:
                continue

            for syn in synonyms:
                if syn in obs_cols and syn not in renames:
                    renames[syn] = canonical
                    break

        if renames:
            adatas[i].obs = adatas[i].obs.rename(columns=renames)
            rename_log[i] = renames
            for old, new in renames.items():
                logger.info(f"  file {i}: renamed obs column '{old}' → '{new}'")

    return adatas, rename_log


def _harmonize_var_columns(adatas: list) -> list:
    """Keep only var columns shared across all datasets before concat.

    Different formats produce different var columns (10X h5 has 'gene_ids',
    'feature_types'; h5ad may have 'highly_variable', etc.). Non-shared
    columns would be filled with NaN by ad.concat, which is confusing.
    We keep only the intersection of columns and log what was dropped.
    """
    if len(adatas) <= 1:
        return adatas

    var_col_sets = [set(a.var.columns) for a in adatas]
    common_cols = var_col_sets[0]
    for s in var_col_sets[1:]:
        common_cols &= s

    all_cols = set().union(*var_col_sets)
    dropped_cols = all_cols - common_cols

    if dropped_cols:
        logger.info(
            f"  Dropping non-shared var columns before merge: "
            f"{sorted(dropped_cols)}"
        )
        for i in range(len(adatas)):
            cols_to_drop = [c for c in adatas[i].var.columns if c not in common_cols]
            if cols_to_drop:
                adatas[i].var = adatas[i].var.drop(columns=cols_to_drop)

    return adatas


def prepare_data(
    input_paths: list[str],
    batch_key: str = "batch",
    gene_id_type: str = "auto",
    gene_intersection_mode: str = "intersection",
    obs_key_mapping: dict = None,
):
    """
    Load and unify one or more datasets into a single AnnData.

    Parameters
    ----------
    input_paths : list of str
        Paths to input files or 10X MTX directories.
    batch_key : str
        Column name for batch labels when concatenating multiple files.
    gene_id_type : str
        'symbol', 'ensembl', or 'auto' (detect from gene names).
    gene_intersection_mode : str
        'intersection' keeps only shared genes, 'union' keeps all (filling
        missing with zeros).
    obs_key_mapping : dict, optional
        Explicit mapping to harmonize obs column names across datasets.
        Format: {canonical_name: [synonym1, synonym2, ...]}.
        E.g. {"cell_type": ["annotation", "CellType", "labels"]}.
        Review per-file obs_columns in the result to decide the mapping.

    Returns
    -------
    tuple of (adata, stats_dict)
    """
    import anndata as ad

    t0 = time.time()

    if not input_paths:
        raise ValueError("No input paths provided.")

    # ── Load all files ──
    adatas = []
    batch_names = []
    per_file_info = []

    for path in input_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input path not found: {path}")

        fmt = _detect_format(path)
        adata_i = _load_single(path, fmt)

        name = os.path.splitext(os.path.basename(path))[0]
        if os.path.isdir(path):
            name = os.path.basename(path.rstrip("/"))

        info = {
            "name": name,
            "path": path,
            "format": fmt,
            "n_cells": adata_i.n_obs,
            "n_genes": adata_i.n_vars,
        }

        # Strip species prefixes from gene names.
        adata_i.var_names = _strip_species_prefix(adata_i.var_names)

        # Resolve gene names: convert Ensembl IDs → symbols if possible.
        adata_i, gene_action = _resolve_gene_names(adata_i)
        if gene_action:
            info["gene_name_action"] = gene_action

        # Detect gene ID type (after resolution attempt).
        if gene_id_type == "auto":
            detected = _detect_gene_id_type(adata_i.var_names)
            info["gene_id_type"] = detected
        else:
            info["gene_id_type"] = gene_id_type

        # Validate count matrix.
        info["matrix_type"] = _validate_counts(adata_i)

        # Record obs columns with sample values for LLM inspection.
        obs_summary = {}
        for col in adata_i.obs.columns:
            series = adata_i.obs[col]
            n_unique = int(series.nunique())
            # Show a few example values so the LLM can understand semantics.
            examples = series.dropna().unique()[:5].tolist()
            examples = [str(v) for v in examples]
            obs_summary[col] = {
                "n_unique": n_unique,
                "dtype": str(series.dtype),
                "examples": examples,
            }
        info["obs_columns"] = obs_summary

        per_file_info.append(info)
        adatas.append(adata_i)
        batch_names.append(name)

        logger.info(
            f"  {name}: {info['n_cells']} cells x {info['n_genes']} genes "
            f"({fmt}, {info['matrix_type']['likely_type']})"
        )

    # ── Single file: return directly ──
    if len(adatas) == 1:
        adata = adatas[0]
        matrix_type = per_file_info[0]["matrix_type"]
        adata.uns["matrix_type"] = matrix_type
        stats = {
            "n_files": 1,
            "per_file": per_file_info,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "batch_key": batch_key if batch_key in adata.obs.columns else None,
            "batches": (
                sorted(adata.obs[batch_key].unique().astype(str).tolist())
                if batch_key in adata.obs.columns
                else None
            ),
            "matrix_type": matrix_type,
            "elapsed_s": round(time.time() - t0, 1),
        }
        return adata, stats

    # ── Multiple files: pre-merge checks ──

    # Check for mixed matrix types — error out if types are incompatible.
    matrix_types = [info["matrix_type"]["likely_type"] for info in per_file_info]
    unique_types = set(matrix_types)
    mixed_types = len(unique_types) > 1

    if mixed_types:
        # Some mixtures are acceptable (e.g. raw_counts + unknown), others are not.
        critical_types = unique_types - {"unknown"}
        if len(critical_types) > 1:
            type_details = [
                f"  {info['name']}: {info['matrix_type']['likely_type']}"
                for info in per_file_info
            ]
            raise ValueError(
                "Cannot merge datasets with different matrix types:\n"
                + "\n".join(type_details)
                + "\nAll files must contain the same type of data "
                "(e.g., all raw counts or all normalized). "
                "Preprocess files to a common state before merging."
            )

    # Check for mixed gene ID types after resolution.
    gene_id_types = [info.get("gene_id_type", "unknown") for info in per_file_info]
    unique_gene_types = set(gene_id_types)
    if len(unique_gene_types) > 1 and "ensembl" in unique_gene_types and "symbol" in unique_gene_types:
        type_details = [
            f"  {info['name']}: {info.get('gene_id_type', '?')}"
            for info in per_file_info
        ]
        logger.warning(
            "Mixed gene ID types across files (some Ensembl, some symbols):\n"
            + "\n".join(type_details)
            + "\nGene intersection may be very low. Consider converting all "
            "to the same ID type."
        )

    # ── Merge ──
    logger.info(f"Merging {len(adatas)} datasets...")

    # Harmonize obs column names (e.g. "celltype" → "cell_type").
    adatas, obs_rename_log = _harmonize_obs_columns(adatas, obs_key_mapping)

    # Harmonize var columns to avoid NaN-filled columns after concat.
    adatas = _harmonize_var_columns(adatas)

    # Compute gene overlap.
    gene_sets = [set(a.var_names) for a in adatas]
    common_genes = gene_sets[0]
    for gs in gene_sets[1:]:
        common_genes &= gs
    common_genes = sorted(common_genes)
    all_genes = sorted(set().union(*gene_sets))

    logger.info(
        f"  Gene overlap: {len(common_genes)} shared / {len(all_genes)} total"
    )

    if gene_intersection_mode == "intersection":
        if len(common_genes) < 200:
            logger.warning(
                f"  Only {len(common_genes)} shared genes — this is very low. "
                "Check gene naming conventions across files. "
                "Consider gene_intersection_mode='union'."
            )
        for i in range(len(adatas)):
            adatas[i] = adatas[i][:, common_genes].copy()

    # Concatenate.
    adata = ad.concat(
        adatas,
        label=batch_key,
        keys=batch_names,
        join="outer" if gene_intersection_mode == "union" else "inner",
        fill_value=0,
    )

    # Make obs_names unique after concat.
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()

    # Store the consensus matrix type in adata.uns for downstream tools.
    # For multi-file merges, all files have the same type (mixed types
    # were rejected earlier).
    consensus_type = (unique_types - {"unknown"}).pop() if (unique_types - {"unknown"}) else "unknown"
    adata.uns["matrix_type"] = {"likely_type": consensus_type}

    stats = {
        "n_files": len(adatas),
        "per_file": per_file_info,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "batch_key": batch_key,
        "batches": batch_names,
        "n_batches": len(batch_names),
        "gene_overlap": {
            "shared": len(common_genes),
            "total_union": len(all_genes),
            "mode": gene_intersection_mode,
        },
        "matrix_types": matrix_types,
        "mixed_matrix_types": mixed_types,
        "obs_columns_renamed": {
            batch_names[i]: renames
            for i, renames in obs_rename_log.items()
        } if obs_rename_log else None,
        "elapsed_s": round(time.time() - t0, 1),
    }

    logger.info(
        f"  Merged: {adata.n_obs} cells x {adata.n_vars} genes, "
        f"{len(batch_names)} batches"
    )

    return adata, stats


class PrepareDataTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "prepare_data"

    @property
    def input_type(self) -> str:
        return "files"

    @property
    def output_type(self) -> str:
        return "counts"

    def get_param_space(self) -> dict:
        return {
            "input_paths": {"type": "list_string", "required": True},
            "batch_key": {"type": "categorical", "choices": ["batch", "sample_id", "study"]},
            "gene_id_type": {"type": "categorical", "choices": ["auto", "symbol", "ensembl"]},
            "gene_intersection_mode": {"type": "categorical", "choices": ["intersection", "union"]},
            "obs_key_mapping": {"type": "object", "description": "Map canonical obs column names to synonyms"},
        }

    def run(self, adata=None, params: dict = None, batch_key: Optional[str] = None):
        """
        Load and unify input datasets.

        Parameters
        ----------
        adata : ignored (data is loaded from files).
        params : dict with:
            input_paths (required): list of file paths.
            batch_key ('batch'): column name for batch labels.
            gene_id_type ('auto'): 'symbol', 'ensembl', or 'auto'.
            gene_intersection_mode ('intersection'): 'intersection' or 'union'.
            obs_key_mapping (None): {canonical: [synonyms]} to harmonize
                obs column names across datasets.
        batch_key : optional override for batch_key in params.

        Returns
        -------
        tuple of (adata, stats_dict)
        """
        params = params or {}
        p = {**_DEFAULTS, **params}

        if batch_key is not None:
            p["batch_key"] = batch_key

        input_paths = p.get("input_paths")
        if not input_paths:
            raise ValueError("input_paths is required.")

        return prepare_data(
            input_paths=input_paths,
            batch_key=p["batch_key"],
            gene_id_type=p["gene_id_type"],
            gene_intersection_mode=p["gene_intersection_mode"],
            obs_key_mapping=p.get("obs_key_mapping"),
        )
