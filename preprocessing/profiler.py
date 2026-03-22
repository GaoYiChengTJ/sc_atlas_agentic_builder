"""
Dataset profiler — extract summary statistics from AnnData for LLM reasoning.

Returns plain dicts serializable to JSON.
"""

import logging

import numpy as np
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


def profile_dataset(adata, batch_key: str = "batch", name: str = "dataset") -> dict:
    """
    Build a profile of a single-cell dataset.

    Returns dict with: basic, counts_dist, genes_dist, mt, batches, gene_stats.
    """
    X = adata.X
    profile = {"name": name}

    # Basic info
    if issparse(X):
        total_counts = np.array(X.sum(axis=1)).flatten()
        genes_per_cell = np.array((X > 0).sum(axis=1)).flatten()
    else:
        total_counts = np.asarray(X.sum(axis=1)).flatten()
        genes_per_cell = np.asarray((X > 0).sum(axis=1)).flatten()

    max_val = float(X.data.max() if issparse(X) else X.max()) if X.shape[0] > 0 else 0

    profile["basic"] = {
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "is_sparse": issparse(X),
        "dtype": str(X.dtype),
        "max_value": round(max_val, 2),
        "likely_raw_counts": bool(max_val > 30 and _looks_like_counts(X)),
        "layers": list(adata.layers.keys()) if adata.layers else [],
        "obsm_keys": list(adata.obsm.keys()) if adata.obsm else [],
    }

    profile["counts_dist"] = _describe(total_counts)
    profile["genes_dist"] = _describe(genes_per_cell)

    # Mitochondrial genes — detect prefix automatically
    mt_prefix = _detect_mt_prefix(adata.var_names)
    if mt_prefix:
        mt_idx = [i for i, g in enumerate(adata.var_names) if g.startswith(mt_prefix)]
        if issparse(X):
            mt_counts = np.array(X[:, mt_idx].sum(axis=1)).flatten()
        else:
            mt_counts = X[:, mt_idx].sum(axis=1).flatten()
        mt_pct = mt_counts / (total_counts + 1e-8) * 100
        profile["mt"] = {"prefix": mt_prefix, "n_genes": len(mt_idx), **_describe(mt_pct)}
    else:
        profile["mt"] = {"prefix": None, "n_genes": 0}

    # Batch composition
    if batch_key in adata.obs.columns:
        bc = adata.obs[batch_key].value_counts()
        profile["batches"] = {
            "n_batches": len(bc),
            "size_min": int(bc.min()), "size_max": int(bc.max()),
            "size_median": int(bc.median()),
        }
    else:
        profile["batches"] = {"n_batches": 0, "batch_key_missing": True}

    logger.info(
        f"[Profiler] {name}: {adata.n_obs}x{adata.n_vars}, "
        f"{profile['batches'].get('n_batches', 0)} batches"
    )
    return profile


def compare_profiles(profiles: list[dict]) -> dict:
    """
    Compare profiles across datasets for unified preprocessing decisions.

    Returns per-dataset summaries and cross-dataset warnings.
    """
    if len(profiles) < 2:
        return {"n_datasets": len(profiles)}

    # Per-dataset summary — everything the LLM needs side-by-side
    per_dataset = {}
    for p in profiles:
        per_dataset[p["name"]] = {
            "n_cells": p["basic"]["n_cells"],
            "n_genes": p["basic"]["n_genes"],
            "likely_raw_counts": p["basic"]["likely_raw_counts"],
            "counts_median": p["counts_dist"]["median"],
            "counts_p5": p["counts_dist"]["p5"],
            "counts_p95": p["counts_dist"]["p95"],
            "genes_median": p["genes_dist"]["median"],
            "mt_prefix": p["mt"].get("prefix"),
            "mt_median": p["mt"].get("median", 0),
            "mt_p95": p["mt"].get("p95", 0),
            "n_batches": p["batches"].get("n_batches", 0),
        }

    # Cross-dataset warnings
    warnings = []

    medians = [d["counts_median"] for d in per_dataset.values()]
    if max(medians) > 3 * max(min(medians), 1):
        warnings.append(f"Sequencing depth varies {min(medians):.0f}–{max(medians):.0f}")

    cells = [d["n_cells"] for d in per_dataset.values()]
    if max(cells) > 10 * max(min(cells), 1):
        warnings.append(f"Cell count imbalance: {min(cells)}–{max(cells)}")

    mt_p95s = [d["mt_p95"] for d in per_dataset.values()]
    if max(mt_p95s) - min(mt_p95s) > 10:
        warnings.append(f"MT% p95 varies {min(mt_p95s):.1f}–{max(mt_p95s):.1f}, consider per-dataset thresholds")

    raw_flags = {n: d["likely_raw_counts"] for n, d in per_dataset.items()}
    if len(set(raw_flags.values())) > 1:
        warnings.append(f"Mixed data states (raw vs normalized): {raw_flags}")

    return {
        "n_datasets": len(profiles),
        "per_dataset": per_dataset,
        "warnings": warnings,
    }


def _detect_mt_prefix(var_names) -> str | None:
    """Auto-detect mitochondrial gene prefix across species."""
    for prefix in ["MT-", "mt-", "Mt-"]:
        if sum(1 for g in var_names if str(g).startswith(prefix)) >= 2:
            return prefix
    return None


def _describe(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {"mean": 0, "median": 0, "std": 0, "p5": 0, "p95": 0}
    return {
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "std": round(float(np.std(arr)), 2),
        "p5": round(float(np.percentile(arr, 5)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
    }


def _looks_like_counts(X, n_sample: int = 1000) -> bool:
    data = X.data if issparse(X) else X.flatten()
    if len(data) == 0:
        return False
    sample = data[:min(n_sample, len(data))]
    nz = sample[sample != 0]
    return bool(len(nz) > 0 and np.mean(np.mod(nz, 1) == 0) > 0.95)
