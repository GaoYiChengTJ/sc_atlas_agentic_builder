"""
Stateless preprocessing operations for single-cell data.

Each function: (adata, **params) -> (adata, stats_dict).
"""

import logging
from typing import Any, Optional

import anndata as ad
import numpy as np
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


def filter_cells_and_genes(
    adata: ad.AnnData,
    min_genes: int = 200,
    max_genes: int = 8000,
    min_counts: int = 500,
    max_counts: Optional[int] = None,
    min_cells_per_gene: int = 3,
    max_mt_percent: Optional[float] = 20.0,
    mt_prefix: str = "MT-",
    batch_key: Optional[str] = None,
    min_cells_per_batch: int = 50,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """QC filter cells and genes using scanpy. Returns (adata, stats)."""
    import scanpy as sc

    n_cells_before, n_genes_before = adata.n_obs, adata.n_vars

    # Compute QC metrics first
    if mt_prefix:
        adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=False)
    else:
        sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=False)

    # Cell filtering — single mask using QC metric columns for consistency
    keep = (
        (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["n_genes_by_counts"] <= max_genes)
        & (adata.obs["total_counts"] >= min_counts)
    )
    if max_counts is not None:
        keep &= adata.obs["total_counts"] <= max_counts
    if max_mt_percent is not None and "pct_counts_mt" in adata.obs:
        keep &= adata.obs["pct_counts_mt"] <= max_mt_percent

    # Batch filtering — combine into the same cell mask
    if batch_key and batch_key in adata.obs.columns:
        bc = adata.obs[batch_key].value_counts()
        valid = bc[bc >= min_cells_per_batch].index
        keep &= adata.obs[batch_key].isin(valid)

    # Single slice for all cell removal
    adata = adata[keep].copy()

    # Gene filtering
    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    # Recompute QC metrics so returned adata has accurate metadata.
    # Use percent_top=[] to avoid IndexError when cells have fewer genes
    # than scanpy's default percent_top=[50, 100, 200, 500].
    if mt_prefix:
        adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=False, percent_top=[])
    else:
        sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=False, percent_top=[])

    logger.info(f"[QC] {n_cells_before}→{adata.n_obs} cells, {n_genes_before}→{adata.n_vars} genes")
    return adata, {
        "n_cells_before": n_cells_before, "n_cells_after": adata.n_obs,
        "n_genes_before": n_genes_before, "n_genes_after": adata.n_vars,
    }


def _run_doubletdetection_single(
    adata_batch: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray]:
    """Run doubletdetection on a single batch."""
    import doubletdetection

    clf = doubletdetection.BoostClassifier(
        n_iters=10, standard_scaling=True, verbose=False,
    )
    labels = clf.fit(adata_batch.X).predict()
    predicted = np.asarray(labels).astype(bool)
    scores = np.asarray(clf.doublet_score())
    return scores, predicted


def detect_doublets(
    adata: ad.AnnData,
    method: str = "scrublet",
    threshold: Optional[float] = None,
    batch_key: Optional[str] = None,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Detect and remove doublets per batch. Returns (adata, stats).

    Parameters
    ----------
    batch_key : optional column in adata.obs identifying the sample/lane.
        When provided, doublet detection runs independently per batch to
        respect the statistical assumptions of the algorithms. When None,
        the entire dataset is treated as a single batch.
    """
    import scanpy as sc

    n_before = adata.n_obs
    stats: dict[str, Any] = {"method": method, "n_before": n_before}

    try:
        if method == "scrublet":
            # sc.pp.scrublet handles batch_key natively — it iterates
            # per batch internally without redundant deep copies.
            adata_work = adata.copy()
            sc.pp.scrublet(adata_work, batch_key=batch_key, threshold=threshold)
            scores = adata_work.obs["doublet_score"].values
            predicted = adata_work.obs["predicted_doublet"].values.astype(bool)

            # Collect threshold info from uns.
            scrub_uns = adata_work.uns.get("scrublet", {})
            effective_threshold = threshold if threshold is not None else scrub_uns.get("threshold")
            stats["threshold"] = float(effective_threshold) if effective_threshold is not None else None
            if "batches" in scrub_uns:
                stats["per_batch"] = scrub_uns["batches"]

        elif method == "doubletdetection":
            # doubletdetection has no native batch_key support — iterate manually.
            if batch_key and batch_key in adata.obs.columns:
                batches = adata.obs[batch_key].unique()
                scores = np.full(adata.n_obs, np.nan, dtype=np.float64)
                predicted = np.zeros(adata.n_obs, dtype=bool)
                per_batch: dict[str, Any] = {}
                for batch_val in batches:
                    mask = (adata.obs[batch_key] == batch_val).values
                    b_scores, b_pred = _run_doubletdetection_single(adata[mask])
                    scores[mask] = b_scores
                    predicted[mask] = b_pred
                    per_batch[str(batch_val)] = {
                        "n_cells": int(mask.sum()),
                        "n_doublets": int(b_pred.sum()),
                    }
                stats["per_batch"] = per_batch
            else:
                if batch_key and batch_key not in adata.obs.columns:
                    logger.warning(
                        f"[Doublet] batch_key '{batch_key}' not in obs, "
                        "running on full dataset as single batch"
                    )
                scores, predicted = _run_doubletdetection_single(adata)
        else:
            logger.warning(f"[Doublet] Unknown method '{method}', skipping")
            return adata, {**stats, "skipped": True}

        # All scoring succeeded — now commit changes to adata.
        adata.obs["doublet_score"] = scores
        adata.obs["predicted_doublet"] = predicted
        n_doublets = int(predicted.sum())
        stats["n_doublets"] = n_doublets

        if n_doublets > 0:
            adata = adata[~predicted].copy()
            logger.info(f"[Doublet] Removed {n_doublets}/{n_before}")

    except ImportError:
        logger.warning(f"[Doublet] {method} not installed, skipping")
        stats["skipped"] = True
    except Exception as e:
        logger.warning(f"[Doublet] {method} failed: {e}")
        stats["error"] = str(e)

    stats["n_after"] = adata.n_obs
    return adata, stats



_VALID_NORM_METHODS = {"total", "scran", "pearson_residuals"}

def normalize(
    adata: ad.AnnData,
    method: str = "total",
    target_sum: Optional[float] = 1e4,
    log_transform: bool = True,
    pearson_residuals_max_cells: int = 30_000,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Normalize expression. Methods: total, scran, pearson_residuals."""
    import scanpy as sc

    if method not in _VALID_NORM_METHODS:
        raise ValueError(
            f"Unknown normalization method '{method}'. "
            f"Choose from: {sorted(_VALID_NORM_METHODS)}"
        )

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    if method == "scran":
        try:
            import scanpy.external as sce
            sce.pp.scran_normalization(adata)
        except (ImportError, AttributeError):
            logger.warning("[Norm] scran unavailable, falling back to total")
            method = "total"
            sc.pp.normalize_total(adata, target_sum=target_sum)
    elif method == "total":
        sc.pp.normalize_total(adata, target_sum=target_sum)
    elif method == "pearson_residuals":
        if adata.n_obs > pearson_residuals_max_cells:
            logger.warning(
                f"[Norm] pearson_residuals on {adata.n_obs} cells will densify "
                f"the matrix and may cause OOM. Consider running after HVG "
                f"selection, or raise pearson_residuals_max_cells "
                f"(current={pearson_residuals_max_cells})."
            )
        try:
            sc.experimental.pp.normalize_pearson_residuals(adata)
            log_transform = False
        except AttributeError:
            logger.warning("[Norm] pearson_residuals unavailable, falling back to total")
            sc.pp.normalize_total(adata, target_sum=target_sum)
            method = "total"

    if log_transform:
        sc.pp.log1p(adata)

    adata.layers["normalized"] = adata.X.copy()
    logger.info(f"[Norm] {method}, log1p={log_transform}")
    return adata, {"method": method, "log_transformed": log_transform}


def select_hvg(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
    batch_key: Optional[str] = None,
    subset: bool = False,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Select highly variable genes. Returns (adata, stats)."""
    import scanpy as sc

    if batch_key and batch_key not in adata.obs.columns:
        logger.warning(
            f"[HVG] batch_key '{batch_key}' not found in adata.obs "
            f"(available: {list(adata.obs.columns)}). Ignoring batch_key."
        )
        batch_key = None

    n_top_genes = min(n_top_genes, adata.n_vars)
    kwargs = dict(n_top_genes=n_top_genes, batch_key=batch_key)

    if flavor == "seurat_v3" and "counts" in adata.layers:
        try:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer="counts", **kwargs)
        except Exception:
            flavor = "seurat"
            sc.pp.highly_variable_genes(adata, flavor=flavor, **kwargs)
    else:
        if flavor == "seurat_v3":
            flavor = "seurat"
        sc.pp.highly_variable_genes(adata, flavor=flavor, **kwargs)

    n_hvg = int(adata.var["highly_variable"].sum())

    if subset:
        adata.raw = adata
        adata = adata[:, adata.var["highly_variable"]].copy()

    logger.info(f"[HVG] {n_hvg} genes (flavor={flavor})")
    return adata, {"n_hvg": n_hvg, "flavor": flavor, "subset": subset}


def run_pca(
    adata: ad.AnnData,
    n_pcs: int = 50,
    scale: bool = True,
    scale_max_value: float = 10.0,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Scale (optional) + PCA. Returns (adata, stats)."""
    import scanpy as sc

    # Preserve sparsity: skip zero-centering when X is sparse.
    # Centering turns all true zeros into negative values, destroying
    # sparsity and inflating memory by 10-20x on large datasets.
    if scale:
        # When .raw exists and .X is sparse, sc.pp.scale without
        # zero_center modifies .X.data in-place, which corrupts .raw.X
        # because they share the same sparse data array. Copy .X first
        # so that .raw retains the pre-scaled (log-normalized) values
        # that rank_genes_groups expects.
        if adata.raw is not None and issparse(adata.X):
            adata.X = adata.X.copy()
        zero_center = not issparse(adata.X)
        sc.pp.scale(adata, max_value=scale_max_value, zero_center=zero_center)

    # Explicitly route HVG usage instead of relying on scanpy's silent
    # detection of the "highly_variable" column.
    use_hvg = "highly_variable" in adata.var.columns
    n_hvg = int(adata.var["highly_variable"].sum()) if use_hvg else adata.n_vars
    n_pcs = min(n_pcs, n_hvg - 1, adata.n_obs - 1)
    # sklearn PCA with sparse input only supports 'arpack' or 'covariance_eigh'.
    # Use 'arpack' for sparse, 'auto' for dense (lets sklearn pick randomized for large data).
    solver = "arpack" if issparse(adata.X) else "auto"
    # use mask_var for scanpy>=1.10, fall back to use_highly_variable.
    try:
        mask_arg = "highly_variable" if use_hvg else None
        sc.pp.pca(adata, n_comps=n_pcs, mask_var=mask_arg, svd_solver=solver)
    except TypeError:
        sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_hvg, svd_solver=solver)

    var_ratio = adata.uns["pca"]["variance_ratio"]
    stats: dict[str, Any] = {
        "n_pcs": n_pcs, "scaled": scale,
        "zero_centered": scale and not issparse(adata.X),
        "use_highly_variable": use_hvg,
        "n_genes_for_pca": n_hvg,
        "variance_explained_pct": round(float(var_ratio.sum() * 100), 1),
    }
    logger.info(f"[PCA] {n_pcs} PCs, {stats['variance_explained_pct']}% variance, hvg={use_hvg}")
    return adata, stats
