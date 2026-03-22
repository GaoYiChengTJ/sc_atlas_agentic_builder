"""
Stateless integration operations for multi-sample single-cell data.

Each function: (adata, **params) -> (adata, stats_dict).
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VALID_METHODS = {"harmony", "scanorama"}
_VALID_HVG_STRATEGIES = {"intersection", "rank", "union"}


def _compute_per_batch_hvg(
    adata,
    batch_key: str,
    n_top_genes: int,
    flavor: str,
    layer: Optional[str],
    min_cells_per_batch: int = 50,
) -> tuple[dict[str, set[str]], str]:
    """
    Compute HVGs per batch. Returns {batch: set_of_genes} and the
    flavor actually used (may change if seurat_v3 falls back).
    """
    import scanpy as sc

    per_batch_hvg: dict[str, set[str]] = {}
    used_flavor = flavor

    for batch in adata.obs[batch_key].unique():
        batch_adata = adata[adata.obs[batch_key] == batch]
        n_cells = batch_adata.n_obs

        if n_cells < min_cells_per_batch:
            logger.warning(
                f"[IntegrationGenes] Batch '{batch}' has {n_cells} cells "
                f"(<{min_cells_per_batch}). Skipping for HVG selection."
            )
            continue

        batch_adata = batch_adata.copy()
        # Cap n_top_genes to what this batch can support.
        batch_n_genes = min(n_top_genes, batch_adata.n_vars, batch_adata.n_obs - 1)

        kwargs = dict(n_top_genes=batch_n_genes, flavor=flavor)
        if layer:
            kwargs["layer"] = layer

        try:
            sc.pp.highly_variable_genes(batch_adata, **kwargs)
        except (ValueError, ArithmeticError):
            # seurat_v3 can fail on small/sparse batches — fall back.
            if flavor == "seurat_v3":
                kwargs.pop("layer", None)
                kwargs["flavor"] = "seurat"
                sc.pp.highly_variable_genes(batch_adata, **kwargs)
                used_flavor = "seurat"
            else:
                raise

        hvg = set(batch_adata.var_names[batch_adata.var["highly_variable"]])
        per_batch_hvg[str(batch)] = hvg

    return per_batch_hvg, used_flavor


def select_integration_genes(
    adata,
    batch_key: str,
    n_top_genes: int = 2000,
    strategy: str = "rank",
    flavor: str = "seurat_v3",
    min_genes_threshold: int = 100,
) -> tuple[Any, dict[str, Any]]:
    """
    Select genes for integration that are variable across batches
    but not driven by batch effects.

    Parameters
    ----------
    batch_key : obs column identifying batches.
    n_top_genes : number of HVGs to select.
    strategy : how to combine per-batch HVG lists.
        - 'rank': rank-based selection across batches (recommended).
        - 'intersection': only genes variable in ALL batches (conservative).
        - 'union': genes variable in ANY batch (permissive).
    flavor : HVG flavor ('seurat_v3' needs counts layer).
    min_genes_threshold : warn if fewer genes than this are selected.
    """
    import scanpy as sc

    if batch_key not in adata.obs.columns:
        raise ValueError(
            f"batch_key='{batch_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    if strategy not in _VALID_HVG_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {sorted(_VALID_HVG_STRATEGIES)}"
        )

    n_batches = adata.obs[batch_key].nunique()
    n_top_genes = min(n_top_genes, adata.n_vars)

    # Use counts layer for seurat_v3 if available.
    layer = "counts" if flavor == "seurat_v3" and "counts" in adata.layers else None
    per_batch_stats: dict[str, int] = {}

    if strategy == "rank":
        # Scanpy's batch_key-aware HVG selection uses rank-based merging.
        kwargs = dict(
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            flavor=flavor,
        )
        if layer:
            kwargs["layer"] = layer
        try:
            sc.pp.highly_variable_genes(adata, **kwargs)
        except (ValueError, ArithmeticError):
            if flavor == "seurat_v3":
                kwargs.pop("layer", None)
                kwargs["flavor"] = "seurat"
                sc.pp.highly_variable_genes(adata, **kwargs)
                flavor = "seurat"
            else:
                raise

    elif strategy in ("intersection", "union"):
        per_batch_hvg, flavor = _compute_per_batch_hvg(
            adata, batch_key, n_top_genes, flavor, layer,
        )

        if not per_batch_hvg:
            raise ValueError("No batches had enough cells for HVG computation.")

        per_batch_stats = {b: len(genes) for b, genes in per_batch_hvg.items()}

        if strategy == "intersection":
            merged = set.intersection(*per_batch_hvg.values())
        else:
            merged = set.union(*per_batch_hvg.values())

        # For union: re-rank by frequency across batches, cap at n_top_genes.
        if strategy == "union" and len(merged) > n_top_genes:
            from collections import Counter
            gene_counts = Counter()
            for genes in per_batch_hvg.values():
                gene_counts.update(genes)
            # Keep genes appearing in the most batches, then by name for stability.
            ranked = sorted(gene_counts.keys(), key=lambda g: (-gene_counts[g], g))
            merged = set(ranked[:n_top_genes])

        adata.var["highly_variable"] = adata.var_names.isin(merged)

    n_hvg = int(adata.var["highly_variable"].sum())

    # Warn if too few genes.
    if n_hvg < min_genes_threshold:
        logger.warning(
            f"[IntegrationGenes] Only {n_hvg} genes selected "
            f"(threshold={min_genes_threshold}). Consider using 'union' "
            "strategy or increasing n_top_genes."
        )

    stats = {
        "n_hvg": n_hvg,
        "strategy": strategy,
        "flavor": flavor,
        "n_batches": n_batches,
        "batch_key": batch_key,
    }
    if per_batch_stats:
        stats["per_batch_hvg_counts"] = per_batch_stats

    logger.info(
        f"[IntegrationGenes] {n_hvg} genes selected "
        f"(strategy={strategy}, {n_batches} batches)"
    )
    return adata, stats


def run_harmony(
    adata,
    batch_key: str,
    n_pcs: Optional[int] = None,
    max_iter: int = 20,
) -> tuple[Any, dict[str, Any]]:
    """Run Harmony integration on PCA embeddings.

    Calls harmonypy directly instead of sc.external.pp.harmony_integrate
    to avoid a shape bug in scanpy's wrapper (it does .T on Z_corr, but
    modern harmonypy already returns (n_cells, n_dims)).
    """
    try:
        import harmonypy
    except ImportError:
        raise ImportError(
            "harmonypy is not installed. Install with: pip install harmonypy"
        )

    if batch_key not in adata.obs.columns:
        raise ValueError(
            f"batch_key='{batch_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    if "X_pca" not in adata.obsm:
        raise ValueError("PCA not found. Run PCA before Harmony.")

    pca = adata.obsm["X_pca"].astype(np.float64)
    if n_pcs is not None:
        pca = pca[:, :min(n_pcs, pca.shape[1])]

    ho = harmonypy.run_harmony(
        pca, adata.obs, batch_key, max_iter_harmony=max_iter,
    )

    # harmonypy Z_corr shape depends on version/backend:
    # - old versions: (n_dims, n_cells) — need .T
    # - modern (PyTorch): (n_cells, n_dims) — already correct
    z = np.asarray(ho.Z_corr)
    if z.shape[0] == pca.shape[1] and z.shape[1] == adata.n_obs:
        z = z.T
    if z.shape[0] != adata.n_obs:
        raise RuntimeError(
            f"Harmony output shape mismatch: expected {adata.n_obs} cells, "
            f"got {z.shape[0]}. Z_corr raw shape: {ho.Z_corr.shape}"
        )

    adata.obsm["X_harmony"] = z

    stats = {
        "method": "harmony",
        "batch_key": batch_key,
        "n_dims": z.shape[1],
        "max_iter": max_iter,
        "output_key": "X_harmony",
    }
    logger.info(f"[Harmony] Done: {stats['n_dims']} dims")
    return adata, stats



def run_scanorama(
    adata,
    batch_key: str,
) -> tuple[Any, dict[str, Any]]:
    """Run Scanorama integration.

    Uses sc.external.pp.scanorama_integrate which takes a single AnnData
    with a batch key column. Cells must be contiguously stored by batch,
    so we sort first if needed.
    """
    try:
        import scanorama  # noqa: F401
    except ImportError:
        raise ImportError(
            "Scanorama is not installed. Install it: pip install scanorama"
        )
    import scanpy as sc

    if batch_key not in adata.obs.columns:
        raise ValueError(
            f"batch_key='{batch_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    if "X_pca" not in adata.obsm:
        raise ValueError("PCA not found. Run PCA before Scanorama.")

    n_batches = adata.obs[batch_key].nunique()

    # Scanorama requires cells from the same batch to be contiguous.
    # Sort by batch_key if not already contiguous.
    batch_values = adata.obs[batch_key].values
    is_contiguous = True
    seen = set()
    prev = None
    for v in batch_values:
        if v != prev:
            if v in seen:
                is_contiguous = False
                break
            seen.add(v)
            prev = v

    if not is_contiguous:
        sort_idx = adata.obs[batch_key].argsort(kind="stable")
        adata = adata[sort_idx].copy()

    sc.external.pp.scanorama_integrate(
        adata,
        key=batch_key,
    )

    if "X_scanorama" not in adata.obsm:
        raise RuntimeError("Scanorama did not produce X_scanorama embedding.")

    n_dims = adata.obsm["X_scanorama"].shape[1]

    stats = {
        "method": "scanorama",
        "batch_key": batch_key,
        "n_batches": n_batches,
        "n_dims": n_dims,
        "output_key": "X_scanorama",
    }
    logger.info(f"[Scanorama] Done: {stats['n_dims']} dims, {n_batches} batches")
    return adata, stats


def run_integration(
    adata,
    method: str,
    batch_key: str,
    **kwargs,
) -> tuple[Any, dict[str, Any]]:
    """
    Dispatch integration to the appropriate method.

    Parameters
    ----------
    method : 'harmony' or 'scanorama'.
    batch_key : obs column for batch.
    **kwargs : method-specific parameters.
    """
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown integration method '{method}'. "
            f"Choose from: {sorted(_VALID_METHODS)}"
        )

    if method == "harmony":
        return run_harmony(adata, batch_key=batch_key, **kwargs)
    elif method == "scanorama":
        return run_scanorama(adata, batch_key=batch_key, **kwargs)


def _validate_eval_inputs(
    adata, batch_key: str, label_key: Optional[str], embed_key: str,
) -> None:
    """Validate inputs for evaluate_integration."""
    if embed_key not in adata.obsm:
        raise ValueError(
            f"embed_key='{embed_key}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )
    if batch_key not in adata.obs.columns:
        raise ValueError(
            f"batch_key='{batch_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )
    if label_key and label_key not in adata.obs.columns:
        raise ValueError(
            f"label_key='{label_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )



def _eval_scib(adata, batch_key, label_key, embed_key) -> dict[str, Any]:
    """Full evaluation using scib-metrics. 40% batch + 60% bio (scIB convention)."""
    import scib_metrics
    from scib_metrics.nearest_neighbors import NeighborsResults
    import scanpy as sc

    batch_scores = {}
    bio_scores = {}
    errors = []

    X = adata.obsm[embed_key]
    batch_labels = adata.obs[batch_key].values
    label_values = adata.obs[label_key].values if label_key else None

    # Compute neighbor graph on the target embedding (needed for graph-based metrics).
    n_neighbors = 15
    sc.pp.neighbors(adata, use_rep=embed_key, n_neighbors=n_neighbors)

    # Extract kNN indices/distances from scanpy's sparse distance matrix.
    # Scanpy stores exactly (n_neighbors - 1) non-zero entries per row in CSR format.
    # Vectorized extraction: reshape the flat arrays into (n_cells, k).
    dist_csr = adata.obsp["distances"].tocsr()
    k_actual = int(dist_csr.nnz / adata.n_obs)  # entries per row
    nn_indices = dist_csr.indices.reshape(adata.n_obs, k_actual).astype(np.int32)
    nn_distances = dist_csr.data.reshape(adata.n_obs, k_actual).astype(np.float64)
    neighbors = NeighborsResults(indices=nn_indices, distances=nn_distances)

    # ── Batch metrics ──
    # silhouette_batch requires cell type labels to measure batch mixing
    # *within* each cell type. Without labels, this metric is meaningless.
    if label_values is not None:
        try:
            batch_scores["asw_batch"] = float(scib_metrics.silhouette_batch(
                X, labels=label_values, batch=batch_labels, rescale=True,
            ))
        except Exception as e:
            batch_scores["asw_batch"] = None
            errors.append(f"ASW_batch: {e}")
    else:
        batch_scores["asw_batch"] = None

    # graph_connectivity checks if same cell type connects across batches.
    if label_values is not None:
        try:
            batch_scores["graph_connectivity"] = float(
                scib_metrics.graph_connectivity(neighbors, labels=label_values)
            )
        except Exception as e:
            batch_scores["graph_connectivity"] = None
            errors.append(f"graph_connectivity: {e}")

    # ── Bio conservation metrics (need labels) ──
    if label_values is not None:
        try:
            bio_scores["asw_label"] = float(scib_metrics.silhouette_label(
                X, labels=label_values, rescale=True,
            ))
        except Exception as e:
            bio_scores["asw_label"] = None
            errors.append(f"ASW_label: {e}")

        try:
            nmi_ari = scib_metrics.nmi_ari_cluster_labels_leiden(
                neighbors, labels=label_values, optimize_resolution=True,
            )
            bio_scores["nmi"] = float(nmi_ari["nmi"])
            bio_scores["ari"] = float(nmi_ari["ari"])
        except Exception as e:
            bio_scores["nmi"] = None
            bio_scores["ari"] = None
            errors.append(f"NMI/ARI: {e}")

        try:
            bio_scores["isolated_label_asw"] = float(
                scib_metrics.isolated_labels(
                    X, labels=label_values, batch=batch_labels, rescale=True,
                )
            )
        except Exception as e:
            bio_scores["isolated_label_asw"] = None
            errors.append(f"isolated_label_asw: {e}")

    if errors:
        logger.warning(f"[EvalIntegration] Some scib metrics failed: {errors}")

    # ── Overall score: 40% batch + 60% bio (scIB convention) ──
    valid_batch = [v for v in batch_scores.values() if v is not None]
    valid_bio = [v for v in bio_scores.values() if v is not None]

    batch_avg = float(np.mean(valid_batch)) if valid_batch else None
    bio_avg = float(np.mean(valid_bio)) if valid_bio else None

    if batch_avg is not None and bio_avg is not None:
        overall = 0.4 * batch_avg + 0.6 * bio_avg
    elif batch_avg is not None:
        overall = batch_avg
    elif bio_avg is not None:
        overall = bio_avg
    else:
        overall = None

    result = {
        "batch_scores": {k: round(v, 4) if v is not None else None for k, v in batch_scores.items()},
        "bio_scores": {k: round(v, 4) if v is not None else None for k, v in bio_scores.items()},
        "batch_avg": round(batch_avg, 4) if batch_avg is not None else None,
        "bio_avg": round(bio_avg, 4) if bio_avg is not None else None,
        "overall_score": round(overall, 4) if overall is not None else None,
        "weighting": "40% batch + 60% bio (scIB convention)",
    }
    if errors:
        result["errors"] = errors
    return result



def evaluate_integration(
    adata,
    batch_key: str,
    label_key: Optional[str] = None,
    embed_key: str = "X_harmony",
    max_cells: int = 50_000,
) -> dict[str, Any]:
    """
    Evaluate integration quality using scib-metrics.

    Uses 40% batch mixing + 60% bio conservation (scIB convention).
    Requires scib-metrics: pip install scib-metrics

    Parameters
    ----------
    batch_key : obs column for batch.
    label_key : optional obs column for cell type labels (needed for bio conservation).
    embed_key : obsm key for the integrated embedding.
    max_cells : subsample for speed on large datasets.
    """
    try:
        import scib_metrics  # noqa: F401
    except ImportError:
        raise ImportError(
            "scib-metrics is required for integration evaluation. "
            "Install with: pip install scib-metrics"
        )

    _validate_eval_inputs(adata, batch_key, label_key, embed_key)

    n_batches = len(np.unique(adata.obs[batch_key].values))

    stats: dict[str, Any] = {
        "embed_key": embed_key,
        "batch_key": batch_key,
        "label_key": label_key,
        "n_cells": adata.n_obs,
        "n_batches": n_batches,
        "n_dims": adata.obsm[embed_key].shape[1],
    }

    # Subsample for memory on large datasets.
    if adata.n_obs > max_cells:
        import scanpy as sc
        sc.pp.subsample(adata, n_obs=max_cells, random_state=0, copy=False)
        stats["subsampled_to"] = max_cells

    metrics = _eval_scib(adata, batch_key, label_key, embed_key)
    stats.update(metrics)

    logger.info(
        f"[EvalIntegration] {embed_key}: "
        f"batch_avg={stats.get('batch_avg')}, "
        f"bio_avg={stats.get('bio_avg')}, "
        f"overall={stats.get('overall_score')}"
    )
    return stats
