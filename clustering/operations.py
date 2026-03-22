"""
Stateless clustering operations for single-cell data.

Each function: (adata, **params) -> (adata, stats_dict).
"""

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_neighbors(
    adata,
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    use_rep: str = "X_pca",
    metric: str = "euclidean",
) -> tuple[Any, dict[str, Any]]:
    """Compute nearest-neighbor graph on a specified representation."""
    import scanpy as sc

    if use_rep not in adata.obsm:
        raise ValueError(
            f"use_rep='{use_rep}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )

    n_dims = adata.obsm[use_rep].shape[1]

    # Only apply n_pcs for PCA where variance decay makes truncation
    # meaningful. Integration embeddings (scVI, Harmony, etc.) are dense
    # latent spaces — all dimensions carry equal weight, so slicing them
    # destroys batch-correction information.
    if use_rep == "X_pca" and n_pcs is not None:
        n_pcs = min(n_pcs, n_dims)
    elif use_rep != "X_pca" and n_pcs is not None:
        logger.warning(
            f"[Neighbors] Ignoring n_pcs={n_pcs} for use_rep='{use_rep}'. "
            "All dimensions of integration embeddings are used."
        )
        n_pcs = None

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        metric=metric,
    )

    stats = {
        "n_neighbors": n_neighbors,
        "n_pcs": n_pcs,
        "use_rep": use_rep,
        "metric": metric,
        "n_dims": n_dims,
    }
    logger.info(f"[Neighbors] {use_rep}, k={n_neighbors}, n_pcs={n_pcs}")
    return adata, stats


def run_leiden(
    adata,
    resolution: float = 1.0,
    key_added: str = "leiden",
    random_state: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """Run Leiden community detection. Requires a neighbor graph."""
    import scanpy as sc

    if "neighbors" not in adata.uns:
        raise ValueError("Neighbor graph not found. Run compute_neighbors first.")

    # Scanpy >= 1.10 defaults to igraph-based leiden (flavor="igraph").
    # Older versions use leidenalg. Try igraph first, fall back to leidenalg.
    kwargs = dict(
        resolution=resolution,
        key_added=key_added,
        random_state=random_state,
    )
    try:
        sc.tl.leiden(adata, flavor="igraph", **kwargs)
    except (ImportError, TypeError):
        # TypeError: older scanpy without flavor param
        # ImportError: igraph not installed
        try:
            sc.tl.leiden(adata, **kwargs)
        except ImportError:
            raise ImportError(
                "Leiden clustering requires either 'igraph' (scanpy>=1.10) "
                "or 'leidenalg' (scanpy<1.10). Install one: "
                "pip install igraph  OR  pip install leidenalg"
            )

    labels = adata.obs[key_added]
    n_clusters = labels.nunique()
    cluster_sizes = labels.value_counts().to_dict()

    stats = {
        "method": "leiden",
        "resolution": resolution,
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "key_added": key_added,
    }
    logger.info(f"[Leiden] resolution={resolution}, {n_clusters} clusters")
    return adata, stats


def compute_umap(
    adata,
    min_dist: float = 0.5,
    spread: float = 1.0,
    random_state: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """Compute UMAP embedding for visualization."""
    import scanpy as sc

    if "neighbors" not in adata.uns:
        raise ValueError("Neighbor graph not found. Run compute_neighbors first.")

    sc.tl.umap(adata, min_dist=min_dist, spread=spread, random_state=random_state)

    stats = {"min_dist": min_dist, "spread": spread}
    logger.info(f"[UMAP] min_dist={min_dist}, spread={spread}")
    return adata, stats


def assess_clusters(
    adata,
    cluster_key: str = "leiden",
    use_rep: str = "X_pca",
    max_cells: int = 50_000,
) -> dict[str, Any]:
    """Compute cluster quality metrics. Returns stats dict (does not modify adata)."""
    try:
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
    except ImportError:
        raise ImportError(
            "Cluster assessment requires scikit-learn. "
            "Install it: pip install scikit-learn"
        )

    if cluster_key not in adata.obs.columns:
        raise ValueError(
            f"Cluster key '{cluster_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )
    if use_rep not in adata.obsm:
        raise ValueError(
            f"use_rep='{use_rep}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )

    # Convert categorical/string labels to integer codes for sklearn.
    labels_series = adata.obs[cluster_key]
    if hasattr(labels_series, "cat"):
        label_codes = labels_series.cat.codes.values
    else:
        label_codes = labels_series.astype("category").cat.codes.values

    X = adata.obsm[use_rep]
    n_clusters = len(np.unique(label_codes))

    # Per-cluster cell counts (use original labels for readability).
    cluster_sizes = labels_series.value_counts().to_dict()

    stats: dict[str, Any] = {
        "cluster_key": cluster_key,
        "use_rep": use_rep,
        "n_clusters": n_clusters,
        "n_cells": adata.n_obs,
        "cluster_sizes": cluster_sizes,
    }

    if n_clusters < 2:
        logger.warning(f"[Assess] Only {n_clusters} cluster(s), skipping metrics")
        stats["warning"] = "fewer than 2 clusters, metrics not computed"
        return stats

    # Stratified subsample to preserve rare clusters.
    if adata.n_obs > max_cells:
        rng = np.random.RandomState(0)
        idx = []
        for code in np.unique(label_codes):
            cluster_idx = np.where(label_codes == code)[0]
            # Each cluster gets at least min_per_cluster cells in the sample.
            min_per_cluster = 20
            n_take = max(min_per_cluster, int(len(cluster_idx) * max_cells / adata.n_obs))
            n_take = min(n_take, len(cluster_idx))
            idx.append(rng.choice(cluster_idx, n_take, replace=False))
        idx = np.concatenate(idx)
        X_sub, labels_sub = X[idx], label_codes[idx]
        stats["subsampled_to"] = len(idx)
    else:
        X_sub, labels_sub = X, label_codes

    stats["silhouette_score"] = round(float(silhouette_score(X_sub, labels_sub)), 4)
    stats["calinski_harabasz_score"] = round(float(calinski_harabasz_score(X_sub, labels_sub)), 1)

    logger.info(
        f"[Assess] {n_clusters} clusters, "
        f"silhouette={stats['silhouette_score']}, "
        f"calinski_harabasz={stats['calinski_harabasz_score']}"
    )
    return stats
