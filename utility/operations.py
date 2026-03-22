"""
Stateless operations for subclustering, subcluster annotation,
and cluster merging.
"""

import logging
from typing import Any, Optional

import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)


def _get_cluster_key(adata: ad.AnnData) -> str:
    """Find the clustering column in adata.obs."""
    for key in ("leiden", "louvain"):
        if key in adata.obs.columns:
            return key
    raise ValueError(
        "No clustering found in adata.obs (looked for 'leiden', 'louvain'). "
        "Run clustering first."
    )


def subcluster(
    adata: ad.AnnData,
    cluster_id: str,
    resolution: float = 0.3,
    use_rep: str = "X_pca",
    n_neighbors: int = 15,
    n_top_marker_genes: int = 10,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """
    Subset a cluster and re-cluster + find markers.

    Returns (adata_subset, stats) where adata_subset contains new leiden
    labels and marker genes for the subclusters.
    """
    import scanpy as sc

    cluster_key = _get_cluster_key(adata)
    mask = adata.obs[cluster_key].astype(str) == str(cluster_id)
    n_cells = int(mask.sum())

    if n_cells == 0:
        available = sorted(adata.obs[cluster_key].unique().astype(str))
        raise ValueError(
            f"Cluster '{cluster_id}' not found or empty. Available: {available}"
        )

    if n_cells < 50:
        raise ValueError(
            f"Cluster '{cluster_id}' has only {n_cells} cells, too few to subcluster. "
            "Minimum 50 cells required."
        )

    # Subset.
    adata_sub = adata[mask].copy()

    # Validate embedding exists.
    if use_rep not in adata_sub.obsm:
        available = list(adata_sub.obsm.keys())
        raise ValueError(
            f"use_rep='{use_rep}' not found in obsm. Available: {available}"
        )

    # Neighbors + clustering.
    k = min(n_neighbors, n_cells - 1)
    # For non-PCA embeddings, use all dimensions (n_pcs would slice them).
    n_pcs = None
    sc.pp.neighbors(adata_sub, use_rep=use_rep, n_neighbors=k, n_pcs=n_pcs)
    sc.tl.leiden(adata_sub, resolution=resolution)

    n_subclusters = len(adata_sub.obs["leiden"].unique())

    # Marker genes for subclusters.
    marker_summary = {}
    if n_subclusters <= 1:
        logger.warning(
            f"[Subcluster] Cluster {cluster_id}: only 1 subcluster found at "
            f"resolution={resolution}. Try a higher resolution."
        )
    else:
        # Defensive check: determine best source for DE.
        use_raw = False
        layer = None
        if adata_sub.raw is not None:
            if adata_sub.raw.n_vars > adata_sub.n_vars:
                # .raw has all genes (set before HVG subsetting) — ideal.
                use_raw = True
            else:
                # .raw has same genes as .var — likely set after HVG subset.
                # DE will be limited to HVGs only.
                use_raw = True
                logger.warning(
                    f"[Subcluster] .raw has same gene count as .var ({adata_sub.n_vars}). "
                    "DE is limited to HVGs. For full-genome DE, ensure .raw is set "
                    "before HVG subsetting in preprocessing."
                )
        elif "counts" in adata_sub.layers:
            # No .raw but raw counts available in a layer.
            layer = "counts"
            logger.info("[Subcluster] No .raw found, using layer='counts' for DE.")
        else:
            # Fall back to .X — may be scaled/transformed.
            logger.warning(
                "[Subcluster] No .raw or counts layer found. "
                "DE will run on .X which may be scaled — fold changes may be distorted."
            )

        sc.tl.rank_genes_groups(
            adata_sub, groupby="leiden", method="wilcoxon",
            use_raw=use_raw, layer=layer, pts=True,
            corr_method="benjamini-hochberg",
        )
        for group in adata_sub.obs["leiden"].unique():
            df = sc.get.rank_genes_groups_df(adata_sub, group=str(group))
            top = df.head(n_top_marker_genes)["names"].tolist()
            marker_summary[str(group)] = top

    # Cluster sizes.
    cluster_sizes = adata_sub.obs["leiden"].value_counts().to_dict()
    cluster_sizes = {str(k): int(v) for k, v in cluster_sizes.items()}

    stats = {
        "original_cluster": str(cluster_id),
        "n_cells": n_cells,
        "n_subclusters": n_subclusters,
        "subcluster_sizes": cluster_sizes,
        "top_markers": marker_summary,
        "use_rep": use_rep,
        "resolution": resolution,
    }
    if n_subclusters <= 1:
        stats["warning"] = (
            f"Only 1 subcluster found at resolution={resolution}. "
            "Try a higher resolution (e.g., 0.5 or 1.0)."
        )

    logger.info(
        f"[Subcluster] Cluster {cluster_id}: {n_cells} cells → "
        f"{n_subclusters} subclusters"
    )
    return adata_sub, stats


def annotate_subclusters(
    adata: ad.AnnData,
    adata_sub: ad.AnnData,
    cluster_id: str,
    label_mapping: dict[str, str],
    fine_label_mapping: Optional[dict[str, str]] = None,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """
    Write subcluster labels back to the main adata.

    Maps sub-cluster IDs (from adata_sub.obs["leiden"]) to cell type labels
    and writes them to adata.obs["cell_type"] using the original cell indices.

    Parameters
    ----------
    adata : main AnnData (full dataset).
    adata_sub : subclustered AnnData (from subcluster()).
    cluster_id : the original cluster that was subclustered.
    label_mapping : {sub_cluster_id: cell_type_name}.
    fine_label_mapping : optional {sub_cluster_id: fine_cell_type_name}.
    """
    # Ensure columns exist.
    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "Unknown"
    if fine_label_mapping and "cell_type_fine" not in adata.obs.columns:
        adata.obs["cell_type_fine"] = "Unknown"

    # Validate that adata_sub indices exist in main adata.
    missing_indices = adata_sub.obs.index.difference(adata.obs.index)
    if len(missing_indices) > 0:
        raise ValueError(
            f"adata_sub contains {len(missing_indices)} cell indices not found in "
            "main adata. Ensure adata_sub was created by SubclusterTool from the "
            "same adata object."
        )

    # Check for unmapped subclusters.
    all_sub_ids = set(adata_sub.obs["leiden"].astype(str).unique())
    mapped_sub_ids = set(str(k) for k in label_mapping.keys())
    unmapped = all_sub_ids - mapped_sub_ids
    if unmapped:
        logger.warning(
            f"[AnnotateSubclusters] Subclusters {sorted(unmapped)} are not in "
            f"label_mapping and will keep their existing cell_type value."
        )

    cells_updated = 0
    label_counts: dict[str, int] = {}

    for sub_id, label in label_mapping.items():
        sub_mask = adata_sub.obs["leiden"].astype(str) == str(sub_id)
        cell_indices = adata_sub.obs.index[sub_mask]
        n = int(sub_mask.sum())

        if n == 0:
            available = sorted(all_sub_ids)
            logger.warning(
                f"[AnnotateSubclusters] Sub-cluster '{sub_id}' not found. "
                f"Available: {available}"
            )
            continue

        # Write to main adata using original cell indices.
        adata.obs.loc[cell_indices, "cell_type"] = label
        cells_updated += n
        label_counts[label] = label_counts.get(label, 0) + n

        if fine_label_mapping and str(sub_id) in fine_label_mapping:
            adata.obs.loc[cell_indices, "cell_type_fine"] = fine_label_mapping[str(sub_id)]

    stats = {
        "original_cluster": str(cluster_id),
        "cells_updated": cells_updated,
        "label_counts": label_counts,
        "has_fine_labels": fine_label_mapping is not None,
    }
    if unmapped:
        stats["unmapped_subclusters"] = sorted(unmapped)

    logger.info(
        f"[AnnotateSubclusters] Cluster {cluster_id}: "
        f"{cells_updated} cells → {label_counts}"
    )
    return adata, stats


def merge_clusters(
    adata: ad.AnnData,
    cluster_ids: list[str],
    label: str,
    fine_label: Optional[str] = None,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """
    Assign the same cell type label to multiple clusters.

    Parameters
    ----------
    cluster_ids : list of cluster IDs to merge under one label.
    label : the cell type label to assign.
    fine_label : optional fine-grained label.
    """
    cluster_key = _get_cluster_key(adata)

    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "Unknown"
    if fine_label and "cell_type_fine" not in adata.obs.columns:
        adata.obs["cell_type_fine"] = "Unknown"

    cluster_ids_str = [str(c) for c in cluster_ids]
    available = set(adata.obs[cluster_key].astype(str).unique())

    # Validate all cluster IDs exist before modifying anything.
    missing = [cid for cid in cluster_ids_str if cid not in available]
    if missing:
        raise ValueError(
            f"Cluster(s) {missing} not found. Available: {sorted(available)}"
        )

    total_cells = 0
    per_cluster = {}

    for cid in cluster_ids_str:
        mask = adata.obs[cluster_key].astype(str) == cid
        n = int(mask.sum())
        adata.obs.loc[mask, "cell_type"] = label
        if fine_label:
            adata.obs.loc[mask, "cell_type_fine"] = fine_label
        total_cells += n
        per_cluster[cid] = n

    stats = {
        "merged_clusters": cluster_ids_str,
        "label": label,
        "fine_label": fine_label,
        "total_cells": total_cells,
        "per_cluster": per_cluster,
    }

    logger.info(f"[MergeClusters] {cluster_ids_str} → '{label}' ({total_cells} cells)")
    return adata, stats


def _apply_mapping_to_column(
    adata: ad.AnnData,
    col: str,
    mapping: dict[str, str],
) -> dict[str, Any]:
    """Apply a label mapping to a single obs column. Returns per-column stats."""
    labels = adata.obs[col].astype(str)
    unique_before = sorted(labels.unique().tolist())

    unknown_keys = [k for k in mapping if k not in unique_before]
    if unknown_keys:
        logger.warning(
            f"[HarmonizeLabels] {col}: mapping keys not found: {unknown_keys}"
        )

    new_labels = labels.replace(mapping)
    adata.obs[col] = new_labels

    unique_after = sorted(new_labels.unique().tolist())
    n_cells_changed = int((labels != new_labels).sum())

    change_counts = {}
    for old, new in mapping.items():
        if old in unique_before:
            n = int((labels == old).sum())
            change_counts[f"{old} → {new}"] = n

    col_stats = {
        "unique_labels_before": unique_before,
        "n_labels_before": len(unique_before),
        "unique_labels_after": unique_after,
        "n_labels_after": len(unique_after),
        "n_cells_changed": n_cells_changed,
        "change_counts": change_counts,
    }
    if unknown_keys:
        col_stats["warning"] = f"Mapping keys not found: {unknown_keys}"

    return col_stats


def harmonize_labels(
    adata: ad.AnnData,
    label_key: str = "cell_type",
    mapping: dict[str, str] = None,
    fine_mapping: Optional[dict[str, str]] = None,
    batch_key: Optional[str] = None,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """
    Harmonize cell type labels across datasets.

    The LLM agent provides the mapping dicts based on its understanding
    of cell type naming conventions.

    Parameters
    ----------
    adata : AnnData with cell type labels in obs.
    label_key : obs column with broad labels (default "cell_type").
    mapping : {old_label: new_label} for broad labels (required).
    fine_mapping : optional {old_label: new_label} for fine labels.
        Applied to "{label_key}_fine" column if it exists.
    batch_key : obs column for batch (used for per-batch reporting).
    """
    if label_key not in adata.obs.columns:
        raise ValueError(
            f"'{label_key}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    if not mapping:
        raise ValueError("mapping is required. The agent should provide {old_label: new_label}.")

    stats: dict[str, Any] = {"label_key": label_key}

    # Apply broad label mapping.
    stats["broad"] = _apply_mapping_to_column(adata, label_key, mapping)
    stats["mapping"] = mapping

    # Apply fine label mapping if provided and column exists.
    fine_key = f"{label_key}_fine"
    if fine_mapping and fine_key in adata.obs.columns:
        stats["fine"] = _apply_mapping_to_column(adata, fine_key, fine_mapping)
        stats["fine_mapping"] = fine_mapping
        stats["fine_key"] = fine_key
    elif fine_mapping and fine_key not in adata.obs.columns:
        logger.warning(
            f"[HarmonizeLabels] fine_mapping provided but '{fine_key}' "
            f"not found in adata.obs. Skipped."
        )
        stats["fine_warning"] = f"'{fine_key}' column not found, fine_mapping skipped."

    # Per-batch breakdown after harmonization.
    if batch_key and batch_key in adata.obs.columns:
        new_labels = adata.obs[label_key].astype(str)
        per_batch = {}
        for batch_val in adata.obs[batch_key].unique():
            batch_mask = adata.obs[batch_key] == batch_val
            batch_labels = new_labels[batch_mask]
            per_batch[str(batch_val)] = sorted(batch_labels.unique().tolist())
        stats["per_batch_labels_after"] = per_batch

    total_changed = stats["broad"]["n_cells_changed"]
    if "fine" in stats:
        total_changed += stats["fine"]["n_cells_changed"]

    logger.info(
        f"[HarmonizeLabels] broad: {stats['broad']['n_labels_before']} → "
        f"{stats['broad']['n_labels_after']} labels, "
        f"{total_changed} total cells changed"
    )
    return adata, stats
