"""
ReflectAnnotationTool — gather structured evidence for the agent to
self-critique its cell type annotations.

Computes raw facts from adata (no interpretation, no rules):
  - Per cluster: assigned label, cell count, top marker genes + scores
  - Per label: total cells, proportion, which clusters share it
  - Cross-cluster: marker overlap between differently-labeled clusters

The LLM reads this evidence and decides whether to revise, subcluster,
merge, or accept the annotations.
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "groupby": "leiden",
    "label_key": "cell_type",
    "n_top_markers": 8,
}


def _get_top_markers_per_cluster(adata, groupby: str, n_top: int) -> tuple:
    """Extract top marker genes per cluster from rank_genes_groups results.

    Handles two storage formats:
      1. DataFrame with 'group' column (from filter_marker_genes)
      2. Scanpy structured numpy record array (from rank_genes_groups)

    Returns (markers_dict, stale_warning) where stale_warning is a string
    if the stored marker results don't match the current cluster IDs, or None.
    """
    import numpy as np
    import pandas as pd

    # Pick the best available key.
    key = "rank_genes_groups_filtered"
    if key not in adata.uns:
        key = "rank_genes_groups"
    if key not in adata.uns:
        return {}, None

    result = adata.uns[key]
    markers = {}

    # ── Format 1: DataFrame from filter_marker_genes ──
    if isinstance(result, pd.DataFrame) and "group" in result.columns:
        groups = [str(g) for g in result["group"].unique()]

        # Column name mapping: filter_marker_genes uses scanpy names
        # ('gene' or 'names', 'scores' or 'score', 'logfoldchanges' or 'log2fc').
        gene_col = "gene" if "gene" in result.columns else "names"
        score_col = "score" if "score" in result.columns else "scores"
        logfc_col = "log2fc" if "log2fc" in result.columns else "logfoldchanges"

        for group in groups:
            group_df = result[result["group"] == group].head(n_top)
            gene_list = []
            for _, row in group_df.iterrows():
                name_str = str(row.get(gene_col, ""))
                if name_str == "nan" or name_str == "":
                    continue
                entry = {"gene": name_str}
                if score_col in row.index and pd.notna(row[score_col]):
                    entry["score"] = round(float(row[score_col]), 3)
                if logfc_col in row.index and pd.notna(row[logfc_col]):
                    entry["log2fc"] = round(float(row[logfc_col]), 3)
                gene_list.append(entry)
            markers[str(group)] = gene_list

    # ── Format 2: scanpy structured record array ──
    else:
        groups = [str(g) for g in result["names"].dtype.names]
        has_logfc = "logfoldchanges" in (
            result.dtype.names if hasattr(result, 'dtype') else result
        )

        for group in groups:
            names = result["names"][group][:n_top]
            scores = result["scores"][group][:n_top]
            logfcs = (
                result["logfoldchanges"][group][:n_top]
                if has_logfc else [None] * n_top
            )

            gene_list = []
            for i, name in enumerate(names):
                name_str = str(name)
                if name_str == "nan" or name_str == "":
                    continue
                entry = {"gene": name_str}

                score_val = scores[i]
                if score_val is not None and not (
                    hasattr(score_val, '__float__') and np.isnan(float(score_val))
                ):
                    entry["score"] = round(float(score_val), 3)

                logfc_val = logfcs[i]
                if logfc_val is not None and not (
                    hasattr(logfc_val, '__float__') and np.isnan(float(logfc_val))
                ):
                    entry["log2fc"] = round(float(logfc_val), 3)

                gene_list.append(entry)

            markers[str(group)] = gene_list

    # ── Check if marker groups match current cluster IDs ──
    stale_warning = None
    if groupby in adata.obs.columns:
        current_ids = set(adata.obs[groupby].astype(str).unique())
        marker_ids = set(groups)
        if current_ids != marker_ids:
            missing = current_ids - marker_ids
            extra = marker_ids - current_ids
            parts = []
            if missing:
                parts.append(f"clusters {sorted(missing)} have no markers")
            if extra:
                parts.append(f"markers exist for removed clusters {sorted(extra)}")
            stale_warning = (
                f"Marker results may be stale — {'; '.join(parts)}. "
                f"Consider re-running run_marker_genes."
            )

    return markers, stale_warning


def _compute_marker_overlap(markers_by_cluster: dict, top_n: int) -> list:
    """Find pairs of clusters that share top marker genes."""
    overlaps = []
    clusters = list(markers_by_cluster.keys())

    for i in range(len(clusters)):
        genes_i = {g["gene"] for g in markers_by_cluster[clusters[i]][:top_n]}
        if not genes_i:
            continue
        for j in range(i + 1, len(clusters)):
            genes_j = {g["gene"] for g in markers_by_cluster[clusters[j]][:top_n]}
            if not genes_j:
                continue
            shared = genes_i & genes_j
            if shared:
                overlaps.append({
                    "cluster_a": clusters[i],
                    "cluster_b": clusters[j],
                    "shared_markers": sorted(shared),
                    "n_shared": len(shared),
                    "jaccard": round(len(shared) / len(genes_i | genes_j), 3),
                })

    overlaps.sort(key=lambda x: x["n_shared"], reverse=True)
    return overlaps


def reflect_annotation(
    adata,
    groupby: str = "leiden",
    label_key: str = "cell_type",
    n_top_markers: int = 8,
) -> dict:
    """
    Gather structured evidence for annotation self-critique.

    Returns a dict of raw facts — no interpretation.
    """
    stats = {}

    # ── Check prerequisites ──
    if groupby not in adata.obs.columns:
        return {"error": f"Cluster column '{groupby}' not found in obs."}
    if label_key not in adata.obs.columns:
        return {"error": f"Label column '{label_key}' not found in obs. Run annotate_cell_types first."}

    clusters = adata.obs[groupby].astype(str)
    labels = adata.obs[label_key].astype(str)
    cluster_ids = sorted(clusters.unique(), key=lambda x: int(x) if x.isdigit() else x)

    # ── Per-cluster summary ──
    markers_by_cluster, stale_warning = _get_top_markers_per_cluster(adata, groupby, n_top_markers)
    if stale_warning:
        stats["warning"] = stale_warning

    per_cluster = {}
    for cid in cluster_ids:
        mask = clusters == cid
        n_cells = int(mask.sum())
        cluster_labels = labels[mask]

        # Safely get mode — handle empty or all-NaN cases.
        mode_result = cluster_labels.mode()
        if len(mode_result) > 0:
            label_value = str(mode_result.iloc[0])
        else:
            label_value = "Unknown"

        entry = {
            "label": label_value,
            "n_cells": n_cells,
            "proportion": round(n_cells / adata.n_obs, 4),
        }

        # If multiple labels exist in this cluster (e.g. after subclustering),
        # report the label composition.
        label_dist = cluster_labels.value_counts()
        if len(label_dist) > 1:
            entry["label_composition"] = {
                str(k): int(v) for k, v in label_dist.items()
            }

        # Top markers for this cluster.
        if cid in markers_by_cluster:
            entry["top_markers"] = markers_by_cluster[cid]

        # Marker scores if available (from score_markers tool).
        score_log = adata.uns.get("score_markers_log")
        if score_log and "score_matrix" in score_log:
            score_matrix = score_log["score_matrix"]
            if cid in score_matrix:
                entry["marker_scores"] = score_matrix[cid]

        # Confidence if available (from check_confidence tool).
        conf_log = adata.uns.get("confidence_log")
        if conf_log and "per_cluster" in conf_log:
            if cid in conf_log["per_cluster"]:
                entry["confidence"] = conf_log["per_cluster"][cid]

        per_cluster[cid] = entry

    stats["per_cluster"] = per_cluster

    # ── Per-label summary ──
    label_counts = labels.value_counts()
    per_label = {}
    for label_val in sorted(label_counts.index):
        label_mask = labels == label_val
        assigned_clusters = sorted(clusters[label_mask].unique(),
                                   key=lambda x: int(x) if x.isdigit() else x)
        per_label[label_val] = {
            "n_cells": int(label_counts[label_val]),
            "proportion": round(int(label_counts[label_val]) / adata.n_obs, 4),
            "clusters": assigned_clusters,
        }

    stats["per_label"] = per_label

    # ── Cross-cluster marker overlap ──
    if markers_by_cluster:
        overlaps = _compute_marker_overlap(markers_by_cluster, top_n=n_top_markers)
        for ov in overlaps:
            ov["label_a"] = per_cluster.get(ov["cluster_a"], {}).get("label", "?")
            ov["label_b"] = per_cluster.get(ov["cluster_b"], {}).get("label", "?")
        stats["marker_overlaps"] = overlaps

    # ── Summary counts ──
    stats["summary"] = {
        "n_clusters": len(cluster_ids),
        "n_labels": len(label_counts),
        "n_cells": adata.n_obs,
        "label_list": sorted(label_counts.index.tolist()),
    }

    return stats


class ReflectAnnotationTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "reflect_annotation"

    @property
    def input_type(self) -> str:
        return "annotated"

    @property
    def output_type(self) -> str:
        return "reflection_evidence"

    def get_param_space(self) -> dict:
        return {
            "label_key": {"type": "categorical", "choices": ["cell_type", "cell_type_fine"]},
            "n_top_markers": {"type": "int", "low": 3, "high": 20},
        }

    def run(self, adata, params: dict = None, batch_key: Optional[str] = None):
        """
        Gather evidence for annotation self-critique.

        Parameters
        ----------
        adata : AnnData with cluster labels and cell type annotations.
        params : dict with:
            groupby ('leiden'): cluster column.
            label_key ('cell_type'): annotation column to review.
            n_top_markers (8): top marker genes to show per cluster.

        Returns
        -------
        tuple of (adata, stats_dict) — adata is unchanged, stats has
        the evidence for the LLM to reason over.
        """
        t0 = time.time()
        params = params or {}
        p = {**_DEFAULTS, **params}

        stats = reflect_annotation(
            adata,
            groupby=p["groupby"],
            label_key=p["label_key"],
            n_top_markers=p["n_top_markers"],
        )

        stats["elapsed_s"] = round(time.time() - t0, 1)
        adata.uns["reflect_annotation_log"] = stats
        logger.info(f"[reflect_annotation] Done: {stats.get('elapsed_s')}s")
        return adata, stats
