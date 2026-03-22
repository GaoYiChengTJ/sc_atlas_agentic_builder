"""
Stateless annotation operations for single-cell data.

Each function provides evidence for cell type annotation.
Label assignment is a separate step driven by the agent.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def score_known_markers(
    adata,
    marker_dict: dict[str, list[str]],
    groupby: str = "leiden",
    use_raw: Optional[bool] = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Score each cluster against known marker gene signatures.

    Parameters
    ----------
    marker_dict : {cell_type: [gene1, gene2, ...]}
        Marker gene lists per expected cell type.
    groupby : obs column with cluster labels.
    use_raw : use adata.raw for scoring. If None, auto-detects.

    Returns
    -------
    (adata, stats) where stats contains:
        - score_matrix: {cluster: {cell_type: mean_score}}
        - best_match: {cluster: cell_type} (highest scoring)
        - genes_found: {cell_type: n_found} how many markers were in the data
    """
    import scanpy as sc

    if groupby not in adata.obs.columns:
        raise ValueError(
            f"groupby='{groupby}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    if use_raw is None:
        use_raw = adata.raw is not None

    # Determine which genes are available.
    if use_raw and adata.raw is not None:
        available_genes = set(adata.raw.var_names)
    else:
        available_genes = set(adata.var_names)

    import re

    genes_found: dict[str, int] = {}
    score_keys: dict[str, str] = {}

    # Use .raw for scoring if available (tests all genes, not just HVGs).
    # Temporarily swap .X if use_raw is requested, since sc.tl.score_genes
    # doesn't reliably support use_raw across scanpy versions.
    scoring_adata = adata
    if use_raw and adata.raw is not None:
        scoring_adata = adata.raw.to_adata()

    for cell_type, genes in marker_dict.items():
        present = [g for g in genes if g in available_genes]
        genes_found[cell_type] = len(present)

        if len(present) < 2:
            logger.warning(
                f"[Annotation] '{cell_type}': only {len(present)} marker(s) found "
                f"out of {len(genes)}. Skipping (need >=2)."
            )
            continue

        if len(present) < 5:
            logger.warning(
                f"[Annotation] '{cell_type}': only {len(present)} markers. "
                "Short lists may produce unreliable scores due to background subtraction."
            )

        # Sanitize score column name: replace non-alphanumeric with underscore.
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", cell_type)
        score_name = f"score_{safe_name}"

        sc.tl.score_genes(
            scoring_adata,
            gene_list=present,
            score_name=score_name,
        )
        # Copy scores back to original adata if we used .raw.
        if scoring_adata is not adata:
            adata.obs[score_name] = scoring_adata.obs[score_name].values

        score_keys[cell_type] = score_name

    if not score_keys:
        raise ValueError(
            "No cell types had enough markers (>=2) in the dataset. "
            f"Genes found per type: {genes_found}"
        )

    # Build cluster × cell_type score matrix using both mean and median.
    clusters = adata.obs[groupby].unique().tolist()
    score_matrix: dict[str, dict[str, float]] = {}
    best_match: dict[str, str] = {}

    for cluster in clusters:
        mask = adata.obs[groupby] == cluster
        cluster_scores: dict[str, float] = {}
        for cell_type, score_name in score_keys.items():
            vals = adata.obs.loc[mask, score_name]
            cluster_scores[cell_type] = {
                "mean": round(float(vals.mean()), 4),
                "median": round(float(vals.median()), 4),
            }
        score_matrix[str(cluster)] = cluster_scores
        # Use median for best_match — more robust to outliers.
        best_match[str(cluster)] = max(
            cluster_scores,
            key=lambda ct: cluster_scores[ct]["median"],
        )

    stats = {
        "score_matrix": score_matrix,
        "best_match": best_match,
        "genes_found": genes_found,
        "scored_types": list(score_keys.keys()),
        "n_clusters": len(clusters),
        "groupby": groupby,
    }
    logger.info(
        f"[Annotation] Scored {len(score_keys)} cell types "
        f"across {len(clusters)} clusters"
    )
    return adata, stats


def run_celltypist(
    adata,
    model: str = "Immune_All_Low.pkl",
    groupby: str = "leiden",
    majority_voting: bool = True,
    auto_download: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """
    Run CellTypist reference-based annotation.

    Parameters
    ----------
    model : CellTypist model name or path to a custom model file.
    groupby : obs column for majority voting.
    majority_voting : if True, refines per-cell predictions to per-cluster
        consensus using majority voting within each cluster.
    auto_download : if True, download the model if not found locally.
        Set to False (default) for restricted environments.
    """
    try:
        import celltypist
        from celltypist import models as ct_models
    except ImportError:
        raise ImportError(
            "CellTypist is not installed. Install it: pip install celltypist"
        )

    if groupby not in adata.obs.columns:
        raise ValueError(
            f"groupby='{groupby}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    # Load model — download only if explicitly allowed.
    try:
        ct_model = ct_models.Model.load(model=model)
    except Exception:
        if not auto_download:
            raise FileNotFoundError(
                f"CellTypist model '{model}' not found locally and "
                "auto_download=False. Either download it manually with "
                f"`celltypist.models.download_models(model='{model}')` "
                "or set auto_download=True."
            )
        logger.info(f"[CellTypist] Downloading model '{model}'...")
        ct_models.download_models(model=model)
        ct_model = ct_models.Model.load(model=model)

    # CellTypist expects normalized + log-transformed data in .X.
    # Use the 'normalized' layer if .X has been modified (e.g., scaled for PCA).
    if "normalized" in adata.layers:
        input_adata = adata.copy()
        input_adata.X = input_adata.layers["normalized"].copy()
        logger.info("[CellTypist] Using layers['normalized'] for input")
    elif adata.raw is not None:
        input_adata = adata.raw.to_adata()
        # Preserve obs columns (cluster labels, etc.)
        input_adata.obs = adata.obs.copy()
        logger.info("[CellTypist] Using .raw for input")
    else:
        logger.warning(
            "[CellTypist] No 'normalized' layer or .raw found. "
            "Using .X directly — ensure it contains normalized log-transformed data."
        )
        input_adata = adata

    predictions = celltypist.annotate(
        input_adata,
        model=ct_model,
        majority_voting=majority_voting,
        over_clustering=groupby if majority_voting else None,
    )

    # Transfer predictions to adata, aligned on cell barcode index.
    result_adata = predictions.to_adata()
    common_idx = adata.obs.index.intersection(result_adata.obs.index)
    if len(common_idx) < adata.n_obs:
        logger.warning(
            f"[CellTypist] Index mismatch: {adata.n_obs} cells in adata, "
            f"{len(common_idx)} matched in predictions."
        )
    adata.obs["celltypist_predicted"] = result_adata.obs.loc[common_idx, "predicted_labels"]
    if majority_voting and "majority_voting" in result_adata.obs.columns:
        adata.obs["celltypist_majority"] = result_adata.obs.loc[common_idx, "majority_voting"]
    if "conf_score" in result_adata.obs.columns:
        adata.obs["celltypist_conf"] = result_adata.obs.loc[common_idx, "conf_score"]

    # Summarize per-cluster predictions.
    label_col = "celltypist_majority" if majority_voting and "celltypist_majority" in adata.obs.columns else "celltypist_predicted"
    cluster_predictions: dict[str, dict[str, Any]] = {}
    for cluster in adata.obs[groupby].unique():
        mask = adata.obs[groupby] == cluster
        labels = adata.obs.loc[mask, label_col].dropna()
        mode_result = labels.mode()
        top_label = mode_result.iloc[0] if len(mode_result) > 0 else "Unknown"
        agreement = float((labels == top_label).mean()) if len(labels) > 0 else 0.0
        cluster_predictions[str(cluster)] = {
            "predicted": top_label,
            "agreement": round(agreement, 3),
            "n_cells": int(mask.sum()),
        }

    stats = {
        "model": model,
        "majority_voting": majority_voting,
        "groupby": groupby,
        "label_column": label_col,
        "cluster_predictions": cluster_predictions,
    }
    logger.info(
        f"[CellTypist] Annotated {adata.n_obs} cells with model '{model}'"
    )
    return adata, stats


def assign_labels(
    adata,
    label_mapping: dict[str, str],
    groupby: str = "leiden",
    key_added: str = "cell_type",
    fine_label_mapping: Optional[dict[str, str]] = None,
    fine_key_added: str = "cell_type_fine",
) -> tuple[Any, dict[str, Any]]:
    """
    Assign cell type labels to clusters based on agent's decision.

    Parameters
    ----------
    label_mapping : {cluster_id: cell_type_label}
        Broad cell type labels decided by the agent.
    groupby : obs column with cluster labels.
    key_added : obs column name for the new labels.
    fine_label_mapping : optional {cluster_id: fine_label} for granular types.
    fine_key_added : obs column for fine labels.
    """
    if groupby not in adata.obs.columns:
        raise ValueError(
            f"groupby='{groupby}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    clusters = set(adata.obs[groupby].unique().astype(str))
    mapped_clusters = set(label_mapping.keys())

    # Warn about unmapped clusters.
    unmapped = clusters - mapped_clusters
    if unmapped:
        logger.warning(
            f"[Annotation] {len(unmapped)} cluster(s) not in label_mapping: "
            f"{sorted(unmapped)}. They will be labeled 'Unknown'."
        )

    # Warn about extra keys not matching any cluster.
    extra = mapped_clusters - clusters
    if extra:
        logger.warning(
            f"[Annotation] {len(extra)} key(s) in label_mapping don't match "
            f"any cluster: {sorted(extra)}"
        )

    # Apply broad labels. Use native dict .map() + fillna for speed.
    cluster_series = adata.obs[groupby].astype(str)
    adata.obs[key_added] = (
        cluster_series.map(label_mapping).fillna("Unknown").astype("category")
    )

    # Apply fine labels if provided.
    if fine_label_mapping:
        adata.obs[fine_key_added] = (
            cluster_series.map(fine_label_mapping).fillna("Unknown").astype("category")
        )

    # Summary stats.
    label_counts = adata.obs[key_added].value_counts().to_dict()
    stats = {
        "key_added": key_added,
        "groupby": groupby,
        "n_labels": len(set(label_mapping.values())),
        "n_clusters_mapped": len(mapped_clusters & clusters),
        "n_unmapped": len(unmapped),
        "label_counts": label_counts,
        "has_fine_labels": fine_label_mapping is not None,
    }
    if fine_label_mapping:
        stats["fine_key_added"] = fine_key_added
        stats["fine_label_counts"] = adata.obs[fine_key_added].value_counts().to_dict()

    logger.info(
        f"[Annotation] Assigned {stats['n_labels']} cell types "
        f"to {stats['n_clusters_mapped']} clusters"
    )
    return adata, stats


def compute_annotation_confidence(
    adata,
    groupby: str = "leiden",
    score_prefix: str = "score_",
    celltypist_col: Optional[str] = None,
    low_confidence_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute per-cluster annotation confidence.

    Confidence is based on:
    - Score gap ratio: (best - second) / (best - worst) per cluster.
      Normalized to the actual score range, not a magic number.
    - CellTypist agreement: within-cluster label consistency.
    - Cross-method agreement: whether marker scores and CellTypist agree.

    Returns
    -------
    dict with per-cluster confidence scores and flags for low-confidence clusters.
    """
    if groupby not in adata.obs.columns:
        raise ValueError(
            f"groupby='{groupby}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    # Find all marker score columns.
    score_cols = [c for c in adata.obs.columns if c.startswith(score_prefix)]
    has_celltypist = celltypist_col is not None and celltypist_col in adata.obs.columns

    # Vectorized: compute per-cluster mean scores in one pass.
    cluster_mean_scores = None
    if score_cols:
        cluster_mean_scores = (
            adata.obs.groupby(groupby, observed=True)[score_cols]
            .mean()
        )

    # Vectorized: compute per-cluster celltypist mode + agreement.
    celltypist_summary = {}
    if has_celltypist:
        for cluster, group_df in adata.obs.groupby(groupby, observed=True):
            labels = group_df[celltypist_col].dropna()
            mode_result = labels.mode()
            top_label = mode_result.iloc[0] if len(mode_result) > 0 else "Unknown"
            agreement = float((labels == top_label).mean()) if len(labels) > 0 else 0.0
            celltypist_summary[str(cluster)] = {
                "label": top_label,
                "agreement": round(agreement, 3),
            }

    # Per-cluster cell counts.
    cluster_sizes = adata.obs[groupby].value_counts().to_dict()

    confidence: dict[str, dict[str, Any]] = {}

    for cluster in adata.obs[groupby].unique():
        cluster_str = str(cluster)
        cluster_conf: dict[str, Any] = {"n_cells": int(cluster_sizes.get(cluster, 0))}

        # Marker score gap (normalized to score range).
        if cluster_mean_scores is not None and len(score_cols) >= 2:
            scores = cluster_mean_scores.loc[cluster]
            sorted_vals = scores.sort_values(ascending=False)
            best_type = sorted_vals.index[0][len(score_prefix):]
            best_score = float(sorted_vals.iloc[0])
            second_score = float(sorted_vals.iloc[1])
            worst_score = float(sorted_vals.iloc[-1])

            score_range = best_score - worst_score
            score_gap = best_score - second_score
            # Normalize gap relative to the full range for this cluster.
            gap_ratio = (score_gap / score_range) if score_range > 0 else 0.0

            cluster_conf["best_marker_type"] = best_type
            cluster_conf["best_score"] = round(best_score, 4)
            cluster_conf["second_score"] = round(second_score, 4)
            cluster_conf["score_gap"] = round(score_gap, 4)
            cluster_conf["gap_ratio"] = round(gap_ratio, 4)
        elif cluster_mean_scores is not None and len(score_cols) == 1:
            col = score_cols[0]
            cell_type = col[len(score_prefix):]
            cluster_conf["best_marker_type"] = cell_type
            cluster_conf["best_score"] = round(float(cluster_mean_scores.loc[cluster, col]), 4)
            cluster_conf["gap_ratio"] = None

        # CellTypist agreement.
        if has_celltypist and cluster_str in celltypist_summary:
            ct = celltypist_summary[cluster_str]
            cluster_conf["celltypist_label"] = ct["label"]
            cluster_conf["celltypist_agreement"] = ct["agreement"]

        # Cross-method agreement: do marker scores and CellTypist agree?
        if "best_marker_type" in cluster_conf and "celltypist_label" in cluster_conf:
            marker_type = cluster_conf["best_marker_type"].lower().replace("_", " ")
            ct_type = cluster_conf["celltypist_label"].lower()
            # Fuzzy match: check if one contains the other.
            cross_agree = marker_type in ct_type or ct_type in marker_type
            cluster_conf["cross_method_agreement"] = cross_agree

        # Overall confidence: combine available signals.
        signals = []
        if "gap_ratio" in cluster_conf and cluster_conf["gap_ratio"] is not None:
            signals.append(cluster_conf["gap_ratio"])
        if "celltypist_agreement" in cluster_conf:
            signals.append(cluster_conf["celltypist_agreement"])
        if "cross_method_agreement" in cluster_conf:
            signals.append(1.0 if cluster_conf["cross_method_agreement"] else 0.0)

        if signals:
            cluster_conf["confidence"] = round(float(np.mean(signals)), 3)
        else:
            cluster_conf["confidence"] = None

        confidence[cluster_str] = cluster_conf

    # Flag low-confidence clusters.
    low_confidence = [
        cid for cid, c in confidence.items()
        if c.get("confidence") is not None and c["confidence"] < low_confidence_threshold
    ]

    stats = {
        "per_cluster": confidence,
        "low_confidence_clusters": low_confidence,
        "low_confidence_threshold": low_confidence_threshold,
        "n_score_columns": len(score_cols),
        "has_celltypist": has_celltypist,
    }

    logger.info(
        f"[Confidence] {len(confidence)} clusters, "
        f"{len(low_confidence)} low-confidence"
    )
    return stats
