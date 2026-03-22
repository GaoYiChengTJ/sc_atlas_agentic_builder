"""
Stateless marker gene operations for single-cell data.

Each function: (adata, **params) -> (adata, stats_dict) or returns results directly.
"""

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_VALID_METHODS = {"wilcoxon", "t-test", "t-test_overestim_var", "logreg"}


def rank_marker_genes(
    adata,
    groupby: str = "leiden",
    method: str = "wilcoxon",
    use_raw: Optional[bool] = None,
    layer: Optional[str] = None,
    corr_method: str = "benjamini-hochberg",
    reference: str = "rest",
    key_added: str = "rank_genes_groups",
) -> tuple[Any, dict[str, Any]]:
    """
    Rank genes for characterizing groups (clusters).

    Parameters
    ----------
    groupby : obs column with cluster labels.
    method : statistical test. 'wilcoxon' (recommended), 't-test',
        't-test_overestim_var', or 'logreg'.
    use_raw : use adata.raw for expression values. If None, automatically
        uses raw when available (to test on all genes, not just HVGs).
    layer : expression layer to use. Overrides use_raw if set.
    corr_method : multiple testing correction ('benjamini-hochberg' or 'bonferroni').
    reference : compare each group against 'rest' (one-vs-rest) or a specific group.
    key_added : key in adata.uns to store results.
    """
    import scanpy as sc

    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {sorted(_VALID_METHODS)}"
        )

    if groupby not in adata.obs.columns:
        raise ValueError(
            f"groupby='{groupby}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    # Warn if both layer and use_raw are explicitly set.
    if layer is not None and use_raw is True:
        logger.warning(
            "[MarkerGenes] Both layer and use_raw are set. "
            f"Scanpy will use layer='{layer}' and ignore use_raw."
        )
        use_raw = False

    # Auto-detect: use .raw if available and no layer specified,
    # so DE tests run on all genes (not just HVGs in .X).
    if use_raw is None and layer is None:
        use_raw = adata.raw is not None

    # Filter out groups with too few cells for reliable DE testing.
    min_cells_per_group = 3
    group_counts = adata.obs[groupby].value_counts()
    small_groups = group_counts[group_counts < min_cells_per_group].index.tolist()
    if small_groups:
        logger.warning(
            f"[MarkerGenes] Excluding {len(small_groups)} group(s) with "
            f"<{min_cells_per_group} cells: {small_groups}. "
            "Too few cells for reliable DE testing."
        )
        adata = adata[~adata.obs[groupby].isin(small_groups)].copy()

    n_groups = adata.obs[groupby].nunique()
    if n_groups < 2:
        raise ValueError(
            f"Need at least 2 groups for DE testing, but only {n_groups} "
            f"remain after filtering small groups."
        )

    kwargs = dict(
        groupby=groupby,
        method=method,
        use_raw=use_raw,
        layer=layer,
        corr_method=corr_method,
        reference=reference,
        key_added=key_added,
    )

    # logreg doesn't support corr_method
    if method == "logreg":
        kwargs.pop("corr_method")

    sc.tl.rank_genes_groups(adata, **kwargs)

    stats = {
        "method": method,
        "groupby": groupby,
        "n_groups": n_groups,
        "use_raw": use_raw,
        "layer": layer,
        "corr_method": corr_method if method != "logreg" else None,
        "reference": reference,
        "key_added": key_added,
        "excluded_small_groups": small_groups if small_groups else None,
    }
    logger.info(f"[MarkerGenes] {method}, {n_groups} groups, groupby={groupby}")
    return adata, stats


def _build_marker_df(adata, key: str = "rank_genes_groups") -> pd.DataFrame:
    """
    Build a single DataFrame from rank_genes_groups results using
    scanpy's vectorized extraction. Handles pts/pts_rest correctly.
    """
    import scanpy as sc

    result = adata.uns[key]
    group_names = result["names"].dtype.names

    frames = []
    for group in group_names:
        df = sc.get.rank_genes_groups_df(adata, group=group, key=key)
        df["group"] = group

        # sc.get.rank_genes_groups_df doesn't include pts/pts_rest.
        # These are DataFrames (not structured arrays), so extract manually.
        if "pts" in result and isinstance(result["pts"], pd.DataFrame):
            pts_df = result["pts"]
            if group in pts_df.columns:
                gene_pct = pts_df[group].reindex(df["names"]).values
                df["pct_group"] = gene_pct
        if "pts_rest" in result and isinstance(result["pts_rest"], pd.DataFrame):
            pts_rest_df = result["pts_rest"]
            if group in pts_rest_df.columns:
                gene_pct_rest = pts_rest_df[group].reindex(df["names"]).values
                df["pct_other"] = gene_pct_rest

        frames.append(df)

    full_df = pd.concat(frames, ignore_index=True)

    # Standardize column name: scanpy uses 'names', we use 'gene'.
    if "names" in full_df.columns:
        full_df = full_df.rename(columns={"names": "gene"})

    return full_df


def filter_marker_genes(
    adata,
    key: str = "rank_genes_groups",
    min_log2fc: float = 0.25,
    max_pval_adj: float = 0.05,
    min_pct_group: float = 0.1,
    max_pct_other: float = 0.5,
) -> tuple[Any, dict[str, Any]]:
    """
    Filter ranked marker genes and store as a single concatenated DataFrame.

    Uses scanpy's vectorized sc.get.rank_genes_groups_df for extraction,
    then applies filters. Result is stored as a single DataFrame with a
    'group' column — safe for write_h5ad and easy to export.

    Parameters
    ----------
    min_log2fc : minimum log2 fold change.
    max_pval_adj : maximum adjusted p-value.
    min_pct_group : minimum fraction of in-group cells expressing the gene.
    max_pct_other : maximum fraction of out-group cells expressing the gene.
        A good marker is highly expressed in-group but NOT in other groups.
    """
    if key not in adata.uns:
        raise ValueError(
            f"Key '{key}' not found in adata.uns. Run rank_marker_genes first."
        )

    df = _build_marker_df(adata, key=key)
    total_before = len(df)

    # Apply filters based on available columns.
    if "logfoldchanges" in df.columns:
        df = df[df["logfoldchanges"] >= min_log2fc]
    if "pvals_adj" in df.columns:
        df = df[df["pvals_adj"] <= max_pval_adj]
    if "pct_group" in df.columns:
        df = df[df["pct_group"] >= min_pct_group]
    if "pct_other" in df.columns:
        df = df[df["pct_other"] <= max_pct_other]

    df = df.reset_index(drop=True)
    total_after = len(df)

    # Store as a single DataFrame — safe for h5ad serialization.
    filtered_key = key + "_filtered"
    adata.uns[filtered_key] = df

    available_cols = set(df.columns.tolist())
    stats = {
        "min_log2fc": min_log2fc,
        "max_pval_adj": max_pval_adj,
        "min_pct_group": min_pct_group,
        "max_pct_other": max_pct_other,
        "filtered_key": filtered_key,
        "total_genes_before": total_before,
        "total_genes_after": total_after,
        "fields_available": {
            "pvals_adj": "pvals_adj" in available_cols,
            "logfoldchanges": "logfoldchanges" in available_cols,
            "pct": "pct_group" in available_cols,
        },
    }
    logger.info(
        f"[MarkerGenes] Filtered: {total_before}→{total_after} genes "
        f"(log2fc>={min_log2fc}, pval<={max_pval_adj}, "
        f"pct_in>={min_pct_group}, pct_out<={max_pct_other})"
    )
    return adata, stats


def extract_top_markers(
    adata,
    key: str = "rank_genes_groups",
    n_genes: int = 10,
) -> dict[str, Any]:
    """
    Extract top N marker genes per cluster as a clean dictionary.

    Supports both:
      - scanpy's structured array format (from rank_genes_groups)
      - a single concatenated DataFrame with 'group' column (from filter_marker_genes)

    Returns
    -------
    dict with:
        - 'markers': {cluster_id: [{gene, score, log2fc, pval_adj}, ...]}
        - 'summary': {cluster_id: [gene1, gene2, ...]} (names only)
        - 'n_genes': number requested
    """
    if key not in adata.uns:
        raise ValueError(
            f"Key '{key}' not found in adata.uns. Run rank_marker_genes first."
        )

    result = adata.uns[key]

    # Ensure we have a DataFrame regardless of input format.
    if isinstance(result, pd.DataFrame) and "group" in result.columns:
        df = result
    else:
        df = _build_marker_df(adata, key=key)

    # Rename scanpy columns to clean agent-facing names.
    rename_map = {"scores": "score", "logfoldchanges": "log2fc"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Columns to include in per-gene entries (besides 'gene').
    stat_cols = [c for c in ["score", "log2fc", "pvals_adj", "pct_group", "pct_other"]
                 if c in df.columns]

    markers: dict[str, list[dict]] = {}
    summary: dict[str, list[str]] = {}

    for group, group_df in df.groupby("group", sort=False):
        top = group_df.head(n_genes)
        subset = top[["gene"] + stat_cols].copy()
        for col in stat_cols:
            if col != "pvals_adj":
                subset[col] = subset[col].round(4)
        records = subset.to_dict("records")
        # Replace NaN with None for JSON-safe output.
        for entry in records:
            for k, v in entry.items():
                if isinstance(v, float) and np.isnan(v):
                    entry[k] = None
        markers[str(group)] = records
        summary[str(group)] = top["gene"].tolist()

    n_groups = len(markers)
    logger.info(f"[MarkerGenes] Extracted top {n_genes} markers for {n_groups} groups")
    return {
        "markers": markers,
        "summary": summary,
        "n_genes": n_genes,
        "n_groups": n_groups,
        "key": key,
    }
