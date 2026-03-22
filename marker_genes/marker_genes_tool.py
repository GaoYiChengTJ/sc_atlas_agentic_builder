"""
GetMarkerGenesTool — marker gene identification as a ToolWrapper.

Takes clustered AnnData, identifies marker genes per cluster,
and returns filtered top markers for annotation.
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper

from .operations import (
    extract_top_markers,
    filter_marker_genes,
    rank_marker_genes,
)

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "groupby": "leiden",
    "method": "wilcoxon",
    "use_raw": None,
    "layer": None,
    "corr_method": "benjamini-hochberg",
    "reference": "rest",
    "n_top_genes": 10,
    "min_log2fc": 0.25,
    "max_pval_adj": 0.05,
    "min_pct_group": 0.1,
    "max_pct_other": 0.5,
    "run_filter": True,
}


class GetMarkerGenesTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "get_marker_genes"

    @property
    def output_type(self) -> str:
        return "marker_genes"

    @property
    def input_type(self) -> str:
        return "clustered"

    def _check_imports(self):
        import scanpy  # noqa: F401

    def get_param_space(self) -> dict:
        return {
            "method": {
                "type": "categorical",
                "choices": ["wilcoxon", "t-test", "t-test_overestim_var", "logreg"],
            },
            "n_top_genes": {"type": "int", "low": 5, "high": 50},
            "min_log2fc": {"type": "float", "low": 0.0, "high": 2.0},
            "max_pval_adj": {"type": "float", "low": 0.001, "high": 0.1},
            "min_pct_group": {"type": "float", "low": 0.05, "high": 0.5},
            "max_pct_other": {"type": "float", "low": 0.1, "high": 0.9},
        }

    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Run marker gene identification pipeline.

        Parameters
        ----------
        adata : AnnData — clustered, with cluster labels in obs.
        params : dict — marker gene parameters. Unset keys use defaults:
            groupby ("leiden"), method ("wilcoxon"), use_raw (None, auto),
            layer (None), corr_method ("benjamini-hochberg"),
            reference ("rest"), n_top_genes (10),
            min_log2fc (0.25), max_pval_adj (0.05),
            min_pct_group (0.1), max_pct_other (0.5), run_filter (True).
        batch_key : unused, kept for ToolWrapper interface compatibility.

        Returns
        -------
        AnnData with rank_genes_groups in uns, plus top_markers dict
        in uns["marker_genes_log"]["top_markers"].
        """
        t0 = time.time()
        p = {**_DEFAULTS, **params}
        log = {}
        step = None

        logger.info(
            f"[marker_genes] Start: {adata.n_obs} cells, "
            f"groupby={p['groupby']}, method={p['method']}"
        )

        try:
            # 1. Rank genes
            step = "rank"
            adata, log["rank"] = rank_marker_genes(
                adata,
                groupby=p["groupby"],
                method=p["method"],
                use_raw=p["use_raw"],
                layer=p["layer"],
                corr_method=p["corr_method"],
                reference=p["reference"],
            )

            # 2. Filter (optional)
            if p["run_filter"]:
                step = "filter"
                adata, log["filter"] = filter_marker_genes(
                    adata,
                    min_log2fc=p["min_log2fc"],
                    max_pval_adj=p["max_pval_adj"],
                    min_pct_group=p["min_pct_group"],
                    max_pct_other=p["max_pct_other"],
                )

            # 3. Extract top markers
            step = "extract"
            # Use filtered results if available, otherwise raw rankings.
            extract_key = "rank_genes_groups"
            if p["run_filter"] and "rank_genes_groups_filtered" in adata.uns:
                extract_key = "rank_genes_groups_filtered"

            log["top_markers"] = extract_top_markers(
                adata,
                key=extract_key,
                n_genes=p["n_top_genes"],
            )

        except Exception as e:
            logger.error(f"[marker_genes] Failed at step '{step}': {e}")
            log["failed_step"] = step
            log["error"] = str(e)
            raise RuntimeError(
                f"Marker gene identification failed at step '{step}': {e}"
            ) from e

        adata.uns["marker_genes_log"] = log
        n_groups = log["rank"]["n_groups"]
        logger.info(
            f"[marker_genes] Done: {n_groups} groups, "
            f"top {p['n_top_genes']} genes each, "
            f"{time.time()-t0:.1f}s"
        )
        return adata
