"""
CheckConfidenceTool — evaluate annotation confidence per cluster.

Compares evidence from marker scoring and CellTypist, flags low-confidence
clusters that need subclustering or manual review.
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper
from .operations import compute_annotation_confidence

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "groupby": "leiden",
    "score_prefix": "score_",
    "celltypist_col": None,
    "low_confidence_threshold": 0.5,
}


class CheckConfidenceTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "check_confidence"

    @property
    def output_type(self) -> str:
        return "confidence_report"

    @property
    def input_type(self) -> str:
        return "clustered"

    def _check_imports(self):
        pass

    def get_param_space(self) -> dict:
        return {
            "low_confidence_threshold": {"type": "float", "low": 0.2, "high": 0.8},
        }

    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Compute per-cluster annotation confidence.

        Parameters
        ----------
        adata : AnnData — with score columns and/or celltypist predictions.
        params : dict with:
            groupby ("leiden"), score_prefix ("score_"),
            celltypist_col (None, auto-detected from celltypist_log),
            low_confidence_threshold (0.5).

        Returns
        -------
        AnnData with confidence_log in uns containing per-cluster
        confidence scores and low_confidence_clusters list.
        """
        t0 = time.time()
        p = {**_DEFAULTS, **params}

        # Auto-detect celltypist column from previous celltypist run.
        celltypist_col = p["celltypist_col"]
        if celltypist_col is None and "celltypist_log" in adata.uns:
            celltypist_col = adata.uns["celltypist_log"].get("label_column")

        logger.info(f"[check_confidence] Start: {adata.n_obs} cells")

        stats = compute_annotation_confidence(
            adata,
            groupby=p["groupby"],
            score_prefix=p["score_prefix"],
            celltypist_col=celltypist_col,
            low_confidence_threshold=p["low_confidence_threshold"],
        )

        adata.uns["confidence_log"] = stats
        logger.info(f"[check_confidence] Done: {time.time()-t0:.1f}s")
        return adata
