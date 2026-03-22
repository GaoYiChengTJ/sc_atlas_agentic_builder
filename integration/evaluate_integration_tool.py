"""
EvaluateIntegrationTool — evaluate batch integration quality.

Uses scib-metrics (required) to compute batch mixing (ASW_batch,
graph connectivity) and bio conservation (ASW_label, NMI, ARI,
isolated label ASW). Overall score: 40% batch + 60% bio (scIB convention).
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper
from .operations import evaluate_integration

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "embed_key": "X_harmony",
    "label_key": None,
    "max_cells": 50_000,
}


class EvaluateIntegrationTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "evaluate_integration"

    @property
    def output_type(self) -> str:
        return "integration_evaluation"

    @property
    def input_type(self) -> str:
        return "integrated"

    def _check_imports(self):
        import scib  # noqa: F401

    def get_param_space(self) -> dict:
        return {
            "embed_key": {
                "type": "categorical",
                "choices": ["X_harmony", "X_scanorama", "X_pca"],
            },
        }

    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Evaluate integration quality using scib-metrics.

        Parameters
        ----------
        adata : AnnData — with integrated embedding in obsm.
        params : dict with:
            embed_key ("X_harmony") — which embedding to evaluate.
            label_key (None) — obs column with cell type labels for bio
                conservation metrics. If None, only batch mixing is computed.
            max_cells (50000) — subsample for speed.
        batch_key : obs column for batch (required).

        Returns
        -------
        (AnnData, stats) where stats contains:
            - batch_scores: {asw_batch, graph_connectivity}
            - bio_scores: {asw_label, nmi, ari, isolated_label_asw} (if labels)
            - batch_avg, bio_avg, overall_score
            - weighting: "40% batch + 60% bio (scIB convention)"
        """
        t0 = time.time()
        p = {**_DEFAULTS, **params}

        if not batch_key:
            raise ValueError("batch_key is required for integration evaluation.")

        logger.info(
            f"[evaluate_integration] Start: {p['embed_key']}, "
            f"{adata.n_obs} cells"
        )

        stats = evaluate_integration(
            adata,
            batch_key=batch_key,
            label_key=p["label_key"],
            embed_key=p["embed_key"],
            max_cells=p["max_cells"],
        )

        stats["elapsed_s"] = round(time.time() - t0, 1)
        adata.uns["evaluate_integration_log"] = stats
        logger.info(f"[evaluate_integration] Done: {stats['elapsed_s']}s")
        return adata, stats
