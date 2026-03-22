"""
SelectIntegrationGenesTool — select genes for batch integration.

OPTIONAL: If preprocessing already ran select_hvg with batch_key,
the rank-based strategy is already applied. Only use this tool when:
  - You want intersection/union strategy instead of rank.
  - You want to re-select genes with different params after exploration.
  - Preprocessing was run without batch_key.
For the standard workflow, skip this and go straight to RunIntegrationTool.
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper
from .operations import select_integration_genes

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "n_top_genes": 2000,
    "strategy": "rank",
    "flavor": "seurat_v3",
}


class SelectIntegrationGenesTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "select_integration_genes"

    @property
    def output_type(self) -> str:
        return "integration_genes"

    @property
    def input_type(self) -> str:
        return "preprocessed"

    def _check_imports(self):
        import scanpy  # noqa: F401

    def get_param_space(self) -> dict:
        return {
            "n_top_genes": {"type": "int", "low": 500, "high": 5000},
            "strategy": {"type": "categorical", "choices": ["rank", "intersection", "union"]},
        }

    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Re-select HVGs with batch-aware strategies for integration.

        This is OPTIONAL. If PreprocessingTool was run with batch_key,
        rank-based HVG selection was already applied. Use this tool only
        when you need intersection/union strategy or different gene counts.

        Parameters
        ----------
        adata : AnnData — preprocessed.
        params : dict with:
            n_top_genes (2000), strategy ("rank"), flavor ("seurat_v3").
            strategy: "rank" (same as preprocessing default),
                      "intersection" (conservative, only genes variable in ALL batches),
                      "union" (permissive, genes variable in ANY batch, re-ranked).
        batch_key : obs column for batch (required).

        Returns
        -------
        AnnData with updated highly_variable column in var.
        """
        t0 = time.time()
        p = {**_DEFAULTS, **params}

        if not batch_key:
            raise ValueError("batch_key is required for integration gene selection.")
        if batch_key not in adata.obs.columns:
            raise ValueError(
                f"batch_key '{batch_key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        logger.info(f"[select_integration_genes] Start: {adata.n_vars} genes")

        adata, stats = select_integration_genes(
            adata,
            batch_key=batch_key,
            n_top_genes=p["n_top_genes"],
            strategy=p["strategy"],
            flavor=p["flavor"],
        )

        stats["elapsed_s"] = round(time.time() - t0, 1)
        adata.uns["select_integration_genes_log"] = stats
        logger.info(f"[select_integration_genes] Done: {stats['elapsed_s']}s")
        return adata, stats
