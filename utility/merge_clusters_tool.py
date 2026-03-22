"""
MergeClustersTool — assign the same cell type label to multiple clusters.

Used when marker analysis shows two clusters are the same cell type.
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper
from .operations import merge_clusters

logger = logging.getLogger(__name__)


class MergeClustersTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "merge_clusters"

    @property
    def input_type(self) -> str:
        return "clustered"

    @property
    def output_type(self) -> str:
        return "annotated"

    def get_param_space(self) -> dict:
        return {
            "cluster_ids": {"type": "list", "required": True},
            "label": {"type": "string", "required": True},
            "fine_label": {"type": "string", "required": False},
        }

    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Merge multiple clusters under one cell type label.

        Parameters
        ----------
        adata : AnnData with existing clustering.
        params : dict with:
            cluster_ids (required): list of cluster IDs to merge.
            label (required): cell type label for merged clusters.
            fine_label: optional fine-grained label.

        Returns
        -------
        tuple of (adata, stats).
        """
        t0 = time.time()

        adata, stats = merge_clusters(
            adata,
            cluster_ids=params["cluster_ids"],
            label=params["label"],
            fine_label=params.get("fine_label"),
        )

        stats["elapsed_s"] = round(time.time() - t0, 1)
        adata.uns.setdefault("merge_clusters_log", []).append(stats)
        logger.info(f"[merge_clusters] Done: {stats['elapsed_s']}s")
        return adata, stats
