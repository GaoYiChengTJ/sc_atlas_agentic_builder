"""
SubclusterTool — subset a cluster and re-cluster + find markers.

Used when a cluster has low annotation confidence or mixed markers.
The agent reviews the sub-markers and then calls AnnotateSubclustersTool.
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper
from .operations import subcluster

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "resolution": 0.3,
    "use_rep": "X_pca",
    "n_neighbors": 15,
    "n_top_marker_genes": 10,
}


class SubclusterTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "subcluster"

    @property
    def input_type(self) -> str:
        return "clustered"

    @property
    def output_type(self) -> str:
        return "subclustered"

    def get_param_space(self) -> dict:
        return {
            "cluster_id": {"type": "string", "required": True},
            "resolution": {"type": "float", "low": 0.05, "high": 2.0},
            "use_rep": {"type": "categorical", "choices": ["X_pca", "X_harmony", "X_scanorama"]},
            "n_neighbors": {"type": "int", "low": 5, "high": 50},
        }

    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Subcluster a specific cluster.

        Parameters
        ----------
        adata : AnnData with existing clustering in obs.
        params : dict with:
            cluster_id (required): which cluster to subcluster.
            resolution (0.3): leiden resolution for subclustering.
            use_rep ("X_pca"): embedding to use.
            n_neighbors (15): neighbors for sub-graph.
            n_top_marker_genes (10): top markers per subcluster.

        Returns
        -------
        tuple of (adata_subset, stats).
        The subset has new leiden labels + marker genes.
        The caller should store adata_subset for annotate_subclusters.
        """
        t0 = time.time()
        p = {**_DEFAULTS, **params}

        adata_sub, stats = subcluster(
            adata,
            cluster_id=str(p["cluster_id"]),
            resolution=p["resolution"],
            use_rep=p["use_rep"],
            n_neighbors=p["n_neighbors"],
            n_top_marker_genes=p["n_top_marker_genes"],
        )

        stats["elapsed_s"] = round(time.time() - t0, 1)
        adata_sub.uns["subcluster_log"] = stats
        logger.info(f"[subcluster] Done: {stats['elapsed_s']}s")
        return adata_sub, stats
