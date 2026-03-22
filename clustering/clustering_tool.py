"""
ClusteringTool — single-cell clustering as a ToolWrapper.

Takes preprocessed AnnData (with PCA or integrated embedding),
returns clustered AnnData with UMAP.
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper

from .operations import (
    assess_clusters,
    compute_neighbors,
    compute_umap,
    run_leiden,
)

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "n_neighbors": 15,
    "n_pcs": None,
    "use_rep": "X_pca",
    "metric": "euclidean",
    "resolution": 1.0,
    "random_state": 0,
    "run_umap": True,
    "umap_min_dist": 0.5,
    "umap_spread": 1.0,
    "run_assessment": True,
}


class ClusteringTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "clustering"

    @property
    def output_type(self) -> str:
        return "clustered"

    @property
    def input_type(self) -> str:
        return "preprocessed"

    def _check_imports(self):
        import scanpy  # noqa: F401

    def get_param_space(self) -> dict:
        return {
            "n_neighbors": {"type": "int", "low": 5, "high": 50},
            "resolution": {"type": "float", "low": 0.1, "high": 3.0},
            "use_rep": {"type": "categorical", "choices": ["X_pca", "X_harmony", "X_scVI"]},
        }

    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Run clustering pipeline.

        Parameters
        ----------
        adata : AnnData — preprocessed, with PCA or integrated embedding.
        params : dict — clustering parameters. Unset keys use defaults:
            n_neighbors (15), n_pcs (None), use_rep ("X_pca"),
            metric ("euclidean"), resolution (1.0), random_state (0),
            run_umap (True), umap_min_dist (0.5), umap_spread (1.0),
            run_assessment (True).
        batch_key : unused, kept for ToolWrapper interface compatibility.

        Returns
        -------
        AnnData with cluster labels in obs, UMAP in obsm, and
        clustering_log in uns.
        """
        t0 = time.time()
        p = {**_DEFAULTS, **params}
        log = {}
        step = None

        logger.info(f"[clustering] Start: {adata.n_obs} cells, use_rep={p['use_rep']}")

        try:
            # 1. Neighbors
            step = "neighbors"
            adata, log["neighbors"] = compute_neighbors(
                adata,
                n_neighbors=p["n_neighbors"],
                n_pcs=p["n_pcs"],
                use_rep=p["use_rep"],
                metric=p["metric"],
            )

            # 2. Clustering
            step = "cluster"
            adata, log["cluster"] = run_leiden(
                adata,
                resolution=p["resolution"],
                key_added="leiden",
                random_state=p["random_state"],
            )

            # 3. UMAP (optional)
            if p["run_umap"]:
                step = "umap"
                adata, log["umap"] = compute_umap(
                    adata,
                    min_dist=p["umap_min_dist"],
                    spread=p["umap_spread"],
                    random_state=p["random_state"],
                )

            # 4. Assessment (optional)
            if p["run_assessment"]:
                step = "assessment"
                log["assessment"] = assess_clusters(
                    adata,
                    cluster_key="leiden",
                    use_rep=p["use_rep"],
                )

        except Exception as e:
            logger.error(f"[clustering] Failed at step '{step}': {e}")
            log["failed_step"] = step
            log["error"] = str(e)
            raise RuntimeError(
                f"Clustering failed at step '{step}': {e}"
            ) from e

        adata.uns["clustering_log"] = log
        n_clusters = log["cluster"]["n_clusters"]
        logger.info(
            f"[clustering] Done: {n_clusters} clusters, "
            f"{time.time()-t0:.1f}s"
        )
        return adata
