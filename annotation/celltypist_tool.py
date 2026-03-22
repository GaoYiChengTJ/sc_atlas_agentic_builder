"""
CellTypistTool — reference-based cell type annotation using CellTypist.
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper
from .operations import run_celltypist

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "groupby": "leiden",
    "model": "Immune_All_Low.pkl",
    "majority_voting": True,
    "auto_download": False,
}


class CellTypistTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "celltypist"

    @property
    def output_type(self) -> str:
        return "celltypist_predictions"

    @property
    def input_type(self) -> str:
        return "clustered"

    def _check_imports(self):
        import celltypist  # noqa: F401

    def get_param_space(self) -> dict:
        return {
            "model": {
                "type": "categorical",
                "choices": [
                    "Immune_All_Low.pkl",
                    "Immune_All_High.pkl",
                    "Human_Lung_Atlas.pkl",
                ],
            },
            "majority_voting": {"type": "categorical", "choices": [True, False]},
        }

    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Run CellTypist reference-based annotation.

        Parameters
        ----------
        adata : AnnData — clustered, with normalized data in .raw or layers.
        params : dict with:
            model ("Immune_All_Low.pkl"), groupby ("leiden"),
            majority_voting (True), auto_download (False).

        Returns
        -------
        AnnData with celltypist predictions in obs and celltypist_log in uns.
        """
        t0 = time.time()
        p = {**_DEFAULTS, **params}

        logger.info(f"[celltypist] Start: {adata.n_obs} cells, model={p['model']}")

        adata, stats = run_celltypist(
            adata,
            model=p["model"],
            groupby=p["groupby"],
            majority_voting=p["majority_voting"],
            auto_download=p["auto_download"],
        )

        adata.uns["celltypist_log"] = stats
        logger.info(f"[celltypist] Done: {time.time()-t0:.1f}s")
        return adata
