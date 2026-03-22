"""
ScoreMarkersTool — score clusters against known marker gene signatures.
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper
from .operations import score_known_markers

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "groupby": "leiden",
    "use_raw": None,
}


class ScoreMarkersTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "score_markers"

    @property
    def output_type(self) -> str:
        return "marker_scores"

    @property
    def input_type(self) -> str:
        return "clustered"

    def _check_imports(self):
        import scanpy  # noqa: F401

    def get_param_space(self) -> dict:
        return {}

    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Score clusters against known marker gene signatures.

        Parameters
        ----------
        adata : AnnData — clustered.
        params : dict with:
            marker_dict : {cell_type: [gene1, gene2, ...]} (required)
            groupby ("leiden"), use_raw (None, auto).

        Returns
        -------
        AnnData with score columns in obs and score_markers_log in uns.
        """
        t0 = time.time()
        p = {**_DEFAULTS, **params}

        if not p.get("marker_dict"):
            raise ValueError("marker_dict is required.")

        logger.info(f"[score_markers] Start: {adata.n_obs} cells")

        adata, stats = score_known_markers(
            adata,
            marker_dict=p["marker_dict"],
            groupby=p["groupby"],
            use_raw=p["use_raw"],
        )

        adata.uns["score_markers_log"] = stats
        logger.info(f"[score_markers] Done: {time.time()-t0:.1f}s")
        return adata
