"""
AnnotateCellTypesTool — LLM-driven cell type label assignment.

The agent reviews marker scores, CellTypist predictions, and confidence
reports, then decides on cell type labels for each cluster. This tool
writes those decisions to adata.obs.
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper
from .operations import assign_labels

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "groupby": "leiden",
    "key_added": "cell_type",
    "fine_key_added": "cell_type_fine",
}


class AnnotateCellTypesTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "annotate_cell_types"

    @property
    def output_type(self) -> str:
        return "annotated"

    @property
    def input_type(self) -> str:
        return "clustered"

    def _check_imports(self):
        pass

    def get_param_space(self) -> dict:
        return {}

    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Assign cell type labels decided by the LLM agent.

        Parameters
        ----------
        adata : AnnData — clustered, with annotation evidence in obs/uns.
        params : dict with:
            label_mapping : {cluster_id: cell_type} (required)
                Broad cell type labels decided by the agent.
            fine_label_mapping : optional {cluster_id: fine_label}
                Granular cell type labels (e.g., "CD8+ cytotoxic T").
            groupby ("leiden"), key_added ("cell_type"),
            fine_key_added ("cell_type_fine").

        Returns
        -------
        AnnData with cell type labels in obs["cell_type"] (and optionally
        obs["cell_type_fine"]) and annotate_cell_types_log in uns.
        """
        t0 = time.time()
        p = {**_DEFAULTS, **params}

        if not p.get("label_mapping"):
            raise ValueError(
                "label_mapping is required. Provide {cluster_id: cell_type} "
                "based on marker gene evidence."
            )

        logger.info(
            f"[annotate_cell_types] Start: {adata.n_obs} cells, "
            f"{len(p['label_mapping'])} labels"
        )

        adata, stats = assign_labels(
            adata,
            label_mapping=p["label_mapping"],
            groupby=p["groupby"],
            key_added=p["key_added"],
            fine_label_mapping=p.get("fine_label_mapping"),
            fine_key_added=p["fine_key_added"],
        )

        adata.uns["annotate_cell_types_log"] = stats
        logger.info(f"[annotate_cell_types] Done: {time.time()-t0:.1f}s")
        return adata
