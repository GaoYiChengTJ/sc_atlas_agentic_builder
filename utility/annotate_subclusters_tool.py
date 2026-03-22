"""
AnnotateSubclustersTool — write subcluster labels back to main adata.

Called after SubclusterTool. The agent reviews sub-markers, decides labels,
and this tool merges them back into the main dataset.
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper
from .operations import annotate_subclusters

logger = logging.getLogger(__name__)


class AnnotateSubclustersTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "annotate_subclusters"

    @property
    def input_type(self) -> str:
        return "subclustered"

    @property
    def output_type(self) -> str:
        return "annotated"

    def get_param_space(self) -> dict:
        return {
            "cluster_id": {"type": "string", "required": True},
            "label_mapping": {"type": "dict", "required": True},
            "fine_label_mapping": {"type": "dict", "required": False},
        }

    def run(self, adata, params: dict, batch_key: Optional[str] = None,
            adata_sub=None):
        """
        Write subcluster labels back to the main adata.

        Parameters
        ----------
        adata : main AnnData (full dataset).
        params : dict with:
            cluster_id (required): the original cluster that was subclustered.
            label_mapping (required): {sub_cluster_id: cell_type_name}.
            fine_label_mapping: optional {sub_cluster_id: fine_label}.
        adata_sub : the subclustered AnnData from SubclusterTool.

        Returns
        -------
        tuple of (adata, stats).
        """
        t0 = time.time()

        if adata_sub is None:
            raise ValueError(
                "adata_sub is required. Pass the subclustered AnnData "
                "returned by SubclusterTool."
            )

        adata, stats = annotate_subclusters(
            adata,
            adata_sub=adata_sub,
            cluster_id=str(params["cluster_id"]),
            label_mapping=params["label_mapping"],
            fine_label_mapping=params.get("fine_label_mapping"),
        )

        stats["elapsed_s"] = round(time.time() - t0, 1)
        adata.uns.setdefault("annotate_subclusters_log", {})[str(params["cluster_id"])] = stats
        logger.info(f"[annotate_subclusters] Done: {stats['elapsed_s']}s")
        return adata, stats
