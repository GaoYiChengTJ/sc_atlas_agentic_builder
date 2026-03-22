"""
HarmonizeLabelsTool — standardize cell type label names across datasets.

After annotating each dataset independently, label names may differ
(e.g., "T cells" vs "T_cell", "Mono" vs "Monocytes"). The LLM agent
provides a mapping dict based on its semantic understanding, and this
tool applies it.
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper
from .operations import harmonize_labels

logger = logging.getLogger(__name__)


class HarmonizeLabelsTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "harmonize_labels"

    @property
    def input_type(self) -> str:
        return "annotated"

    @property
    def output_type(self) -> str:
        return "harmonized"

    def get_param_space(self) -> dict:
        return {
            "label_key": {"type": "string", "default": "cell_type"},
            "mapping": {"type": "dict", "required": True},
            "fine_mapping": {"type": "dict", "required": False},
        }

    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Apply label renaming mappings to standardize cell type names.

        The agent provides the mappings based on its understanding of
        cell type naming conventions (handles abbreviations, synonyms,
        formatting differences).

        Parameters
        ----------
        adata : AnnData with cell type labels.
        params : dict with:
            label_key ("cell_type"): obs column with broad labels.
            mapping (required): {old_label: new_label} for broad labels.
            fine_mapping: optional {old_label: new_label} for fine labels
                (applied to "{label_key}_fine" column).
        batch_key : obs column for batch (used for per-batch reporting).

        Returns
        -------
        tuple of (adata, stats).
        """
        t0 = time.time()

        adata, stats = harmonize_labels(
            adata,
            label_key=params.get("label_key", "cell_type"),
            mapping=params.get("mapping"),
            fine_mapping=params.get("fine_mapping"),
            batch_key=batch_key,
        )

        stats["elapsed_s"] = round(time.time() - t0, 1)
        adata.uns["harmonize_labels_log"] = stats
        logger.info(f"[harmonize_labels] Done: {stats['elapsed_s']}s")
        return adata, stats
