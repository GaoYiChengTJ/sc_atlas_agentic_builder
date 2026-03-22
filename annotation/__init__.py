"""
Cell type annotation tools for cell atlas construction.

Four tools, each mapping to one agent decision:

    from annotation import (
        ScoreMarkersTool,       # score clusters against marker gene lists
        CellTypistTool,         # reference-based annotation (optional)
        CheckConfidenceTool,    # evaluate confidence, flag ambiguous clusters
        AnnotateCellTypesTool,  # LLM decides and writes cell type labels
    )
"""

from .score_markers_tool import ScoreMarkersTool
from .celltypist_tool import CellTypistTool
from .check_confidence_tool import CheckConfidenceTool
from .annotate_cell_types_tool import AnnotateCellTypesTool

__all__ = [
    "ScoreMarkersTool",
    "CellTypistTool",
    "CheckConfidenceTool",
    "AnnotateCellTypesTool",
]
