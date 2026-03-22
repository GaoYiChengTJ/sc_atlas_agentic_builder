"""
Marker gene identification tool for cell atlas construction.

    from marker_genes import GetMarkerGenesTool

    result = GetMarkerGenesTool().run(adata, params={...})
"""

from .marker_genes_tool import GetMarkerGenesTool

__all__ = ["GetMarkerGenesTool"]
