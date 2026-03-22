"""
Single-cell clustering tool for cell atlas construction.

    from clustering import ClusteringTool

    clustered = ClusteringTool().run(adata, params={...})
"""

from .clustering_tool import ClusteringTool

__all__ = ["ClusteringTool"]
