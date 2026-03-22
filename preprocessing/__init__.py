"""
Single-cell preprocessing tool for cell atlas construction.

    from preprocessing import PreprocessingTool, profile_dataset

    profile = profile_dataset(adata, batch_key="batch")
    preprocessed = PreprocessingTool().run(adata, params={...}, batch_key="batch")
"""

from .preprocessing_tool import PreprocessingTool
from .profiler import profile_dataset, compare_profiles

__all__ = ["PreprocessingTool", "profile_dataset", "compare_profiles"]
