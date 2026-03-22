"""
PreprocessingTool — single-cell dataset preprocessing as a ToolWrapper.

Takes raw or partially-processed data and returns preprocessed AnnData
with PCA. Automatically detects the data state (raw counts, normalized,
log-normalized) via adata.uns["matrix_type"] (set by prepare_data) or
by inspecting the matrix, and skips steps that have already been applied.

The calling agent decides all parameters; this tool just executes.
"""

import logging
import time
from typing import Any, Optional

import numpy as np
from scipy.sparse import issparse

from ..base import ToolWrapper

from .operations import (
    detect_doublets,
    filter_cells_and_genes,
    normalize,
    run_pca,
    select_hvg,
)

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "min_genes": 200, "max_genes": 8000, "min_counts": 500,
    "max_counts": None, "max_mt_percent": 20.0, "mt_prefix": "MT-",
    "min_cells_per_gene": 3, "min_cells_per_batch": 50,
    "run_doublet_detection": True, "doublet_method": "scrublet",
    "doublet_threshold": None,
    "normalization_method": "total", "target_sum": 1e4, "log_transform": True,
    "n_top_genes": 2000, "hvg_flavor": "seurat_v3",
    "n_pcs": 50, "scale_before_pca": True, "scale_max_value": 10.0,
}


def _detect_matrix_type(adata) -> str:
    """Detect data state from adata.uns['matrix_type'] or by inspecting .X.

    Returns one of: 'raw_counts', 'normalized', 'log_normalized',
    'scaled_or_residuals', 'unknown'.
    """
    # Prefer the annotation left by prepare_data.
    mt = adata.uns.get("matrix_type")
    if mt and isinstance(mt, dict) and mt.get("likely_type"):
        return mt["likely_type"]

    # Fall back to inspecting the matrix directly.
    X = adata.X
    if issparse(X):
        data = X.data
    else:
        data = X.ravel()

    if len(data) == 0:
        return "unknown"

    # Sample for speed.
    if len(data) > 10_000:
        rng = np.random.default_rng(42)
        data = rng.choice(np.asarray(data, dtype=np.float64), size=10_000, replace=False)
    else:
        data = np.asarray(data, dtype=np.float64)

    if np.any(data < 0):
        return "scaled_or_residuals"
    if np.allclose(data, np.round(data), atol=1e-6):
        return "raw_counts"
    if float(np.max(data)) < 20:
        return "log_normalized"
    return "normalized"


class PreprocessingTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "preprocessing"

    @property
    def output_type(self) -> str:
        return "preprocessed"

    @property
    def input_type(self) -> str:
        return "counts"

    def _check_imports(self):
        import scanpy  # noqa: F401

    def get_param_space(self) -> dict:
        return {
            "min_genes": {"type": "int", "low": 50, "high": 1000},
            "max_genes": {"type": "int", "low": 3000, "high": 20000},
            "min_counts": {"type": "int", "low": 100, "high": 2000},
            "max_mt_percent": {"type": "float", "low": 5.0, "high": 50.0},
            "normalization_method": {"type": "categorical", "choices": ["total", "scran", "pearson_residuals"]},
            "n_top_genes": {"type": "int", "low": 500, "high": 10000},
            "hvg_flavor": {"type": "categorical", "choices": ["seurat", "seurat_v3", "cell_ranger"]},
            "n_pcs": {"type": "int", "low": 5, "high": 100},
            "scale_before_pca": {"type": "categorical", "choices": [True, False]},
            "run_doublet_detection": {"type": "categorical", "choices": [True, False]},
        }

    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Run preprocessing pipeline, adapting to the data state.

        Detects whether the input is raw counts, normalized, or
        log-normalized (via adata.uns['matrix_type'] from prepare_data,
        or by inspecting .X) and skips steps already applied:

          raw_counts       → QC, doublets, normalize, log1p, HVG, PCA
          normalized       → QC, doublets, (skip normalize), log1p, HVG, PCA
          log_normalized   → QC, doublets, (skip normalize+log1p), HVG, PCA
          scaled_or_residuals → (skip QC+doublets+normalize+log1p), HVG, PCA

        Parameters
        ----------
        adata : AnnData
        params : dict — preprocessing parameters. Unset keys use defaults.
        batch_key : optional obs column for batch.

        Returns
        -------
        AnnData with layers["counts"] (if raw), layers["normalized"],
        .raw, PCA.
        """
        t0 = time.time()
        p = {**_DEFAULTS, **params}
        log: dict[str, Any] = {}
        step = None

        if batch_key and batch_key not in adata.obs.columns:
            logger.warning(
                f"[preprocessing] batch_key '{batch_key}' not found in "
                f"adata.obs (available: {list(adata.obs.columns)}). "
                "Batch-aware steps will run without batch correction."
            )
            batch_key = None

        # Detect data state.
        data_state = _detect_matrix_type(adata)
        log["detected_data_state"] = data_state

        skip_qc = data_state == "scaled_or_residuals"
        skip_doublets = data_state == "scaled_or_residuals"
        skip_normalize = data_state in ("normalized", "log_normalized", "scaled_or_residuals")
        skip_log1p = data_state in ("log_normalized", "scaled_or_residuals")

        skipped_steps = []
        if skip_qc:
            skipped_steps.append("qc")
        if skip_doublets:
            skipped_steps.append("doublets")
        if skip_normalize:
            skipped_steps.append("normalization")
        if skip_log1p:
            skipped_steps.append("log1p")

        if skipped_steps:
            logger.info(
                f"[preprocessing] Data state: {data_state} — "
                f"skipping: {', '.join(skipped_steps)}"
            )
        else:
            logger.info(f"[preprocessing] Data state: {data_state} — running full pipeline")

        log["skipped_steps"] = skipped_steps
        logger.info(f"[preprocessing] Start: {adata.n_obs}x{adata.n_vars}")

        try:
            # 1. QC
            if not skip_qc:
                step = "qc"
                adata, log["qc"] = filter_cells_and_genes(
                    adata, min_genes=p["min_genes"], max_genes=p["max_genes"],
                    min_counts=p["min_counts"], max_counts=p.get("max_counts"),
                    min_cells_per_gene=p["min_cells_per_gene"],
                    max_mt_percent=p["max_mt_percent"], mt_prefix=p["mt_prefix"],
                    batch_key=batch_key, min_cells_per_batch=p["min_cells_per_batch"],
                )
            else:
                log["qc"] = {"skipped": True, "reason": f"data is {data_state}"}

            # 2. Doublets
            if not skip_doublets and p["run_doublet_detection"]:
                step = "doublet"
                adata, log["doublet"] = detect_doublets(
                    adata, method=p["doublet_method"],
                    threshold=p.get("doublet_threshold"),
                    batch_key=batch_key,
                )
            elif skip_doublets:
                log["doublet"] = {"skipped": True, "reason": f"data is {data_state}"}

            # 3. Normalize
            if not skip_normalize:
                step = "norm"
                adata, log["norm"] = normalize(
                    adata, method=p["normalization_method"],
                    target_sum=p["target_sum"],
                    log_transform=p["log_transform"] and not skip_log1p,
                )
            elif not skip_log1p:
                # Data is normalized but not log-transformed — just do log1p.
                step = "norm"
                import scanpy as sc
                if "counts" not in adata.layers:
                    adata.layers["counts"] = adata.X.copy()
                sc.pp.log1p(adata)
                adata.layers["normalized"] = adata.X.copy()
                log["norm"] = {
                    "method": "skipped (already normalized)",
                    "log_transformed": True,
                }
                logger.info("[Norm] Skipped normalization, applied log1p only")
            else:
                # Both normalize and log1p are skipped.
                if "normalized" not in adata.layers:
                    adata.layers["normalized"] = adata.X.copy()
                log["norm"] = {
                    "method": "skipped",
                    "log_transformed": False,
                    "reason": f"data is {data_state}",
                }

            # Snapshot .raw AFTER normalization/log1p but BEFORE HVG
            # subsetting and scaling, so that rank_genes_groups (which
            # defaults to .raw) sees log-transformed values for all genes.
            adata.raw = adata

            # 4. HVG
            step = "hvg"
            adata, log["hvg"] = select_hvg(
                adata, n_top_genes=p["n_top_genes"],
                flavor=p["hvg_flavor"], batch_key=batch_key,
            )

            # 5. PCA
            step = "pca"
            adata, log["pca"] = run_pca(
                adata, n_pcs=p["n_pcs"],
                scale=p["scale_before_pca"],
                scale_max_value=p.get("scale_max_value", 10.0),
            )

        except Exception as e:
            logger.error(f"[preprocessing] Failed at step '{step}': {e}")
            log["failed_step"] = step
            log["error"] = str(e)
            raise RuntimeError(
                f"Preprocessing failed at step '{step}': {e}"
            ) from e

        adata.uns["preprocessing_log"] = log
        logger.info(f"[preprocessing] Done: {adata.n_obs}x{adata.n_vars}, {time.time()-t0:.1f}s")
        return adata
