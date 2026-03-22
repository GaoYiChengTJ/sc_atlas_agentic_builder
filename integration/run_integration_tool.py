"""
RunIntegrationTool — run batch integration on multi-sample data.

Supports Harmony (fast, PCA-based) and Scanorama (MNN-based).
Writes corrected embedding to adata.obsm.
"""

import logging
import time
from typing import Optional

from ..base import ToolWrapper
from .operations import run_integration

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "method": "harmony",
    # Harmony params
    "n_pcs": None,
    "max_iter": 20,
}


class RunIntegrationTool(ToolWrapper):

    @property
    def name(self) -> str:
        return "run_integration"

    @property
    def output_type(self) -> str:
        return "integrated"

    @property
    def input_type(self) -> str:
        return "preprocessed"

    def _check_imports(self):
        errors = []
        try:
            import harmonypy  # noqa: F401
        except ImportError:
            errors.append("harmonypy")
        try:
            import scanorama  # noqa: F401
        except ImportError:
            errors.append("scanorama")
        if len(errors) == 2:
            raise ImportError(
                "No integration library installed. Install at least one: "
                "pip install harmonypy  OR  pip install scanorama"
            )

    def get_param_space(self) -> dict:
        return {
            "method": {"type": "categorical", "choices": ["harmony", "scanorama"]},
            "max_iter": {"type": "int", "low": 5, "high": 50},
        }

    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Run batch integration.

        Parameters
        ----------
        adata : AnnData — preprocessed with PCA.
        params : dict with:
            method ("harmony"), plus method-specific params.
            Harmony: n_pcs (None), max_iter (20).
            Scanorama: no extra params.
        batch_key : obs column for batch (required).

        Returns
        -------
        (AnnData, stats) with integrated embedding in obsm
        (X_harmony or X_scanorama).
        """
        t0 = time.time()
        p = {**_DEFAULTS, **params}

        if not batch_key:
            raise ValueError("batch_key is required for integration.")

        method = p["method"]
        logger.info(f"[run_integration] Start: {method}, {adata.n_obs} cells")

        # Build method-specific kwargs.
        if method == "harmony":
            kwargs = {k: p[k] for k in ["n_pcs", "max_iter"] if p.get(k) is not None}
        elif method == "scanorama":
            kwargs = {}
        else:
            kwargs = {}

        adata, stats = run_integration(adata, method=method, batch_key=batch_key, **kwargs)

        stats["elapsed_s"] = round(time.time() - t0, 1)
        adata.uns["run_integration_log"] = stats
        logger.info(f"[run_integration] Done: {method}, {stats['elapsed_s']}s")
        return adata, stats
