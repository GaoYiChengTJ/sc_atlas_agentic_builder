"""
Abstract base class for single-cell analysis tool wrappers.

Shared by preprocessing, annotation, and other tool modules.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class ToolWrapper(ABC):
    """
    Unified interface for single-cell analysis tools.

    Each subclass wraps one analysis step and exposes:
      - name: unique tool identifier
      - input_type: what kind of input is expected
      - output_type: what kind of output is produced
      - get_param_space(): parameter search space for optimization
      - run(): execute the tool with given parameters
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier (e.g., 'preprocessing', 'annotation')."""
        ...

    @property
    @abstractmethod
    def output_type(self) -> str:
        """What this tool produces (e.g., 'preprocessed', 'annotated')."""
        ...

    @property
    def input_type(self) -> str:
        """
        What this tool needs in adata.X.

        Common values:
          - 'counts': raw integer counts
          - 'preprocessed': normalized, HVG-selected, with PCA
          - 'normalized': normalized + log-transformed expression
          - 'pca_only': only uses obsm['X_pca'], X content irrelevant

        Override in subclasses. Default is 'counts'.
        """
        return "counts"

    @abstractmethod
    def get_param_space(self) -> dict:
        """
        Return the parameter search space.

        Returns a dict of param_name -> dict with keys:
          - type: 'int', 'float', 'float_log', 'categorical'
          - low, high: for int/float/float_log
          - choices: for categorical
        """
        ...

    @abstractmethod
    def run(self, adata, params: dict, batch_key: Optional[str] = None):
        """
        Run the tool with the given parameters.

        Parameters
        ----------
        adata : AnnData
            Input data.
        params : dict
            Parameter values from the optimizer.
        batch_key : optional
            Column in adata.obs identifying batches.

        Returns
        -------
        AnnData with processed representation.
        """
        ...

    def is_available(self) -> bool:
        """Check if the tool's dependencies are installed."""
        try:
            self._check_imports()
            return True
        except ImportError:
            return False

    def _check_imports(self):
        """Override to import tool-specific packages (raises ImportError if missing)."""
        pass

    def run_timed(self, adata, params: dict, batch_key: Optional[str] = None):
        """Run the tool and return (result_adata, elapsed_seconds)."""
        t0 = time.time()
        result = self.run(adata, params, batch_key)
        elapsed = time.time() - t0
        logger.info(f"[{self.name}] Completed in {elapsed:.1f}s")
        return result, elapsed
