"""
Demo: run the sc-atlas-agentic-builder pipeline with Claude API tool_use.

Claude autonomously drives the analysis — choosing parameters, interpreting
results, and deciding next steps. This script implements a minimal agent
loop (API call → tool execution → result feedback) without external
agent frameworks.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m demo.run_with_claude

    # Or with a custom dataset:
    python -m demo.run_with_claude --input my_data.h5ad --batch-key sample_id
"""

import argparse
import json
import logging
import os
import sys

# Add the grandparent dir (test/) to sys.path so sc_atlas_agentic_builder
# is importable as a top-level package with working relative imports.
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Tool definitions for Claude API ──

TOOLS = [
    {
        "name": "prepare_data",
        "description": (
            "Load and unify input datasets into a single AnnData object. "
            "Handles multiple formats: h5ad, 10X h5, 10X MTX directory, CSV/TSV, loom. "
            "For multiple files: detects format, converts Ensembl IDs to gene symbols, "
            "strips species prefixes, intersects (or unions) genes, and concatenates "
            "with batch labels. Returns per-file obs column info (names, dtypes, "
            "example values) so you can inspect them and provide obs_key_mapping "
            "to harmonize column names across datasets. "
            "Validates whether the matrix contains raw counts vs normalized data. "
            "Does NOT preprocess — call run_preprocessing next."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "input_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Path(s) to input files or 10X MTX directories",
                },
                "batch_key": {
                    "type": "string",
                    "description": "Column name for batch labels when merging multiple files",
                    "default": "batch",
                },
                "gene_id_type": {
                    "type": "string",
                    "enum": ["symbol", "ensembl", "auto"],
                    "description": "Gene identifier type (auto-detected if 'auto')",
                    "default": "auto",
                },
                "gene_intersection_mode": {
                    "type": "string",
                    "enum": ["intersection", "union"],
                    "description": "How to handle non-overlapping genes across files",
                    "default": "intersection",
                },
                "obs_key_mapping": {
                    "type": "object",
                    "description": (
                        "Harmonize obs column names across datasets. "
                        "Dict of {canonical_name: [synonym1, synonym2, ...]}. "
                        "E.g. {'cell_type': ['annotation', 'CellType', 'labels']}. "
                        "Review the per-file obs_columns in the result to decide "
                        "which columns represent the same concept."
                    ),
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
            "required": ["input_paths"],
        },
    },
    {
        "name": "run_preprocessing",
        "description": (
            "Preprocess raw scRNA-seq data: QC filtering, doublet removal, "
            "normalization, HVG selection, and PCA. Returns preprocessing stats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_genes": {"type": "integer", "description": "Min genes per cell", "default": 200},
                "max_genes": {"type": "integer", "description": "Max genes per cell", "default": 8000},
                "min_counts": {"type": "integer", "description": "Min UMI counts per cell", "default": 500},
                "max_mt_percent": {"type": "number", "description": "Max mitochondrial %", "default": 20.0},
                "normalization_method": {
                    "type": "string", "enum": ["total", "scran", "pearson_residuals"], "default": "total",
                },
                "n_top_genes": {"type": "integer", "description": "Number of HVGs", "default": 2000},
                "n_pcs": {"type": "integer", "description": "Number of PCs", "default": 50},
            },
        },
    },
    {
        "name": "select_integration_genes",
        "description": (
            "OPTIONAL: Re-select HVGs with batch-aware strategies for integration. "
            "Skip if preprocessing already ran with batch_key (rank strategy applied). "
            "Use only when you need intersection/union strategy or different gene counts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_top_genes": {
                    "type": "integer",
                    "description": "Number of HVGs to select",
                    "default": 2000,
                },
                "strategy": {
                    "type": "string",
                    "enum": ["rank", "intersection", "union"],
                    "description": "How to combine per-batch HVG lists",
                    "default": "rank",
                },
                "flavor": {
                    "type": "string",
                    "enum": ["seurat_v3", "seurat", "cell_ranger"],
                    "description": "HVG flavor",
                    "default": "seurat_v3",
                },
            },
        },
    },
    {
        "name": "run_integration",
        "description": (
            "Run batch integration on multi-sample data. Supports Harmony "
            "(fast, PCA-based) and Scanorama (MNN-based). Writes corrected "
            "embedding to adata.obsm. Only needed for multi-batch datasets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["harmony", "scanorama"],
                    "description": "Integration method",
                    "default": "harmony",
                },
                "n_pcs": {
                    "type": "integer",
                    "description": "Number of PCs for Harmony (default: all)",
                },
                "max_iter": {
                    "type": "integer",
                    "description": "Max iterations for Harmony",
                    "default": 20,
                },
            },
        },
    },
    {
        "name": "evaluate_integration",
        "description": (
            "Evaluate integration quality using scib-metrics. Computes batch "
            "mixing (ASW_batch, graph connectivity) and bio conservation "
            "(ASW_label, NMI, ARI) metrics. Requires scib-metrics. "
            "Provide label_key if cell type labels are available for bio metrics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "embed_key": {
                    "type": "string",
                    "description": "Embedding to evaluate (X_harmony, X_scanorama, X_pca)",
                    "default": "X_harmony",
                },
                "label_key": {
                    "type": "string",
                    "description": "Obs column with cell type labels for bio conservation metrics",
                },
            },
        },
    },
    {
        "name": "run_clustering",
        "description": (
            "Cluster cells: compute neighbors, Leiden clustering, UMAP, and "
            "quality assessment. Returns cluster sizes and silhouette score. "
            "Use use_rep to specify which embedding (X_pca for single-batch, "
            "X_harmony/X_scanorama for integrated data)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "use_rep": {
                    "type": "string",
                    "description": "Embedding to use (X_pca, X_harmony, X_scanorama)",
                    "default": "X_pca",
                },
                "n_neighbors": {"type": "integer", "description": "Number of neighbors", "default": 15},
                "resolution": {"type": "number", "description": "Leiden resolution", "default": 1.0},
            },
        },
    },
    {
        "name": "run_marker_genes",
        "description": (
            "Find marker genes per cluster using differential expression. "
            "Returns top marker genes for each cluster."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["wilcoxon", "t-test", "t-test_overestim_var", "logreg"],
                    "default": "wilcoxon",
                },
                "n_top_genes": {"type": "integer", "description": "Top N markers per cluster", "default": 10},
                "min_log2fc": {"type": "number", "description": "Min log2 fold change", "default": 0.25},
            },
        },
    },
    {
        "name": "score_markers",
        "description": (
            "Score clusters against known marker gene signatures. "
            "Provide a marker_dict mapping cell type names to lists of marker genes. "
            "Returns per-cluster scores and best matches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "marker_dict": {
                    "type": "object",
                    "description": "Dict of {cell_type: [gene1, gene2, ...]}",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
            "required": ["marker_dict"],
        },
    },
    {
        "name": "check_confidence",
        "description": (
            "Evaluate annotation confidence per cluster. Compares marker scores "
            "and CellTypist predictions (if available). Returns per-cluster "
            "confidence scores and flags low-confidence clusters that need "
            "subclustering or manual review."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "low_confidence_threshold": {
                    "type": "number",
                    "description": "Threshold below which clusters are flagged (default 0.5)",
                    "default": 0.5,
                },
            },
        },
    },
    {
        "name": "annotate_cell_types",
        "description": (
            "Assign cell type labels to clusters based on your analysis of "
            "marker genes, scores, and confidence. Provide a label_mapping "
            "dict of {cluster_id: cell_type_name}. Optionally provide "
            "fine_label_mapping for granular annotations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "label_mapping": {
                    "type": "object",
                    "description": "Dict of {cluster_id: cell_type_label}",
                    "additionalProperties": {"type": "string"},
                },
                "fine_label_mapping": {
                    "type": "object",
                    "description": "Optional dict of {cluster_id: fine_label}",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["label_mapping"],
        },
    },
    {
        "name": "subcluster",
        "description": (
            "Subcluster a specific cluster that has low annotation confidence "
            "or mixed markers. Subsets the data, re-clusters at the given "
            "resolution, and returns sub-cluster marker genes. Call "
            "annotate_subclusters afterwards to assign labels to subclusters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cluster_id": {
                    "type": "string",
                    "description": "The cluster ID to subcluster (e.g., '0', '5')",
                },
                "resolution": {
                    "type": "number",
                    "description": "Leiden resolution for subclustering (lower = fewer subclusters)",
                    "default": 0.3,
                },
                "use_rep": {
                    "type": "string",
                    "description": "Embedding to use (X_pca, X_harmony, X_scanorama)",
                    "default": "X_pca",
                },
            },
            "required": ["cluster_id"],
        },
    },
    {
        "name": "annotate_subclusters",
        "description": (
            "Assign cell type labels to subclusters of a previously subclustered "
            "cluster. Must call subcluster first. The sub-labels are merged "
            "back into the main dataset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cluster_id": {
                    "type": "string",
                    "description": "The original cluster ID that was subclustered",
                },
                "label_mapping": {
                    "type": "object",
                    "description": "Dict of {sub_cluster_id: cell_type_name}",
                    "additionalProperties": {"type": "string"},
                },
                "fine_label_mapping": {
                    "type": "object",
                    "description": "Optional fine-grained labels",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["cluster_id", "label_mapping"],
        },
    },
    {
        "name": "merge_clusters",
        "description": (
            "Merge multiple clusters into a single cell type label. "
            "Use when marker analysis shows two clusters are the same cell type."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "cluster_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of cluster IDs to merge",
                },
                "label": {
                    "type": "string",
                    "description": "Cell type label for the merged clusters",
                },
                "fine_label": {
                    "type": "string",
                    "description": "Optional fine-grained label",
                },
            },
            "required": ["cluster_ids", "label"],
        },
    },
    {
        "name": "harmonize_labels",
        "description": (
            "Standardize cell type label names across datasets. "
            "After annotating each dataset independently, label names may differ "
            "(e.g., 'T cells' vs 'T_cell', 'Mono' vs 'Monocytes'). "
            "Provide a mapping dict to rename labels. Optionally provide "
            "fine_mapping for fine-grained labels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mapping": {
                    "type": "object",
                    "description": "Dict of {old_label: new_label} for broad cell type labels",
                    "additionalProperties": {"type": "string"},
                },
                "fine_mapping": {
                    "type": "object",
                    "description": "Optional dict of {old_label: new_label} for fine labels",
                    "additionalProperties": {"type": "string"},
                },
                "label_key": {
                    "type": "string",
                    "description": "Obs column with labels (default 'cell_type')",
                    "default": "cell_type",
                },
            },
            "required": ["mapping"],
        },
    },
    {
        "name": "annotate_batch",
        "description": (
            "Annotate a single batch independently. Subsets the data to the "
            "specified batch, runs clustering + marker genes, and returns "
            "cluster markers. Use this to annotate each batch before integration. "
            "After reviewing markers, call assign_batch_labels to write labels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "batch_value": {
                    "type": "string",
                    "description": "The batch value to annotate (e.g., 'sample_1')",
                },
                "resolution": {
                    "type": "number",
                    "description": "Leiden resolution for clustering",
                    "default": 0.8,
                },
                "use_rep": {
                    "type": "string",
                    "description": "Embedding to use (X_pca)",
                    "default": "X_pca",
                },
                "n_top_genes": {
                    "type": "integer",
                    "description": "Top marker genes per cluster",
                    "default": 10,
                },
            },
            "required": ["batch_value"],
        },
    },
    {
        "name": "assign_batch_labels",
        "description": (
            "Assign cell type labels to a previously annotated batch. "
            "Must call annotate_batch first. Labels are merged back into "
            "the main dataset."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "batch_value": {
                    "type": "string",
                    "description": "The batch value that was annotated",
                },
                "label_mapping": {
                    "type": "object",
                    "description": "Dict of {cluster_id: cell_type_name}",
                    "additionalProperties": {"type": "string"},
                },
                "fine_label_mapping": {
                    "type": "object",
                    "description": "Optional fine-grained labels",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["batch_value", "label_mapping"],
        },
    },
]


class PipelineExecutor:
    """Executes tool calls from Claude and manages adata state."""

    def __init__(self, adata, batch_key=None):
        self.adata = adata
        self.batch_key = batch_key
        # Store subclustered adata objects keyed by cluster_id.
        self._stored_subclusters: dict[str, "anndata.AnnData"] = {}
        # Store per-batch annotated adata objects keyed by batch_value.
        self._stored_batches: dict[str, "anndata.AnnData"] = {}

        from sc_atlas_agentic_builder.preprocessing import PreprocessingTool
        from sc_atlas_agentic_builder.integration import (
            SelectIntegrationGenesTool,
            RunIntegrationTool,
            EvaluateIntegrationTool,
        )
        from sc_atlas_agentic_builder.clustering import ClusteringTool
        from sc_atlas_agentic_builder.marker_genes import GetMarkerGenesTool
        from sc_atlas_agentic_builder.annotation import (
            ScoreMarkersTool,
            CellTypistTool,
            CheckConfidenceTool,
            AnnotateCellTypesTool,
        )
        from sc_atlas_agentic_builder.utility import (
            PrepareDataTool,
            SubclusterTool,
            AnnotateSubclustersTool,
            MergeClustersTool,
            HarmonizeLabelsTool,
        )

        self.prepare_data = PrepareDataTool()
        self.preprocessing = PreprocessingTool()
        self.select_integration_genes = SelectIntegrationGenesTool()
        self.integration = RunIntegrationTool()
        self.evaluate_integration = EvaluateIntegrationTool()
        self.clustering = ClusteringTool()
        self.marker_genes = GetMarkerGenesTool()
        self.score_markers = ScoreMarkersTool()
        self.celltypist = CellTypistTool()
        self.check_confidence = CheckConfidenceTool()
        self.annotate_cell_types = AnnotateCellTypesTool()
        self.subcluster = SubclusterTool()
        self.annotate_subclusters = AnnotateSubclustersTool()
        self.merge_clusters = MergeClustersTool()
        self.harmonize_labels = HarmonizeLabelsTool()

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return JSON result string."""
        try:
            if tool_name == "prepare_data":
                self.adata, stats = self.prepare_data.run(
                    params=tool_input, batch_key=self.batch_key,
                )
                # Update batch_key if the tool set one.
                if stats.get("batch_key"):
                    self.batch_key = stats["batch_key"]
                return json.dumps({
                    "status": "success",
                    "n_files": stats["n_files"],
                    "n_cells": stats["n_cells"],
                    "n_genes": stats["n_genes"],
                    "batch_key": stats.get("batch_key"),
                    "batches": stats.get("batches"),
                    "gene_overlap": stats.get("gene_overlap"),
                    "matrix_type": stats.get("matrix_type") or stats.get("matrix_types"),
                    "mixed_matrix_types": stats.get("mixed_matrix_types", False),
                    "per_file": stats.get("per_file"),
                    "elapsed_s": stats.get("elapsed_s"),
                })

            elif tool_name == "run_preprocessing":
                self.adata = self.preprocessing.run(
                    self.adata, params=tool_input, batch_key=self.batch_key,
                )
                log = self.adata.uns["preprocessing_log"]
                return json.dumps({
                    "status": "success",
                    "qc": log["qc"],
                    "pca": log.get("pca", {}),
                    "shape": f"{self.adata.n_obs} cells x {self.adata.n_vars} genes",
                })

            elif tool_name == "select_integration_genes":
                self.adata, stats = self.select_integration_genes.run(
                    self.adata, params=tool_input, batch_key=self.batch_key,
                )
                return json.dumps({
                    "status": "success",
                    "n_hvg": stats["n_hvg"],
                    "strategy": stats["strategy"],
                    "n_batches": stats["n_batches"],
                })

            elif tool_name == "run_integration":
                self.adata, stats = self.integration.run(
                    self.adata, params=tool_input, batch_key=self.batch_key,
                )
                return json.dumps({
                    "status": "success",
                    "method": stats["method"],
                    "output_key": stats["output_key"],
                    "n_dims": stats["n_dims"],
                    "elapsed_s": stats.get("elapsed_s"),
                })

            elif tool_name == "evaluate_integration":
                self.adata, stats = self.evaluate_integration.run(
                    self.adata, params=tool_input, batch_key=self.batch_key,
                )
                return json.dumps({
                    "status": "success",
                    "batch_scores": stats.get("batch_scores"),
                    "bio_scores": stats.get("bio_scores"),
                    "overall_score": stats.get("overall_score"),
                    "errors": stats.get("errors"),
                })

            elif tool_name == "run_clustering":
                self.adata = self.clustering.run(self.adata, params=tool_input)
                log = self.adata.uns["clustering_log"]
                return json.dumps({
                    "status": "success",
                    "cluster": log["cluster"],
                    "assessment": log.get("assessment", {}),
                })

            elif tool_name == "run_marker_genes":
                self.adata = self.marker_genes.run(self.adata, params=tool_input)
                log = self.adata.uns["marker_genes_log"]
                return json.dumps({
                    "status": "success",
                    "top_markers": log["top_markers"]["summary"],
                    "n_groups": log["top_markers"]["n_groups"],
                })

            elif tool_name == "score_markers":
                self.adata = self.score_markers.run(self.adata, params={
                    "marker_dict": tool_input["marker_dict"],
                })
                log = self.adata.uns["score_markers_log"]
                return json.dumps({
                    "status": "success",
                    "best_match": log["best_match"],
                    "score_matrix": log["score_matrix"],
                })

            elif tool_name == "check_confidence":
                self.adata = self.check_confidence.run(self.adata, params=tool_input)
                log = self.adata.uns["confidence_log"]
                return json.dumps({
                    "status": "success",
                    "per_cluster": log["per_cluster"],
                    "low_confidence_clusters": log["low_confidence_clusters"],
                })

            elif tool_name == "annotate_cell_types":
                self.adata = self.annotate_cell_types.run(self.adata, params={
                    "label_mapping": tool_input["label_mapping"],
                    "fine_label_mapping": tool_input.get("fine_label_mapping"),
                })
                log = self.adata.uns["annotate_cell_types_log"]
                return json.dumps({
                    "status": "success",
                    "assignment": log,
                })

            elif tool_name == "subcluster":
                cluster_id = str(tool_input["cluster_id"])
                adata_sub, stats = self.subcluster.run(
                    self.adata, params=tool_input,
                )
                # Store the subset for later annotate_subclusters call.
                self._stored_subclusters[cluster_id] = adata_sub
                return json.dumps({
                    "status": "success",
                    "original_cluster": stats["original_cluster"],
                    "n_cells": stats["n_cells"],
                    "n_subclusters": stats["n_subclusters"],
                    "subcluster_sizes": stats["subcluster_sizes"],
                    "top_markers": stats["top_markers"],
                    "warning": stats.get("warning"),
                })

            elif tool_name == "annotate_subclusters":
                cluster_id = str(tool_input["cluster_id"])
                if cluster_id not in self._stored_subclusters:
                    return json.dumps({
                        "status": "error",
                        "message": f"Cluster '{cluster_id}' was not subclustered. "
                                   f"Call subcluster first. "
                                   f"Available: {list(self._stored_subclusters.keys())}",
                    })
                adata_sub = self._stored_subclusters[cluster_id]
                self.adata, stats = self.annotate_subclusters.run(
                    self.adata, params=tool_input, adata_sub=adata_sub,
                )
                # Clean up stored subcluster.
                del self._stored_subclusters[cluster_id]
                return json.dumps({
                    "status": "success",
                    "original_cluster": stats["original_cluster"],
                    "cells_updated": stats["cells_updated"],
                    "label_counts": stats["label_counts"],
                    "unmapped_subclusters": stats.get("unmapped_subclusters"),
                })

            elif tool_name == "merge_clusters":
                self.adata, stats = self.merge_clusters.run(
                    self.adata, params=tool_input,
                )
                return json.dumps({
                    "status": "success",
                    "merged_clusters": stats["merged_clusters"],
                    "label": stats["label"],
                    "total_cells": stats["total_cells"],
                })

            elif tool_name == "harmonize_labels":
                self.adata, stats = self.harmonize_labels.run(
                    self.adata, params=tool_input, batch_key=self.batch_key,
                )
                return json.dumps({
                    "status": "success",
                    "broad": stats.get("broad", {}),
                    "fine": stats.get("fine"),
                    "per_batch_labels_after": stats.get("per_batch_labels_after"),
                })

            elif tool_name == "annotate_batch":
                batch_value = str(tool_input["batch_value"])
                resolution = tool_input.get("resolution", 0.8)
                use_rep = tool_input.get("use_rep", "X_pca")
                n_top_genes = tool_input.get("n_top_genes", 10)

                if not self.batch_key:
                    return json.dumps({
                        "status": "error",
                        "message": "No batch_key set. Cannot annotate per batch.",
                    })

                mask = self.adata.obs[self.batch_key].astype(str) == batch_value
                n_cells = int(mask.sum())
                if n_cells == 0:
                    available = sorted(self.adata.obs[self.batch_key].unique().astype(str))
                    return json.dumps({
                        "status": "error",
                        "message": f"Batch '{batch_value}' not found. Available: {available}",
                    })

                # Subset to batch.
                adata_batch = self.adata[mask].copy()

                # Cluster using ClusteringTool.
                adata_batch = self.clustering.run(adata_batch, params={
                    "use_rep": use_rep,
                    "resolution": resolution,
                    "n_neighbors": min(15, n_cells - 1),
                })

                # Find markers using GetMarkerGenesTool.
                adata_batch = self.marker_genes.run(adata_batch, params={
                    "method": "wilcoxon",
                    "n_top_genes": n_top_genes,
                    "min_log2fc": 0.25,
                })

                # Store for assign_batch_labels.
                self._stored_batches[batch_value] = adata_batch

                cluster_log = adata_batch.uns.get("clustering_log", {})
                marker_log = adata_batch.uns.get("marker_genes_log", {})

                return json.dumps({
                    "status": "success",
                    "batch_value": batch_value,
                    "n_cells": n_cells,
                    "n_clusters": cluster_log.get("cluster", {}).get("n_clusters", 0),
                    "cluster_sizes": cluster_log.get("cluster", {}).get("cluster_sizes", {}),
                    "top_markers": marker_log.get("top_markers", {}).get("summary", {}),
                    "silhouette": cluster_log.get("assessment", {}).get("silhouette_score"),
                })

            elif tool_name == "assign_batch_labels":
                batch_value = str(tool_input["batch_value"])
                label_mapping = tool_input["label_mapping"]
                fine_label_mapping = tool_input.get("fine_label_mapping")

                if batch_value not in self._stored_batches:
                    return json.dumps({
                        "status": "error",
                        "message": f"Batch '{batch_value}' was not annotated. "
                                   f"Call annotate_batch first. "
                                   f"Available: {list(self._stored_batches.keys())}",
                    })

                adata_batch = self._stored_batches[batch_value]

                # Ensure columns exist in main adata.
                if "cell_type" not in self.adata.obs.columns:
                    self.adata.obs["cell_type"] = "Unknown"
                if fine_label_mapping and "cell_type_fine" not in self.adata.obs.columns:
                    self.adata.obs["cell_type_fine"] = "Unknown"

                cells_updated = 0
                label_counts = {}

                for cluster_id, label in label_mapping.items():
                    cluster_mask = adata_batch.obs["leiden"].astype(str) == str(cluster_id)
                    cell_indices = adata_batch.obs.index[cluster_mask]
                    n = int(cluster_mask.sum())
                    if n == 0:
                        continue
                    self.adata.obs.loc[cell_indices, "cell_type"] = label
                    cells_updated += n
                    label_counts[label] = label_counts.get(label, 0) + n

                    if fine_label_mapping and str(cluster_id) in fine_label_mapping:
                        self.adata.obs.loc[cell_indices, "cell_type_fine"] = fine_label_mapping[str(cluster_id)]

                # Clean up.
                del self._stored_batches[batch_value]

                return json.dumps({
                    "status": "success",
                    "batch_value": batch_value,
                    "cells_updated": cells_updated,
                    "label_counts": label_counts,
                })

            else:
                return json.dumps({"status": "error", "message": f"Unknown tool: {tool_name}"})

        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})


def _convert_tools_to_openai_format(tools):
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def run_agent_loop(
    adata,
    batch_key=None,
    model="claude-sonnet-4-20250514",
    api_base=None,
    api_key=None,
):
    """
    Main agent loop: Claude decides, we execute, repeat until done.

    Supports both Anthropic native API and OpenAI-compatible endpoints.

    Parameters
    ----------
    api_base : optional API base URL. If it ends with /v1 or contains
        a non-Anthropic host, uses OpenAI-compatible client.
    api_key : optional API key. Falls back to env vars.
    """
    executor = PipelineExecutor(adata, batch_key=batch_key)

    # Detect which client to use.
    use_openai = api_base is not None and "/v1" in api_base

    if use_openai:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key=api_key or "dummy")
        logger.info(f"Using OpenAI-compatible endpoint: {api_base}")
    else:
        import anthropic
        kwargs = {}
        if api_base:
            kwargs["base_url"] = api_base
        if api_key:
            kwargs["api_key"] = api_key
        client = anthropic.Anthropic(**kwargs)
        logger.info("Using Anthropic native API")

    system_prompt = (
        "You are a single-cell genomics expert. You have access to tools for "
        "data loading, preprocessing, integration, clustering, marker gene "
        "identification, and cell type annotation of scRNA-seq data.\n\n"
        "Your workflow:\n\n"
        "--- DATA LOADING ---\n"
        "0. If data is not yet loaded: call prepare_data with input file paths.\n"
        "   - Supports h5ad, 10X h5, 10X MTX directory, CSV/TSV, loom.\n"
        "   - Auto-detects format, converts Ensembl IDs to gene symbols, "
        "strips species prefixes.\n"
        "   - Returns per-file obs_columns with dtypes and example values. "
        "Review these to understand what metadata each dataset has.\n"
        "   - If different datasets use different column names for the same "
        "concept (e.g. 'celltype' vs 'annotation', 'donor' vs 'patient_id'), "
        "call prepare_data again with obs_key_mapping to harmonize them: "
        "{'cell_type': ['celltype', 'annotation'], 'batch': ['donor', 'patient_id']}.\n"
        "   - For multiple files: validates that all have the same matrix type "
        "(rejects mixing raw counts with normalized data).\n"
        "   - Stores matrix_type in adata.uns for downstream tools.\n\n"
        "--- PREPROCESSING ---\n"
        "1. Call run_preprocessing. It auto-detects the data state from "
        "adata.uns['matrix_type'] and adapts:\n"
        "   - raw_counts: full pipeline (QC, doublets, normalize, log1p, HVG, PCA)\n"
        "   - normalized: skip normalization, still apply log1p + HVG + PCA\n"
        "   - log_normalized: skip normalization + log1p, do HVG + PCA\n"
        "   - scaled_or_residuals: skip QC + doublets + normalization + log1p, "
        "do HVG + PCA\n"
        "   Check detected_data_state and skipped_steps in the result.\n"
        "   If data IS already preprocessed (X_pca exists): skip this step.\n\n"
        "--- PER-BATCH ANNOTATION (multi-batch only) ---\n"
        "2. Annotate each batch independently to get coarse cell type labels "
        "for integration evaluation:\n"
        "   a. For each batch: call annotate_batch(batch_value=...) to cluster "
        "and find markers\n"
        "   b. Review the top markers. Use your knowledge of canonical markers "
        "to identify cell types (e.g., CD3D/CD3E = T cells, CD14/LYZ = Monocytes)\n"
        "   c. Call assign_batch_labels(batch_value=..., label_mapping={...}) "
        "to assign coarse labels. Keep labels broad (T cells, B cells, Monocytes, "
        "NK cells, etc.)\n"
        "   d. Repeat for ALL batches\n"
        "   e. Review all unique labels across batches. If naming differs "
        "(e.g., 'Mono' vs 'Monocytes'), call harmonize_labels with a mapping "
        "dict to standardize\n\n"
        "--- INTEGRATION (multi-batch only) ---\n"
        "3. Run run_integration (harmony or scanorama) to correct batch effects.\n"
        "   Optionally call evaluate_integration with label_key='cell_type' "
        "to check batch mixing and bio conservation.\n\n"
        "--- FINAL ANNOTATION ---\n"
        "4. Cluster the cells on the integrated embedding (X_harmony/X_scanorama "
        "for multi-batch, X_pca for single-batch)\n"
        "5. Find marker genes per cluster with run_marker_genes\n"
        "6. Build a marker_dict from canonical markers and call score_markers\n"
        "7. Call check_confidence to identify ambiguous clusters\n"
        "8. Call annotate_cell_types with label_mapping and fine_label_mapping\n"
        "9. If any cluster has confidence < 0.5: call subcluster, review "
        "sub-markers, then call annotate_subclusters\n"
        "10. If two clusters are the same cell type: call merge_clusters\n\n"
        "--- NOTES ---\n"
        "- For single-batch data: skip steps 2-3, go directly to step 4.\n"
        "- Adapt parameters based on dataset size and results.\n"
        "- If clustering quality is poor (low silhouette), try different resolution.\n"
        "- Use your knowledge of canonical marker genes for annotation.\n"
        "- Keep per-batch annotation coarse — final annotation (steps 4-10) "
        "produces the detailed labels."
    )

    has_batches = batch_key is not None
    is_preprocessed = "X_pca" in adata.obsm
    matrix_type = adata.uns.get("matrix_type", {}).get("likely_type", "unknown")

    user_message = (
        f"Please analyze this scRNA-seq dataset:\n"
        f"- Shape: {adata.n_obs} cells x {adata.n_vars} genes\n"
        f"- Matrix type: {matrix_type}\n"
        f"- Batch column: {batch_key or 'none (single batch)'}\n"
        f"- Obs columns: {list(adata.obs.columns)}\n"
        f"- Obsm keys: {list(adata.obsm.keys())}\n"
    )
    if has_batches:
        n_batches = len(adata.obs[batch_key].unique())
        batch_values = sorted(adata.obs[batch_key].unique().astype(str))
        user_message += f"- Batches ({n_batches}): {batch_values}\n"
    if is_preprocessed:
        user_message += (
            f"\nData is already preprocessed (PCA exists with "
            f"{adata.obsm['X_pca'].shape[1]} PCs). Skip preprocessing.\n"
        )

    user_message += (
        f"\nRun the full pipeline: "
        f"{'preprocess -> ' if not is_preprocessed else ''}"
        f"{'per-batch annotation -> harmonize -> integrate -> ' if has_batches else ''}"
        f"cluster -> markers -> annotate.\n"
        f"Choose appropriate parameters for this dataset size."
    )

    logger.info("Starting agent loop...")
    logger.info(f"Model: {model}")
    logger.info(f"Dataset: {adata.n_obs} cells x {adata.n_vars} genes\n")

    max_turns = 50

    if use_openai:
        # ── OpenAI-compatible endpoint ──
        openai_tools = _convert_tools_to_openai_format(TOOLS)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        for turn in range(max_turns):
            logger.info(f"--- Turn {turn + 1} ---")

            response = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                tools=openai_tools,
                messages=messages,
            )

            choice = response.choices[0]
            message = choice.message
            # Print text content.
            if message.content:
                logger.info(f"Claude: {message.content}")

            # Process tool calls.
            if message.tool_calls:
                # Bedrock rejects empty text content blocks, so ensure
                # the assistant message has non-empty content or None.
                if not message.content:
                    message.content = None
                messages.append(message)
                for tc in message.tool_calls:
                    tool_input = json.loads(tc.function.arguments)
                    logger.info(f"Tool call: {tc.function.name}({json.dumps(tool_input, indent=2)})")
                    result = executor.execute(tc.function.name, tool_input)
                    logger.info(f"Result: {result[:500]}...")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            elif choice.finish_reason == "stop":
                logger.info("\nAgent finished.")
                break
            else:
                if not message.content:
                    message.content = None
                messages.append(message)
        else:
            logger.info("\nMax turns reached.")

    else:
        # ── Anthropic native API ──
        messages = [{"role": "user", "content": user_message}]

        for turn in range(max_turns):
            logger.info(f"--- Turn {turn + 1} ---")

            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )

            assistant_content = response.content
            tool_results = []
            for block in assistant_content:
                if block.type == "text":
                    logger.info(f"Claude: {block.text}")
                elif block.type == "tool_use":
                    logger.info(f"Tool call: {block.name}({json.dumps(block.input, indent=2)})")
                    result = executor.execute(block.name, block.input)
                    logger.info(f"Result: {result[:500]}...")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "assistant", "content": assistant_content})

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            elif response.stop_reason == "end_turn":
                logger.info("\nAgent finished.")
                break
        else:
            logger.info("\nMax turns reached.")

    return executor.adata


def main():
    from sc_atlas_agentic_builder.utility import PrepareDataTool

    parser = argparse.ArgumentParser(description="Run sc-atlas-agentic-builder pipeline with Claude API")
    parser.add_argument("--input", type=str, nargs="+", default=['/SPXvePFS/share-users/ycgao/test/cell_atlas_agent/data/Merge_data_uncorrected.h5ad'],
                        help="Path(s) to .h5ad file(s). Single file = pre-merged, "
                             "multiple files = separate datasets to merge.")
    parser.add_argument("--batch-key", type=str, default='study', help="Batch column in obs")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514", help="Claude model")
    parser.add_argument("--api-base", type=str, default=None, help="API base URL (for OpenAI-compatible endpoints)")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--output", type=str, default="annotated.h5ad", help="Output path")
    args = parser.parse_args()

    # Load or create data.
    if args.input:
        adata, stats = PrepareDataTool().run(params={
            "input_paths": args.input,
            "batch_key": args.batch_key or "batch",
        })
        batch_key = stats.get("batch_key") or args.batch_key
        matrix_type = (stats.get("matrix_type") or {}).get("likely_type", "unknown")
        n_batches = stats.get("n_batches")
        batch_info = f", {n_batches} batches" if n_batches else ""
        logger.info(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes{batch_info}, matrix_type={matrix_type}")
    else:
        logger.info("No input file, creating demo dataset...")
        from sc_atlas_agentic_builder.demo.create_demo_data import create_demo_pbmc
        adata = create_demo_pbmc(n_cells=3000)
        batch_key = "sample_id"

    logger.info(f"Dataset: {adata.n_obs} cells x {adata.n_vars} genes\n")

    # Run the agent loop.
    adata = run_agent_loop(
        adata,
        batch_key=batch_key,
        model=args.model,
        api_base=args.api_base,
        api_key=args.api_key,
    )

    # Save result. Clean uns of non-serializable objects first.
    import copy
    save_adata = adata.copy()
    # Remove complex nested logs that h5py can't serialize.
    keys_to_remove = [k for k in save_adata.uns if k.endswith("_log")]
    for k in keys_to_remove:
        del save_adata.uns[k]
    # Save logs as JSON sidecar instead.
    logs = {}
    for k in keys_to_remove:
        if k in adata.uns:
            logs[k] = adata.uns[k]
    log_path = args.output.replace(".h5ad", "_logs.json")
    try:
        with open(log_path, "w") as f:
            json.dump(logs, f, indent=2, default=str)
        logger.info(f"\nLogs saved to {log_path}")
    except Exception as e:
        logger.warning(f"Could not save logs: {e}")

    save_adata.write_h5ad(args.output)
    logger.info(f"AnnData saved to {args.output}")

    if "cell_type" in adata.obs.columns:
        logger.info(f"Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
