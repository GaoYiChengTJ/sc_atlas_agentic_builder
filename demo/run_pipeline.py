"""
Demo: run the full cell atlas pipeline on synthetic data.

No agent framework needed — just plain Python.

Usage:
    python -m demo.run_pipeline

Three modes:
    1. Script mode (this file) — you make decisions, tools execute.
    2. Claude API mode — Claude makes decisions via tool_use.
    3. Interactive mode — step by step in a notebook.
"""

import json
import logging
import sys
import os

_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    from sc_atlas_agentic_builder.demo.create_demo_data import create_demo_pbmc
    from sc_atlas_agentic_builder.utility import PrepareDataTool
    from sc_atlas_agentic_builder.preprocessing import PreprocessingTool
    from sc_atlas_agentic_builder.clustering import ClusteringTool
    from sc_atlas_agentic_builder.marker_genes import GetMarkerGenesTool
    from sc_atlas_agentic_builder.annotation import AnnotationTool

    # ── Create demo data ──
    logger.info("=" * 60)
    logger.info("STEP 0: Create demo dataset")
    logger.info("=" * 60)
    adata = create_demo_pbmc(n_cells=3000)
    logger.info(f"  {adata.n_obs} cells x {adata.n_vars} genes")
    logger.info(f"  True labels: {adata.obs['true_label'].value_counts().to_dict()}")

    # ── Step 0.5: Prepare data ──
    # In a real workflow, prepare_data loads from files. Here we save and
    # reload to exercise the full path including matrix type detection.
    import tempfile, os
    tmpfile = os.path.join(tempfile.mkdtemp(), "demo.h5ad")
    adata.write_h5ad(tmpfile)
    adata, prep_stats = PrepareDataTool().run(params={
        "input_paths": [tmpfile],
        "batch_key": "sample_id",
    })
    logger.info(f"  Matrix type: {prep_stats['matrix_type']['likely_type']}")
    os.unlink(tmpfile)

    # ── Step 1: Preprocess ──
    # PreprocessingTool auto-detects the data state from adata.uns["matrix_type"]
    # and skips steps that have already been applied.
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Preprocessing")
    logger.info("=" * 60)
    adata = PreprocessingTool().run(adata, params={
        "min_genes": 5,
        "max_genes": 5000,
        "min_counts": 5,
        "max_mt_percent": 30.0,
        "min_cells_per_gene": 1,
        "normalization_method": "total",
        "n_top_genes": 300,
        "n_pcs": 20,
    }, batch_key="sample_id")

    pp_log = adata.uns["preprocessing_log"]
    logger.info(f"  Data state: {pp_log['detected_data_state']}")
    logger.info(f"  Skipped steps: {pp_log['skipped_steps'] or 'none'}")
    qc = pp_log["qc"]
    if not qc.get("skipped"):
        logger.info(f"  QC: {qc['n_cells_before']} -> {qc['n_cells_after']} cells")
    logger.info(f"  PCA: {pp_log['pca']['variance_explained_pct']}% variance")

    # ── Step 2: Cluster ──
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Clustering")
    logger.info("=" * 60)
    adata = ClusteringTool().run(adata, params={
        "use_rep": "X_pca",
        "n_neighbors": 15,
        "resolution": 0.8,
        "run_umap": True,
        "run_assessment": True,
    })

    cl = adata.uns["clustering_log"]["cluster"]
    logger.info(f"  {cl['n_clusters']} clusters found")
    logger.info(f"  Cluster sizes: {cl['cluster_sizes']}")

    assess = adata.uns["clustering_log"].get("assessment", {})
    if "silhouette_score" in assess:
        logger.info(f"  Silhouette score: {assess['silhouette_score']}")

    # ── Step 3: Marker genes ──
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Marker gene identification")
    logger.info("=" * 60)
    adata = GetMarkerGenesTool().run(adata, params={
        "method": "wilcoxon",
        "n_top_genes": 10,
        "min_log2fc": 0.25,
    })

    top_markers = adata.uns["marker_genes_log"]["top_markers"]["summary"]
    logger.info("  Top markers per cluster:")
    for cluster, genes in top_markers.items():
        logger.info(f"    Cluster {cluster}: {', '.join(genes[:5])}")

    # ── Step 4a: Annotation — score markers ──
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4a: Annotation — gather evidence")
    logger.info("=" * 60)

    # This is what the LLM agent would construct from its knowledge.
    marker_dict = {
        "T cells":     ["CD3D", "CD3E", "IL7R", "LTB"],
        "CD8 T cells": ["CD8A", "CD8B", "GZMB", "CD3D"],
        "B cells":     ["MS4A1", "CD79A", "CD79B", "CD19"],
        "Monocytes":   ["CD14", "LYZ", "S100A8", "S100A9"],
        "NK cells":    ["NKG7", "GNLY", "PRF1", "NCAM1"],
        "DC":          ["FCER1A", "CST3", "CLEC10A"],
    }

    adata = AnnotationTool().run(adata, params={
        "marker_dict": marker_dict,
    })

    ann_log = adata.uns["annotation_log"]
    logger.info("  Best match per cluster:")
    for cluster, ct in ann_log["marker_scores"]["best_match"].items():
        logger.info(f"    Cluster {cluster} -> {ct}")

    confidence = ann_log["confidence"]
    if confidence["low_confidence_clusters"]:
        logger.info(f"  Low confidence clusters: {confidence['low_confidence_clusters']}")
    else:
        logger.info("  All clusters have good confidence!")

    # ── Step 4b: Annotation — assign labels ──
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4b: Annotation — assign labels")
    logger.info("=" * 60)

    # Agent decides labels based on evidence.
    best_match = ann_log["marker_scores"]["best_match"]
    label_mapping = {cluster: ct for cluster, ct in best_match.items()}

    adata = AnnotationTool().run(adata, params={
        "label_mapping": label_mapping,
    })

    assignment = adata.uns["annotation_log"]["assignment"]
    logger.info(f"  Assigned {assignment['n_labels']} cell types:")
    for label, count in assignment["label_counts"].items():
        logger.info(f"    {label}: {count} cells")

    # ── Evaluate against ground truth ──
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION: Compare to ground truth")
    logger.info("=" * 60)

    # Simple accuracy check.
    if "true_label" in adata.obs.columns:
        # Map true labels to the same naming convention.
        true_to_pred = {
            "CD4_T": "T cells",
            "CD8_T": "CD8 T cells",
            "B_cell": "B cells",
            "Monocyte": "Monocytes",
            "NK": "NK cells",
            "DC": "DC",
        }
        adata.obs["true_mapped"] = adata.obs["true_label"].map(true_to_pred)
        matches = (adata.obs["cell_type"] == adata.obs["true_mapped"])
        accuracy = matches.mean()
        logger.info(f"  Overall accuracy: {accuracy:.1%}")

        # Per-type accuracy.
        for true_type in sorted(adata.obs["true_mapped"].unique()):
            mask = adata.obs["true_mapped"] == true_type
            type_acc = matches[mask].mean()
            logger.info(f"    {true_type}: {type_acc:.1%}")

    logger.info("\n" + "=" * 60)
    logger.info("DONE!")
    logger.info("=" * 60)
    logger.info(f"  Final: {adata.n_obs} cells, {adata.obs['cell_type'].nunique()} cell types")
    logger.info(f"  Stored in adata.obs['cell_type']")

    return adata


if __name__ == "__main__":
    adata = main()
