# sc-atlas-agentic-builder: Agentic Single-Cell Atlas Construction

## Overview

End-to-end scRNA-seq analysis from raw counts to annotated cell atlas, driven by an LLM agent (Claude) that makes all biological decisions (parameter selection, cell type identification, subclustering).

## Environment Setup

### Python Version

```
python >= 3.10
```

### Dependencies (pinned)

```
pip install \
  scanpy==1.12 \
  anndata==0.12.10 \
  harmonypy==0.2.0 \
  scanorama==1.7.4 \
  scib-metrics==0.5.9 \
  scikit-learn==1.7.1 \
  leidenalg==0.11.0 \
  python-igraph==1.0.0 \
  openai==2.26.0 \
  numpy==2.2.6 \
  scipy==1.16.3 \
  pandas==2.3.1
```

### API Configuration

The pipeline requires access to a Claude-compatible API endpoint. Set via CLI arguments:

```bash
--api-base <URL>       # OpenAI-compatible endpoint
--api-key <KEY>        # API key
--model <MODEL_NAME>   # e.g., claude-sonnet-4-20250514
```

## Demo Data Generation

The pipeline includes a built-in synthetic PBMC-like dataset generator (`demo/create_demo_data.py`). No external data files are required.

### Dataset Specification

- **3,000 cells**, **528 genes** (25 marker genes + 500 background + 3 mitochondrial)
- **2 batches**: `batch_0`, `batch_1`
- **6 cell types** with canonical markers:

| Cell Type | Fraction | Canonical Markers |
|-----------|----------|-------------------|
| CD4 T cells | 25% | CD3D, CD3E, IL7R, LTB |
| CD8 T cells | 15% | CD3D, CD3E, CD8A, CD8B, GZMB |
| B cells | 20% | MS4A1, CD79A, CD79B, CD19 |
| Monocytes | 20% | CD14, LYZ, S100A8, S100A9 |
| NK cells | 12% | NKG7, GNLY, PRF1, NCAM1 |
| Dendritic cells | 8% | FCER1A, CST3, CLEC10A |

### Generate Standalone

```bash
cd sc_atlas_agentic_builder
python -m demo.create_demo_data
# Output: demo_pbmc.h5ad (3000 cells x 528 genes)
```

## Run

### Option 1: Demo Data (no external files needed)

```bash
cd sc_atlas_agentic_builder
python -m demo.run_with_claude
```

This automatically generates the demo dataset and runs the full pipeline.

### Option 2: Custom Single File

```bash
python -m demo.run_with_claude \
  --input /path/to/merged_data.h5ad \
  --batch-key sample_id
```

### Option 3: Multiple Separate Files

```bash
python -m demo.run_with_claude \
  --input sample1.h5ad sample2.h5ad sample3.h5ad \
  --batch-key batch
```

For multiple files: `prepare_data` auto-detects format, converts Ensembl IDs to symbols, strips species prefixes, validates matrix types (rejects mixing raw + normalized), harmonizes obs column names (via LLM-provided mapping), intersects genes, and concatenates. Then `run_preprocessing` auto-detects the data state and adapts its pipeline accordingly.

## Expected Output

### Output Files

| File | Description |
|------|-------------|
| `annotated.h5ad` | Annotated AnnData with cell type labels |
| `annotated_logs.json` | Full execution log (all tool calls and results) |

### Expected Console Output (Demo Data)

```
No input file, creating demo dataset...
Dataset: 3000 cells x 528 genes

Starting agent loop...
Model: claude-sonnet-4-20250514

--- Turn 1 ---
Claude: [runs preprocessing]
Result: {"status": "success", "qc": {"n_cells_before": 3000, "n_cells_after": ~2200}, ...}

--- Turn 2 ---
Claude: [runs integration — harmony]
Result: {"status": "success", "method": "harmony", "embed_key": "X_harmony", ...}

--- Turn 3 ---
Claude: [runs clustering on X_harmony]
Result: {"status": "success", "cluster": {"n_clusters": 4-6, "silhouette_score": ~0.05-0.15}, ...}

--- Turn 4 ---
Claude: [runs marker genes]
Result: {"status": "success", "top_markers": {"0": ["CD3E", "CD3D", ...], "1": ["CD79B", ...], ...}}

--- Turn 5 ---
Claude: [scores markers with canonical marker dict]
Result: {"status": "success", "best_match": {"0": "T_cells", "1": "B_cells", ...}}

--- Turn 6 ---
Claude: [checks confidence]
Result: {"status": "success", "per_cluster": {...}, "low_confidence_clusters": [...]}

--- Turn 7 ---
Claude: [assigns cell type labels]
Result: {"status": "success", "assignment": {"n_labels": 4-6, "label_counts": {...}}}

Agent finished.
Logs saved to annotated_logs.json
AnnData saved to annotated.h5ad
Cell types: {<cell_type>: <count>, ...}
```

### Expected Results (Demo Data)

The agent should complete in **7-15 turns** and identify the following major cell types:

| Expected Cell Type | Markers Used | Approximate Cell Count |
|-------------------|--------------|----------------------|
| T cells (or CD4/CD8 split) | CD3D, CD3E, IL7R, CD8A, CD8B | ~800-1200 |
| B cells | CD79A, CD79B, MS4A1, CD19 | ~400-600 |
| Monocytes | CD14, LYZ, S100A8, S100A9 | ~400-600 |
| NK cells | NKG7, GNLY, PRF1, NCAM1 | ~200-350 |

Notes:
- Exact cluster counts and cell numbers vary by run due to QC filtering and LLM parameter choices
- The T cell cluster may appear as one mixed cluster or be split into CD4/CD8 depending on resolution
- Dendritic cells (8% of input) may merge with monocytes or form a separate small cluster
- The agent may subcluster low-confidence clusters, adding 2-4 extra turns

### Verification Checklist

After the pipeline completes, verify:

1. **Output file exists**: `annotated.h5ad` is written
2. **Cell type labels assigned**: `adata.obs["cell_type"]` column exists with non-null values
3. **Major types identified**: At least 3 of {T cells, B cells, Monocytes, NK cells} are present
4. **Pipeline log complete**: `annotated_logs.json` contains entries for preprocessing, clustering, marker genes, and annotation steps
5. **No crashes**: Pipeline exits cleanly with "Agent finished." message

```python
# Programmatic verification
import anndata as ad
import json

adata = ad.read_h5ad("annotated.h5ad")
logs = json.load(open("annotated_logs.json"))

assert "cell_type" in adata.obs.columns, "Missing cell_type column"
assert adata.obs["cell_type"].notna().all(), "Some cells lack labels"

cell_types = set(adata.obs["cell_type"].unique())
expected = {"T_cells", "B_cells", "Monocytes", "NK_cells"}
# Allow flexible naming — check substring matches
matched = sum(1 for e in expected if any(e.lower().replace("_", "") in ct.lower().replace("_", "") for ct in cell_types))
assert matched >= 3, f"Only matched {matched}/4 expected types. Found: {cell_types}"

print(f"Cells: {adata.n_obs}")
print(f"Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")
print("Verification PASSED")
```

## Pipeline Architecture

```
Input: files (.h5ad, 10X .h5, MTX dir, CSV/TSV, loom)
  |
  v
[PrepareDataTool] load, detect format, convert Ensembl→symbol,
  |               strip species prefixes, validate matrix type,
  |               harmonize obs columns (via LLM-provided mapping),
  |               intersect/union genes, concatenate with batch labels
  |               → stores matrix_type in adata.uns
  v
[PreprocessingTool] auto-adapts to data state:
  |   raw_counts     → QC, doublets, normalize, log1p, HVG, PCA
  |   normalized     → QC, doublets, skip normalize, log1p, HVG, PCA
  |   log_normalized → QC, doublets, skip normalize+log1p, HVG, PCA
  |   scaled/resid.  → skip QC+doublets+normalize+log1p, HVG, PCA
  v
  (multi-batch only)
[annotate_batch x N] per-batch clustering + coarse annotation
  |
  v
[HarmonizeLabelsTool] standardize label names across batches
  |
  v
[RunIntegrationTool] Harmony or Scanorama
  |
  v
[EvaluateIntegrationTool] scib-metrics (batch mixing + bio conservation)
  |
  v
[ClusteringTool] KNN -> Leiden -> UMAP -> assessment
  |
  v
[GetMarkerGenesTool] differential expression -> filter -> top markers
  |
  v
[ScoreMarkersTool] score clusters against marker gene signatures
  |
  v
[CheckConfidenceTool] per-cluster confidence scoring
  |
  v
[AnnotateCellTypesTool] LLM assigns cell type labels
  |
  v
[ReflectAnnotationTool] gather evidence for self-critique:
  |                      per-cluster markers vs labels,
  |                      cross-cluster marker overlaps,
  |                      cell proportions per label
  |                      → LLM reviews: accept / revise / subcluster / merge
  |
  v  (if low-confidence clusters)
[SubclusterTool] -> [AnnotateSubclustersTool] iterative refinement
  |
  v
Output: annotated.h5ad + annotated_logs.json
```

## Module Structure

```
sc_atlas_agentic_builder/
  base.py                      # ToolWrapper abstract interface
  preprocessing/               # QC, normalization, HVG, PCA (auto-adapts to data state)
    operations.py              #   filter_cells_and_genes, detect_doublets, normalize, select_hvg, run_pca
    preprocessing_tool.py      #   PreprocessingTool(ToolWrapper) — reads adata.uns["matrix_type"]
  integration/                 # Batch correction + evaluation
    operations.py              #   select_integration_genes, run_harmony, run_scanorama, evaluate_integration
    select_genes_tool.py       #   SelectIntegrationGenesTool (optional)
    run_integration_tool.py    #   RunIntegrationTool
    evaluate_integration_tool.py # EvaluateIntegrationTool
  clustering/                  # Community detection + visualization
    operations.py              #   compute_neighbors, run_leiden, compute_umap, assess_clusters
    clustering_tool.py         #   ClusteringTool
  marker_genes/                # Differential expression
    operations.py              #   rank_marker_genes, filter_marker_genes, extract_top_markers
    marker_genes_tool.py       #   GetMarkerGenesTool
  annotation/                  # Cell type assignment
    operations.py              #   score_known_markers, run_celltypist, assign_labels, compute_annotation_confidence
    score_markers_tool.py      #   ScoreMarkersTool
    celltypist_tool.py         #   CellTypistTool (optional)
    check_confidence_tool.py   #   CheckConfidenceTool
    annotate_cell_types_tool.py # AnnotateCellTypesTool
  utility/                     # Data preparation + refinement operations
    prepare_data_tool.py       #   PrepareDataTool — load, validate, unify datasets
    operations.py              #   subcluster, annotate_subclusters, merge_clusters, harmonize_labels
    subcluster_tool.py         #   SubclusterTool
    annotate_subclusters_tool.py # AnnotateSubclustersTool
    merge_clusters_tool.py     #   MergeClustersTool
    harmonize_labels_tool.py   #   HarmonizeLabelsTool
    reflect_annotation_tool.py #   ReflectAnnotationTool — evidence for self-critique
  demo/                        # Runner + test data
    run_with_claude.py         #   Agent loop, executor, tool definitions
    create_demo_data.py        #   Synthetic PBMC dataset generator
```
