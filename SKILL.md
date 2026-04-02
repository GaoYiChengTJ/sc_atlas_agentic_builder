---
name: sc-atlas-agentic-builder
description: Run the full single-cell RNA-seq cell atlas construction pipeline (preprocessing, per-batch annotation, integration, clustering, marker gene identification, and cell type annotation). Use when user asks to analyze scRNA-seq data, build a cell atlas, annotate cell types, preprocess single-cell data, or run the sc pipeline.
argument-hint: [path/to/data.h5ad] [--batch-key sample_id]
---

# CellAtlasAgent: Agentic Single-Cell Atlas Construction

End-to-end scRNA-seq analysis from raw counts to annotated cell atlas, driven by an LLM agent that makes all biological decisions (parameter selection, cell type identification, subclustering).

**User input**: `$ARGUMENTS`

## Pipeline Architecture

```
Input: files (.h5ad, 10X .h5, MTX dir, CSV/TSV, loom)
  |
  v
[PrepareDataTool] load, detect format, convert Ensembl→symbol,
  |               strip species prefixes, validate matrix type,
  |               harmonize obs columns, intersect/union genes,
  |               concatenate with batch labels
  |               → stores matrix_type in adata.uns
  v
[PreprocessingTool] auto-adapts to data state:
  |   raw_counts     → QC, doublets, normalize, log1p, HVG, PCA
  |   normalized     → QC, doublets, skip normalize, log1p, HVG, PCA
  |   log_normalized → QC, doublets, skip normalize+log1p, HVG, PCA
  |   scaled/resid.  → skip QC+doublets+normalize+log1p, HVG, PCA
  v
  (multi-batch only — per-batch annotation for integration eval)
[annotate_batch x N] per-batch clustering + coarse annotation
  |
  v
[assign_batch_labels x N] write coarse labels per batch
  |
  v
[HarmonizeLabelsTool] standardize label names across batches
  |
  v
[RunIntegrationTool] Harmony or Scanorama → X_harmony or X_scanorama
  |
  v
[EvaluateIntegrationTool] scib-metrics (batch mixing + bio conservation)
  |                        ← requires cell_type labels from per-batch annotation
  v
  (final annotation on integrated data)
[ClusteringTool] KNN → Leiden → UMAP → assessment
  |
  v
[GetMarkerGenesTool] differential expression → filter → top markers
  |
  v
[ScoreMarkersTool] score clusters against marker gene signatures
  |
  v
[CellTypistTool] (optional) reference-based annotation
  |
  v
[CheckConfidenceTool] per-cluster confidence scoring
  |
  v
[AnnotateCellTypesTool] assign cell type labels
  |
  v
[ReflectAnnotationTool] gather evidence for self-critique:
  |                      per-cluster markers vs labels,
  |                      cross-cluster marker overlaps,
  |                      cell proportions per label
  |                      → LLM reviews and decides: accept / revise / subcluster / merge
  |
  v  (if issues found — max 2 reflection rounds)
[AnnotateCellTypesTool] revise labels
  |
  v  (if low-confidence clusters)
[SubclusterTool] → [AnnotateSubclustersTool] iterative refinement
  |
  v  (if clusters share a type)
[MergeClustersTool] merge clusters with same label
  |
  v
Output: annotated.h5ad + annotated_logs.json
```

## Tools Reference

The pipeline is at `/SPXvePFS/share-users/ycgao/test/sc_atlas_agentic_builder/`.

### PrepareDataTool (`utility.prepare_data_tool`)
- Load and unify input datasets (h5ad, 10X h5, 10X MTX, CSV/TSV, loom)
- Auto-detects format, converts Ensembl IDs → gene symbols, strips species prefixes
- Validates matrix type (raw counts vs normalized vs log-normalized); rejects mixing raw + normalized
- For multiple files: intersects/unions genes, concatenates with batch labels
- Reports per-file obs columns with example values — review these and provide `obs_key_mapping` to harmonize column names
- Stores `matrix_type` in `adata.uns` for downstream tools
- Key params: `input_paths`, `batch_key`, `gene_id_type`, `gene_intersection_mode`, `obs_key_mapping`

### PreprocessingTool (`preprocessing.preprocessing_tool`)
- QC filtering, doublet removal, normalization, HVG selection, PCA
- **Auto-adapts to data state**: reads `adata.uns["matrix_type"]` and skips steps already applied
- Key params: `min_genes`, `max_genes`, `max_mt_percent`, `mt_prefix`, `normalization_method`, `n_top_genes`, `n_pcs`, `run_doublet_detection`, `doublet_method`
- Needs `batch_key` for per-batch doublet detection
- Check `detected_data_state` and `skipped_steps` in the log
- If data IS already preprocessed (X_pca exists): skip this step

### annotate_batch (implemented in `demo/run_with_claude.py` PipelineExecutor)
- Annotate a single batch independently before integration
- Subsets data to the specified batch, runs ClusteringTool + GetMarkerGenesTool
- Key params: `batch_value`, `resolution` (default 0.8), `use_rep` (default X_pca), `n_top_genes`
- Returns: n_clusters, cluster_sizes, top_markers, silhouette score
- After reviewing markers, call assign_batch_labels to write coarse labels
- **Purpose**: provides cell_type labels needed for bio conservation metrics in evaluate_integration

### assign_batch_labels (implemented in `demo/run_with_claude.py` PipelineExecutor)
- Assign cell type labels to a previously annotated batch
- Must call annotate_batch first
- Key params: `batch_value`, `label_mapping` (required), `fine_label_mapping` (optional)
- Labels are merged back into the main adata.obs["cell_type"]
- Keep labels **coarse** (T cells, B cells, Monocytes, NK cells, etc.)

### HarmonizeLabelsTool (`utility.harmonize_labels_tool`)
- Standardize label names across batches after per-batch annotation
- Key params: `mapping` ({old_label: new_label}), `fine_mapping`, `label_key`
- Use when batches have inconsistent naming (e.g., "Mono" vs "Monocytes", "T cells" vs "T_cell")

### SelectIntegrationGenesTool (`integration.select_genes_tool`) — optional
- Re-select HVGs with batch-aware strategies for integration
- Key params: `n_top_genes`, `strategy` (rank/intersection/union), `flavor`
- Skip if preprocessing already ran with batch_key (rank strategy applied)

### RunIntegrationTool (`integration.run_integration_tool`)
- Batch correction for multi-batch datasets
- Methods: `harmony` (fast, recommended default) or `scanorama` (MNN-based)
- Key params: `method`, `n_pcs`, `max_iter`
- **Requires** `batch_key`
- Produces new embedding: `X_harmony` or `X_scanorama` in obsm

### EvaluateIntegrationTool (`integration.evaluate_integration_tool`)
- Evaluate integration quality using scib-metrics
- Computes batch mixing (ASW_batch, graph connectivity) and bio conservation (ASW_label, NMI, ARI)
- Key params: `embed_key`, `label_key`
- **Provide `label_key="cell_type"`** to evaluate bio conservation (requires per-batch annotation labels)

### ClusteringTool (`clustering.clustering_tool`)
- Neighbors, Leiden clustering, UMAP, quality assessment
- Key params: `use_rep` (X_pca, X_harmony, X_scanorama), `n_neighbors`, `resolution`, `run_umap`, `run_assessment`
- Check `silhouette_score` in assessment — if <0.1, try different resolution

### GetMarkerGenesTool (`marker_genes.marker_genes_tool`)
- Differential expression per cluster (wilcoxon recommended)
- Key params: `method`, `n_top_genes`, `min_log2fc`, `groupby`
- Check `top_markers["summary"]` for canonical markers

### ScoreMarkersTool (`annotation.score_markers_tool`)
- Score clusters against known marker gene signatures
- Requires `marker_dict`: `{cell_type: [gene1, gene2, ...]}`
- Build marker_dict from your knowledge of canonical markers for the tissue type
- Returns `best_match` and `score_matrix` per cluster

### CellTypistTool (`annotation.celltypist_tool`) — optional
- Reference-based annotation, only for well-studied tissues with available models
- Key params: `model`, `majority_voting`, `auto_download`
- Skip for non-model organisms or unusual tissue types

### CheckConfidenceTool (`annotation.check_confidence_tool`)
- Evaluate annotation confidence per cluster
- Flags `low_confidence_clusters` that need subclustering
- Check `cross_method_agreement` if both markers and CellTypist were run

### AnnotateCellTypesTool (`annotation.annotate_cell_types_tool`)
- Write cell type labels decided by the agent
- Requires `label_mapping`: `{cluster_id: cell_type}`
- Optional `fine_label_mapping` for granular annotations
- Key params: `groupby`, `key_added`, `fine_key_added`

### SubclusterTool (`utility.subcluster_tool`)
- Subcluster a single cluster for fine-grained analysis
- Key params: `cluster_id`, `resolution`, `use_rep`, `n_neighbors`, `n_top_marker_genes`
- Returns subset adata with new leiden labels + marker genes
- Use for low-confidence clusters or large heterogeneous clusters

### AnnotateSubclustersTool (`utility.annotate_subclusters_tool`)
- Assign labels to subclusters after SubclusterTool
- Sub-labels are merged back into the main dataset
- Key params: `cluster_id`, `label_mapping`, `fine_label_mapping`

### ReflectAnnotationTool (`utility.reflect_annotation_tool`)
- Gather structured evidence for the agent to self-critique its annotations
- Returns per-cluster summary (label, cell count, top markers, marker scores, confidence)
- Returns per-label summary (total cells, proportion, which clusters share the label)
- Returns cross-cluster marker overlaps (shared top markers, Jaccard similarity)
- **No rules or flags** — raw evidence only, the LLM decides what to do
- Call AFTER annotate_cell_types, review output, then revise/accept (max 2 rounds)
- Key params: `label_key` (default 'cell_type'), `n_top_markers` (default 8)

### MergeClustersTool (`utility.merge_clusters_tool`)
- Assign same label to multiple clusters when marker analysis shows they are the same cell type
- Key params: `cluster_ids`, `label`, `fine_label`

## Workflow

### For single-batch data: skip steps 2-3, go directly from step 1 to step 4.

```
Step 0: Prepare data (PrepareDataTool)
  → Load file(s), auto-detect format
  → Check matrix_type: raw_counts? normalized? log_normalized?
  → Review per-file obs_columns — provide obs_key_mapping if needed
  → For multi-file: check gene overlap

Step 1: Preprocess (PreprocessingTool, auto-adapts to data state)
  → Adapt params to dataset size (small: lower min_genes; large: higher n_pcs)
  → Check detected_data_state and skipped_steps in the log
  → Check QC stats: how many cells filtered?
  → Skip if X_pca already exists

Step 2: Per-batch annotation (multi-batch only)
  → For each batch (or representative batches for large datasets):
    a. annotate_batch(batch_value=...) → cluster + markers
    b. Review top markers, identify coarse cell types
    c. assign_batch_labels(batch_value=..., label_mapping={...})
  → Keep labels COARSE (T cells, B cells, Monocytes, NK cells, etc.)
  → After all batches: harmonize_labels if naming differs across batches
  → PURPOSE: provides cell_type column for evaluate_integration bio metrics

Step 3: Integrate (multi-batch only)
  → RunIntegrationTool: harmony (fast) or scanorama (MNN-based)
  → EvaluateIntegrationTool with label_key="cell_type" to check:
    - batch mixing (ASW_batch, graph connectivity)
    - bio conservation (ASW_label, NMI, ARI)
  → Use integrated embedding (X_harmony/X_scanorama) for final clustering

Step 4: Cluster (ClusteringTool)
  → Start with resolution=0.8
  → Use use_rep="X_harmony" if integration was run, else "X_pca"
  → Check silhouette score and cluster sizes
  → If poor separation, adjust resolution

Step 5: Find marker genes (GetMarkerGenesTool)
  → Use wilcoxon, top 10-15 genes
  → Review top_markers["summary"] per cluster

Step 6: Score markers (ScoreMarkersTool)
  → Build marker_dict from canonical markers matching the top genes and tissue context
  → Review best_match and score_matrix

Step 7: CellTypist (optional)
  → Only for well-studied tissues with available models

Step 8: Check confidence (CheckConfidenceTool)
  → Review low_confidence_clusters
  → If confidence < 0.5: subcluster with SubclusterTool, then AnnotateSubclustersTool

Step 9: Annotate (AnnotateCellTypesTool)
  → Assign label_mapping based on all evidence
  → Use fine_label_mapping for granular types
  → Mark ambiguous clusters as "Unknown" for review

Step 10: Reflect (ReflectAnnotationTool)
  → Review per-cluster markers vs assigned labels
  → Check cross-cluster marker overlaps (should clusters be merged?)
  → Check cell proportions per label (anything surprising?)
  → Check for missing expected cell types
  → If issues found: revise with AnnotateCellTypesTool, then reflect again
  → Max 2 reflection rounds

Step 11: Refine (if needed)
  → If two clusters are the same cell type: MergeClustersTool
  → If a cluster has mixed markers: SubclusterTool → AnnotateSubclustersTool
```

## How to run the tools

Each step should be executed as a Python script via the Bash tool. Use **pickle** for intermediate saves to avoid h5ad serialization issues. Run related steps in the **same Python process** where possible.

### Imports

```python
import sys
sys.path.insert(0, "/SPXvePFS/share-users/ycgao/test")

from sc_atlas_agentic_builder.utility import (
    PrepareDataTool, SubclusterTool, AnnotateSubclustersTool,
    MergeClustersTool, HarmonizeLabelsTool, ReflectAnnotationTool,
)
from sc_atlas_agentic_builder.preprocessing import PreprocessingTool
from sc_atlas_agentic_builder.integration import (
    SelectIntegrationGenesTool, RunIntegrationTool, EvaluateIntegrationTool,
)
from sc_atlas_agentic_builder.clustering import ClusteringTool
from sc_atlas_agentic_builder.marker_genes import GetMarkerGenesTool
from sc_atlas_agentic_builder.annotation import (
    ScoreMarkersTool, CellTypistTool, CheckConfidenceTool, AnnotateCellTypesTool,
)
```

### Per-batch annotation pattern (Step 2)

For multi-batch datasets, annotate each batch independently before integration:

```python
clustering = ClusteringTool()
marker_genes = GetMarkerGenesTool()

# Initialize cell_type column
adata.obs["cell_type"] = "Unknown"

for batch_value in adata.obs[batch_key].unique():
    # Subset to batch
    mask = adata.obs[batch_key] == batch_value
    adata_batch = adata[mask].copy()
    n_cells = mask.sum()

    # Cluster the batch
    adata_batch = clustering.run(adata_batch, params={
        "use_rep": "X_pca",
        "resolution": 0.8,
        "n_neighbors": min(15, n_cells - 1),
    })

    # Find markers
    adata_batch = marker_genes.run(adata_batch, params={
        "method": "wilcoxon", "n_top_genes": 10, "min_log2fc": 0.25,
    })

    # Review markers and assign coarse labels
    summary = adata_batch.uns["marker_genes_log"]["top_markers"]["summary"]
    # ... agent decides label_mapping based on markers ...

    # Write labels back to main adata
    for cluster_id, label in label_mapping.items():
        cluster_mask = adata_batch.obs["leiden"].astype(str) == str(cluster_id)
        cell_indices = adata_batch.obs.index[cluster_mask]
        adata.obs.loc[cell_indices, "cell_type"] = label

# Harmonize labels if naming differs across batches
adata, stats = HarmonizeLabelsTool().run(adata, params={
    "mapping": {"Mono": "Monocytes", "T_cell": "T cells", ...},
}, batch_key=batch_key)
```

### Integration evaluation (Step 3)

```python
# Integrate
adata, stats = RunIntegrationTool().run(adata, params={
    "method": "harmony",
}, batch_key=batch_key)

# Evaluate — label_key requires per-batch annotation from Step 2
adata, eval_stats = EvaluateIntegrationTool().run(adata, params={
    "embed_key": "X_harmony",
    "label_key": "cell_type",  # from per-batch annotation
}, batch_key=batch_key)
# → batch_scores: ASW_batch, graph_connectivity
# → bio_scores: ASW_label, NMI, ARI
```

### Saving output

```python
import json, copy

save_adata = adata.copy()

# Remove non-serializable logs from uns
keys_to_remove = [k for k in save_adata.uns if k.endswith("_log")]
logs = {k: adata.uns[k] for k in keys_to_remove if k in adata.uns}
for k in keys_to_remove:
    del save_adata.uns[k]

# Save logs as JSON sidecar
with open("annotated_logs.json", "w") as f:
    json.dump(logs, f, indent=2, default=str)

# Save h5ad
save_adata.write_h5ad("annotated.h5ad")
```

## Implementation notes

1. **Use pickle for intermediate saves** — h5ad serialization fails on nested dicts with numpy arrays in `adata.uns`. Use `pickle.dump(adata, open('intermediate.pkl', 'wb'))`.
2. **Save logs as JSON sidecar** — remove all `*_log` keys from `adata.uns` before writing h5ad, save them separately as `annotated_logs.json`.
3. **Check disk space** — avoid `/tmp` if low on space; use the project filesystem.
4. **For large datasets (>100k cells)** — split steps across Bash calls using pickle for intermediate files.
5. **For many batches (>20)** — consider annotating representative batches only, or group similar batches for per-batch annotation.
6. **Cross-reference with existing metadata** — if the dataset has existing cell type labels (e.g. `celltype`, `celltype_article`), use `pd.crosstab(adata.obs['leiden'], adata.obs['celltype'])` to validate annotations.

## Key decision points for the agent

1. **Data loading** — review prepare_data results: check matrix_type, obs_columns per file, gene overlap. Provide obs_key_mapping to align column names across datasets.
2. **Preprocessing params** — auto-adapts to data state, but adapt QC params to dataset size (small: lower min_genes; large: higher n_pcs). Skip if X_pca exists.
3. **Per-batch annotation** — keep labels coarse (broad cell types). Purpose is to provide labels for integration evaluation, not final annotation.
4. **Integration** — use for multi-batch datasets. Harmony is fast and usually sufficient. Scanorama for datasets with minimal overlap. Skip for single-batch.
5. **Resolution** — affects number of clusters. Too low = merged types, too high = over-split. Start at 0.8, adjust based on silhouette and marker clarity.
6. **Marker dict** — use canonical markers from training knowledge matching observed top genes and tissue context.
7. **Low confidence** — subcluster ambiguous clusters with SubclusterTool, don't force labels. Mark as "Unknown" if unclear.
8. **CellTypist** — only use when a relevant pretrained model exists for the tissue type.

## Module structure

```
sc_atlas_agentic_builder/
  base.py                        # ToolWrapper abstract interface
  preprocessing/
    operations.py                #   filter_cells_and_genes, detect_doublets, normalize, select_hvg, run_pca
    preprocessing_tool.py        #   PreprocessingTool — reads adata.uns["matrix_type"]
  integration/
    operations.py                #   select_integration_genes, run_harmony, run_scanorama, evaluate_integration
    select_genes_tool.py         #   SelectIntegrationGenesTool (optional)
    run_integration_tool.py      #   RunIntegrationTool
    evaluate_integration_tool.py #   EvaluateIntegrationTool
  clustering/
    operations.py                #   compute_neighbors, run_leiden, compute_umap, assess_clusters
    clustering_tool.py           #   ClusteringTool
  marker_genes/
    operations.py                #   rank_marker_genes, filter_marker_genes, extract_top_markers
    marker_genes_tool.py         #   GetMarkerGenesTool
  annotation/
    operations.py                #   score_known_markers, run_celltypist, assign_labels, compute_annotation_confidence
    score_markers_tool.py        #   ScoreMarkersTool
    celltypist_tool.py           #   CellTypistTool (optional)
    check_confidence_tool.py     #   CheckConfidenceTool
    annotate_cell_types_tool.py  #   AnnotateCellTypesTool
  utility/
    prepare_data_tool.py         #   PrepareDataTool — load, validate, unify datasets
    operations.py                #   subcluster, annotate_subclusters, merge_clusters, harmonize_labels
    subcluster_tool.py           #   SubclusterTool
    annotate_subclusters_tool.py #   AnnotateSubclustersTool
    merge_clusters_tool.py       #   MergeClustersTool
    harmonize_labels_tool.py     #   HarmonizeLabelsTool
    reflect_annotation_tool.py   #   ReflectAnnotationTool — gather evidence for self-critique
  demo/
    run_with_claude.py           #   Agent loop, executor, tool definitions
    create_demo_data.py          #   Synthetic PBMC dataset generator
```
