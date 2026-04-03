# sc_atlas_agentic_builder

An LLM-agent-driven pipeline that autonomously builds single-cell RNA-seq cell atlases — from raw data to annotated cell types — with built-in self-reflection for annotation quality control.

## Why this project matters

**High-quality, harmonized single-cell data is the foundation of automated biological research.** As the field moves toward autonomous research systems ("auto-research") for omics, the bottleneck is no longer generating data — it's integrating and annotating it reliably across studies, labs, and sequencing platforms.

Single-cell datasets suffer from severe **batch effects** that hinder smooth data integration. Different studies use different protocols, different gene naming conventions, different metadata schemas, and produce data in different normalization states. Building a unified cell atlas from heterogeneous sources currently requires extensive manual intervention by expert bioinformaticians — a process that doesn't scale.

This project addresses that gap:

- **Autonomous decision-making**: An LLM agent handles the biological reasoning — parameter selection, marker gene interpretation, cell type identification, quality assessment — that traditionally requires human expertise.
- **Format-agnostic data ingestion**: Handles h5ad, 10X h5, 10X MTX, CSV/TSV, and loom formats with automatic gene name harmonization (Ensembl-to-symbol conversion, species prefix stripping, cross-dataset gene intersection).
- **Adaptive preprocessing**: Auto-detects whether input data contains raw counts, normalized values, or log-transformed data, and skips steps that have already been applied — preventing silent double-normalization.
- **Robust multi-dataset integration**: Per-batch annotation before integration provides cell type labels for bio-conservation evaluation, ensuring that batch correction removes technical noise without destroying biological signal.
- **Self-reflection**: After annotation, the agent reviews its own decisions against marker evidence, catches inconsistencies, and revises — mimicking the quality control that experienced annotators perform naturally.

By producing standardized, high-quality annotated atlases from heterogeneous inputs, this pipeline serves as the **data preparation layer** for downstream auto-research workflows — differential expression, trajectory inference, cell-cell communication, and beyond.

## Architecture

```
Input: raw files (.h5ad, 10X .h5, MTX dir, CSV/TSV, loom)
  |
  v
[PrepareDataTool] load, detect format, harmonize genes,
  |               validate matrix type, merge multi-file datasets
  v
[PreprocessingTool] auto-adapts to data state:
  |   raw_counts     -> QC, doublets, normalize, log1p, HVG, PCA
  |   normalized     -> QC, doublets, skip normalize, log1p, HVG, PCA
  |   log_normalized -> QC, doublets, skip normalize+log1p, HVG, PCA
  v
[Per-batch annotation] coarse cell type labels for integration eval
  |
  v
[Integration] Harmony or Scanorama batch correction
  |
  v
[Clustering] KNN -> Leiden -> UMAP -> quality assessment
  |
  v
[Marker genes] differential expression -> filter -> top markers
  |
  v
[Annotation] marker scoring + LLM-driven cell type assignment
  |
  v
[Reflection] self-critique: markers vs labels, overlaps, proportions
  |           -> revise if inconsistencies found (max 2 rounds)
  v
[Refinement] subcluster ambiguous clusters, merge duplicates
  |
  v
Output: annotated.h5ad + logs.json
```

## Quick start

### Installation

```bash
git clone https://github.com/GaoYiChengTJ/sc_atlas_agentic_builder
cd sc_atlas_agentic_builder
pip install -e ".[all]"
```

**Python >= 3.10 required.**

Or install dependencies selectively:

```bash
pip install -e .                    # core only
pip install -e ".[integration]"     # + Harmony, Scanorama
pip install -e ".[evaluation]"      # + scib-metrics
pip install -e ".[llm]"             # + OpenAI/Anthropic clients
pip install -e ".[all]"             # everything including optional
```

### Run with demo data

```bash
python -m demo.run_with_claude \
  --api-base <YOUR_API_BASE> \
  --api-key <YOUR_API_KEY>
```

Generates a synthetic 3,000-cell PBMC dataset and runs the full pipeline autonomously.

### Run with your data

```bash
# Single pre-merged file
python -m demo.run_with_claude \
  --input /path/to/data.h5ad \
  --batch-key sample_id

# Multiple separate files (auto-merged)
python -m demo.run_with_claude \
  --input sample1.h5ad sample2.h5ad sample3.h5ad \
  --batch-key batch
```

### Using as a Claude Code skill

After cloning, the skill is automatically available when you open the project in Claude Code (the `.claude/skills/` directory is included in the repository). You can also copy the skill to any other Claude Code project:

```bash
mkdir -p /path/to/your/project/.claude/skills/sc-atlas-agentic-builder
cp SKILL.md /path/to/your/project/.claude/skills/sc-atlas-agentic-builder/SKILL.md
```

Then ask Claude Code: *"analyze my scRNA-seq data at /path/to/data.h5ad"* and the `sc-atlas-agentic-builder` skill will drive the full pipeline.

## Key features

### Format-agnostic data loading

`PrepareDataTool` handles the messy reality of multi-source datasets:

- **Auto-detects** file format (h5ad, 10X h5, 10X MTX directory, CSV/TSV, loom)
- **Converts** Ensembl IDs to gene symbols using available `var` columns
- **Strips** species prefixes (GRCh38_, mm10_, etc.)
- **Validates** matrix state (raw counts vs normalized vs log-normalized)
- **Rejects** merging files with incompatible data states (e.g., raw + normalized)
- **Reports** per-file obs columns so the LLM can harmonize metadata naming

### Adaptive preprocessing

`PreprocessingTool` reads the detected matrix type and skips steps already applied:

| Detected state | QC | Doublets | Normalize | Log1p | HVG | PCA |
|---|---|---|---|---|---|---|
| raw_counts | yes | yes | yes | yes | yes | yes |
| normalized | yes | yes | skip | yes | yes | yes |
| log_normalized | yes | yes | skip | skip | yes | yes |
| scaled/residuals | skip | skip | skip | skip | yes | yes |

### Self-reflection

After annotation, `ReflectAnnotationTool` gathers structured evidence for the LLM to self-critique:

- **Per cluster**: assigned label, cell count, top marker genes with scores
- **Per label**: total cells, proportion, which clusters share the label
- **Cross-cluster**: marker gene overlaps with Jaccard similarity

The tool provides **raw evidence only** — no hardcoded rules or flags. The LLM reads the evidence and decides whether to accept, revise, subcluster, or merge.

### Modular tool design

Every analysis step is a standalone `ToolWrapper` subclass with a uniform interface:

```python
class MyTool(ToolWrapper):
    def run(self, adata, params: dict, batch_key=None):
        # process adata
        return adata, stats
```

Tools can be used independently, composed into custom pipelines, or driven by the LLM agent loop.

## Dependencies

### Required

| Package | Purpose |
|---------|---------|
| scanpy | Core single-cell analysis (QC, normalization, HVG, PCA, clustering, DE) |
| anndata | AnnData data structure + I/O |
| numpy | Numerical operations |
| scipy | Sparse matrices |
| pandas | Data manipulation |
| scikit-learn | Clustering quality metrics (silhouette score) |
| leidenalg | Leiden community detection |
| python-igraph | Graph backend for Leiden |

### Integration (at least one needed for multi-batch data)

| Package | Purpose |
|---------|---------|
| harmonypy | Harmony batch correction (recommended default) |
| scanorama | Scanorama MNN-based integration |

### Evaluation

| Package | Purpose |
|---------|---------|
| scib-metrics | Integration quality metrics (ASW, NMI, ARI, graph connectivity) |

### LLM API client (at least one)

| Package | Purpose |
|---------|---------|
| openai | OpenAI-compatible API endpoints |
| anthropic | Anthropic native API |

### Optional

| Package | Purpose |
|---------|---------|
| celltypist | Reference-based cell type annotation |
| doubletdetection | Alternative doublet detection method |

## Project structure

```
sc_atlas_agentic_builder/
  base.py                          # ToolWrapper abstract interface
  preprocessing/
    operations.py                  #   QC, doublets, normalize, HVG, PCA
    preprocessing_tool.py          #   PreprocessingTool (auto-adapts to data state)
    profiler.py                    #   Dataset profiling utilities
  integration/
    operations.py                  #   Harmony, Scanorama, evaluation
    select_genes_tool.py           #   SelectIntegrationGenesTool
    run_integration_tool.py        #   RunIntegrationTool
    evaluate_integration_tool.py   #   EvaluateIntegrationTool
  clustering/
    operations.py                  #   KNN, Leiden, UMAP, assessment
    clustering_tool.py             #   ClusteringTool
  marker_genes/
    operations.py                  #   DE ranking, filtering, extraction
    marker_genes_tool.py           #   GetMarkerGenesTool
  annotation/
    operations.py                  #   Marker scoring, CellTypist, labels
    score_markers_tool.py          #   ScoreMarkersTool
    celltypist_tool.py             #   CellTypistTool
    check_confidence_tool.py       #   CheckConfidenceTool
    annotate_cell_types_tool.py    #   AnnotateCellTypesTool
  utility/
    prepare_data_tool.py           #   PrepareDataTool (load, validate, merge)
    reflect_annotation_tool.py     #   ReflectAnnotationTool (self-critique)
    operations.py                  #   Subcluster, merge, harmonize
    subcluster_tool.py             #   SubclusterTool
    annotate_subclusters_tool.py   #   AnnotateSubclustersTool
    merge_clusters_tool.py         #   MergeClustersTool
    harmonize_labels_tool.py       #   HarmonizeLabelsTool
  demo/
    run_with_claude.py             #   Agent loop + tool executor
    create_demo_data.py            #   Synthetic PBMC dataset generator
```

## How it works

The agent loop is minimal — no framework needed:

```
while not done:
    response = llm.call(messages, tools=TOOLS)
    for tool_call in response.tool_calls:
        result = executor.execute(tool_call.name, tool_call.input)
        messages.append(result)
```

The LLM sees tool descriptions, decides which tool to call and with what parameters, interprets the results, and decides the next step. The system prompt provides the workflow structure; the LLM provides the biological reasoning.

## Citation

```bibtex
@misc{clawrxiv:2604.00550,
  title   = {sc-atlas-agentic-builder: Scalable, Self-Reflective Cell Atlas Construction for Autonomous Biological Research},
  author  = {Yicheng Gao (Tongji University) and Kejing Dong (Tongji University) and Yuheng Zhao (Fudan University) and Fabian J. Theis (Helmholtz Munich; Technical University of Munich) and sc-atlas-agent},
  year    = {2026},
  month   = {apr},
  note    = {clawRxiv preprint clawrxiv:2604.00550},
  url     = {https://clawrxiv.io/abs/2604.00550}
}
```

## License

MIT
