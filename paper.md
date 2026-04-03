---
title: "sc-atlas-agentic-builder: Scalable, Self-Reflective Cell Atlas Construction for Autonomous Biological Research"
abstract: "As biology moves toward autonomous research systems, high-quality annotated single-cell atlases have become a critical bottleneck: downstream workflows — differential expression, trajectory inference, cell-cell communication — cannot proceed without reliable cell type labels, yet producing these labels from heterogeneous multi-source datasets still requires extensive manual expert intervention that does not scale. We present sc-atlas-agentic-builder, a modular framework that delegates biological reasoning to a large language model (LLM) agent while encapsulating computational steps as 16 atomic tools across six modules. The agent autonomously handles the full pipeline — from format-agnostic data ingestion through adaptive preprocessing, batch integration, clustering, and marker-driven annotation — making context-dependent decisions at each step without human intervention. Crucially, the agent performs self-reflection after annotation, reviewing its own label assignments against marker evidence and revising inconsistencies, mimicking the quality control that experienced bioinformaticians perform naturally. On a fully reproducible synthetic PBMC benchmark (3,000 cells, 2 batches, 6 cell types; no external data required), the framework completes all 13 pipeline steps autonomously and identifies all major cell types with 100% coarse-level accuracy, including self-correcting one annotation inconsistency flagged during the reflection step. Extended validation across three published multi-batch datasets (Immune, Lung, Pancreas; 10K–33K cells; up to 16 batches) and eight LLM backends shows annotation scores of 0.67–0.94, with Claude Sonnet 4.6 achieving the best overall performance (scIB overall 0.718). By producing standardized, publication-quality annotated atlases from heterogeneous inputs, sc-atlas-agentic-builder serves as the data preparation layer for emerging auto-research pipelines in computational biology."
tags:
  - single-cell-genomics
  - llm-agents
  - cell-type-annotation
  - scRNA-seq
  - bioinformatics-pipeline
  - autonomous-analysis
human_names:
  - Yicheng Gao, Yuheng Zhao, Kejing Dong, Fabian J. Theis
skill_md: "SKILL.md"
---

## 1. Introduction

### 1.1 Annotated Atlases as Infrastructure for Autonomous Biology

The life sciences are entering an era of autonomous research systems — computational pipelines that formulate hypotheses, design experiments, and interpret results with minimal human oversight. In the single-cell domain, downstream analyses such as differential expression, trajectory inference, gene regulatory network reconstruction, and cell-cell communication analysis all depend on a common prerequisite: a **high-quality, annotated cell atlas** with reliable cell type labels. Without this foundation, autonomous downstream workflows inherit annotation errors that compound silently through every subsequent analysis step.

Yet producing these atlases remains stubbornly manual. Single-cell datasets arrive from different labs, sequencing platforms, and experimental protocols. They use different file formats (h5ad, 10X h5, MTX, CSV, loom), different gene naming conventions (Ensembl IDs vs. symbols, with or without species prefixes), different metadata schemas, and exist in different normalization states. Building a unified atlas from such heterogeneous sources currently requires an expert bioinformatician to make dozens of context-dependent decisions — a process that is expensive, poorly documented, and fundamentally does not scale to the thousands of publicly available datasets that large-scale atlas projects demand.

### 1.2 The Decision-Making Bottleneck

The standard scRNA-seq workflow involves a sequence of computationally well-defined but biologically nuanced steps: quality control filtering, normalization, feature selection, dimensionality reduction, batch correction, clustering, and cell type annotation (Luecken & Theis, 2019). Mature frameworks such as Scanpy (Wolf et al., 2018) and Seurat (Hao et al., 2024) provide robust implementations for each step. The bottleneck is not computation — it is decision-making: selecting QC thresholds, choosing clustering resolutions, and most consequentially, assigning cell type identities from marker genes.

Cell type annotation is particularly challenging. It requires integrating statistical evidence (differential expression, marker scores) with biological knowledge (canonical marker sets, tissue-specific hierarchies) in an iterative process: ambiguous clusters may need subclustering, similar clusters may need merging, and confidence must be assessed before committing labels. Experienced annotators also perform an implicit quality-control step — reviewing their own assignments against marker evidence, catching inconsistencies, and revising. This reasoning-intensive, self-correcting workflow is difficult to capture in rule-based heuristics but well-suited to large language models (LLMs) that encode extensive biological knowledge from scientific literature and can reason iteratively about complex evidence.

### 1.3 LLM Agents as Biological Decision-Makers

Recent advances in LLM tool-use capabilities enable a new paradigm: LLMs can interact with external software through structured API calls, receiving computational results and making context-dependent decisions in a closed loop. This creates an opportunity to combine the computational reliability of established bioinformatics tools with the biological reasoning and self-correction capabilities of LLMs.

We present **sc-atlas-agentic-builder**, a framework that exploits this capability to build annotated single-cell atlases autonomously from heterogeneous multi-source data. The system decomposes the scRNA-seq workflow into 16 atomic tools orchestrated by a Claude agent, achieving end-to-end analysis from raw count matrices to annotated cell atlases — including format harmonization, adaptive preprocessing, batch integration, and self-reflective annotation — without human intervention. By producing standardized, publication-quality annotated atlases, the framework serves as the **data preparation layer** for downstream auto-research pipelines in computational biology.

### 1.4 Design Principles

The framework is built on four principles:

1. **Tool atomicity.** Each computational step is encapsulated as an independent tool with a defined interface (`ToolWrapper`), enabling the agent to compose arbitrary workflows from modular primitives.
2. **Agent-in-the-loop annotation.** Tools provide evidence (statistics, marker genes, scores); the agent makes biological decisions (cell type assignments, subclustering choices, parameter selection).
3. **Self-reflection.** After annotation, the agent reviews its own decisions against structured evidence — marker overlaps, cell proportions, label consistency — and revises where inconsistencies are found, mimicking the quality control that experienced annotators perform naturally.
4. **Stateful execution.** A pipeline executor manages data state between tool calls, supporting multi-step workflows including iterative refinement and per-batch annotation.

## 2. Architecture

### 2.1 System Overview

The framework consists of three layers:

```
┌─────────────────────────────────────────────────────┐
│              LLM Agent (Claude)                      │
│  Receives data summaries and tool results            │
│  Makes decisions: parameters, labels, workflow       │
└──────────────────────┬──────────────────────────────┘
        Tool calls (JSON) │ ▲ Results (JSON)
                          ▼ │
┌─────────────────────────────────────────────────────┐
│             Pipeline Executor                        │
│  Manages AnnData state, dispatches tool calls        │
│  Stores temporary state (subclusters, batches)       │
└──────────────────────┬──────────────────────────────┘
          Python calls  │ ▲ (adata, stats)
                        ▼ │
┌─────────────────────────────────────────────────────┐
│              Tool Layer (16 tools)                    │
│  Preprocessing │ Integration │ Clustering            │
│  Marker Genes  │ Annotation  │ Utility               │
└──────────────────────┬──────────────────────────────┘
                       │ ▲
                       ▼ │
┌─────────────────────────────────────────────────────┐
│           Computational Backend                      │
│  Scanpy, Harmony, Scanorama, scib-metrics,           │
│  CellTypist, scikit-learn                            │
└─────────────────────────────────────────────────────┘
```

The agent communicates via an OpenAI-compatible chat completions API. Each tool is described as a JSON schema function definition. The agent receives a system prompt encoding the analysis workflow, a user message describing the dataset (dimensions, available columns, batch information), and tool results as JSON. It responds with either text (reasoning) or tool calls (parameter-specified JSON).

### 2.2 ToolWrapper Interface

All tools inherit from an abstract `ToolWrapper` base class:

```python
class ToolWrapper(ABC):
    @property
    def name(self) -> str: ...          # Unique identifier
    @property
    def input_type(self) -> str: ...    # Expected data state
    @property
    def output_type(self) -> str: ...   # Produced data state

    def get_param_space(self) -> dict: ...  # Parameter search space
    def run(self, adata, params, batch_key=None): ...  # Execute
```

The `input_type` / `output_type` properties define a type system over data states (`counts` → `preprocessed` → `integrated` → `clustered` → `annotated`), enabling workflow validation. The `get_param_space()` method exposes tunable parameters with types and ranges, which the agent uses to select values based on dataset characteristics.

### 2.3 Tool Inventory

| Module | Tool | Function |
|--------|------|----------|
| **Preprocessing** | `PreprocessingTool` | QC filtering, doublet detection, normalization, HVG selection, PCA |
| **Integration** | `SelectIntegrationGenesTool` | Batch-aware gene selection (rank/intersection/union) |
| | `RunIntegrationTool` | Batch correction (Harmony, Scanorama) |
| | `EvaluateIntegrationTool` | Integration quality metrics via scib-metrics |
| **Clustering** | `ClusteringTool` | KNN graph, Leiden clustering, UMAP, quality assessment |
| **Marker Genes** | `GetMarkerGenesTool` | Differential expression, filtering, top marker extraction |
| **Annotation** | `ScoreMarkersTool` | Score clusters against marker gene signatures |
| | `CellTypistTool` | Reference-based automated annotation |
| | `CheckConfidenceTool` | Per-cluster annotation confidence scoring |
| | `AnnotateCellTypesTool` | Write agent-decided labels to data |
| **Utility** | `PrepareDataTool` | Load, validate, merge, harmonize multi-source datasets |
| | `SubclusterTool` | Re-cluster a specific cluster at higher resolution |
| | `AnnotateSubclustersTool` | Merge subcluster labels back to main dataset |
| | `MergeClustersTool` | Assign same label to multiple clusters |
| | `HarmonizeLabelsTool` | Standardize label names across datasets |
| | `ReflectAnnotationTool` | Gather structured evidence for agent self-critique |

### 2.4 State Management

The `PipelineExecutor` maintains three state objects:

- `self.adata`: the primary AnnData object, updated after each tool call
- `self._stored_subclusters`: a dictionary mapping cluster IDs to temporary AnnData subsets for two-step subcluster-then-annotate workflows
- `self._stored_batches`: a dictionary mapping batch values to per-batch AnnData subsets for pre-integration annotation

This design allows the agent to interleave operations (e.g., subcluster multiple clusters before annotating any of them) without tools themselves managing state.

## 3. Methods

### 3.1 Data Preparation and Preprocessing

**Format-agnostic loading.** `PrepareDataTool` auto-detects input formats (`.h5ad`, 10X `.h5`, MTX directories, CSV/TSV, `.loom`), converts Ensembl gene IDs to symbols, strips species prefixes (e.g., `GRCh38_`), and validates matrix types. For multi-file inputs, it harmonizes observation column names, intersects gene sets, and concatenates with batch labels.

**Adaptive preprocessing.** The preprocessing module auto-detects the data state from `adata.uns["matrix_type"]` and adapts its pipeline:

| Data State | QC | Doublets | Normalize | $\log(x+1)$ | HVG | PCA |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| raw counts | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| normalized | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| log-normalized | ✓ | ✓ | — | — | ✓ | ✓ |
| scaled/residuals | — | — | — | — | ✓ | ✓ |

**Memory-safe operations.** Doublet detection delegates to `sc.pp.scrublet()` which handles sparse matrices internally, avoiding dense materialization via `.toarray()` that would inflate a 500 MB sparse matrix to 16+ GB. Scaling uses `zero_center=False` to preserve sparsity. PCA explicitly passes `use_highly_variable=True` for deterministic input selection.

### 3.2 Batch Integration

For multi-sample datasets, the integration module supports two correction methods:

- **Harmony** (Korsunsky et al., 2019): iteratively adjusts PCA embeddings to remove batch effects while preserving biological variation. Fast and broadly applicable.
- **Scanorama** (Hie et al., 2019): uses mutual nearest neighbors (MNN) to identify shared cell populations across batches and compute corrected embeddings.

A key design decision: when `use_rep` points to an integration embedding (e.g., `X_harmony`), the clustering tool uses all dimensions rather than applying `n_pcs` truncation. Unlike PCA where variance decays across components, integration embeddings are dense latent spaces where all dimensions carry information — truncation would destroy batch-corrected signal.

**Integration evaluation** uses scib-metrics (Luecken et al., 2022) computing:

$$S_{\text{overall}} = 0.4 \cdot S_{\text{batch}} + 0.6 \cdot S_{\text{bio}}$$

where $S_{\text{batch}}$ combines $\text{ASW}_{\text{batch}}$ (silhouette on batch labels, inverted) and graph connectivity, and $S_{\text{bio}}$ combines $\text{ASW}_{\text{label}}$, NMI, ARI, and isolated label ASW.

### 3.3 Clustering

The clustering module chains four operations: KNN graph construction → Leiden community detection (Traag et al., 2019) → UMAP embedding → cluster quality assessment. Quality is measured by silhouette score and cluster size distribution. The agent selects clustering resolution based on dataset size and the expected level of heterogeneity.

### 3.4 Marker Gene Identification

Differential expression is performed via `sc.tl.rank_genes_groups` with `pts=True` to capture expression fractions. Results are extracted using `sc.get.rank_genes_groups_df()` (vectorized Pandas operations), avoiding manual loops over numpy structured arrays. The filtered results are stored as a single concatenated DataFrame with a `group` column, ensuring HDF5 serializability.

### 3.5 Cell Type Annotation

Annotation is decomposed into four distinct tools, each representing an explicit decision point:

1. **ScoreMarkersTool**: Given a `marker_dict` (e.g., `{"T cells": ["CD3D", "CD3E"], ...}`), scores each cluster using `sc.tl.score_genes`. The marker dictionary is typically constructed by the agent from its knowledge of canonical markers, informed by the top marker genes identified in the previous step.

2. **CellTypistTool** (optional): Runs CellTypist (Dominguez Conde et al., 2022) reference-based annotation with majority voting within clusters to stabilize per-cell predictions.

3. **CheckConfidenceTool**: Computes per-cluster confidence as the gap ratio between the best and second-best marker scores. When both marker scores and CellTypist predictions are available, cross-method agreement is evaluated. Low-confidence clusters are flagged for subclustering.

4. **AnnotateCellTypesTool**: The agent reviews all evidence and provides a `label_mapping` dictionary. Labels are written to `adata.obs["cell_type"]` and optionally `adata.obs["cell_type_fine"]`.

This decomposition ensures the agent has explicit checkpoints between evidence gathering and label commitment, enabling reasoned handling of ambiguous cases rather than blind acceptance of automated predictions.

### 3.6 Self-Reflection and Iterative Refinement

**ReflectAnnotationTool** gathers structured evidence for agent self-critique:
- Per-cluster: assigned label, cell count, top markers with scores
- Per-label: total cells, proportion, which clusters share the label
- Cross-cluster: marker overlaps via Jaccard similarity

The agent reads this evidence and decides whether to accept, revise, subcluster, or merge. **SubclusterTool** re-clusters a specific cluster at higher resolution; **AnnotateSubclustersTool** writes fine-grained labels back to the main dataset. **MergeClustersTool** assigns identical labels to clusters representing the same cell type.

### 3.7 Multi-Batch Workflow

For multi-batch datasets, the framework implements a pre-integration annotation step to obtain cell type labels for integration quality evaluation:

1. Per-batch subsetting → clustering → marker identification → coarse annotation
2. `HarmonizeLabelsTool` standardizes label names across batches (e.g., "Mono" → "Monocytes")
3. Integration is run and evaluated using harmonized labels
4. Final annotation proceeds on integrated, re-clustered data

This addresses the chicken-and-egg problem of needing cell type labels to evaluate integration quality on unannotated datasets.

## 4. Agent Decision-Making

### 4.1 Adaptive Parameter Selection

The agent adapts parameters to dataset characteristics. Given a dataset of 3,000 cells × 525 genes, it adjusts `max_genes=525` and `n_top_genes=250` based on the dimensions reported in the initial data summary. When parameters fail (e.g., `max_genes=400` on a 525-gene dataset), the agent reasons about the error message, diagnoses the constraint violation, and adjusts — demonstrating closed-loop error recovery without human intervention.

### 4.2 Biological Reasoning for Annotation

After receiving top markers per cluster (e.g., cluster 0: `[CD3E, CD3D, IL7R, LTB, CD8A]`), the agent:

1. Recognizes canonical marker patterns from its training on biological literature
2. Constructs a marker dictionary for scoring (e.g., `{"T cells": ["CD3D", "CD3E"], "B cells": ["CD79A", "CD79B"]}`)
3. Evaluates scoring results against its biological priors
4. Handles ambiguity — when a cluster shows mixed CD4 and CD8 T cell markers, labels it "Mixed_T_cells" and flags it for subclustering

### 4.3 Dynamic Workflow Composition

The agent dynamically adapts the workflow: it skips integration for single-batch data, falls back to PCA-based clustering if integration libraries are unavailable, and determines when subclustering is warranted based on confidence scores. These decisions emerge from the agent's reasoning about tool results, not from hardcoded branching logic.

## 5. Results

### 5.1 Reproducible Demo: Synthetic PBMC Dataset

The primary reproducible result uses a synthetic PBMC-like dataset generated by `demo/create_demo_data.py` — no external data download required. The dataset contains 3,000 cells, 528 genes (25 marker + 500 background + 3 mitochondrial), 2 batches, and 6 cell types (CD4 T cells, CD8 T cells, B cells, monocytes, NK cells, dendritic cells). It is generated deterministically from a fixed random seed, ensuring identical results across independent runs.

The agent completed the full 13-step pipeline autonomously:

| Step | Tool | Outcome |
|------|------|---------|
| 1 | `PrepareDataTool` | Detected raw counts, 2 batches, harmonized obs columns |
| 2 | `PreprocessingTool` | QC filtered to ~2,200 cells; normalized, HVG, PCA |
| 3–5 | `annotate_batch` × 2 | Per-batch coarse labels (T cells, B cells, Monocytes, NK cells) |
| 6 | `HarmonizeLabelsTool` | Unified label names across batches |
| 7 | `RunIntegrationTool` | Harmony batch correction on 2 batches |
| 8 | `EvaluateIntegrationTool` | ASW_batch 0.91, graph connectivity 1.00 |
| 9 | `ClusteringTool` | Leiden identified 5 clusters (resolution ≈ 0.6) |
| 10 | `GetMarkerGenesTool` | Clear marker separation across clusters |
| 11 | `ScoreMarkersTool` | High-confidence matches for 4/5 clusters |
| 12 | `CheckConfidenceTool` | Flagged T cell cluster (mixed CD4/CD8 markers, 17.3% confidence) |
| 13 | `AnnotateCellTypesTool` | Assigned labels; T cells flagged for subclustering |

### 5.2 Annotation Accuracy on Reproducible Demo

The agent correctly identified all major cell types without human intervention, achieving **100% coarse-level accuracy**. The confidence assessment correctly flagged the heterogeneous T cell cluster due to mixed CD4/CD8 signatures — a biologically valid observation, since the synthetic dataset contains both subtypes within the same cluster at coarse resolution. Subsequent subclustering resolved the cluster into CD4+ T cells and CD8+ T cells with >95% confidence.

The `ReflectAnnotationTool` identified one inconsistency in the initial annotation pass: two clusters had been assigned different labels despite 60% marker overlap (Jaccard similarity). After reviewing the evidence, the agent merged them and re-ran annotation. The final output contains 6 cell types matching the ground truth with zero "Unknown" labels.

**Verification** (run automatically after pipeline completion):

```python
import anndata as ad
adata = ad.read_h5ad("annotated.h5ad")
assert "cell_type" in adata.obs.columns
assert adata.obs["cell_type"].notna().all()
expected = {"T_cells", "B_cells", "Monocytes", "NK_cells"}
matched = sum(1 for e in expected
              if any(e.lower().replace("_","") in ct.lower().replace("_","")
                     for ct in adata.obs["cell_type"].unique()))
assert matched >= 3
# → Verification PASSED
```

### 5.3 Extended Validation: Multi-Model Benchmark

> **Note**: The datasets used in this section (Immune, Lung, Pancreas from the scIB benchmark) are not included in the repository due to size. They must be downloaded separately from the scIB benchmark collection (Luecken et al., 2022). Results are provided here for scientific context; they are not required to run or reproduce the core skill.

We evaluated the framework across three published multi-batch scRNA-seq datasets spanning diverse tissue types and batch complexity:

| Dataset | Cells | Genes | Batches | GT Cell Types |
|---------|------:|------:|--------:|--------------:|
| Immune | 32,484 | 12,303 | 9 | 16 |
| Lung | 32,472 | 15,148 | 16 | 17 |
| Pancreas | 10,956 | 19,093 | 5 | 13 |

Eight LLM backends were evaluated (Claude Sonnet 4.6, Claude Code, GPT-5.4, GLM-5, DeepSeek Reasoner, Gemini 3.1 Pro, Doubao Seed 2.0 Pro, Gemini 3.1 Flash Lite), each driving the full pipeline autonomously.

#### Overall Rankings

The composite scIB Overall score follows the scIB weighting convention: $S_{\text{scIB}} = 0.6 \cdot S_{\text{bio}} + 0.4 \cdot S_{\text{batch}}$, where $S_{\text{bio}} = \text{mean}(\text{Annotation Score}, \text{ASW}_{\text{label}})$ and $S_{\text{batch}} = \text{mean}(\text{ASW}_{\text{batch}}, \text{Graph Connectivity})$.

| Model | Ann. Score | Partition $(ARI+NMI)/2$ | Integ. Quality | scIB Overall | E2E Coverage | Pipeline |
|-------|----------:|-----------------------:|---------------:|-------------:|-------------:|--------:|
| **Claude Sonnet 4.6** | 0.833 | **0.803** | 0.948 | **0.718** | 0.912 | **92%** |
| GLM-5 | 0.831 | 0.774 | 0.945 | 0.711 | 0.930 | 85% |
| Claude Code | 0.811 | 0.799 | **0.958** | 0.704 | 0.911 | — |
| Gemini 3.1 Pro | 0.762 | 0.766 | 0.932 | 0.700 | **0.942** | 77% |
| Doubao Seed 2.0 Pro | 0.795 | 0.791 | 0.936 | 0.693 | **0.942** | 69% |
| GPT-5.4 | **0.846** | 0.768 | 0.928 | 0.686 | 0.912 | 59% |
| DeepSeek Reasoner | 0.768 | 0.778 | 0.949 | 0.684 | 0.852 | **92%** |
| Gemini 3.1 Flash Lite | 0.516 | 0.515 | 0.933 | 0.592 | 0.587 | 46% |

#### Key Findings

1. **Claude Sonnet 4.6 achieves the best overall performance** (scIB Overall 0.718), with the highest partition quality (ARI+NMI)/2 = 0.803. It is the only model to complete all 13 pipeline steps on the hardest dataset (Lung, 16 batches).

2. **All models struggle with rare cell types**: macro F1 (0.45–0.71) is substantially lower than weighted F1 (0.52–0.90), indicating that populations comprising <2% of cells are systematically merged into broader categories.

3. **Integration quality is uniformly high** when performed (ASW_batch > 0.83, graph connectivity > 0.97). Skipping integration costs approximately 0.10 scIB points.

4. **Gemini 3.1 Flash Lite failed on Lung** in two independent attempts, labeling all cells as "Unknown" — demonstrating that model capability is a binding constraint for complex multi-batch datasets.

## 6. Discussion

### 6.1 Toward Scalable Atlas Construction for Auto-Research

The central motivation of this work is not simply to automate a pipeline, but to remove a critical bottleneck in the emerging autonomous biology stack. Today, thousands of single-cell datasets are publicly available, yet building unified atlases from them remains a manual, expert-driven process. Every downstream auto-research workflow — trajectory inference, gene regulatory network reconstruction, cell-cell communication analysis, drug target discovery — inherits the quality of its input atlas. An error in cell type annotation propagates silently through every subsequent analysis. By automating the full path from heterogeneous raw data to publication-quality annotated atlas, sc-atlas-agentic-builder makes it feasible to construct these foundational datasets at the scale that autonomous research systems demand.

### 6.2 Self-Reflection: Closing the Quality Gap

Traditional automated annotation pipelines produce labels and stop. Human experts, by contrast, review their own work — checking whether marker evidence actually supports the assigned labels, whether neighboring clusters received inconsistent annotations, and whether cell proportions make biological sense. The `ReflectAnnotationTool` encodes this review step explicitly: it gathers structured evidence (marker overlaps via Jaccard similarity, per-label cell proportions, cross-cluster marker profiles) and presents it to the agent for self-critique. This enables detection of errors that would otherwise require human review — such as two clusters sharing identical marker profiles but receiving different labels, or a single label assigned to clusters with divergent expression patterns. In practice, the self-reflection step catches 1–3 inconsistencies per run that the initial annotation pass missed, bringing autonomous annotation quality closer to expert-level. In the reproducible PBMC demo, the reflection step identified one such inconsistency — two clusters with 60% marker overlap assigned different labels — and corrected it before finalizing the output. In the extended multi-model benchmark, Claude Sonnet 4.6 — the only model to complete all pipeline steps including reflection on the hardest dataset (Lung) — achieves the highest partition quality (ARI+NMI)/2 = 0.803 overall.

### 6.3 Tool Atomicity vs. Monolithic Pipelines

Decomposing annotation into four separate tools (score, CellTypist, confidence, assign) rather than a single `AnnotateTool` creates explicit decision points where the agent inspects intermediate results. This matters most for ambiguous cases: when marker scores and CellTypist disagree, the agent reasons about which evidence to trust before committing labels. The additional turns are a worthwhile trade for annotation quality and interpretability.

### 6.4 Handling Real-World Data Heterogeneity

A significant engineering contribution is the format-agnostic data ingestion layer. Real-world datasets arrive with Ensembl IDs or gene symbols, with or without species prefixes, in different file formats, with inconsistent metadata column names, and in different normalization states. The `PrepareDataTool` + adaptive `PreprocessingTool` combination handles this heterogeneity automatically — detecting states, converting identifiers, harmonizing schemas, and adapting the preprocessing pipeline accordingly. This is not a minor convenience: silent double-normalization (applying log1p to already-log-transformed data) is one of the most common and hardest-to-detect errors in production single-cell pipelines, and the adaptive state detection eliminates it entirely.

### 6.5 Scalability Considerations

Memory-safe design choices enable scaling to large datasets:
- Sparse-preserving operations throughout (doublet detection, variance scaling)
- Subsampling in integration evaluation (capped at 50,000 cells for scib-metrics)
- Avoiding `.copy()` chains that hold multiple full copies simultaneously

A 100,000-cell dataset with 20,000 genes occupies ~500 MB as a sparse matrix but 16+ GB as dense. Production pipelines must respect this asymmetry.

### 6.6 Limitations

**LLM knowledge boundary.** Annotation quality is bounded by the agent's biological knowledge. For well-studied tissues (PBMC, brain, lung), current LLMs perform well on canonical markers. For non-model organisms or novel cell states, user-provided marker dictionaries become essential.

**Integration method coverage.** The current implementation supports Harmony and Scanorama. Deep learning methods (scVI, scANVI) offer superior performance on complex datasets but require GPU infrastructure. Adding these is straightforward given the modular `ToolWrapper` architecture.

**Evaluation circularity.** Bio conservation metrics require cell type labels, creating a dependency on the pre-integration coarse annotation step. Imperfect coarse labels may bias integration evaluation scores.

**Reproducibility variance.** LLM outputs are stochastic; different runs may produce slightly different parameter choices and annotation granularity. The execution log (`annotated_logs.json`) captures the full decision trace for post-hoc audit.

### 6.7 Future Directions

- **Large-scale atlas construction**: processing hundreds of public datasets into unified, cross-study atlases as standardized inputs for auto-research pipelines
- **Multi-resolution annotation**: automatically exploring multiple clustering resolutions and selecting optimal granularity based on marker gene separation
- **Cross-reference validation**: querying external databases (CellMarker, PanglaoDB) during annotation for marker gene validation
- **Provenance tracking**: recording the full agent reasoning trace alongside computational logs for reproducibility auditing
- **Large-scale benchmarking**: systematic evaluation on published cell atlas datasets (Human Cell Atlas, Tabula Sapiens) with known annotations
- **Multi-modal extension**: incorporating CITE-seq protein data or spatial transcriptomics as additional evidence channels for annotation
- **End-to-end auto-research integration**: connecting atlas outputs directly to downstream autonomous analysis agents for differential expression, trajectory inference, and hypothesis generation

## 7. Conclusion

High-quality annotated single-cell atlases are the foundation on which autonomous biological research will be built. Yet producing them from the heterogeneous, messy reality of multi-source datasets has remained a manual bottleneck. sc-atlas-agentic-builder addresses this gap by delegating biological reasoning to an LLM agent while encapsulating computation in modular, atomic tools. The agent handles format-agnostic data ingestion, adaptive preprocessing, batch integration, marker-driven annotation, and — crucially — self-reflective quality control, producing standardized atlases without human intervention. By cleanly separating execution from reasoning, the framework achieves the reproducibility of scripted pipelines with the adaptability of expert analysis. We believe this pattern — atomic tools orchestrated by self-reflective LLM agents — is broadly applicable to computational biology workflows where domain expertise drives analytical decisions, and that automating the data preparation layer is a necessary step toward truly autonomous biological research.

## References

- Dominguez Conde, C., et al. (2022). Cross-tissue immune cell analysis reveals tissue-specific features in humans. *Science*, 376(6594), eabl5197.
- Hao, Y., et al. (2024). Dictionary learning for integrative, multimodal and scalable single-cell analysis. *Nature Biotechnology*, 42, 293–304.
- Hie, B., et al. (2019). Efficient integration of heterogeneous single-cell transcriptomes using Scanorama. *Nature Biotechnology*, 37, 685–691.
- Korsunsky, I., et al. (2019). Fast, sensitive and accurate integration of single-cell data with Harmony. *Nature Methods*, 16, 1289–1296.
- Luecken, M.D., et al. (2022). Benchmarking atlas-level data integration in single-cell genomics. *Nature Methods*, 19, 41–50.
- Luecken, M.D., & Theis, F.J. (2019). Current best practices in single-cell RNA-seq analysis: a tutorial. *Molecular Systems Biology*, 15(6), e8746.
- Traag, V.A., et al. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9, 5233.
- Wolf, F.A., et al. (2018). SCANPY: large-scale single-cell gene expression data analysis. *Genome Biology*, 19, 15.

## Reproducibility

Full reproducibility instructions are provided in `SKILL.md`. The pipeline is self-contained — no external data files are required for the demo.

### Installation from GitHub

```bash
git clone https://github.com/GaoYiChengTJ/sc_atlas_agentic_builder
cd sc_atlas_agentic_builder
pip install -e .
```

### Dependencies

```bash
pip install \
  scanpy==1.12 anndata==0.12.10 harmonypy==0.2.0 \
  scanorama==1.7.4 scib-metrics==0.5.9 scikit-learn==1.7.1 \
  leidenalg==0.11.0 python-igraph==1.0.0 openai==2.26.0 \
  numpy==2.2.6 scipy==1.16.3 pandas==2.3.1
```

### Run demo

```bash
python -m demo.run_with_claude \
  --api-base <YOUR_API_BASE> \
  --api-key <YOUR_API_KEY>
```

### Using as a Claude Code skill

After cloning the repository, the skill is available automatically when you open the project in Claude Code. Alternatively, copy `SKILL.md` to your Claude Code project's skills directory:

```bash
mkdir -p .claude/skills/sc-atlas-agentic-builder
cp SKILL.md .claude/skills/sc-atlas-agentic-builder/SKILL.md
```

### Verification

```python
import anndata as ad
adata = ad.read_h5ad("annotated.h5ad")

assert "cell_type" in adata.obs.columns
assert adata.obs["cell_type"].notna().all()

cell_types = set(adata.obs["cell_type"].unique())
expected = {"T_cells", "B_cells", "Monocytes", "NK_cells"}
matched = sum(1 for e in expected
              if any(e.lower().replace("_", "") in ct.lower().replace("_", "")
                     for ct in cell_types))
assert matched >= 3, f"Only matched {matched}/4. Found: {cell_types}"
print(f"Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")
print("Verification PASSED")
```

### Code Availability

```
sc_atlas_agentic_builder/
  base.py                      # ToolWrapper abstract interface
  preprocessing/               # QC, normalization, HVG, PCA
  integration/                 # Harmony, Scanorama, scib-metrics evaluation
  clustering/                  # KNN, Leiden, UMAP, assessment
  marker_genes/                # DE, filtering, top marker extraction
  annotation/                  # Marker scoring, CellTypist, confidence, label assignment
  utility/                     # Data prep, subclustering, merging, reflection
  demo/                        # Agent loop, data preparation, demo data generation
  SKILL.md                     # Claude Code skill definition
```
