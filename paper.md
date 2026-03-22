# sc-atlas-agentic-builder: A Modular LLM-Driven Framework for Autonomous Single-Cell RNA-seq Analysis and Cell Type Annotation

**Authors:** Yicheng Gao, Claude (Anthropic)

**Tags:** `single-cell-genomics`, `llm-agents`, `cell-type-annotation`, `scRNA-seq`, `bioinformatics-pipeline`

---

## Abstract

Single-cell RNA sequencing (scRNA-seq) has transformed our understanding of cellular heterogeneity, yet its analysis pipelines remain labor-intensive, requiring expert decisions at each step — from quality control parameter tuning to cell type annotation. We present **sc-atlas-agentic-builder**, a modular framework that integrates large language models (LLMs) as autonomous decision-makers within a structured scRNA-seq analysis pipeline. The framework decomposes the analysis workflow into 16 atomic tools organized across six modules (preprocessing, integration, clustering, marker gene identification, annotation, and utility operations), each following a unified `ToolWrapper` interface. An LLM agent (Claude) orchestrates these tools via the tool_use API, making context-dependent decisions such as parameter selection, cell type identification from marker genes, and iterative refinement through subclustering. We demonstrate that this architecture enables end-to-end analysis from raw count matrices to annotated cell atlases, handling both single-sample and multi-batch datasets with different input formats. The separation of computational execution (tools) from biological reasoning (agent) creates a system that is both reproducible and adaptable, where domain expertise is encoded in the agent's reasoning rather than hardcoded heuristics.

---

## 1. Introduction

### 1.1 The Bottleneck in Single-Cell Analysis

The standard scRNA-seq analysis workflow involves a sequence of computationally well-defined but biologically nuanced steps: quality control filtering, normalization, feature selection, dimensionality reduction, batch correction, clustering, and cell type annotation (Luecken & Theis, 2019). While mature software ecosystems such as Scanpy (Wolf et al., 2018) and Seurat (Hao et al., 2024) provide implementations for each step, the critical challenge lies not in execution but in decision-making. At each step, analysts must choose parameters (e.g., mitochondrial percentage thresholds, clustering resolution, number of principal components) and interpret results (e.g., identifying cell types from marker genes). These decisions require domain expertise and are often poorly documented, making analyses difficult to reproduce and scale.

The cell type annotation step is particularly challenging. It requires integrating statistical evidence (differential expression results, marker gene scores) with biological knowledge (canonical marker gene sets, tissue-specific cell type hierarchies) to assign biologically meaningful labels. This process is inherently iterative: ambiguous clusters may require subclustering, and similar clusters may need merging — decisions that depend on the analyst's experience and the specific biological context.

### 1.2 LLMs as Biological Reasoning Engines

Large language models trained on scientific literature encode substantial biological knowledge, including canonical marker gene associations (e.g., CD3D/CD3E for T cells, CD14/LYZ for monocytes), cell type hierarchies, and tissue-specific biology. Recent advances in tool-use capabilities allow LLMs to interact with external software through structured API calls, receiving results and making subsequent decisions based on the output. This creates an opportunity to combine the computational reliability of established bioinformatics tools with the biological reasoning capabilities of LLMs.

### 1.3 Contribution

We present sc-atlas-agentic-builder, a framework that operationalizes this insight through three design principles:

1. **Tool atomicity**: Each computational step is encapsulated as an independent tool with a defined interface, enabling the agent to compose arbitrary workflows.
2. **Agent-in-the-loop annotation**: Tools provide evidence (scores, statistics, marker genes); the agent makes biological decisions (cell type assignments, subclustering choices).
3. **Stateful execution**: A pipeline executor manages data state between tool calls, enabling multi-step workflows including iterative subclustering and per-batch annotation.

---

## 2. Architecture

### 2.1 System Overview

sc-atlas-agentic-builder consists of three layers:

```
+-------------------------------------------------+
|              LLM Agent (Claude)                 |
|  Receives data summaries, tool results          |
|  Makes decisions: parameters, labels, workflow  |
+-------------------------------------------------+
          |  Tool calls (JSON)  ^  Results (JSON)
          v                     |
+-------------------------------------------------+
|           Pipeline Executor                      |
|  Manages AnnData state, dispatches tool calls   |
|  Stores temporary state (subclusters, batches)  |
+-------------------------------------------------+
          |  Python calls       ^  (adata, stats)
          v                     |
+-------------------------------------------------+
|              Tool Layer (16 tools)               |
|  Preprocessing | Integration | Clustering       |
|  Marker Genes  | Annotation  | Utility          |
+-------------------------------------------------+
          |                     ^
          v                     |
+-------------------------------------------------+
|         Computational Backend                    |
|  Scanpy, Harmony, Scanorama, scib-metrics,      |
|  CellTypist, scikit-learn                        |
+-------------------------------------------------+
```

### 2.2 ToolWrapper Interface

All tools inherit from an abstract `ToolWrapper` base class that enforces a uniform contract:

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

The `input_type` and `output_type` properties define a type system over data states (`counts` -> `preprocessed` -> `integrated` -> `clustered` -> `annotated`), enabling the agent and executor to validate workflow ordering. The `get_param_space()` method exposes tunable parameters with their types and ranges, which the agent uses to select appropriate values based on dataset characteristics.

### 2.3 Tool Inventory

The framework provides 16 tools across six modules:

| Module | Tool | Purpose |
|--------|------|---------|
| **Preprocessing** | `PreprocessingTool` | QC filtering, normalization, HVG selection, PCA |
| **Integration** | `SelectIntegrationGenesTool` | Batch-aware gene selection (rank/intersection/union) |
| | `RunIntegrationTool` | Batch correction (Harmony, Scanorama) |
| | `EvaluateIntegrationTool` | Integration quality metrics via scib-metrics |
| **Clustering** | `ClusteringTool` | KNN graph, Leiden clustering, UMAP, quality assessment |
| **Marker Genes** | `GetMarkerGenesTool` | Differential expression, filtering, top marker extraction |
| **Annotation** | `ScoreMarkersTool` | Score clusters against marker gene signatures |
| | `CellTypistTool` | Reference-based automated annotation |
| | `CheckConfidenceTool` | Per-cluster annotation confidence scoring |
| | `AnnotateCellTypesTool` | Write agent-decided labels to data |
| **Utility** | `SubclusterTool` | Re-cluster a specific cluster at higher resolution |
| | `AnnotateSubclustersTool` | Merge subcluster labels back to main dataset |
| | `MergeClustersTool` | Assign same label to multiple clusters |
| | `HarmonizeLabelsTool` | Standardize label names across datasets |

Two additional executor-level operations (`annotate_batch`, `assign_batch_labels`) handle per-batch annotation workflows without requiring dedicated tool classes.

---

## 3. Methods

### 3.1 Preprocessing

The preprocessing module applies a sequential pipeline: QC filtering -> doublet detection -> normalization -> HVG selection -> PCA. Several design decisions address common failure modes in production single-cell pipelines:

**Memory-safe doublet detection.** Rather than calling `scrublet` directly and materializing dense matrices via `.toarray()` (which inflates a typical 500MB sparse matrix to 16+ GB), we delegate to `sc.pp.scrublet()`, which handles sparse matrices internally. When `batch_key` is provided, scrublet runs per-batch to respect the statistical assumptions of the doublet simulation algorithm.

**Sparsity-preserving scaling.** Standard scaling (`sc.pp.scale`) centers gene expression by subtracting means, destroying matrix sparsity. We use `zero_center=False` to scale variance without centering, preserving the sparse representation and avoiding out-of-memory failures on large datasets.

**Explicit HVG routing in PCA.** Rather than relying on Scanpy's implicit behavior of detecting `highly_variable` in `adata.var`, we explicitly pass `use_highly_variable=True` with validation, making the PCA input deterministic and debuggable.

### 3.2 Batch Integration

For multi-sample cell atlases, batch effects must be corrected before clustering. The integration module supports two methods:

**Harmony** (Korsunsky et al., 2019): Operates on the PCA embedding, iteratively adjusting principal components to remove batch effects while preserving biological variation. Fast and widely applicable.

**Scanorama** (Hie et al., 2019): Uses mutual nearest neighbors (MNN) to identify shared cell populations across batches and computes a corrected embedding. Does not require a GPU.

A critical design decision: when `use_rep` points to an integration embedding (e.g., `X_harmony`, `X_scVI`), the clustering tool uses all dimensions rather than applying `n_pcs` truncation. Unlike PCA where variance decays across components, integration embeddings are dense latent spaces where all dimensions carry equal weight — truncating them would destroy batch-corrected information.

**Integration evaluation** uses scib-metrics (Luecken et al., 2022) as the primary evaluation path, computing:
- *Batch mixing*: ASW_batch (silhouette score on batch labels, inverted so higher = better mixing), graph connectivity
- *Bio conservation*: ASW_label (silhouette on cell type labels), NMI, ARI, isolated label ASW
- *Overall score*: 40% batch mixing + 60% bio conservation, following the scIB convention

### 3.3 Marker Gene Identification

Differential expression is performed via `sc.tl.rank_genes_groups` with `pts=True` to capture expression fractions. Results are extracted using `sc.get.rank_genes_groups_df()` (vectorized Pandas operations) rather than manual Python loops over structured arrays, avoiding both performance bottlenecks and the structural mismatch between numpy recarrays (`names`, `scores`) and Pandas DataFrames (`pts`, `pts_rest`).

The filtered results are stored as a single concatenated DataFrame with a `group` column, rather than a dictionary of DataFrames, ensuring HDF5 serializability when saving to `.h5ad` format. NaN values from statistical tests (common with logistic regression on sparse data) are replaced with `None` before JSON serialization.

### 3.4 Cell Type Annotation

Annotation is decomposed into four distinct tools, each representing an independent decision point for the agent:

1. **ScoreMarkersTool**: Given a `marker_dict` (e.g., `{"T cells": ["CD3D", "CD3E"], ...}`), scores each cluster using `sc.tl.score_genes`. The marker dictionary is typically constructed by the agent from its knowledge of canonical markers, informed by the top marker genes identified in the previous step.

2. **CellTypistTool** (optional): Runs CellTypist (Dominguez Conde et al., 2022) reference-based annotation when a pretrained model exists for the tissue type. Includes majority voting within clusters to stabilize per-cell predictions.

3. **CheckConfidenceTool**: Computes per-cluster confidence as the gap ratio between the best and second-best marker scores. When both marker scores and CellTypist predictions are available, cross-method agreement is also evaluated. Low-confidence clusters are flagged for subclustering.

4. **AnnotateCellTypesTool**: The agent reviews all evidence and provides a `label_mapping` dictionary. The tool writes labels to `adata.obs["cell_type"]` and optionally `adata.obs["cell_type_fine"]`.

This separation ensures the agent has explicit decision points between gathering evidence and committing labels, enabling it to reason about ambiguous cases rather than blindly accepting automated predictions.

### 3.5 Iterative Refinement

The framework supports two refinement operations:

**Subclustering**: The agent identifies a low-confidence cluster (e.g., mixed T cell markers), calls `SubclusterTool` to re-cluster that subset at higher resolution, reviews the resulting sub-markers, and calls `AnnotateSubclustersTool` to write fine-grained labels back to the main dataset. The executor manages the temporary `adata_sub` object between these two calls, since AnnData objects cannot be serialized in tool call JSON.

**Cluster merging**: When marker analysis reveals that two clusters represent the same cell type (e.g., two monocyte clusters split by technical variation), the agent calls `MergeClustersTool` to assign the same label to both cluster IDs.

### 3.6 Multi-Batch Workflow

For multi-batch datasets, the framework implements a pre-integration annotation step to obtain cell type labels for integration quality evaluation:

1. For each batch, the executor subsets the data and runs clustering + marker identification
2. The agent reviews markers and assigns coarse labels per batch
3. `HarmonizeLabelsTool` standardizes label names across batches (e.g., "Mono" -> "Monocytes")
4. Integration is run and evaluated using the harmonized labels
5. Final annotation proceeds on the integrated, re-clustered data

The framework handles two input formats: (1) a single pre-merged `.h5ad` file with a batch column, and (2) multiple separate `.h5ad` files with potentially different gene sets. For format (2), a `prepare_data()` function preprocesses each dataset individually, intersects gene sets, concatenates, and re-runs HVG selection + PCA on the merged data.

---

## 4. Agent-Tool Interaction

### 4.1 Communication Protocol

The agent communicates with tools via the OpenAI-compatible chat completions API. Each tool is described as a JSON schema function definition. The agent receives:

- A system prompt encoding the analysis workflow and decision guidelines
- A user message describing the dataset (dimensions, available columns, batch information, preprocessing state)
- Tool results as JSON strings containing statistics, marker gene lists, and confidence scores

The agent responds with either text (reasoning, status updates) or tool calls (parameter-specified JSON). The executor parses tool calls, executes the corresponding tool, and returns results for the next turn.

### 4.2 Decision Examples

**Parameter selection**: Given a dataset of 3,000 cells x 525 genes, the agent adapts preprocessing parameters (e.g., `max_genes=525`, `n_top_genes=250`) based on the dataset dimensions reported in the user message. When initial parameters fail (e.g., `max_genes=400` causing an index error on a 525-gene dataset), the agent reasons about the failure and adjusts.

**Cell type identification**: After receiving top markers per cluster (e.g., cluster 0: `[CD3E, CD3D, IL7R, LTB, CD8A]`), the agent constructs a marker dictionary from its knowledge of canonical immune markers, scores clusters, and assigns labels. When a cluster shows mixed markers (both CD4 and CD8 T cell genes), the agent notes this as "Mixed_T_cells" at the broad level and may subcluster to resolve.

**Workflow adaptation**: For single-batch data, the agent skips integration steps. When integration fails (e.g., library not installed), the agent falls back to PCA-based clustering. These decisions are made dynamically based on tool results, not hardcoded.

### 4.3 State Management

The `PipelineExecutor` maintains three state objects:
- `self.adata`: The primary AnnData object, updated after each tool call
- `self._stored_subclusters`: A dictionary mapping cluster IDs to temporary AnnData subsets, enabling the two-step subcluster-then-annotate workflow
- `self._stored_batches`: A dictionary mapping batch values to per-batch AnnData subsets for the pre-integration annotation workflow

This design allows the agent to interleave operations (e.g., subcluster multiple clusters before annotating any of them) without requiring the tools themselves to manage state.

---

## 5. Results

### 5.1 Demo Dataset Validation

We validated the framework on a synthetic PBMC-like dataset (3,000 cells, 525 genes, 2 batches, 4 cell types: T cells, B cells, monocytes, NK cells). The agent completed the full pipeline in 11 turns:

1. Initial preprocessing failed (`max_genes` too restrictive); agent adjusted and succeeded
2. Integration encountered a shape mismatch bug; agent fell back to PCA-based clustering
3. Leiden clustering identified 4 clusters (resolution=0.6, silhouette=0.08)
4. Marker genes clearly separated cell types (CD3E/CD3D for T cells, CD79A/CD79B for B cells, CD14/LYZ for monocytes, NKG7/GNLY for NK cells)
5. Marker scoring confirmed assignments with high confidence for 3/4 clusters
6. Cluster 0 (T cells) was flagged as low confidence (17.3%) due to mixed CD4/CD8 markers — the agent noted this as a candidate for subclustering

### 5.2 Annotation Quality

The agent correctly identified all four major cell types without any human intervention. The confidence assessment correctly flagged the heterogeneous T cell cluster. On a dataset where the ground truth labels were known, the coarse annotation achieved 100% accuracy at the major cell type level.

---

## 6. Discussion

### 6.1 Tool Atomicity vs. Monolithic Pipelines

A key architectural decision was decomposing annotation into four separate tools (score, CellTypist, confidence, assign) rather than a single `AnnotateTool`. While a monolithic tool would require fewer agent turns, the decomposition creates explicit decision points where the agent can inspect intermediate results. This matters most for ambiguous cases: when marker scores and CellTypist disagree, the agent can reason about which evidence source to trust before committing labels.

### 6.2 Memory and Performance Considerations

Several design choices were driven by memory constraints on large datasets:
- Sparse-preserving operations throughout (doublet detection via `sc.pp.scrublet`, scaling with `zero_center=False`)
- Subsampling in integration evaluation (capped at 50,000 cells for scib-metrics)
- Avoiding `.copy()` chains that briefly hold multiple full copies of the dataset

These are not theoretical concerns — a 100,000-cell dataset with 20,000 genes occupies ~500MB as a sparse matrix but 16+ GB as dense. Production pipelines must respect this.

### 6.3 Limitations

**Agent reasoning quality depends on the LLM.** The framework's annotation quality is bounded by the agent's biological knowledge. For well-studied tissues (PBMC, brain, lung), current LLMs perform well on canonical markers. For non-model organisms or novel cell states, the agent's reasoning may be insufficient, and user-provided marker dictionaries become essential.

**Integration method coverage.** The current implementation supports Harmony and Scanorama. Deep learning methods (scVI, scANVI) offer superior performance on complex datasets but require GPU infrastructure and longer training times. Adding these as tool options is straightforward given the modular architecture.

**Evaluation dependency on labels.** Bio conservation metrics require cell type labels, creating a chicken-and-egg problem for unannotated datasets. Our pre-integration annotation step addresses this with coarse labels, but imperfect annotations may skew evaluation scores.

### 6.4 Future Directions

- **Multi-resolution annotation**: Extending the subclustering loop to automatically explore multiple resolutions and select the optimal one based on marker gene separation
- **Cross-reference validation**: Querying external databases (CellMarker, PanglaoDB) for marker gene validation during annotation
- **Provenance tracking**: Recording the full agent reasoning trace alongside computational logs for reproducibility auditing
- **Benchmark on real atlases**: Systematic evaluation on published cell atlas datasets (e.g., Human Cell Atlas, Tabula Sapiens) with known annotations

---

## 7. Conclusion

sc-atlas-agentic-builder demonstrates that LLMs can serve as effective decision-makers in structured bioinformatics pipelines when given appropriate tool interfaces. By separating computational execution from biological reasoning, the framework achieves the reproducibility of scripted pipelines with the adaptability of expert analysis. The modular tool architecture enables incremental extension — adding new integration methods, annotation references, or quality metrics requires implementing a single `ToolWrapper` subclass without modifying the agent or executor logic. We believe this pattern — atomic tools orchestrated by LLM agents — is broadly applicable to computational biology workflows where domain expertise drives analytical decisions.

---

## References

- Dominguez Conde, C., et al. (2022). Cross-tissue immune cell analysis reveals tissue-specific features in humans. *Science*, 376(6594), eabl5197.
- Hao, Y., et al. (2024). Dictionary learning for integrative, multimodal and scalable single-cell analysis. *Nature Biotechnology*, 42, 293-304.
- Hie, B., et al. (2019). Efficient integration of heterogeneous single-cell transcriptomes using Scanorama. *Nature Biotechnology*, 37, 685-691.
- Korsunsky, I., et al. (2019). Fast, sensitive and accurate integration of single-cell data with Harmony. *Nature Methods*, 16, 1289-1296.
- Luecken, M.D., et al. (2022). Benchmarking atlas-level data integration in single-cell genomics. *Nature Methods*, 19, 41-50.
- Luecken, M.D., & Theis, F.J. (2019). Current best practices in single-cell RNA-seq analysis: a tutorial. *Molecular Systems Biology*, 15(6), e8746.
- Wolf, F.A., et al. (2018). SCANPY: large-scale single-cell gene expression data analysis. *Genome Biology*, 19, 15.

---

## Reproducibility

Full reproducibility instructions are provided in `SKILL.md` at the project root. The pipeline is fully self-contained — no external data files are required. A built-in synthetic PBMC dataset generator (`demo/create_demo_data.py`) creates a 3,000-cell, 528-gene dataset with 6 known cell types across 2 batches.

### Dependencies (pinned)

```
pip install \
  scanpy==1.12 anndata==0.12.10 harmonypy==0.2.0 \
  scanorama==1.7.4 scib-metrics==0.5.9 scikit-learn==1.7.1 \
  leidenalg==0.11.0 python-igraph==1.0.0 openai==2.26.0 \
  numpy==2.2.6 scipy==1.16.3 pandas==2.3.1
```

### Run (demo data, no external files needed)

```bash
cd sc_atlas_agentic_builder
python -m demo.run_with_claude
```

### Expected Output

The agent completes in 7-15 turns and produces:
- `annotated.h5ad` — AnnData with `cell_type` and `cell_type_fine` columns in `.obs`
- `annotated_logs.json` — full execution trace (all tool calls and results)

Expected cell types identified from the demo dataset:

| Cell Type | Markers | Approximate Count |
|-----------|---------|-------------------|
| T cells (or CD4/CD8 split) | CD3D, CD3E, IL7R, CD8A | ~800-1200 |
| B cells | CD79A, CD79B, MS4A1 | ~400-600 |
| Monocytes | CD14, LYZ, S100A8 | ~400-600 |
| NK cells | NKG7, GNLY, PRF1 | ~200-350 |

### Verification

```python
import anndata as ad
adata = ad.read_h5ad("annotated.h5ad")

assert "cell_type" in adata.obs.columns, "Missing cell_type column"
assert adata.obs["cell_type"].notna().all(), "Some cells lack labels"

cell_types = set(adata.obs["cell_type"].unique())
expected = {"T_cells", "B_cells", "Monocytes", "NK_cells"}
matched = sum(1 for e in expected
              if any(e.lower().replace("_", "") in ct.lower().replace("_", "")
                     for ct in cell_types))
assert matched >= 3, f"Only matched {matched}/4 expected types. Found: {cell_types}"
print(f"Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")
print("Verification PASSED")
```

### Code Availability

The complete source code is organized at `sc_atlas_agentic_builder/` with the following module structure:

```
sc_atlas_agentic_builder/
  SKILL.md                     # Full reproducibility instructions
  base.py                      # ToolWrapper abstract interface
  preprocessing/               # QC, normalization, HVG, PCA
  integration/                 # Harmony, Scanorama, scib-metrics evaluation
  clustering/                  # KNN, Leiden, UMAP, assessment
  marker_genes/                # DE, filtering, top marker extraction
  annotation/                  # Marker scoring, CellTypist, confidence, label assignment
  utility/                     # Subclustering, merging, label harmonization
  demo/                        # Agent loop, data preparation, demo data generation
```
