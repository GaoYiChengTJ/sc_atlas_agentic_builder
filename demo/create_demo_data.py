"""
Create a small synthetic PBMC-like dataset for testing the pipeline.
"""

import numpy as np
import anndata as ad
import scipy.sparse as sp


def create_demo_pbmc(n_cells: int = 3000, seed: int = 42) -> ad.AnnData:
    """
    Create a synthetic scRNA-seq dataset with known cell types.

    Cell types (with canonical markers):
        - CD4 T cells:  CD3D, CD3E, IL7R, LTB
        - CD8 T cells:  CD3D, CD3E, CD8A, CD8B, GZMB
        - B cells:      MS4A1, CD79A, CD79B, CD19
        - Monocytes:    CD14, LYZ, S100A8, S100A9
        - NK cells:     NKG7, GNLY, PRF1, NCAM1
        - DC:           FCER1A, CST3, CLEC10A
    """
    rng = np.random.RandomState(seed)

    cell_types = {
        "CD4_T":     {"frac": 0.25, "markers": ["CD3D", "CD3E", "IL7R", "LTB"]},
        "CD8_T":     {"frac": 0.15, "markers": ["CD3D", "CD3E", "CD8A", "CD8B", "GZMB"]},
        "B_cell":    {"frac": 0.20, "markers": ["MS4A1", "CD79A", "CD79B", "CD19"]},
        "Monocyte":  {"frac": 0.20, "markers": ["CD14", "LYZ", "S100A8", "S100A9"]},
        "NK":        {"frac": 0.12, "markers": ["NKG7", "GNLY", "PRF1", "NCAM1"]},
        "DC":        {"frac": 0.08, "markers": ["FCER1A", "CST3", "CLEC10A"]},
    }

    # Collect all marker genes + background genes.
    all_markers = set()
    for ct in cell_types.values():
        all_markers.update(ct["markers"])
    all_markers = sorted(all_markers)

    n_background = 500
    background_genes = [f"GENE_{i}" for i in range(n_background)]
    gene_names = all_markers + background_genes
    n_genes = len(gene_names)

    # Generate cells.
    labels = []
    batches = []
    X_dense = np.zeros((n_cells, n_genes), dtype=np.float32)

    idx = 0
    for ct_name, ct_info in cell_types.items():
        n_ct = int(n_cells * ct_info["frac"])
        for i in range(n_ct):
            if idx >= n_cells:
                break

            # Background expression: sparse low counts.
            X_dense[idx, :] = rng.poisson(0.3, n_genes).astype(np.float32)

            # Marker expression: higher counts for this cell type.
            for marker in ct_info["markers"]:
                gene_idx = gene_names.index(marker)
                X_dense[idx, gene_idx] = rng.poisson(8, 1)[0] + rng.randint(2, 6)

            # Shared markers get intermediate expression in related types.
            if ct_name in ("CD4_T", "CD8_T"):
                for shared in ["CD3D", "CD3E"]:
                    gene_idx = gene_names.index(shared)
                    X_dense[idx, gene_idx] = rng.poisson(10, 1)[0] + 3

            labels.append(ct_name)
            batches.append(f"batch_{i % 2}")
            idx += 1

    # Fill remaining cells with random type.
    while idx < n_cells:
        ct_name = rng.choice(list(cell_types.keys()))
        ct_info = cell_types[ct_name]
        X_dense[idx, :] = rng.poisson(0.3, n_genes).astype(np.float32)
        for marker in ct_info["markers"]:
            gene_idx = gene_names.index(marker)
            X_dense[idx, gene_idx] = rng.poisson(8, 1)[0] + rng.randint(2, 6)
        labels.append(ct_name)
        batches.append(f"batch_{idx % 2}")
        idx += 1

    # Add mitochondrial genes.
    mt_genes = ["MT-CO1", "MT-CO2", "MT-ND1"]
    for mt in mt_genes:
        mt_expr = rng.poisson(2, n_cells).astype(np.float32)
        X_dense = np.column_stack([X_dense, mt_expr])
        gene_names.append(mt)

    # Build AnnData.
    X_sparse = sp.csr_matrix(X_dense)
    adata = ad.AnnData(
        X=X_sparse,
        obs={
            "true_label": labels,
            "sample_id": batches,
        },
    )
    adata.var_names = gene_names
    adata.obs_names = [f"cell_{i:05d}" for i in range(n_cells)]

    return adata


if __name__ == "__main__":
    adata = create_demo_pbmc()
    print(f"Created: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"Cell types: {adata.obs['true_label'].value_counts().to_dict()}")
    print(f"Batches: {adata.obs['sample_id'].value_counts().to_dict()}")
    adata.write_h5ad("demo_pbmc.h5ad")
    print("Saved to demo_pbmc.h5ad")
