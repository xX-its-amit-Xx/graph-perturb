# graph-perturb cookbook

Four runnable, narrated notebooks that exercise the **real** `graph-perturb` API end to end:
load single-cell perturbation data, build a *swappable* biological knowledge graph, train a GNN
(or VAE), and print real metrics from `compute_metrics` / `evaluate_model`.

## Notebooks

| Notebook | Shows |
|----------|-------|
| [`01_predict_heldout_single.ipynb`](01_predict_heldout_single.ipynb) | Full pipeline: load data → build a GO BP graph → `make_splits` → train the GraphSAGE GNN → predict a **held-out single CRISPRa perturbation**. Plots predicted-vs-true delta (scatter + top DE genes) and prints per-condition and per-split Pearson / MSE / overlap@20. |
| [`02_compare_backends.ipynb`](02_compare_backends.ipynb) | **Swap the prior.** Build GO vs Reactome vs STRING over the *same* gene universe and split, train the same GNN on each, and print a backend-vs-backend table via `evaluate.compare_backends`. |
| [`03_gnn_vs_vae.ipynb`](03_gnn_vs_vae.ipynb) | **Does the graph help?** Same split, train the GNN and the conditional VAE, print a head-to-head table via `evaluate.compare_models`, plus a discussion of where message passing wins. |
| [`04_custom_graph.ipynb`](04_custom_graph.ipynb) | **Bring your own graph.** Build a custom edge-list CSV from a correlation prior and plug it in via `get_graph_source("custom", ...)`; train and compare to a built-in backend. Bonus: the **GEARS-format adapter** (`gears` backend / `load_gears_graph`) and the lossless **AnnData round-trip** (`processed_to_anndata` / `anndata_to_processed`) for scanpy workflows. |

## Real-data note

Every notebook calls `load_norman` to fetch the **real Norman et al. 2019 (Cell) CRISPRa
Perturb-seq** dataset (scPerturb-harmonized AnnData, log-normalized, HVG-subset, condition labels
normalized to `ctrl` / `GENE` / `A+B`). The metrics printed are computed on this real data when it
is available.

## Disk / offline fallback behavior

`load_norman` only downloads when there is no local `.h5ad` and the target volume passes a free-disk
guard (≥ 4 GB). On a small or air-gapped machine that raises, and each notebook **automatically falls
back** to `make_synthetic_norman` — a tiny, fully-valid `ProcessedPerturbData` built from **real**
pathway gene symbols so the GO/Reactome/STRING backends still find genuine edges. Each notebook
**prints which mode it is in** (`REAL Norman Perturb-seq` vs `SYNTHETIC offline stand-in`).

Likewise, the graph backends download their source dumps (GO GAF, Reactome, STRING) on first use and
cache them under `~/.graph_perturb_cache` (override with `$GRAPH_PERTURB_CACHE`); if a dump is
unavailable offline they log a warning and fall back to deterministic co-expression edges, so every
cell runs without network access.

Epoch counts are intentionally small (≈ 5–6) so the notebooks finish on CPU in minutes. They are
demonstrations of the API and workflow, not publication-grade benchmarks — increase `train.epochs`,
use the real data, and average over seeds for real numbers.
