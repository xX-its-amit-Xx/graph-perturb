"""Shared pytest fixtures for the graph-perturb test suite.

All fixtures are fast and fully offline: synthetic Norman data, a custom-CSV
graph backend, a real backend (``go_bp``) exercised via its offline
co-expression fallback, and tiny configs. Nothing here downloads data, trains
for more than a couple of epochs, or touches the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from graph_perturb.config import (
    Config,
    EvalConfig,
    GraphConfig,
    ModelConfig,
    SplitConfig,
    TrainConfig,
)
from graph_perturb.data.norman import make_synthetic_norman
from graph_perturb.graphs.registry import get_graph_source


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def processed():
    """A small synthetic :class:`ProcessedPerturbData` with singles + combos."""
    return make_synthetic_norman(n_genes=24, n_conditions=14, seed=0, n_cells_per_condition=12)


@pytest.fixture
def gene_names(processed):
    return list(processed.gene_names)


# --------------------------------------------------------------------------- #
# graphs
# --------------------------------------------------------------------------- #
@pytest.fixture
def custom_csv(tmp_path, gene_names):
    """Write a tiny weighted edge-list CSV over the synthetic gene universe."""
    path = tmp_path / "edges.csv"
    lines = ["source,target,weight"]
    # Chain a handful of real edges plus one edge to a gene not in the universe
    # (must be dropped by build) to exercise filtering.
    for i in range(len(gene_names) - 1):
        lines.append(f"{gene_names[i]},{gene_names[i + 1]},{0.5 + 0.01 * i:.3f}")
    lines.append(f"{gene_names[0]},NOT_A_REAL_GENE,0.9")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def custom_graph(custom_csv, gene_names, tmp_path):
    """A :class:`PerturbationGraph` built from the custom CSV backend."""
    source = get_graph_source(
        "custom",
        cache_dir=tmp_path / "cache",
        undirected=True,
        add_self_loops=True,
        path=str(custom_csv),
        weight_col="weight",
    )
    return source.build(gene_names, use_cache=False)


@pytest.fixture
def real_graph(gene_names, tmp_path):
    """A real backend (``go_bp``) built via its OFFLINE co-expression fallback.

    The GAF download is unavailable in CI, so ``_load_edges`` deterministically
    falls back to co-expression ring edges -- exactly the air-gapped path the
    backend documents. No network access occurs.
    """
    source = get_graph_source(
        "go_bp",
        cache_dir=tmp_path / "cache_go",
        undirected=True,
        add_self_loops=True,
        gaf_url="file:///definitely/not/a/real/path.gaf.gz",
    )
    return source.build(gene_names, use_cache=False)


# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_train_cfg():
    return TrainConfig(
        epochs=2,
        batch_size=16,
        lr=1e-2,
        weight_decay=0.0,
        early_stop_patience=10,
        grad_clip=1.0,
        num_workers=0,
        device="cpu",
        log_every=100,
        seed=0,
    )


@pytest.fixture
def tiny_gnn_cfg():
    return ModelConfig(name="gnn", hidden_dim=16, n_layers=2, dropout=0.0, attention_heads=2)


@pytest.fixture
def tiny_vae_cfg():
    return ModelConfig(name="vae", hidden_dim=16, n_layers=2, dropout=0.0, latent_dim=8)


@pytest.fixture
def split_cfg():
    return SplitConfig(val_frac=0.2, test_single_frac=0.3, test_combo_frac=0.5, seed=0)


@pytest.fixture
def tiny_config(tiny_train_cfg, tiny_gnn_cfg, split_cfg):
    cfg = Config()
    cfg.experiment_name = "test-tiny"
    cfg.graph = GraphConfig(name="custom")
    cfg.model = tiny_gnn_cfg
    cfg.train = tiny_train_cfg
    cfg.split = split_cfg
    cfg.eval = EvalConfig(overlap_k=5, splits=("test_single", "test_combo"))
    return cfg


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def rng():
    return np.random.default_rng(0)
