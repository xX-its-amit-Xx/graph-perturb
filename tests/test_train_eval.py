"""Tests for the training loop + evaluation, on a tiny synthetic pipeline."""

from __future__ import annotations

import math

import pytest

from graph_perturb.data.dataset import build_dataloaders
from graph_perturb.data.splits import make_splits
from graph_perturb.evaluate import evaluate_model, metrics_table
from graph_perturb.metrics import PerturbationMetrics
from graph_perturb.models import build_model
from graph_perturb.train import train_model


def _train_tiny(processed, graph, model_cfg, train_cfg, split_cfg, eval_cfg):
    splits = make_splits(processed, split_cfg)
    loaders = build_dataloaders(processed, graph, splits, train_cfg)
    model = build_model(model_cfg, num_genes=processed.n_genes, graph=graph)
    history = train_model(model, loaders, train_cfg, ckpt_dir=None)
    results = evaluate_model(model, processed, graph, splits, eval_cfg)
    return history, results, splits


def test_train_returns_history(processed, custom_graph, tiny_gnn_cfg, tiny_train_cfg, split_cfg):
    from graph_perturb.config import EvalConfig

    history, results, _ = _train_tiny(
        processed, custom_graph, tiny_gnn_cfg, tiny_train_cfg, split_cfg,
        EvalConfig(overlap_k=5, splits=("test_single", "test_combo")),
    )
    assert len(history["history"]) >= 1
    assert math.isfinite(history["best_val"])
    assert "ckpt_path" in history


def test_evaluate_returns_metrics_for_splits(
    processed, custom_graph, tiny_gnn_cfg, tiny_train_cfg, split_cfg
):
    from graph_perturb.config import EvalConfig

    eval_cfg = EvalConfig(overlap_k=5, splits=("test_single", "test_combo"))
    _, results, splits = _train_tiny(
        processed, custom_graph, tiny_gnn_cfg, tiny_train_cfg, split_cfg, eval_cfg
    )
    assert len(results) >= 1
    for name, m in results.items():
        assert name in ("test_single", "test_combo")
        assert isinstance(m, PerturbationMetrics)
        assert math.isfinite(m.pearson_delta)
        assert math.isfinite(m.mse)
        assert m.n_conditions >= 1
    table = metrics_table(results)
    assert "pearson_delta" in table


def test_vae_train_eval(processed, custom_graph, tiny_vae_cfg, tiny_train_cfg, split_cfg):
    from graph_perturb.config import EvalConfig

    eval_cfg = EvalConfig(overlap_k=5, splits=("test_single", "test_combo"))
    # VAE is graph-free; evaluate_model still accepts a graph and ignores topology.
    _, results, _ = _train_tiny(
        processed, custom_graph, tiny_vae_cfg, tiny_train_cfg, split_cfg, eval_cfg
    )
    assert len(results) >= 1
    for m in results.values():
        assert math.isfinite(m.mse)


@pytest.mark.skip(reason="needs network/real data")
def test_real_norman_pipeline():
    from graph_perturb.config import Config
    from graph_perturb.train import run_training

    run_training(Config())
