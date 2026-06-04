"""Tests for the GNN and VAE models: forward shapes, loss, one-step descent."""

from __future__ import annotations

import math

import torch
from torch_geometric.loader import DataLoader

from graph_perturb.data.dataset import PerturbationDataset
from graph_perturb.models import build_model


def _overfit_batch(processed, graph, batch_size=8):
    """A single fixed batch over a couple of conditions for overfit tests."""
    conds = processed.condition_labels()[:2]
    ds = PerturbationDataset(processed, graph, conds)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


def test_gnn_forward_shape(processed, custom_graph, tiny_gnn_cfg):
    model = build_model(tiny_gnn_cfg, num_genes=processed.n_genes, graph=custom_graph)
    batch = _overfit_batch(processed, custom_graph)
    model.eval()
    out = model(batch)
    b = int(batch.batch.max().item()) + 1
    assert out.shape == (b, processed.n_genes)


def test_vae_forward_shape(processed, custom_graph, tiny_vae_cfg):
    model = build_model(tiny_vae_cfg, num_genes=processed.n_genes)
    batch = _overfit_batch(processed, custom_graph)
    model.eval()
    out = model(batch)
    b = int(batch.batch.max().item()) + 1
    assert out.shape == (b, processed.n_genes)
    assert model.requires_graph is False


def test_gnn_loss_finite_and_logs(processed, custom_graph, tiny_gnn_cfg):
    model = build_model(tiny_gnn_cfg, num_genes=processed.n_genes, graph=custom_graph)
    batch = _overfit_batch(processed, custom_graph)
    loss, logs = model.loss(batch)
    assert loss.dim() == 0
    assert math.isfinite(float(loss.detach()))
    assert "mse" in logs and "loss" in logs


def test_vae_loss_has_kl(processed, custom_graph, tiny_vae_cfg):
    model = build_model(tiny_vae_cfg, num_genes=processed.n_genes)
    batch = _overfit_batch(processed, custom_graph)
    loss, logs = model.loss(batch)
    assert math.isfinite(float(loss.detach()))
    assert "kl" in logs
    assert "mse" in logs


def _one_step_decreases(model, batch, steps=30, lr=1e-2):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss0, _ = model.loss(batch)
    initial = float(loss0.detach())
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, _ = model.loss(batch)
        loss.backward()
        opt.step()
    final = float(model.loss(batch)[0].detach())
    return initial, final


def test_gnn_overfits_tiny_batch(processed, custom_graph, tiny_gnn_cfg):
    torch.manual_seed(0)
    model = build_model(tiny_gnn_cfg, num_genes=processed.n_genes, graph=custom_graph)
    batch = _overfit_batch(processed, custom_graph)
    initial, final = _one_step_decreases(model, batch)
    assert final < initial


def test_vae_overfits_tiny_batch(processed, custom_graph, tiny_vae_cfg):
    torch.manual_seed(0)
    model = build_model(tiny_vae_cfg, num_genes=processed.n_genes)
    batch = _overfit_batch(processed, custom_graph)
    initial, final = _one_step_decreases(model, batch)
    assert final < initial
