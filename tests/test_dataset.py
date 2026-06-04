"""Tests for the PyG dataset, dataloaders, and batch_to_dense."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.loader import DataLoader

from graph_perturb.data.dataset import PerturbationDataset, build_dataloaders
from graph_perturb.data.splits import make_splits
from graph_perturb.models.base import batch_to_dense


def test_dataset_item_shapes(processed, custom_graph):
    conds = processed.condition_labels()[:3]
    ds = PerturbationDataset(processed, custom_graph, conds)
    assert len(ds) > 0
    item = ds[0]
    g = processed.n_genes
    assert item.x.shape == (g, 2)
    assert item.y.shape == (g,)
    assert item.edge_index.shape[0] == 2
    assert torch.equal(item.edge_index, custom_graph.edge_index)
    assert isinstance(item.condition, list) and len(item.condition) == 1


def test_dataset_pert_flag_column(processed, custom_graph):
    # x[:, 1] is the perturbation one-hot; must match the condition's targets.
    cond = next(c for c in processed.condition_labels() if not processed.is_combo(c))
    ds = PerturbationDataset(processed, custom_graph, [cond])
    item = ds[0]
    flag = item.x[:, 1]
    target_genes = processed.perturbed_genes_per_condition[cond]
    expected_idx = {custom_graph.gene_to_idx[g] for g in target_genes}
    on = set(torch.nonzero(flag).flatten().tolist())
    assert on == expected_idx


def test_dataset_gene_order_mismatch_raises(processed, custom_graph):
    # Build a graph whose gene order differs, expect AssertionError.
    from graph_perturb.graphs.base import PerturbationGraph

    reordered = list(reversed(processed.gene_names))
    g = PerturbationGraph(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        gene_names=reordered,
        gene_to_idx={n: i for i, n in enumerate(reordered)},
    )
    with pytest.raises(AssertionError):
        PerturbationDataset(processed, g, processed.condition_labels()[:1])


def test_build_dataloaders_keys(processed, custom_graph, split_cfg, tiny_train_cfg):
    splits = make_splits(processed, split_cfg)
    loaders = build_dataloaders(processed, custom_graph, splits, tiny_train_cfg)
    assert "train" in loaders
    assert set(loaders).issubset({"train", "val", "test_single", "test_combo"})
    for key, loader in loaders.items():
        assert isinstance(loader, DataLoader)


def test_batch_to_dense_reshapes(processed, custom_graph):
    conds = processed.condition_labels()[:2]
    ds = PerturbationDataset(processed, custom_graph, conds)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    g = processed.n_genes
    b = int(batch.batch.max().item()) + 1
    y_dense = batch_to_dense(batch.y, batch, g)
    assert y_dense.shape == (b, g)
    x_dense = batch_to_dense(batch.x, batch, g)
    assert x_dense.shape == (b, g, 2)
