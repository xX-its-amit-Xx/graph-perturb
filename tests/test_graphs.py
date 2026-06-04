"""Tests for the swappable knowledge-graph layer (base + registry + backends)."""

from __future__ import annotations

import torch
import pytest

from graph_perturb.graphs.base import PerturbationGraph
from graph_perturb.graphs.registry import available_graphs, get_graph_source


def _edge_set(g: PerturbationGraph) -> set[tuple[int, int]]:
    ei = g.edge_index
    return {(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])}


def test_build_matches_gene_order(custom_graph, gene_names):
    assert list(custom_graph.gene_names) == list(gene_names)
    assert custom_graph.num_nodes == len(gene_names)
    assert custom_graph.gene_to_idx[gene_names[0]] == 0
    assert custom_graph.edge_index.dtype == torch.long
    assert custom_graph.edge_index.shape[0] == 2


def test_undirected_edges_symmetric(custom_graph):
    edges = _edge_set(custom_graph)
    for a, b in edges:
        if a != b:  # ignore self loops
            assert (b, a) in edges


def test_self_loops_present_when_enabled(custom_graph):
    edges = _edge_set(custom_graph)
    for i in range(custom_graph.num_nodes):
        assert (i, i) in edges


def test_self_loops_absent_when_disabled(custom_csv, gene_names, tmp_path):
    source = get_graph_source(
        "custom",
        cache_dir=tmp_path / "c2",
        add_self_loops=False,
        undirected=True,
        path=str(custom_csv),
    )
    g = source.build(gene_names, use_cache=False)
    edges = _edge_set(g)
    assert not any(a == b for a, b in edges)


def test_perturbation_vector_one_hot(custom_graph, gene_names):
    targets = [gene_names[2], gene_names[5]]
    vec = custom_graph.perturbation_vector(targets)
    assert vec.shape == (custom_graph.num_nodes,)
    assert vec.sum().item() == pytest.approx(2.0)
    assert vec[2].item() == 1.0 and vec[5].item() == 1.0


def test_perturbation_vector_ignores_unknown(custom_graph):
    vec = custom_graph.perturbation_vector(["NOT_A_GENE_AT_ALL"])
    assert vec.sum().item() == 0.0


def test_custom_backend_reads_edges(custom_graph, gene_names):
    # The CSV chains gene[i]-gene[i+1]; that adjacency must survive build.
    edges = _edge_set(custom_graph)
    assert (0, 1) in edges
    assert custom_graph.edge_weight is not None  # weight_col supplied
    assert custom_graph.edge_weight.shape[0] == custom_graph.num_edges
    # The edge to a gene outside the universe never produces a node/edge: every
    # edge endpoint is a valid in-universe node id.
    n = custom_graph.num_nodes
    assert int(custom_graph.edge_index.max()) < n
    assert "NOT_A_REAL_GENE" not in custom_graph.gene_to_idx


def test_save_load_roundtrip(custom_graph, tmp_path):
    path = tmp_path / "g.pt"
    custom_graph.save(path)
    loaded = PerturbationGraph.load(path)
    assert loaded.gene_names == custom_graph.gene_names
    assert loaded.gene_to_idx == custom_graph.gene_to_idx
    assert torch.equal(loaded.edge_index, custom_graph.edge_index)
    assert loaded.name == custom_graph.name
    if custom_graph.edge_weight is not None:
        assert torch.allclose(loaded.edge_weight, custom_graph.edge_weight)


def test_real_backend_offline_fallback(real_graph, gene_names):
    # go_bp falls back to co-expression edges with no network; still a valid graph.
    assert real_graph.name == "go_bp"
    assert list(real_graph.gene_names) == list(gene_names)
    assert real_graph.num_edges > 0
    edges = _edge_set(real_graph)
    for i in range(real_graph.num_nodes):
        assert (i, i) in edges  # self loops enabled


def test_registry_available_and_lookup(tmp_path, custom_csv, gene_names):
    names = available_graphs()
    for expected in ("custom", "go_bp", "reactome", "string", "gears"):
        assert expected in names
    src = get_graph_source("custom", cache_dir=tmp_path, path=str(custom_csv))
    assert src.name == "custom"


def test_unknown_backend_raises_keyerror():
    with pytest.raises(KeyError):
        get_graph_source("nope")


def test_isolated_nodes_become_nodes(tmp_path, gene_names):
    # An empty edge list still yields all genes as (self-looped) nodes.
    csv = tmp_path / "empty.csv"
    csv.write_text("source,target\n", encoding="utf-8")
    src = get_graph_source("custom", cache_dir=tmp_path, path=str(csv))
    g = src.build(gene_names, use_cache=False)
    assert g.num_nodes == len(gene_names)
