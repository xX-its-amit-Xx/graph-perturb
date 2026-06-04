"""Tests for the data layer: synthetic loader + AnnData round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from graph_perturb.data import anndata_to_processed, processed_to_anndata
from graph_perturb.data.norman import make_synthetic_norman


def test_synthetic_invariants(processed):
    d = processed
    assert d.expression.dtype == np.float32
    assert d.expression.shape == (d.n_cells, d.n_genes)
    assert d.conditions.shape[0] == d.n_cells
    assert d.control_mean.shape == (d.n_genes,)
    # ctrl must be present and excluded from condition_labels by default.
    assert "ctrl" in set(d.conditions.tolist())
    assert "ctrl" not in d.condition_labels()
    assert len(d.gene_names) == len(set(d.gene_names))


def test_synthetic_has_singles_and_combos(processed):
    labels = processed.condition_labels()
    assert any(not processed.is_combo(c) for c in labels)
    assert any(processed.is_combo(c) for c in labels)


def test_synthetic_deltas_finite(processed):
    for c in processed.condition_labels():
        delta = processed.delta(c)
        assert delta.shape == (processed.n_genes,)
        assert np.all(np.isfinite(delta))


def test_perturbed_gene_map_consistent(processed):
    for c in processed.condition_labels():
        genes = processed.perturbed_genes_per_condition[c]
        assert all(g in processed.gene_names for g in genes)
        assert len(genes) == (2 if processed.is_combo(c) else 1)


def test_control_mean_matches_control_cells(processed):
    mask = processed.conditions == "ctrl"
    expected = processed.expression[mask].mean(axis=0)
    assert np.allclose(processed.control_mean, expected, atol=1e-5)


def test_anndata_roundtrip_preserves(processed):
    adata = processed_to_anndata(processed)
    back = anndata_to_processed(adata)
    assert back.gene_names == processed.gene_names
    assert np.allclose(back.expression, processed.expression, atol=1e-5)
    assert np.array_equal(
        np.asarray(back.conditions, dtype=object), np.asarray(processed.conditions, dtype=object)
    )
    assert np.allclose(back.control_mean, processed.control_mean, atol=1e-5)
    for cond, genes in processed.perturbed_genes_per_condition.items():
        assert sorted(back.perturbed_genes_per_condition[cond]) == sorted(genes)


def test_anndata_hvg_path_keeps_perturbation_genes():
    d = make_synthetic_norman(n_genes=40, n_conditions=16, seed=1)
    adata = processed_to_anndata(d)
    n_top = 15
    back = anndata_to_processed(adata, n_top_genes=n_top)
    assert len(back.gene_names) <= d.n_genes
    # All perturbation-target genes must survive HVG subsetting.
    targeted = {g for genes in d.perturbed_genes_per_condition.values() for g in genes}
    for g in targeted:
        assert g in back.gene_names
    assert back.expression.shape[1] == len(back.gene_names)


def test_anndata_missing_condition_key_raises(processed):
    adata = processed_to_anndata(processed)
    with pytest.raises(KeyError):
        anndata_to_processed(adata, condition_key="does_not_exist")
