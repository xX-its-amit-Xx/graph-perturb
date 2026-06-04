"""Tests for graph_perturb.metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from graph_perturb.metrics import (
    PerturbationMetrics,
    compute_metrics,
    overlap_at_k,
    pearson_per_condition,
)


def test_perfect_prediction(rng):
    true = rng.normal(size=(8, 30))
    pred = true.copy()
    m = compute_metrics(pred, true, k=20)
    assert isinstance(m, PerturbationMetrics)
    assert m.pearson_delta == pytest.approx(1.0, abs=1e-9)
    assert m.pearson_delta_genes == pytest.approx(1.0, abs=1e-9)
    assert m.mse == pytest.approx(0.0, abs=1e-12)
    assert m.overlap_at_20 == pytest.approx(1.0, abs=1e-9)
    assert m.n_conditions == 8


def test_overlap_at_20_perfect_with_k20():
    # 25 genes, distinct magnitudes -> top-20 well defined and identical.
    true = np.arange(25, dtype=float)[None, :]
    pred = true.copy()
    assert overlap_at_k(pred, true, k=20) == pytest.approx(1.0)


def test_shape_mismatch_raises():
    a = np.zeros((3, 10))
    b = np.zeros((3, 9))
    with pytest.raises(ValueError):
        compute_metrics(a, b)


def test_constant_vectors_give_zero_no_nan():
    # Constant rows -> zero variance -> safe pearson returns 0.0, never nan.
    pred = np.full((4, 12), 3.0)
    true = np.full((4, 12), -7.0)
    rs = pearson_per_condition(pred, true)
    assert rs.shape == (4,)
    assert np.all(rs == 0.0)
    m = compute_metrics(pred, true)
    assert m.pearson_delta == 0.0
    assert not math.isnan(m.pearson_delta)
    assert not math.isnan(m.pearson_delta_genes)


def test_negative_correlation_recovered():
    true = np.array([[1.0, 2.0, 3.0, 4.0]])
    pred = -true
    assert pearson_per_condition(pred, true)[0] == pytest.approx(-1.0, abs=1e-9)


def test_overlap_k_clamped_to_n_genes():
    # k larger than the gene count is clamped; perfect match still scores 1.
    true = np.array([[0.0, 1.0, 2.0]])
    assert overlap_at_k(true, true, k=20) == pytest.approx(1.0)
