"""Perturbation-prediction metrics.

All metrics operate on *deltas* (post-perturbation expression minus control
mean), since that is what the literature reports and what is biologically
meaningful. Inputs are ``[n_conditions, n_genes]`` arrays where each row is the
mean predicted/true delta for one perturbation condition.

Reported metrics
----------------
* ``pearson_delta``      - mean per-condition Pearson r across genes
* ``pearson_delta_genes``- mean per-gene Pearson r across conditions (the
                            "per-gene Pearson on the delta" from the spec)
* ``mse``                - mean squared error on the delta
* ``overlap_at_k``       - top-k differentially-expressed gene recovery (Jaccard
                            -free overlap fraction), default k=20
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _as2d(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    return a


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two 1-D vectors; 0.0 if either is constant."""
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def pearson_per_condition(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred, true = _as2d(pred), _as2d(true)
    return np.array([_safe_pearson(pred[i], true[i]) for i in range(pred.shape[0])])


def pearson_per_gene(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred, true = _as2d(pred), _as2d(true)
    if pred.shape[0] < 2:
        # Not enough conditions to correlate across; fall back to per-condition.
        return pearson_per_condition(pred, true)
    return np.array([_safe_pearson(pred[:, j], true[:, j]) for j in range(pred.shape[1])])


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    pred, true = _as2d(pred), _as2d(true)
    return float(np.mean((pred - true) ** 2))


def overlap_at_k(pred: np.ndarray, true: np.ndarray, k: int = 20) -> float:
    """Mean fraction of the true top-k DE genes (by |delta|) recovered in pred top-k."""
    pred, true = _as2d(pred), _as2d(true)
    n_genes = true.shape[1]
    k = min(k, n_genes)
    if k == 0:
        return 0.0
    overlaps = []
    for i in range(true.shape[0]):
        true_top = set(np.argsort(np.abs(true[i]))[-k:].tolist())
        pred_top = set(np.argsort(np.abs(pred[i]))[-k:].tolist())
        overlaps.append(len(true_top & pred_top) / k)
    return float(np.mean(overlaps))


@dataclass
class PerturbationMetrics:
    pearson_delta: float
    pearson_delta_genes: float
    mse: float
    overlap_at_20: float
    n_conditions: int

    def as_dict(self) -> dict:
        return asdict(self)


def compute_metrics(pred: np.ndarray, true: np.ndarray, k: int = 20) -> PerturbationMetrics:
    """Compute the full metric suite for predicted vs. true mean deltas."""
    pred, true = _as2d(pred), _as2d(true)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs true {true.shape}")
    return PerturbationMetrics(
        pearson_delta=float(np.mean(pearson_per_condition(pred, true))),
        pearson_delta_genes=float(np.mean(pearson_per_gene(pred, true))),
        mse=mse(pred, true),
        overlap_at_20=overlap_at_k(pred, true, k=k),
        n_conditions=int(pred.shape[0]),
    )
