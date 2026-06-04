"""Model-agnostic evaluation for graph-perturb.

Evaluation mirrors how the perturbation-prediction literature reports results:
for every held-out condition we compare the model's **predicted mean delta**
(averaged over that condition's cells) against the **true mean delta**
(``ProcessedPerturbData.delta``), then summarize with
:func:`graph_perturb.metrics.compute_metrics`.

The same code path serves the GNN and the VAE because it only relies on the
shared :class:`~graph_perturb.models.base.PerturbationModel` ``forward`` returning
a dense ``[B, num_genes]`` delta and on the data-layer
:class:`~graph_perturb.data.PerturbationDataset` yielding one PyG ``Data`` per
condition cell.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from .config import EvalConfig
from .metrics import PerturbationMetrics, compute_metrics
from .train import resolve_device

logger = logging.getLogger(__name__)

_METRIC_COLUMNS = [
    "pearson_delta",
    "pearson_delta_genes",
    "mse",
    "overlap_at_20",
    "n_conditions",
]


def _condition_labels_for_split(splits, split_name: str) -> list[str]:
    """Pull the list of condition labels for ``split_name`` off a splits object."""
    if hasattr(splits, split_name):
        return list(getattr(splits, split_name))
    as_dict = splits.as_dict() if hasattr(splits, "as_dict") else dict(splits)
    if split_name not in as_dict:
        raise KeyError(
            f"unknown split {split_name!r}; available: {sorted(as_dict)}"
        )
    return list(as_dict[split_name])


def _batch_conditions(batch, fallback: list[str], start: int) -> list[str]:
    """Recover the per-sample condition labels for one batch.

    Prefers an explicit ``batch.condition`` attribute (PyG collates per-sample
    Python attributes into a list); otherwise falls back to positional slicing
    of ``fallback`` using ``start`` and the batch size (valid only when the
    loader is unshuffled, which evaluation enforces).
    """
    cond = getattr(batch, "condition", None)
    if cond is not None:
        if isinstance(cond, str):
            return [cond]
        # Each sample stores its label as a length-1 list (``data.condition =
        # [condition]``); PyG collates these into a list of single-element lists.
        # Unwrap one level so we recover flat string labels.
        out: list[str] = []
        for c in cond:
            if isinstance(c, (list, tuple)):
                out.extend(str(x) for x in c)
            else:
                out.append(str(c))
        return out
    bs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else 1
    return fallback[start : start + bs]


@torch.no_grad()
def _predicted_mean_delta(
    model,
    data,
    graph,
    conditions: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, list[str]]:
    """Run ``model`` over each condition's cells and average to ``[n_cond, G]``.

    Returns the prediction matrix (rows aligned to the returned condition order)
    and that condition order. Conditions absent from ``data`` are skipped.
    """
    from torch_geometric.loader import DataLoader

    from .data import PerturbationDataset

    present = [c for c in conditions if np.any(data.conditions == c)]
    missing = [c for c in conditions if c not in present]
    if missing:
        logger.warning("skipping %d conditions absent from data: %s", len(missing), missing)
    if not present:
        return np.empty((0, data.n_genes), dtype=np.float64), []

    dataset = PerturbationDataset(data, graph, present)
    sample_conditions = list(getattr(dataset, "conditions", []))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sums: dict[str, np.ndarray] = {c: np.zeros(data.n_genes, dtype=np.float64) for c in present}
    counts: dict[str, int] = {c: 0 for c in present}

    model.eval()
    seen = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)  # [B, G]
        pred_np = pred.detach().cpu().numpy().astype(np.float64)
        batch_conds = _batch_conditions(batch, sample_conditions, seen)
        for row, cond in zip(pred_np, batch_conds):
            sums[cond] += row
            counts[cond] += 1
        seen += pred_np.shape[0]

    order = [c for c in present if counts[c] > 0]
    pred_matrix = np.stack([sums[c] / counts[c] for c in order], axis=0)
    return pred_matrix, order


def _true_mean_delta(data, conditions: Iterable[str]) -> np.ndarray:
    """Stack true mean deltas (``data.delta``) for ``conditions`` into ``[n, G]``."""
    return np.stack([np.asarray(data.delta(c), dtype=np.float64) for c in conditions], axis=0)


def evaluate_model(
    model,
    data,
    graph,
    splits,
    eval_cfg: EvalConfig,
) -> dict[str, PerturbationMetrics]:
    """Compute metrics for each split in ``eval_cfg.splits``.

    For every split we form the predicted and true ``[n_conditions, n_genes]``
    mean-delta matrices and pass them to :func:`compute_metrics` with
    ``k = eval_cfg.overlap_k``. Splits that contain no usable conditions are
    skipped (with a warning) rather than producing a degenerate metric.
    """
    from .config import TrainConfig

    device = resolve_device(getattr(model, "_train_cfg", TrainConfig()))
    model.to(device)

    results: dict[str, PerturbationMetrics] = {}
    for split_name in eval_cfg.splits:
        conditions = _condition_labels_for_split(splits, split_name)
        if not conditions:
            logger.warning("split %r is empty; skipping", split_name)
            continue

        pred, order = _predicted_mean_delta(
            model, data, graph, conditions, device, batch_size=64
        )
        if pred.shape[0] == 0:
            logger.warning("no predictable conditions in split %r; skipping", split_name)
            continue

        true = _true_mean_delta(data, order)
        results[split_name] = compute_metrics(pred, true, k=eval_cfg.overlap_k)
        logger.info(
            "split %s: pearson_delta=%.4f mse=%.4g overlap@%d=%.3f (%d conditions)",
            split_name,
            results[split_name].pearson_delta,
            results[split_name].mse,
            eval_cfg.overlap_k,
            results[split_name].overlap_at_20,
            results[split_name].n_conditions,
        )
    return results


def metrics_table(results: Mapping[str, PerturbationMetrics]) -> str:
    """Render ``{split -> PerturbationMetrics}`` as a fixed-width table string."""
    if not results:
        return "(no results)"
    rows = {name: m.as_dict() for name, m in results.items()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df = df.reindex(columns=_METRIC_COLUMNS)
    df.index.name = "split"
    return df.to_string(float_format=lambda v: f"{v:.4f}")


def _results_to_frame(results: Mapping[str, PerturbationMetrics]) -> pd.DataFrame:
    rows = {name: m.as_dict() for name, m in results.items()}
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=_METRIC_COLUMNS)
    df.index.name = "split"
    return df


def compare_backends(
    results_by_backend: Mapping[str, Mapping[str, PerturbationMetrics]],
) -> pd.DataFrame:
    """Combine ``{backend_label -> {split -> metrics}}`` into one tidy DataFrame.

    Used to build the README's "graph backend swap" comparison table. The result
    has a (backend, split) MultiIndex and one column per metric.
    """
    return _stack_labeled_results(results_by_backend, label_name="backend")


def compare_models(
    results_by_model: Mapping[str, Mapping[str, PerturbationMetrics]],
) -> pd.DataFrame:
    """Combine ``{model_label -> {split -> metrics}}`` into one tidy DataFrame.

    Used to build the README's GNN-vs-VAE comparison table. The result has a
    (model, split) MultiIndex and one column per metric.
    """
    return _stack_labeled_results(results_by_model, label_name="model")


def _stack_labeled_results(
    labeled: Mapping[str, Mapping[str, PerturbationMetrics]],
    label_name: str,
) -> pd.DataFrame:
    frames = []
    for label, results in labeled.items():
        df = _results_to_frame(results).reset_index()
        df.insert(0, label_name, label)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=[label_name, "split", *_METRIC_COLUMNS])
    combined = pd.concat(frames, ignore_index=True)
    return combined.set_index([label_name, "split"])
