"""Model-agnostic training loop for graph-perturb.

The same loop trains the GraphSAGE GNN and the conditional VAE because both
conform to :class:`graph_perturb.models.base.PerturbationModel`: training only
ever calls ``model.loss(batch) -> (scalar, logs)`` and ``model.forward`` is left
to evaluation. Optimization is plain Adam with gradient clipping, mean
validation-loss early stopping, and best-checkpointing of the model
``state_dict`` together with the resolved :class:`~graph_perturb.config.Config`.

``run_training`` is the importable, side-effect-light entry point that wires the
whole pipeline (graph -> data -> splits -> dataset/loaders -> model -> train).
``cli.py`` is responsible for argument parsing; nothing here touches argv.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from .config import Config, TrainConfig

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and Torch RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(train_cfg: TrainConfig) -> torch.device:
    """Resolve ``train_cfg.device`` (``auto`` | ``cpu`` | ``cuda``) to a device.

    ``auto`` selects CUDA when available, otherwise CPU. An explicit ``cuda``
    request on a machine without CUDA falls back to CPU with a warning rather
    than crashing, so configs are portable across the CPU-only research box and
    a GPU node.
    """
    want = (train_cfg.device or "auto").lower()
    if want == "cpu":
        return torch.device("cpu")
    if want == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("device='cuda' requested but CUDA unavailable; using CPU")
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch(batch: Any, device: torch.device) -> Any:
    """Move a PyG ``Batch`` (or any object exposing ``.to``) onto ``device``."""
    to = getattr(batch, "to", None)
    if callable(to):
        return batch.to(device)
    return batch


@torch.no_grad()
def _evaluate_loss(model, loader, device: torch.device) -> tuple[float, dict[str, float]]:
    """Mean loss (and mean of each log component) over ``loader``.

    Batches are weighted by their sample count so the reported mean is the true
    per-sample average even when the final batch is smaller.
    """
    model.eval()
    total = 0.0
    total_logs: dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        bs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else 1
        loss, logs = model.loss(batch)
        total += float(loss.detach()) * bs
        for key, val in logs.items():
            total_logs[key] = total_logs.get(key, 0.0) + float(val) * bs
        n += bs
    if n == 0:
        return float("nan"), {}
    return total / n, {k: v / n for k, v in total_logs.items()}


def train_model(
    model,
    loaders: Mapping[str, Any],
    train_cfg: TrainConfig,
    ckpt_dir: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Train ``model`` on ``loaders['train']``, validating on ``loaders['val']``.

    Parameters
    ----------
    model:
        Any :class:`~graph_perturb.models.base.PerturbationModel`.
    loaders:
        Mapping with at least ``"train"`` and ``"val"`` PyG dataloaders.
    train_cfg:
        Optimization hyper-parameters (lr, weight decay, epochs, grad clip,
        early-stopping patience/min-delta, logging cadence, seed, device).
    ckpt_dir:
        Directory to write ``best.pt`` (state_dict + cfg). When ``None`` the best
        weights are still tracked in memory and restored at the end, but nothing
        is written to disk.

    Returns
    -------
    dict
        ``{"best_val": float, "best_epoch": int, "history": [...], "ckpt_path": str|None}``
        where each history entry is ``{"epoch", "train_loss", "val_loss", ...}``.
    """
    set_seed(train_cfg.seed)
    device = resolve_device(train_cfg)
    model.to(device)

    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    ckpt_path: Optional[Path] = None
    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "best.pt"

    best_val = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    history: list[dict[str, float]] = []

    logger.info(
        "training on %s for up to %d epochs (lr=%g, wd=%g, batch grad_clip=%g)",
        device,
        train_cfg.epochs,
        train_cfg.lr,
        train_cfg.weight_decay,
        train_cfg.grad_clip,
    )

    for epoch in range(train_cfg.epochs):
        model.train()
        running = 0.0
        running_logs: dict[str, float] = {}
        n_samples = 0
        for step, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            bs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else 1

            optimizer.zero_grad(set_to_none=True)
            loss, logs = model.loss(batch)
            loss.backward()
            if train_cfg.grad_clip and train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            running += float(loss.detach()) * bs
            for key, val in logs.items():
                running_logs[key] = running_logs.get(key, 0.0) + float(val) * bs
            n_samples += bs

            if train_cfg.log_every and (step % train_cfg.log_every == 0):
                logger.info(
                    "epoch %d step %d: loss=%.4f", epoch, step, float(loss.detach())
                )

        train_loss = running / max(1, n_samples)
        train_logs = {k: v / max(1, n_samples) for k, v in running_logs.items()}

        if val_loader is not None:
            val_loss, val_logs = _evaluate_loss(model, val_loader, device)
        else:
            val_loss, val_logs = train_loss, {}

        record: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        }
        record.update({f"train_{k}": v for k, v in train_logs.items()})
        record.update({f"val_{k}": v for k, v in val_logs.items()})
        history.append(record)
        logger.info(
            "epoch %d: train_loss=%.4f val_loss=%.4f", epoch, train_loss, val_loss
        )

        # Early stopping + best checkpointing on validation loss.
        if val_loss < best_val - train_cfg.early_stop_min_delta:
            best_val = float(val_loss)
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                _save_checkpoint(ckpt_path, model, train_cfg, epoch, best_val)
                logger.info("saved best checkpoint to %s (val_loss=%.4f)", ckpt_path, best_val)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= train_cfg.early_stop_patience:
                logger.info(
                    "early stopping at epoch %d (no val improvement for %d epochs)",
                    epoch,
                    epochs_no_improve,
                )
                break

    # Restore best weights so the returned model is the one we report on.
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = len(history) - 1
        best_val = history[-1]["val_loss"] if history else float("nan")

    return {
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "history": history,
        "ckpt_path": str(ckpt_path) if ckpt_path is not None else None,
    }


def _save_checkpoint(
    path: str | Path,
    model,
    train_cfg: TrainConfig,
    epoch: int,
    val_loss: float,
) -> None:
    """Persist the model ``state_dict`` plus enough metadata to reload it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": train_cfg,
            "num_genes": getattr(model, "num_genes", None),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(path: str | Path, model):
    """Load a ``best.pt`` checkpoint into ``model`` and return ``(model, blob)``.

    The model architecture must already match the saved one (build it via
    :func:`graph_perturb.models.build_model` with the same config). Returns the
    model with weights loaded plus the raw checkpoint dict for inspection.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(state)
    return model, blob


def run_training(cfg: Config) -> tuple[Any, Any, Any, Any, Mapping[str, Any], dict[str, Any]]:
    """Wire the full pipeline and train, returning every stage's artifact.

    Steps
    -----
    1. Load processed perturbation data (Norman by default).
    2. Build condition-level train/val/test splits.
    3. Build the knowledge graph over the data's gene universe (graph models
       only; the VAE is graph-free).
    4. Build PyG dataloaders.
    5. Build and train the model.

    Returns
    -------
    ``(model, graph, data, splits, loaders, history)``
        ``graph`` is ``None`` for graph-free models. ``history`` is the dict
        returned by :func:`train_model`.
    """
    from .data import build_dataloaders, load_norman, make_splits
    from .graphs.registry import get_graph_source
    from .models import build_model

    set_seed(cfg.train.seed)

    # 1. Data ---------------------------------------------------------------
    logger.info("loading data backend %r", cfg.data.name)
    data = load_norman(cfg.data)

    # 2. Splits -------------------------------------------------------------
    splits = make_splits(data, cfg.split)
    logger.info(
        "splits: train=%d val=%d test_single=%d test_combo=%d",
        len(splits.train),
        len(splits.val),
        len(splits.test_single),
        len(splits.test_combo),
    )

    # 3. Graph (only built for graph-consuming models) ----------------------
    graph = None
    if cfg.model.name.lower() != "vae":
        logger.info("building knowledge graph backend %r", cfg.graph.name)
        source = get_graph_source(
            cfg.graph.name,
            cache_dir=cfg.graph.cache_dir,
            undirected=cfg.graph.undirected,
            add_self_loops=cfg.graph.add_self_loops,
            **cfg.graph.options,
        )
        graph = source.build(data.gene_names)

    # 4. Dataloaders --------------------------------------------------------
    loaders = build_dataloaders(data, graph, splits, cfg.train)

    # 5. Model + train ------------------------------------------------------
    model = build_model(cfg.model, num_genes=data.n_genes, graph=graph)
    history = train_model(model, loaders, cfg.train, ckpt_dir=cfg.train.ckpt_dir)

    return model, graph, data, splits, loaders, history
