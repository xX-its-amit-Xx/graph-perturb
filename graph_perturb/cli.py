"""Typer command-line interface for graph-perturb.

Exposes the package as the ``graph-perturb`` console script (see
``[project.scripts]`` in ``pyproject.toml``)::

    graph-perturb graphs
    graph-perturb build-graph --backend go_bp --synthetic
    graph-perturb train --synthetic
    graph-perturb train graph=reactome model=vae train.epochs=5
    graph-perturb evaluate --ckpt checkpoints/best.pt --synthetic
    graph-perturb compare-backends --backends go_bp,reactome,string --synthetic
    graph-perturb bakeoff --synthetic

Design notes
------------
* All heavy imports (torch, scanpy, hydra, the package's own train/eval modules)
  happen *inside* the command bodies so that ``graph-perturb --help`` stays
  instant and importing this module never pulls in the scientific stack.
* Every command has a ``--synthetic`` fast path that fabricates a tiny dataset
  via :func:`graph_perturb.data.norman.make_synthetic_norman` and uses the
  offline graph fallback, so the commands double as <1 min CPU smoke tests with
  no downloads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="graph-perturb",
    add_completion=False,
    no_args_is_help=True,
    help="GNN perturbation-response prediction over a swappable knowledge graph.",
)

logger = logging.getLogger(__name__)

# Tiny but non-trivial knobs shared by every --synthetic smoke path.
_SYNTH_GENES = 80
_SYNTH_CONDITIONS = 30
_SYNTH_EPOCHS = 3


# --------------------------------------------------------------------------- #
# helpers (kept private; all heavy imports are local to the functions)
# --------------------------------------------------------------------------- #
def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _configs_dir() -> Path:
    """Absolute path to the repo's ``configs/`` directory (sibling of the package)."""
    return Path(__file__).resolve().parent.parent / "configs"


def _compose_config(config_name: str, overrides: list[str]):
    """Compose a :class:`graph_perturb.config.Config` from the hydra ``configs/``.

    Uses :func:`hydra.initialize_config_dir` + :func:`hydra.compose` so the same
    composition works whether the package is installed or run from a checkout.
    Dotlist ``overrides`` (e.g. ``["graph=reactome", "train.epochs=5"]``) are
    applied exactly as on a hydra command line.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from .config import Config

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(_configs_dir()), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    raw = OmegaConf.to_container(cfg, resolve=True)
    return _dict_to_config(raw)


def _dict_to_config(raw: dict):
    """Map a plain composed dict onto the structured :class:`Config` dataclasses.

    We build the nested dataclasses explicitly (rather than ``OmegaConf`` schema
    merging) so the CLI works without registering structured configs, and so we
    can coerce list/tuple fields like ``eval.splits`` to the declared types.
    """
    from .config import (
        Config,
        DataConfig,
        EvalConfig,
        GraphConfig,
        ModelConfig,
        SplitConfig,
        TrainConfig,
    )

    g = raw.get("graph", {}) or {}
    d = raw.get("data", {}) or {}
    s = raw.get("split", {}) or {}
    m = raw.get("model", {}) or {}
    t = raw.get("train", {}) or {}
    e = raw.get("eval", {}) or {}

    eval_splits = e.get("splits", ("test_single", "test_combo"))
    return Config(
        graph=GraphConfig(
            name=g.get("name", "go_bp"),
            cache_dir=g.get("cache_dir"),
            undirected=bool(g.get("undirected", True)),
            add_self_loops=bool(g.get("add_self_loops", True)),
            options=dict(g.get("options", {}) or {}),
        ),
        data=DataConfig(
            name=d.get("name", "norman"),
            data_dir=d.get("data_dir"),
            n_top_genes=int(d.get("n_top_genes", 2000)),
            include_perturbation_genes=bool(d.get("include_perturbation_genes", True)),
            min_cells_per_condition=int(d.get("min_cells_per_condition", 20)),
            seed=int(d.get("seed", 0)),
        ),
        split=SplitConfig(
            val_frac=float(s.get("val_frac", 0.1)),
            test_single_frac=float(s.get("test_single_frac", 0.2)),
            test_combo_frac=float(s.get("test_combo_frac", 0.3)),
            seed=int(s.get("seed", 0)),
        ),
        model=ModelConfig(
            name=m.get("name", "gnn"),
            hidden_dim=int(m.get("hidden_dim", 128)),
            n_layers=int(m.get("n_layers", 2)),
            dropout=float(m.get("dropout", 0.1)),
            sage_aggr=m.get("sage_aggr", "mean"),
            attention_heads=int(m.get("attention_heads", 4)),
            latent_dim=int(m.get("latent_dim", 64)),
            kl_weight=float(m.get("kl_weight", 1e-3)),
            extra=dict(m.get("extra", {}) or {}),
        ),
        train=TrainConfig(
            epochs=int(t.get("epochs", 40)),
            batch_size=int(t.get("batch_size", 64)),
            lr=float(t.get("lr", 1e-3)),
            weight_decay=float(t.get("weight_decay", 1e-5)),
            early_stop_patience=int(t.get("early_stop_patience", 8)),
            early_stop_min_delta=float(t.get("early_stop_min_delta", 1e-4)),
            grad_clip=float(t.get("grad_clip", 1.0)),
            num_workers=int(t.get("num_workers", 0)),
            device=t.get("device", "auto"),
            ckpt_dir=t.get("ckpt_dir", "checkpoints"),
            log_every=int(t.get("log_every", 10)),
            seed=int(t.get("seed", 0)),
        ),
        eval=EvalConfig(
            overlap_k=int(e.get("overlap_k", 20)),
            splits=tuple(eval_splits),
        ),
        output_dir=raw.get("output_dir", "outputs"),
        experiment_name=raw.get("experiment_name", "default"),
    )


def _synthetic_config(
    *,
    graph_name: str = "go_bp",
    model_name: str = "gnn",
    epochs: int = _SYNTH_EPOCHS,
):
    """A tiny, fully-offline :class:`Config` for smoke tests."""
    from .config import Config, GraphConfig, ModelConfig, TrainConfig

    cfg = Config()
    cfg.experiment_name = f"synthetic-{graph_name}-{model_name}"
    cfg.graph = GraphConfig(name=graph_name)
    cfg.model = ModelConfig(name=model_name, hidden_dim=32, n_layers=2, latent_dim=16)
    cfg.train = TrainConfig(
        epochs=epochs,
        batch_size=16,
        early_stop_patience=epochs,  # effectively disable early stop for smoke runs
        log_every=50,
        device="cpu",
        ckpt_dir="checkpoints",
    )
    return cfg


def _synthetic_pipeline(cfg, data_dir: Optional[str] = None):
    """Build a synthetic dataset + offline graph + split + dataloaders for ``cfg``.

    Returns ``(data, graph, splits, loaders)``. Used by the ``--synthetic`` paths
    of ``train``/``compare-backends``/``bakeoff`` so they never hit the network.
    """
    from .data.dataset import build_dataloaders
    from .data.norman import make_synthetic_norman
    from .data.splits import make_splits, summarize_splits
    from .graphs.registry import get_graph_source

    data = make_synthetic_norman(
        n_genes=_SYNTH_GENES,
        n_conditions=_SYNTH_CONDITIONS,
        seed=cfg.data.seed,
    )
    source = get_graph_source(
        cfg.graph.name,
        cache_dir=cfg.graph.cache_dir or data_dir,
        undirected=cfg.graph.undirected,
        add_self_loops=cfg.graph.add_self_loops,
        **cfg.graph.options,
    )
    # use_cache=False keeps smoke runs hermetic and avoids writing cache blobs.
    graph = source.build(data.gene_names, use_cache=False)
    splits = make_splits(data, cfg.split)
    logger.info("synthetic splits: %s", summarize_splits(splits))
    loaders = build_dataloaders(data, graph, splits, cfg.train)
    return data, graph, splits, loaders


def _train_on_pipeline(cfg, data, graph, splits, loaders):
    """Build + train a model on an already-prepared pipeline; return the model."""
    from .models import build_model
    from .train import train_model

    model = build_model(cfg.model, num_genes=data.n_genes, graph=graph)
    train_model(model, loaders, cfg.train, ckpt_dir=None)
    return model


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
@app.command()
def graphs() -> None:
    """List the registered (swappable) knowledge-graph backends."""
    from .graphs.registry import available_graphs, get_graph_source

    # Touch the registry so all backends import & register themselves.
    try:
        get_graph_source("__none__")
    except KeyError:
        pass

    names = available_graphs()
    typer.echo("Available graph backends:")
    for name in names:
        typer.echo(f"  - {name}")
    if not names:
        typer.echo("  (none registered)")


@app.command("build-graph")
def build_graph(
    backend: str = typer.Option("go_bp", "--backend", "-b", help="Registry key of the graph backend."),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Cache/data dir override."),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Use a fabricated tiny dataset + offline graph fallback."
    ),
    n_top_genes: int = typer.Option(2000, "--n-top-genes", help="HVGs for the real Norman path."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
) -> None:
    """Build the chosen graph over the gene universe and print its repr + metadata."""
    _setup_logging(verbose)

    from .graphs.registry import get_graph_source

    if synthetic:
        from .data.norman import make_synthetic_norman

        data = make_synthetic_norman(n_genes=_SYNTH_GENES, n_conditions=_SYNTH_CONDITIONS)
    else:
        from .config import DataConfig
        from .data.norman import load_norman

        data = load_norman(
            DataConfig(name="norman", data_dir=data_dir, n_top_genes=n_top_genes)
        )

    source = get_graph_source(backend, cache_dir=data_dir)
    graph = source.build(data.gene_names, use_cache=not synthetic)

    typer.echo(repr(graph))
    typer.echo("metadata:")
    for k, v in graph.metadata.items():
        typer.echo(f"  {k}: {v}")


@app.command()
def train(
    overrides: list[str] = typer.Argument(
        None,
        help="Hydra dotlist overrides for the real path, e.g. 'graph=reactome' 'train.epochs=5'.",
    ),
    config_name: str = typer.Option("config", "--config-name", help="Top-level hydra config name."),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Fast offline smoke run on fabricated data (tiny epochs)."
    ),
    graph_backend: str = typer.Option("go_bp", "--graph", help="Graph backend for --synthetic."),
    model_name: str = typer.Option("gnn", "--model", help="Model (gnn|vae) for --synthetic."),
    epochs: int = typer.Option(_SYNTH_EPOCHS, "--epochs", help="Epochs for --synthetic."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
) -> None:
    """Train a model. ``--synthetic`` runs a tiny offline smoke test; otherwise hydra-composed."""
    _setup_logging(verbose)

    from .evaluate import evaluate_model, metrics_table

    if synthetic:
        cfg = _synthetic_config(graph_name=graph_backend, model_name=model_name, epochs=epochs)
        data, graph, splits, loaders = _synthetic_pipeline(cfg)
        model = _train_on_pipeline(cfg, data, graph, splits, loaders)
    else:
        from .train import run_training

        cfg = _compose_config(config_name, overrides or [])
        model, graph, data, splits, loaders, _history = run_training(cfg)

    results = evaluate_model(model, data, graph, splits, cfg.eval)
    typer.echo(f"\nFinal metrics ({cfg.experiment_name}):")
    typer.echo(metrics_table(results))


@app.command()
def evaluate(
    ckpt: Path = typer.Option(..., "--ckpt", exists=False, help="Path to a best.pt checkpoint."),
    overrides: list[str] = typer.Argument(None, help="Hydra overrides to rebuild the pipeline."),
    config_name: str = typer.Option("config", "--config-name"),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Rebuild a synthetic pipeline instead of the real one."
    ),
    graph_backend: str = typer.Option("go_bp", "--graph", help="Graph backend for --synthetic."),
    model_name: str = typer.Option("gnn", "--model", help="Model (gnn|vae) for --synthetic."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
) -> None:
    """Load a checkpoint, rebuild the matching pipeline, and print the metrics table."""
    _setup_logging(verbose)

    from .evaluate import evaluate_model, metrics_table
    from .models import build_model
    from .train import load_checkpoint

    if synthetic:
        cfg = _synthetic_config(graph_name=graph_backend, model_name=model_name)
        data, graph, splits, _loaders = _synthetic_pipeline(cfg)
    else:
        from .data.dataset import build_dataloaders
        from .data.norman import load_norman
        from .data.splits import make_splits
        from .graphs.registry import get_graph_source

        cfg = _compose_config(config_name, overrides or [])
        data = load_norman(cfg.data)
        source = get_graph_source(
            cfg.graph.name,
            cache_dir=cfg.graph.cache_dir,
            undirected=cfg.graph.undirected,
            add_self_loops=cfg.graph.add_self_loops,
            **cfg.graph.options,
        )
        graph = source.build(data.gene_names)
        splits = make_splits(data, cfg.split)
        build_dataloaders(data, graph, splits, cfg.train)  # validate shapes early

    model = build_model(cfg.model, num_genes=data.n_genes, graph=graph)
    load_checkpoint(str(ckpt), model)

    results = evaluate_model(model, data, graph, splits, cfg.eval)
    typer.echo(f"\nMetrics for {ckpt}:")
    typer.echo(metrics_table(results))


@app.command("compare-backends")
def compare_backends_cmd(
    backends: str = typer.Option(
        "go_bp,reactome,string", "--backends", help="Comma-separated backend keys."
    ),
    overrides: list[str] = typer.Argument(None, help="Hydra overrides (real path only)."),
    config_name: str = typer.Option("config", "--config-name"),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Train tiny offline models on each backend (smoke test)."
    ),
    model_name: str = typer.Option("gnn", "--model", help="Model trained on every backend."),
    epochs: int = typer.Option(_SYNTH_EPOCHS, "--epochs", help="Epochs for --synthetic."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
) -> None:
    """Train the SAME model on each backend over the SAME split; print a backend-vs-backend table."""
    _setup_logging(verbose)

    from .data.dataset import build_dataloaders
    from .evaluate import compare_backends, evaluate_model
    from .graphs.registry import get_graph_source

    names = [b.strip() for b in backends.split(",") if b.strip()]
    if not names:
        raise typer.BadParameter("no backends given")

    if synthetic:
        from .data.norman import make_synthetic_norman
        from .data.splits import make_splits

        cfg = _synthetic_config(model_name=model_name, epochs=epochs)
        data = make_synthetic_norman(
            n_genes=_SYNTH_GENES, n_conditions=_SYNTH_CONDITIONS, seed=cfg.data.seed
        )
        splits = make_splits(data, cfg.split)
    else:
        from .data.norman import load_norman
        from .data.splits import make_splits

        cfg = _compose_config(config_name, overrides or [])
        cfg.model.name = model_name
        data = load_norman(cfg.data)
        splits = make_splits(data, cfg.split)

    results_by_label: dict[str, dict] = {}
    for backend in names:
        typer.echo(f"[compare-backends] training {model_name} on {backend} ...")
        source = get_graph_source(
            backend,
            cache_dir=cfg.graph.cache_dir,
            undirected=cfg.graph.undirected,
            add_self_loops=cfg.graph.add_self_loops,
        )
        graph = source.build(data.gene_names, use_cache=not synthetic)
        loaders = build_dataloaders(data, graph, splits, cfg.train)
        model = _train_on_pipeline(cfg, data, graph, splits, loaders)
        results_by_label[backend] = evaluate_model(model, data, graph, splits, cfg.eval)

    typer.echo("\nBackend comparison:")
    typer.echo(compare_backends(results_by_label).to_string())


@app.command()
def bakeoff(
    overrides: list[str] = typer.Argument(None, help="Hydra overrides (real path only)."),
    config_name: str = typer.Option("config", "--config-name"),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Tiny offline GNN-vs-VAE smoke test."
    ),
    graph_backend: str = typer.Option("go_bp", "--graph", help="Graph backend (GNN uses it)."),
    epochs: int = typer.Option(_SYNTH_EPOCHS, "--epochs", help="Epochs for --synthetic."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
) -> None:
    """Train a GNN and a VAE on the SAME split; print a GNN-vs-VAE metrics table."""
    _setup_logging(verbose)

    from .data.dataset import build_dataloaders
    from .evaluate import compare_models, evaluate_model
    from .graphs.registry import get_graph_source

    if synthetic:
        from .data.norman import make_synthetic_norman
        from .data.splits import make_splits

        base = _synthetic_config(graph_name=graph_backend, epochs=epochs)
        data = make_synthetic_norman(
            n_genes=_SYNTH_GENES, n_conditions=_SYNTH_CONDITIONS, seed=base.data.seed
        )
        splits = make_splits(data, base.split)
        use_cache = False
    else:
        from .data.norman import load_norman
        from .data.splits import make_splits

        base = _compose_config(config_name, overrides or [])
        data = load_norman(base.data)
        splits = make_splits(data, base.split)
        use_cache = True

    source = get_graph_source(
        graph_backend,
        cache_dir=base.graph.cache_dir,
        undirected=base.graph.undirected,
        add_self_loops=base.graph.add_self_loops,
    )
    graph = source.build(data.gene_names, use_cache=use_cache)

    results_by_label: dict[str, dict] = {}
    for model_name in ("gnn", "vae"):
        typer.echo(f"[bakeoff] training {model_name} ...")
        cfg = base
        cfg.model.name = model_name
        loaders = build_dataloaders(data, graph, splits, cfg.train)
        model = _train_on_pipeline(cfg, data, graph, splits, loaders)
        results_by_label[model_name] = evaluate_model(model, data, graph, splits, cfg.eval)

    typer.echo("\nGNN vs VAE:")
    typer.echo(compare_models(results_by_label).to_string())


if __name__ == "__main__":  # pragma: no cover
    app()
