"""Hydra structured configs (dataclasses) for graph-perturb.

These dataclasses define the schema that the YAML files in ``configs/`` fill in.
Using structured configs gives us validation and IDE completion while still
allowing full command-line override (``graph=reactome model=vae train.epochs=5``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GraphConfig:
    name: str = "go_bp"            # registry key: go_bp | reactome | string | gears | custom
    cache_dir: Optional[str] = None
    undirected: bool = True
    add_self_loops: bool = True
    # backend-specific knobs (e.g. STRING score threshold, GO evidence codes,
    # GEARS/custom file paths) are passed through verbatim.
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    name: str = "norman"
    data_dir: Optional[str] = None        # defaults to cache dir / "norman"
    n_top_genes: int = 2000               # HVG selection; bounds the node set
    include_perturbation_genes: bool = True  # always keep targeted genes as nodes
    min_cells_per_condition: int = 20
    seed: int = 0


@dataclass
class SplitConfig:
    # Held-out test sets per the spec: unseen single perturbations AND unseen
    # combinations are evaluated separately.
    val_frac: float = 0.1
    test_single_frac: float = 0.2     # fraction of single-pert conditions held out
    test_combo_frac: float = 0.3      # fraction of combinatorial conditions held out
    seed: int = 0


@dataclass
class ModelConfig:
    name: str = "gnn"                 # gnn | vae
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    # GNN-specific
    sage_aggr: str = "mean"
    attention_heads: int = 4
    # VAE-specific
    latent_dim: int = 64
    kl_weight: float = 1e-3
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 0
    device: str = "auto"              # auto | cpu | cuda
    ckpt_dir: str = "checkpoints"
    log_every: int = 10
    seed: int = 0


@dataclass
class EvalConfig:
    overlap_k: int = 20
    splits: tuple[str, ...] = ("test_single", "test_combo")


@dataclass
class Config:
    graph: GraphConfig = field(default_factory=GraphConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "outputs"
    experiment_name: str = "default"
