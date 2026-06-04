"""graph-perturb: GNN perturbation-response prediction over a swappable knowledge graph."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config
from .graphs.base import GraphSource, PerturbationGraph
from .graphs.registry import available_graphs, get_graph_source, register_graph
from .metrics import PerturbationMetrics, compute_metrics

__all__ = [
    "__version__",
    "Config",
    "GraphSource",
    "PerturbationGraph",
    "get_graph_source",
    "available_graphs",
    "register_graph",
    "compute_metrics",
    "PerturbationMetrics",
]
