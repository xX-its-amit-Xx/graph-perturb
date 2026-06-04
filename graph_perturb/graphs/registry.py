"""Backend registry + cache-dir resolution.

Graph backends register themselves by name so hydra configs can select one with
a plain string (``graph.name: go_bp``). The cache dir defaults to the user's
home (``~/.graph_perturb_cache``) rather than the repo, because raw GO/Reactome/
STRING dumps are large and the repo may live on a small volume. Override with the
``GRAPH_PERTURB_CACHE`` env var or by passing ``cache_dir=`` explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Type

from .base import GraphSource

_REGISTRY: dict[str, Type[GraphSource]] = {}


def default_cache_dir() -> Path:
    env = os.environ.get("GRAPH_PERTURB_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".graph_perturb_cache"


def register_graph(name: str) -> Callable[[Type[GraphSource]], Type[GraphSource]]:
    def deco(cls: Type[GraphSource]) -> Type[GraphSource]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _ensure_backends_imported() -> None:
    """Import backend modules so their @register_graph decorators run.

    Registration is a side effect of importing each backend module; we trigger
    it lazily (here and in :func:`get_graph_source`) to avoid an import cycle at
    package-load time.
    """
    from . import gene_ontology, reactome, string_ppi, gears_adapter, custom  # noqa: F401


def available_graphs() -> list[str]:
    _ensure_backends_imported()
    return sorted(_REGISTRY)


def get_graph_source(name: str, **kwargs) -> GraphSource:
    """Instantiate a registered backend by name.

    Raises a helpful error listing available backends on an unknown name.
    """
    # Import side-effect modules lazily so registration happens on first use
    # without creating import cycles at package import time.
    _ensure_backends_imported()

    if name not in _REGISTRY:
        raise KeyError(
            f"unknown graph backend {name!r}; available: {available_graphs()}"
        )
    return _REGISTRY[name](**kwargs)
