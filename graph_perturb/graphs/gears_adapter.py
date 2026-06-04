"""GEARS-format gene knowledge graph backend.

`GEARS <https://github.com/snap-stanford/GEARS>`_ (Roohani et al.) drives its
perturbation GNN with a gene-gene knowledge graph -- either a Gene Ontology
co-function graph or a co-expression graph derived from the training data. In
the wild that graph ships in one of a few interchangeable serializations, all
of which this adapter understands:

(a) ``networkx`` -- a pickled :class:`networkx.Graph` / ``.gpickle`` whose nodes
    are gene symbols (or ids resolvable via ``node_map``) and whose edges may
    carry an ``"importance"`` / ``"weight"`` attribute.
(b) ``edgelist`` -- a CSV/TSV co-expression edge list (GEARS' ``gene_path``
    style) with columns ``source,target`` (aliases ``gene1/gene2``,
    ``source_gene/target_gene``) and an optional ``importance``/``weight``.
(c) ``pyg`` -- a torch / numpy bundle (``.pt`` or ``.npz``) holding an
    ``edge_index`` ``[2, E]`` integer array plus a ``node_map`` mapping integer
    node ids to gene symbols (and optionally ``edge_weight``/``edge_attr``).

The serialization is chosen by the ``format`` option (``"auto"`` infers it from
the file extension). Integer node ids are translated to gene symbols through an
optional ``node_map`` (an inline ``dict`` or a path to a JSON/CSV mapping); for
the ``pyg`` format the embedded ``node_map`` is used when none is supplied.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from .base import GraphSource, PerturbationGraph
from .registry import register_graph

logger = logging.getLogger(__name__)

_SOURCE_ALIASES = ("source", "gene1", "source_gene", "from", "node1", "tf")
_TARGET_ALIASES = ("target", "gene2", "target_gene", "to", "node2", "gene")
_WEIGHT_ALIASES = ("weight", "importance", "score", "coexpression", "value")


@register_graph("gears")
class GearsGraph(GraphSource):
    """Ingest a GEARS-format gene-coexpression / GO graph.

    Options
    -------
    path:
        Required. Path to the GEARS graph file.
    format:
        One of ``"auto"`` (default), ``"networkx"``, ``"edgelist"``, ``"pyg"``.
        ``"auto"`` infers from the extension: ``.pkl``/``.pickle``/``.gpickle``
        -> networkx, ``.csv``/``.tsv``/``.tab`` -> edgelist,
        ``.pt``/``.pth``/``.npz`` -> pyg.
    node_map:
        Optional mapping from raw node id (often an integer index) to gene
        symbol. Either an inline ``dict`` or a path to a ``.json`` / ``.csv``
        file (a two-column ``id,gene`` CSV, or a JSON object). For ``pyg`` files
        an embedded ``node_map`` is used when this option is omitted.
    """

    def _resolve_format(self, path: Path) -> str:
        fmt = self.options.get("format", "auto")
        if fmt != "auto":
            return fmt
        ext = path.suffix.lower()
        if ext in {".pkl", ".pickle", ".gpickle"}:
            return "networkx"
        if ext in {".csv", ".tsv", ".tab"}:
            return "edgelist"
        if ext in {".pt", ".pth", ".npz"}:
            return "pyg"
        raise ValueError(
            f"cannot infer GEARS graph format from extension {ext!r} for {path}; "
            "pass format= explicitly (networkx|edgelist|pyg)"
        )

    def _load_node_map(self) -> dict | None:
        node_map = self.options.get("node_map")
        if node_map is None:
            return None
        if isinstance(node_map, dict):
            return {str(k): str(v) for k, v in node_map.items()}
        nm_path = Path(node_map).expanduser()
        if not nm_path.exists():
            raise FileNotFoundError(f"node_map file not found: {nm_path}")
        if nm_path.suffix.lower() == ".json":
            with nm_path.open() as fh:
                raw = json.load(fh)
            return {str(k): str(v) for k, v in raw.items()}
        sep = "\t" if nm_path.suffix.lower() in {".tsv", ".tab"} else ","
        df = pd.read_csv(nm_path, sep=sep, header=None)
        return {str(k): str(v) for k, v in zip(df.iloc[:, 0], df.iloc[:, 1])}

    @staticmethod
    def _pick(columns, aliases, default):
        lower = {str(c).lower(): c for c in columns}
        for a in aliases:
            if a in lower:
                return lower[a]
        return default

    def _load_networkx(
        self, path: Path
    ) -> tuple[list[tuple[str, str]], list[float] | None]:
        import networkx as nx  # local import: optional heavy dep

        try:
            graph = nx.read_gpickle(path)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - read_gpickle removed in nx>=3; raw-pickle fallback
            with path.open("rb") as fh:
                graph = pickle.load(fh)
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"expected a networkx Graph in {path}, got {type(graph)!r}")

        edges: list[tuple[str, str]] = []
        weights: list[float] = []
        any_weight = False
        for a, b, attrs in graph.edges(data=True):
            edges.append((str(a), str(b)))
            w = None
            for key in _WEIGHT_ALIASES:
                if key in attrs:
                    w = float(attrs[key])
                    any_weight = True
                    break
            weights.append(1.0 if w is None else w)
        return edges, (weights if any_weight else None)

    def _load_edgelist(
        self, path: Path
    ) -> tuple[list[tuple[str, str]], list[float] | None]:
        sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        df = pd.read_csv(path, sep=sep)
        src_col = self._pick(df.columns, _SOURCE_ALIASES, None)
        tgt_col = self._pick(df.columns, _TARGET_ALIASES, None)
        if src_col is None or tgt_col is None:
            raise ValueError(
                f"could not find source/target columns in {path} "
                f"(columns: {list(df.columns)}); expected one of "
                f"{_SOURCE_ALIASES} and {_TARGET_ALIASES}"
            )
        wt_col = self._pick(df.columns, _WEIGHT_ALIASES, None)
        df = df.dropna(subset=[src_col, tgt_col])
        edges = list(zip(df[src_col].astype(str), df[tgt_col].astype(str)))
        weights = [float(w) for w in df[wt_col].tolist()] if wt_col is not None else None
        return edges, weights

    def _load_pyg(
        self, path: Path
    ) -> tuple[list[tuple[str, str]], list[float] | None]:
        ext = path.suffix.lower()
        if ext == ".npz":
            import numpy as np

            blob = dict(np.load(path, allow_pickle=True))
            edge_index = blob["edge_index"]
            node_map = blob.get("node_map")
            if node_map is not None:
                node_map = node_map.item() if hasattr(node_map, "item") else node_map
            edge_weight = blob.get("edge_weight")
        else:
            import torch

            blob = torch.load(path, weights_only=False)
            edge_index = blob["edge_index"]
            node_map = blob.get("node_map")
            edge_weight = blob.get("edge_weight", blob.get("edge_attr"))
            if hasattr(edge_index, "tolist"):
                edge_index = edge_index.tolist()
            if edge_weight is not None and hasattr(edge_weight, "tolist"):
                edge_weight = edge_weight.tolist()

        # node_map may map id->symbol or symbol->id; normalise to id->symbol.
        external = self._load_node_map()
        if external is not None:
            node_map = external
        id_to_gene: dict[str, str] | None = None
        if node_map is not None:
            keys = list(node_map.keys())
            looks_id_keyed = all(str(k).lstrip("-").isdigit() for k in keys) if keys else False
            if looks_id_keyed:
                id_to_gene = {str(k): str(v) for k, v in node_map.items()}
            else:
                id_to_gene = {str(v): str(k) for k, v in node_map.items()}

        src_row, dst_row = edge_index[0], edge_index[1]

        def _name(n) -> str:
            key = str(int(n)) if not isinstance(n, str) else n
            if id_to_gene is not None and key in id_to_gene:
                return id_to_gene[key]
            return key

        edges = [(_name(s), _name(d)) for s, d in zip(src_row, dst_row)]
        weights = (
            [float(w) for w in edge_weight] if edge_weight is not None else None
        )
        if weights is not None and len(weights) != len(edges):
            logger.warning(
                "pyg edge_weight length %d != n_edges %d; dropping weights",
                len(weights),
                len(edges),
            )
            weights = None
        return edges, weights

    def _load_edges(
        self, genes: set[str] | None
    ) -> tuple[list[tuple[str, str]], list[float] | None]:
        path = self.options.get("path")
        if path is None:
            raise ValueError(
                "GearsGraph requires a 'path' option pointing at a GEARS graph "
                "file (got path=None)."
            )
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"GEARS graph file not found: {path}")

        fmt = self._resolve_format(path)
        if fmt == "networkx":
            edges, weights = self._load_networkx(path)
        elif fmt == "edgelist":
            edges, weights = self._load_edgelist(path)
        elif fmt == "pyg":
            edges, weights = self._load_pyg(path)
        else:
            raise ValueError(
                f"unknown GEARS graph format {fmt!r}; expected one of "
                "networkx|edgelist|pyg"
            )

        # node_map translation for networkx/edgelist (pyg handles its own).
        if fmt != "pyg":
            external = self._load_node_map()
            if external is not None:
                edges = [
                    (external.get(a, a), external.get(b, b)) for a, b in edges
                ]

        if genes is not None:
            kept_edges: list[tuple[str, str]] = []
            kept_weights: list[float] = []
            for k, (a, b) in enumerate(edges):
                if a in genes and b in genes:
                    kept_edges.append((a, b))
                    if weights is not None:
                        kept_weights.append(weights[k])
            edges = kept_edges
            weights = kept_weights if weights is not None else None

        logger.info(
            "loaded %d GEARS edges from %s (format=%s, weighted=%s)",
            len(edges),
            path,
            fmt,
            weights is not None,
        )
        return edges, weights


def load_gears_graph(path: str | Path, **opts) -> PerturbationGraph:
    """Convenience wrapper used by the cookbook: load a GEARS graph by path.

    When no explicit ``genes`` universe is supplied, the graph is built over the
    GEARS file's *own* node universe (every gene symbol appearing as an edge
    endpoint), so the returned :class:`PerturbationGraph` is self-contained.

    Parameters
    ----------
    path:
        Path to the GEARS graph file.
    **opts:
        Forwarded to :class:`GearsGraph`. Recognised extras consumed here:
        ``genes`` (an explicit gene universe), ``cache_dir``, ``undirected``,
        ``add_self_loops``, ``use_cache``; everything else (``format``,
        ``node_map`` ...) flows through as backend options.
    """
    genes = opts.pop("genes", None)
    cache_dir = opts.pop("cache_dir", None)
    undirected = opts.pop("undirected", True)
    add_self_loops = opts.pop("add_self_loops", True)
    use_cache = opts.pop("use_cache", True)

    source = GearsGraph(
        cache_dir=cache_dir,
        undirected=undirected,
        add_self_loops=add_self_loops,
        path=path,
        **opts,
    )

    if genes is None:
        edges, _ = source._load_edges(None)
        seen: dict[str, None] = {}
        for a, b in edges:
            seen.setdefault(a, None)
            seen.setdefault(b, None)
        genes = list(seen)
        logger.info("load_gears_graph: derived %d-gene universe from graph", len(genes))

    return source.build(genes, use_cache=use_cache)
