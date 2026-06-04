"""Custom user-supplied edge-list knowledge graph backend.

This is the "plug in a CUSTOM user knowledge graph" adapter referenced by the
cookbook. A user hands ``graph-perturb`` an arbitrary gene-gene edge list as a
CSV/TSV file and it is treated like any other :class:`~graph_perturb.graphs.base.GraphSource`
backend (GO, Reactome, STRING, GEARS ...).

The file is expected to have (at least) a source-gene column and a target-gene
column, with one edge per row::

    source,target,weight
    TP53,MDM2,0.9
    MYC,MAX,0.8

Column names are configurable via options. An optional weight column carries
edge weights (e.g. a confidence score); when absent the graph is unweighted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import GraphSource
from .registry import register_graph

logger = logging.getLogger(__name__)


@register_graph("custom")
class CustomEdgeListGraph(GraphSource):
    """Ingest a user knowledge graph from a CSV/TSV edge list.

    Options
    -------
    path:
        Required. Path to the edge-list file (``.csv`` or ``.tsv``; the
        separator is inferred from the extension, tab for ``.tsv``/``.tab``,
        comma otherwise).
    source_col:
        Name of the source-gene column. Default ``"source"``.
    target_col:
        Name of the target-gene column. Default ``"target"``.
    weight_col:
        Optional name of an edge-weight column. If ``None`` (default) or absent
        from the file, the graph is unweighted.
    """

    def _load_edges(
        self, genes: set[str] | None
    ) -> tuple[list[tuple[str, str]], list[float] | None]:
        path = self.options.get("path")
        if path is None:
            raise ValueError(
                "CustomEdgeListGraph requires a 'path' option pointing at a "
                "CSV/TSV edge list (got path=None)."
            )
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"custom edge-list file not found: {path}")

        source_col = self.options.get("source_col", "source")
        target_col = self.options.get("target_col", "target")
        weight_col = self.options.get("weight_col")

        sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        df = pd.read_csv(path, sep=sep)

        for col in (source_col, target_col):
            if col not in df.columns:
                raise ValueError(
                    f"column {col!r} not found in {path} (columns: {list(df.columns)})"
                )
        if weight_col is not None and weight_col not in df.columns:
            raise ValueError(
                f"weight column {weight_col!r} not found in {path} "
                f"(columns: {list(df.columns)})"
            )

        cols = [source_col, target_col]
        if weight_col is not None:
            cols.append(weight_col)
        df = df[cols].dropna(subset=[source_col, target_col])
        df[source_col] = df[source_col].astype(str)
        df[target_col] = df[target_col].astype(str)

        if genes is not None:
            df = df[df[source_col].isin(genes) & df[target_col].isin(genes)]

        edges: list[tuple[str, str]] = list(
            zip(df[source_col].tolist(), df[target_col].tolist())
        )
        weights: list[float] | None = (
            [float(w) for w in df[weight_col].tolist()] if weight_col is not None else None
        )

        logger.info(
            "loaded %d custom edges from %s (weighted=%s)",
            len(edges),
            path,
            weights is not None,
        )
        return edges, weights
