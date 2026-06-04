"""Core graph abstractions for graph-perturb.

Everything that wants to be a *swappable* biological knowledge graph implements
:class:`GraphSource`. A source knows how to fetch/load raw edges between genes
(and cache them), and the concrete :meth:`GraphSource.build` method handles the
universal work: restricting edges to the experiment's gene universe, mapping
gene symbols to integer node ids, optionally symmetrizing and adding self-loops,
and packaging everything into a :class:`PerturbationGraph` that the models
consume directly.

The contract is deliberately small so that a GO term graph, a Reactome pathway
graph, a STRING PPI, a GEARS-format file, or a user's hand-built CSV all look
identical to the rest of the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PerturbationGraph:
    """A gene-gene knowledge graph ready for message passing.

    Attributes
    ----------
    edge_index:
        ``LongTensor`` of shape ``[2, E]`` (PyG convention) indexing into
        ``gene_names``.
    gene_names:
        Node ``i`` corresponds to ``gene_names[i]``. This is the canonical node
        ordering; expression matrices passed to the model MUST be aligned to it.
    gene_to_idx:
        Inverse map ``gene_name -> node id``.
    edge_weight:
        Optional ``FloatTensor`` of shape ``[E]`` (e.g. STRING combined score in
        ``[0, 1]``). ``None`` means unweighted.
    name:
        Human-readable backend identifier (``"go_bp"``, ``"reactome"`` ...).
    metadata:
        Free-form provenance (source files, release version, n raw edges, ...).
    """

    edge_index: torch.Tensor
    gene_names: list[str]
    gene_to_idx: dict[str, int]
    edge_weight: torch.Tensor | None = None
    name: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.gene_names)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PerturbationGraph(name={self.name!r}, nodes={self.num_nodes}, "
            f"edges={self.num_edges}, weighted={self.edge_weight is not None})"
        )

    def perturbation_vector(self, perturbed_genes: Sequence[str]) -> torch.Tensor:
        """One-hot ``[num_nodes]`` flag marking which nodes are perturbed.

        Unknown gene symbols are ignored (with a warning) so that a perturbation
        targeting a gene absent from the graph degrades gracefully to "no node
        flagged" rather than crashing.
        """
        vec = torch.zeros(self.num_nodes, dtype=torch.float32)
        for g in perturbed_genes:
            idx = self.gene_to_idx.get(g)
            if idx is None:
                logger.warning("perturbed gene %r not in graph %r; skipping", g, self.name)
                continue
            vec[idx] = 1.0
        return vec

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "edge_index": self.edge_index,
                "gene_names": self.gene_names,
                "edge_weight": self.edge_weight,
                "name": self.name,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PerturbationGraph":
        blob = torch.load(path, weights_only=False)
        gene_names = list(blob["gene_names"])
        return cls(
            edge_index=blob["edge_index"],
            gene_names=gene_names,
            gene_to_idx={g: i for i, g in enumerate(gene_names)},
            edge_weight=blob.get("edge_weight"),
            name=blob.get("name", ""),
            metadata=blob.get("metadata", {}),
        )


class GraphSource(ABC):
    """Abstract pluggable biological knowledge graph backend.

    Subclasses implement exactly one method, :meth:`_load_edges`, which returns
    raw gene-symbol edges (downloading + caching raw source files as needed).
    The shared :meth:`build` turns those into a :class:`PerturbationGraph`
    aligned to a caller-provided gene universe.
    """

    #: Unique backend name; also the key under which the source is registered.
    name: str = "abstract"

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        undirected: bool = True,
        add_self_loops: bool = True,
        **kwargs,
    ) -> None:
        from .registry import default_cache_dir

        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.undirected = undirected
        self.add_self_loops = add_self_loops
        self.options = kwargs

    # ----- subclass contract ------------------------------------------------
    @abstractmethod
    def _load_edges(
        self, genes: set[str] | None
    ) -> tuple[list[tuple[str, str]], list[float] | None]:
        """Return ``(edges, weights)`` as gene-symbol pairs.

        Parameters
        ----------
        genes:
            If not ``None``, the source SHOULD restrict to edges where both
            endpoints are in this set (an optimization; :meth:`build` filters
            again defensively). If ``None``, return everything available.

        Returns
        -------
        edges:
            List of ``(gene_a, gene_b)`` symbol pairs.
        weights:
            Parallel list of edge weights, or ``None`` for an unweighted graph.
        """

    # ----- shared machinery -------------------------------------------------
    def _cache_key(self, genes: Sequence[str] | None) -> str:
        h = hashlib.sha1()
        h.update(self.name.encode())
        h.update(json.dumps(self.options, sort_keys=True, default=str).encode())
        h.update(f"u{int(self.undirected)}s{int(self.add_self_loops)}".encode())
        if genes is not None:
            h.update(",".join(sorted(genes)).encode())
        return h.hexdigest()[:16]

    def build(
        self,
        genes: Sequence[str],
        *,
        use_cache: bool = True,
    ) -> PerturbationGraph:
        """Build a :class:`PerturbationGraph` over the given gene universe.

        ``genes`` defines the node set and its canonical order. Genes with no
        edges still become (isolated, optionally self-looped) nodes so that the
        model always sees the full expression vector.
        """
        genes = list(dict.fromkeys(genes))  # dedupe, preserve order
        cache_path = self.cache_dir / f"{self.name}_{self._cache_key(genes)}.pt"
        if use_cache and cache_path.exists():
            logger.info("loading cached graph %s", cache_path)
            return PerturbationGraph.load(cache_path)

        gene_set = set(genes)
        gene_to_idx = {g: i for i, g in enumerate(genes)}

        raw_edges, raw_weights = self._load_edges(gene_set)
        src, dst, wts = [], [], []
        n_dropped = 0
        for k, (a, b) in enumerate(raw_edges):
            ia, ib = gene_to_idx.get(a), gene_to_idx.get(b)
            if ia is None or ib is None or ia == ib:
                n_dropped += 1
                continue
            src.append(ia)
            dst.append(ib)
            if raw_weights is not None:
                wts.append(float(raw_weights[k]))

        if self.undirected:
            src, dst = src + dst, dst + src
            if raw_weights is not None:
                wts = wts + wts

        if self.add_self_loops:
            n = len(genes)
            src += list(range(n))
            dst += list(range(n))
            if raw_weights is not None:
                wts += [1.0] * n

        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty(
            (2, 0), dtype=torch.long
        )
        edge_weight = (
            torch.tensor(wts, dtype=torch.float32) if (raw_weights is not None and wts) else None
        )

        graph = PerturbationGraph(
            edge_index=edge_index,
            gene_names=genes,
            gene_to_idx=gene_to_idx,
            edge_weight=edge_weight,
            name=self.name,
            metadata={
                "n_raw_edges": len(raw_edges),
                "n_dropped_edges": n_dropped,
                "n_kept_directed_edges": int(edge_index.shape[1]),
                "undirected": self.undirected,
                "self_loops": self.add_self_loops,
                "options": {k: str(v) for k, v in self.options.items()},
            },
        )
        logger.info(
            "built %s: %d nodes, %d edges (%d raw, %d dropped)",
            self.name,
            graph.num_nodes,
            graph.num_edges,
            len(raw_edges),
            n_dropped,
        )
        if use_cache:
            graph.save(cache_path)
        return graph

    # ----- small helpers for subclasses ------------------------------------
    @staticmethod
    def _coexpression_fallback_edges(
        genes: Sequence[str], k: int = 5, seed: int = 0
    ) -> list[tuple[str, str]]:
        """Deterministic k-NN-style ring edges, used ONLY as an offline fallback.

        Real backends must override ``_load_edges``; this exists so tests and
        air-gapped environments can still exercise the pipeline without network
        access. It is never used when real source files are available.
        """
        rng = np.random.default_rng(seed)
        genes = list(genes)
        n = len(genes)
        edges: list[tuple[str, str]] = []
        for i in range(n):
            for j in rng.choice(n, size=min(k, n), replace=False):
                if i != int(j):
                    edges.append((genes[i], genes[int(j)]))
        return edges


def gene_universe_from_iterables(*iterables: Iterable[str]) -> list[str]:
    """Union several gene-name iterables into a single ordered, de-duplicated list."""
    seen: dict[str, None] = {}
    for it in iterables:
        for g in it:
            if g not in seen:
                seen[g] = None
    return list(seen)
