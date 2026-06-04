"""Reactome pathway knowledge-graph backend.

Connects genes that participate in the same Reactome pathway. Raw data come from
Reactome's ``NCBI2Reactome_All_Levels.txt`` mapping, which links NCBI (Entrez)
gene identifiers to every pathway they belong to (across all hierarchy levels).

Source
------
* https://reactome.org/download/current/NCBI2Reactome_All_Levels.txt
  (Reactome, https://reactome.org; data CC0 1.0)

File format (tab-separated, no header)::

    0  NCBI/Entrez gene id
    1  Reactome Pathway stable id (R-HSA-...)
    2  URL
    3  Pathway name
    4  Evidence code (e.g. TAS, IEA)
    5  Species (e.g. "Homo sapiens")

The first column is an NCBI gene id rather than a symbol; the human Reactome
pathway stable ids carry the ``R-HSA-`` species prefix, so we identify human
records both by the ``Homo sapiens`` species column (when present) and by that
prefix. Because downstream node ids are gene *symbols*, we accept an optional
``ncbi_to_symbol`` mapping in :attr:`options`; without it we keep the Entrez id
as the node label (callers building an Entrez-keyed gene universe still work,
and the symbol-keyed universe simply drops unmapped ids in
:meth:`GraphSource.build`). Genes co-occurring in a pathway form a clique;
pathways larger than ``max_pathway_size`` are skipped.
"""

from __future__ import annotations

import itertools
import logging
import urllib.request
from pathlib import Path

from .base import GraphSource
from .registry import register_graph

logger = logging.getLogger(__name__)

#: Default Reactome NCBI-gene-to-pathway mapping download URL.
NCBI2REACTOME_URL = (
    "https://reactome.org/download/current/NCBI2Reactome_All_Levels.txt"
)

_HUMAN_SPECIES = "Homo sapiens"
_HUMAN_PREFIX = "R-HSA-"

# Tab-separated column indices (0-based).
_COL_GENE = 0
_COL_PATHWAY = 1
_COL_SPECIES = 5


@register_graph("reactome")
class ReactomeGraph(GraphSource):
    """Gene-gene edges from shared Reactome (human) pathway membership.

    Options (via :attr:`options` / ``GraphConfig.options``)
    -------------------------------------------------------
    max_pathway_size: int, default 200
        Skip pathways with more than this many member genes.
    ncbi_to_symbol: Mapping[str | int, str] | None, default None
        Optional Entrez-id -> gene-symbol map; node labels are mapped through it
        when provided.
    url: str, default :data:`NCBI2REACTOME_URL`
        Override the download location.
    """

    def _download(self) -> Path:
        url = self.options.get("url", NCBI2REACTOME_URL)
        dest = self.cache_dir / Path(url).name
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        logger.info("downloading Reactome mapping from %s -> %s", url, dest)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - documented HTTP source
        tmp.replace(dest)
        return dest

    def _parse(self, path: Path, genes: set[str] | None) -> dict[str, set[str]]:
        """Return ``{pathway_id: {gene_label, ...}}`` for human pathways."""
        ncbi_to_symbol = self.options.get("ncbi_to_symbol") or {}

        def to_label(raw_id: str) -> str:
            if not ncbi_to_symbol:
                return raw_id
            return (
                ncbi_to_symbol.get(raw_id)
                or ncbi_to_symbol.get(raw_id.lstrip("0"))
                or (ncbi_to_symbol.get(int(raw_id)) if raw_id.isdigit() else None)
                or raw_id
            )

        pathway_to_genes: dict[str, set[str]] = {}
        with open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                cols = line.rstrip("\n").split("\t")
                if len(cols) <= _COL_PATHWAY:
                    continue
                pathway = cols[_COL_PATHWAY].strip()
                species = cols[_COL_SPECIES].strip() if len(cols) > _COL_SPECIES else ""
                is_human = species == _HUMAN_SPECIES or pathway.startswith(_HUMAN_PREFIX)
                if not is_human:
                    continue
                raw_gene = cols[_COL_GENE].strip()
                if not raw_gene or not pathway:
                    continue
                label = to_label(raw_gene)
                if genes is not None and label not in genes:
                    continue
                pathway_to_genes.setdefault(pathway, set()).add(label)
        return pathway_to_genes

    def _load_edges(
        self, genes: set[str] | None
    ) -> tuple[list[tuple[str, str]], list[float] | None]:
        max_pathway_size = int(self.options.get("max_pathway_size", 200))
        try:
            path = self._download()
            pathway_to_genes = self._parse(path, genes)
        except Exception as exc:  # noqa: BLE001 - best-effort, offline fallback
            logger.warning(
                "Reactome mapping unavailable (%s); falling back to co-expression edges",
                exc,
            )
            if genes is None:
                return [], None
            return self._coexpression_fallback_edges(sorted(genes)), None

        edges: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        n_pathways_kept = 0
        for members in pathway_to_genes.values():
            if len(members) < 2 or len(members) > max_pathway_size:
                continue
            n_pathways_kept += 1
            for a, b in itertools.combinations(sorted(members), 2):
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    edges.append(key)

        logger.info(
            "reactome: %d human pathways (<= %d genes) -> %d unique edges",
            n_pathways_kept,
            max_pathway_size,
            len(edges),
        )
        if not edges and genes is not None:
            logger.warning("reactome produced no edges; using co-expression fallback")
            return self._coexpression_fallback_edges(sorted(genes)), None
        return edges, None
