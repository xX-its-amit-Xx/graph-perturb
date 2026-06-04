"""STRING protein-protein interaction knowledge-graph backend.

Produces *weighted* gene-gene edges from STRING's ``combined_score`` (an
integer in ``[0, 1000]`` reflecting confidence that two proteins functionally
interact). Scores are normalized to ``[0, 1]`` and used as edge weights.

Sources
-------
* Links: https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz
* Info (Ensembl protein id -> gene symbol):
  https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz
  (STRING, https://string-db.org; data CC-BY 4.0; ``9606`` = Homo sapiens)

Links file (space-separated, with header)::

    protein1 protein2 combined_score

The protein ids carry the ``9606.`` taxon prefix (e.g. ``9606.ENSP00000269305``).
The info file maps that id to a ``preferred_name`` (gene symbol)::

    #string_protein_id    preferred_name    protein_size    annotation

We translate both endpoints to gene symbols, keep edges whose normalized score
meets ``score_threshold``, and return the parallel weight list.
"""

from __future__ import annotations

import gzip
import logging
import urllib.request
from pathlib import Path

from .base import GraphSource
from .registry import register_graph

logger = logging.getLogger(__name__)

#: STRING release version used to template the download URLs.
DEFAULT_VERSION = "12.0"
#: Homo sapiens NCBI taxon id (STRING file prefix).
HUMAN_TAXON = "9606"

_LINKS_URL_TMPL = (
    "https://stringdb-downloads.org/download/protein.links.v{version}/"
    "{taxon}.protein.links.v{version}.txt.gz"
)
_INFO_URL_TMPL = (
    "https://stringdb-downloads.org/download/protein.info.v{version}/"
    "{taxon}.protein.info.v{version}.txt.gz"
)


@register_graph("string")
class StringPPIGraph(GraphSource):
    """Weighted gene-gene edges from STRING combined confidence scores.

    Options (via :attr:`options` / ``GraphConfig.options``)
    -------------------------------------------------------
    score_threshold: float, default 0.7
        Minimum *normalized* combined score (0-1 scale; 0.7 == raw 700) for an
        edge to be kept.
    version: str, default :data:`DEFAULT_VERSION`
        STRING release used in the download URLs.
    links_url / info_url: str, optional
        Explicit overrides for the two download URLs.
    """

    def _links_url(self) -> str:
        return self.options.get("links_url") or _LINKS_URL_TMPL.format(
            version=self.options.get("version", DEFAULT_VERSION), taxon=HUMAN_TAXON
        )

    def _info_url(self) -> str:
        return self.options.get("info_url") or _INFO_URL_TMPL.format(
            version=self.options.get("version", DEFAULT_VERSION), taxon=HUMAN_TAXON
        )

    def _download(self, url: str) -> Path:
        dest = self.cache_dir / Path(url).name
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        logger.info("downloading STRING file from %s -> %s", url, dest)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - documented HTTP source
        tmp.replace(dest)
        return dest

    @staticmethod
    def _open(path: Path):
        opener = gzip.open if path.suffix == ".gz" else open
        return opener(path, "rt", encoding="utf-8", errors="replace")

    def _parse_info(self, path: Path) -> dict[str, str]:
        """Return ``{string_protein_id: gene_symbol}``."""
        mapping: dict[str, str] = {}
        with self._open(path) as fh:
            header = fh.readline().lstrip("#").rstrip("\n").split("\t")
            try:
                id_col = header.index("string_protein_id")
            except ValueError:
                id_col = 0
            try:
                name_col = header.index("preferred_name")
            except ValueError:
                name_col = 1
            ncols = max(id_col, name_col) + 1
            for line in fh:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < ncols:
                    continue
                pid = cols[id_col].strip()
                symbol = cols[name_col].strip()
                if pid and symbol:
                    mapping[pid] = symbol
        return mapping

    def _load_edges(
        self, genes: set[str] | None
    ) -> tuple[list[tuple[str, str]], list[float] | None]:
        threshold = float(self.options.get("score_threshold", 0.7))
        try:
            info_path = self._download(self._info_url())
            links_path = self._download(self._links_url())
            id_to_symbol = self._parse_info(info_path)
        except Exception as exc:  # noqa: BLE001 - best-effort, offline fallback
            logger.warning(
                "STRING files unavailable (%s); falling back to co-expression edges",
                exc,
            )
            if genes is None:
                return [], None
            return self._coexpression_fallback_edges(sorted(genes)), None

        edges: list[tuple[str, str]] = []
        weights: list[float] = []
        seen: set[tuple[str, str]] = set()
        try:
            with self._open(links_path) as fh:
                header = fh.readline().split()
                try:
                    p1_col = header.index("protein1")
                    p2_col = header.index("protein2")
                    score_col = header.index("combined_score")
                except ValueError:
                    p1_col, p2_col, score_col = 0, 1, 2
                ncols = max(p1_col, p2_col, score_col) + 1
                for line in fh:
                    cols = line.split()
                    if len(cols) < ncols:
                        continue
                    try:
                        weight = int(cols[score_col]) / 1000.0
                    except ValueError:
                        continue
                    if weight < threshold:
                        continue
                    a = id_to_symbol.get(cols[p1_col])
                    b = id_to_symbol.get(cols[p2_col])
                    if not a or not b or a == b:
                        continue
                    if genes is not None and (a not in genes or b not in genes):
                        continue
                    key = (a, b) if a <= b else (b, a)
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append(key)
                    weights.append(weight)
        except Exception as exc:  # noqa: BLE001 - best-effort, offline fallback
            logger.warning(
                "failed parsing STRING links (%s); falling back to co-expression edges",
                exc,
            )
            if genes is None:
                return [], None
            return self._coexpression_fallback_edges(sorted(genes)), None

        logger.info(
            "string: %d weighted edges (normalized score >= %.3f)", len(edges), threshold
        )
        if not edges and genes is not None:
            logger.warning("string produced no edges; using co-expression fallback")
            return self._coexpression_fallback_edges(sorted(genes)), None
        return edges, weights
