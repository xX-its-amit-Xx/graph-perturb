"""Gene Ontology (Biological Process) knowledge-graph backend.

Connects genes that are co-annotated to the same GO Biological Process (BP)
term. Raw annotations come from the GO Consortium's human Gene Association File
(GAF 2.2), which maps gene products (UniProt accessions, with a gene symbol in
the ``DB_Object_Symbol`` column) to GO terms together with an evidence code and
GO aspect (``P`` = biological process, ``F`` = molecular function, ``C`` =
cellular component).

Source
------
* GAF: http://current.geneontology.org/annotations/goa_human.gaf.gz
  (GO Consortium, http://geneontology.org; data CC-BY 4.0)

The GAF format is documented at
https://geneontology.org/docs/go-annotation-file-gaf-format-2.2/ . Relevant
0-based columns used here::

    1  DB_Object_Symbol   (gene symbol)
    3  Qualifier          (e.g. NOT, may be empty)
    4  GO_ID              (GO:xxxxxxx)
    6  Evidence_Code      (IEA, IDA, ...)
    8  Aspect             (P | F | C)

We keep aspect ``P`` (BP), drop ``NOT`` qualifiers, optionally filter by
evidence code, group symbols by GO term, skip terms larger than
``max_term_size`` (hubs add little signal and quadratically many edges), and
emit an (unweighted) clique edge for every co-annotated gene pair within a term.
"""

from __future__ import annotations

import gzip
import itertools
import logging
import urllib.request
from pathlib import Path

from .base import GraphSource
from .registry import register_graph

logger = logging.getLogger(__name__)

#: Default human GAF (GO annotations) download URL.
GAF_URL = "http://current.geneontology.org/annotations/goa_human.gaf.gz"

#: GAF column indices (0-based) per the GAF 2.2 specification.
_COL_SYMBOL = 2
_COL_QUALIFIER = 3
_COL_GO_ID = 4
_COL_EVIDENCE = 6
_COL_ASPECT = 8


@register_graph("go_bp")
class GeneOntologyGraph(GraphSource):
    """Gene-gene edges from shared GO Biological Process annotations.

    Options (via :attr:`options` / ``GraphConfig.options``)
    -------------------------------------------------------
    max_term_size: int, default 200
        Skip GO terms annotating more than this many genes (avoids dense hubs).
    evidence_codes: Iterable[str] | None, default None
        If given, keep only annotations whose evidence code is in this set
        (e.g. ``{"IDA", "IMP", "IPI"}`` for experimental evidence only).
    gaf_url: str, default :data:`GAF_URL`
        Override the GAF download location (e.g. a pinned release).
    """

    def _download_gaf(self) -> Path:
        url = self.options.get("gaf_url", GAF_URL)
        dest = self.cache_dir / Path(url).name
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        logger.info("downloading GO GAF from %s -> %s", url, dest)
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - documented HTTP source
        tmp.replace(dest)
        return dest

    def _parse_gaf(self, path: Path, genes: set[str] | None) -> dict[str, set[str]]:
        """Return ``{GO_term: {gene_symbol, ...}}`` for BP annotations."""
        evidence = self.options.get("evidence_codes")
        evidence_set = {str(e).upper() for e in evidence} if evidence else None
        term_to_genes: dict[str, set[str]] = {}

        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line or line.startswith("!"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) <= _COL_ASPECT:
                    continue
                if cols[_COL_ASPECT] != "P":
                    continue
                qualifier = cols[_COL_QUALIFIER]
                if qualifier and "NOT" in qualifier.upper():
                    continue
                if evidence_set is not None and cols[_COL_EVIDENCE].upper() not in evidence_set:
                    continue
                symbol = cols[_COL_SYMBOL].strip()
                go_id = cols[_COL_GO_ID].strip()
                if not symbol or not go_id:
                    continue
                if genes is not None and symbol not in genes:
                    continue
                term_to_genes.setdefault(go_id, set()).add(symbol)
        return term_to_genes

    def _load_edges(
        self, genes: set[str] | None
    ) -> tuple[list[tuple[str, str]], list[float] | None]:
        max_term_size = int(self.options.get("max_term_size", 200))
        try:
            path = self._download_gaf()
            term_to_genes = self._parse_gaf(path, genes)
        except Exception as exc:  # noqa: BLE001 - best-effort, offline fallback
            logger.warning(
                "GO GAF unavailable (%s); falling back to co-expression edges", exc
            )
            if genes is None:
                return [], None
            return self._coexpression_fallback_edges(sorted(genes)), None

        edges: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        n_terms_kept = 0
        for members in term_to_genes.values():
            if len(members) < 2 or len(members) > max_term_size:
                continue
            n_terms_kept += 1
            for a, b in itertools.combinations(sorted(members), 2):
                key = (a, b)
                if key not in seen:
                    seen.add(key)
                    edges.append(key)

        logger.info(
            "go_bp: %d BP terms (<= %d genes) -> %d unique edges",
            n_terms_kept,
            max_term_size,
            len(edges),
        )
        if not edges and genes is not None:
            logger.warning("go_bp produced no edges; using co-expression fallback")
            return self._coexpression_fallback_edges(sorted(genes)), None
        return edges, None
