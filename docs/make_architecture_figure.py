"""Render the graph-perturb message-passing architecture to ``docs/architecture.png``.

Standalone, dependency-light (matplotlib only), and headless (Agg backend, no
display). It draws the full forward path of
:class:`~graph_perturb.models.gnn.GraphSAGEPerturbation`::

    node features [baseline, pert flag]
        -> input projection
        -> stacked GraphSAGE layers (message passing over the swappable graph)
        -> attention readout
        -> per-gene delta head

Run with::

    python docs/make_architecture_figure.py

The output is written next to this script at ``docs/architecture.png`` and is
embedded by the top-level ``README.md``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never opens a window

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --------------------------------------------------------------------------- #
# palette (kept small + colour-blind friendly)
# --------------------------------------------------------------------------- #
C_INPUT = "#cfe8ff"
C_INPUT_EDGE = "#2f6fb0"
C_GNN = "#d6f0d0"
C_GNN_EDGE = "#3f8f3a"
C_READOUT = "#ffe3b3"
C_READOUT_EDGE = "#c9842b"
C_HEAD = "#f3d0e0"
C_HEAD_EDGE = "#b04a78"
C_GRAPH = "#ece7f6"
C_GRAPH_EDGE = "#6a4fb0"
C_TEXT = "#1b1b1b"


def _box(ax, x, y, w, h, text, *, face, edge, fontsize=10, weight="normal"):
    """Draw a rounded box centred at (x, y) with wrapped centred text."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6,
        facecolor=face,
        edgecolor=edge,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=C_TEXT,
        weight=weight,
        zorder=4,
        wrap=True,
    )
    return patch


def _arrow(ax, x0, y0, x1, y1, *, color="#444444", style="-|>", lw=1.8):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            mutation_scale=16,
            linewidth=lw,
            color=color,
            zorder=2,
        )
    )


def _draw_knowledge_graph(ax, cx, cy, r):
    """A small illustrative gene-gene graph (the swappable biological prior)."""
    import math

    n = 7
    pts = []
    for i in range(n):
        ang = 2 * math.pi * i / n - math.pi / 2
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    # a handful of edges so it reads as a network, not a ring
    edges = [(0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 6), (1, 5), (0, 1)]
    for a, b in edges:
        ax.plot(
            [pts[a][0], pts[b][0]],
            [pts[a][1], pts[b][1]],
            color=C_GRAPH_EDGE,
            linewidth=1.0,
            alpha=0.55,
            zorder=3,
        )
    for (px, py) in pts:
        ax.plot(
            px,
            py,
            "o",
            markersize=7,
            markerfacecolor="#ffffff",
            markeredgecolor=C_GRAPH_EDGE,
            markeredgewidth=1.4,
            zorder=4,
        )


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13.5, 6.0), dpi=150)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    ax.text(
        50,
        57.5,
        "graph-perturb: message passing over a swappable knowledge graph",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
        color=C_TEXT,
    )

    y = 30  # main row baseline

    # 1) input node features ------------------------------------------------- #
    _box(
        ax,
        10,
        y,
        15,
        20,
        "Node features\n[N x 2]\n\nbaseline expr\n+ pert flag",
        face=C_INPUT,
        edge=C_INPUT_EDGE,
        weight="bold",
    )
    ax.text(
        10,
        y - 13.5,
        "one row per gene\n(control mean +\none-hot perturbation)",
        ha="center",
        va="center",
        fontsize=8,
        color="#555555",
    )

    # 2) input projection ---------------------------------------------------- #
    _box(
        ax,
        27,
        y,
        12,
        12,
        "Input\nprojection\n2 -> H",
        face=C_INPUT,
        edge=C_INPUT_EDGE,
    )

    # 3) stacked GraphSAGE layers (drawn as offset stacked cards) ------------ #
    base_x, base_y = 47, y
    for k in range(2, -1, -1):
        off = k * 1.4
        face = C_GNN if k == 0 else "#e8f6e3"
        _box(
            ax,
            base_x + off,
            base_y + off,
            16,
            16,
            "" if k != 0 else "GraphSAGE x L\n\nSAGEConv\nLayerNorm + ReLU\n+ residual",
            face=face,
            edge=C_GNN_EDGE,
            fontsize=9,
            weight="bold" if k == 0 else "normal",
        )
    ax.text(
        base_x,
        base_y - 11.5,
        "message passing over the\nchosen graph topology",
        ha="center",
        va="center",
        fontsize=8,
        color="#555555",
    )

    # 4) attention readout --------------------------------------------------- #
    _box(
        ax,
        70,
        y,
        13,
        16,
        "Attention\nreadout\n\nmulti-head\ngated\nrefinement",
        face=C_READOUT,
        edge=C_READOUT_EDGE,
        weight="bold",
    )

    # 5) per-gene delta head ------------------------------------------------- #
    _box(
        ax,
        89,
        y,
        14,
        18,
        "Per-gene\ndelta head\n[B x num_genes]\n\nMLP -> scalar\nper node",
        face=C_HEAD,
        edge=C_HEAD_EDGE,
        weight="bold",
    )
    ax.text(
        89,
        y - 12.5,
        "predicted\nexpression delta",
        ha="center",
        va="center",
        fontsize=8,
        color="#555555",
    )

    # main-flow arrows ------------------------------------------------------- #
    _arrow(ax, 17.5, y, 21, y)        # input -> proj
    _arrow(ax, 33, y, 39, y)          # proj  -> sage stack
    _arrow(ax, 55.5, y, 63.5, y)      # sage  -> readout
    _arrow(ax, 76.5, y, 82, y)        # readout -> head

    # swappable knowledge-graph callout feeding the GNN ---------------------- #
    gx, gy = 47, 50
    _box(
        ax,
        gx,
        gy,
        30,
        6.5,
        "Swappable knowledge graph:  GO BP | Reactome | STRING | GEARS | custom",
        face=C_GRAPH,
        edge=C_GRAPH_EDGE,
        fontsize=9,
        weight="bold",
    )
    _draw_knowledge_graph(ax, gx + 19.5, gy - 0.2, 2.6)
    # graph edges fed into the SAGE stack (same topology for every sample)
    _arrow(ax, gx, gy - 3.4, base_x, base_y + 8.5, color=C_GRAPH_EDGE, style="-|>", lw=1.6)
    ax.text(
        gx + 9,
        (gy + base_y) / 2 + 3,
        "edge_index\n(shared by all\nsamples in batch)",
        ha="center",
        va="center",
        fontsize=7.5,
        color=C_GRAPH_EDGE,
    )

    # loss / objective note -------------------------------------------------- #
    ax.text(
        89,
        7,
        "trained with MSE on the delta\n(optional correlation term)",
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
        color="#555555",
    )

    fig.tight_layout()
    return fig


def main() -> None:
    out_path = Path(__file__).resolve().parent / "architecture.png"
    fig = build_figure()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
