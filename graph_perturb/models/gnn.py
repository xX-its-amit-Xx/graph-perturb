"""GraphSAGE + attention GNN for per-gene perturbation-response prediction.

The model performs message passing over a *swappable* biological knowledge graph
(:class:`~graph_perturb.graphs.base.PerturbationGraph`) whose topology is shared
by every sample in a batch (PyG stacks them block-diagonally). Each node carries
two input features -- the control-mean baseline expression and a binary
perturbation flag -- and the network predicts a scalar expression **delta** per
node, returned densely as ``[B, num_genes]``.

Architecture
------------
``x [N, 2]`` -> ``input_proj`` -> ``hidden_dim``
   -> ``n_layers`` x (SAGEConv -> ReLU -> Dropout)
   -> :class:`~graph_perturb.models.layers.AttentionReadout` (gated refinement)
   -> per-node MLP head -> scalar delta ``[N]``
   -> :func:`~graph_perturb.models.base.batch_to_dense` -> ``[B, num_genes]``
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from ..config import ModelConfig
from .base import PerturbationModel, batch_to_dense
from .layers import AttentionReadout

logger = logging.getLogger(__name__)


class GraphSAGEPerturbation(PerturbationModel):
    """GraphSAGE encoder + attention readout predicting per-gene deltas.

    Parameters
    ----------
    num_genes:
        Number of nodes per graph (the gene universe size). Must match the graph
        the dataset was aligned to.
    cfg:
        :class:`~graph_perturb.config.ModelConfig`. Uses ``hidden_dim``,
        ``n_layers``, ``dropout``, ``sage_aggr`` and ``attention_heads``.
    graph:
        Optional :class:`~graph_perturb.graphs.base.PerturbationGraph`. Stored for
        provenance / convenience; topology actually used at runtime comes from the
        batched ``edge_index`` so the model stays agnostic to which backend graph
        was selected. ``forward`` does not require it.
    """

    requires_graph: bool = True

    def __init__(self, num_genes: int, cfg: ModelConfig, graph=None) -> None:
        super().__init__(num_genes=num_genes)
        self.cfg = cfg
        self.graph = graph
        hidden = int(cfg.hidden_dim)
        n_layers = int(cfg.n_layers)
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.hidden_dim = hidden
        self.n_layers = n_layers
        self.dropout = float(cfg.dropout)
        self.sage_aggr = str(cfg.sage_aggr)
        # Optional correlation term weight; off by default. Documented below.
        self.corr_weight = float(getattr(cfg, "extra", {}).get("corr_weight", 0.0))

        # Input features are [baseline_expr, pert_flag] -> 2 dims.
        self.input_proj = nn.Linear(2, hidden)

        self.convs = nn.ModuleList(
            SAGEConv(hidden, hidden, aggr=self.sage_aggr) for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))

        self.readout = AttentionReadout(
            in_dim=hidden,
            heads=int(cfg.attention_heads),
            out_dim=hidden,
            dropout=self.dropout,
        )

        # Per-node head -> scalar delta.
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        self.readout.reset_parameters()
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch) -> torch.Tensor:
        """Return predicted delta as dense ``[B, num_genes]``.

        Uses ``batch.x`` ``[N, 2]``, ``batch.edge_index`` (block-diagonal) and
        ``batch.batch`` (node -> sample id). ``batch.edge_weight`` is accepted if
        present but, as documented, ``SAGEConv`` does not use edge weights; we
        pass it through only so weighted graphs do not break the call.
        """
        x = batch.x
        edge_index = batch.edge_index
        batch_index = getattr(batch, "batch", None)
        if batch_index is None:
            batch_index = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        edge_weight = getattr(batch, "edge_weight", None)

        h = F.relu(self.input_proj(x))
        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            # SAGEConv ignores edge_weight; forwarding it keeps weighted graphs
            # working without raising. Some PyG versions accept the kwarg, others
            # do not -- guard accordingly.
            if edge_weight is not None:
                try:
                    h = conv(h, edge_index, edge_weight=edge_weight)
                except TypeError:
                    h = conv(h, edge_index)
            else:
                h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + h_in  # residual connection for deeper stacks / CPU stability

        h = self.readout(h, batch_index, self.num_genes)  # [N, hidden]
        delta = self.head(h).squeeze(-1)  # [N]
        return batch_to_dense(delta, batch, self.num_genes)  # [B, num_genes]

    def loss(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        """MSE on the delta, plus an optional per-sample correlation penalty.

        The base contract is MSE. If ``cfg.extra['corr_weight'] > 0`` we add
        ``corr_weight * (1 - mean_b pearson(pred_b, target_b))``, which encourages
        the predicted delta profile of each cell/pseudobulk to track the true one
        (a common, evaluation-aligned objective for perturbation models). It is
        fully optional and defaults to disabled, preserving the base behaviour.
        """
        pred = self.forward(batch)
        target = batch_to_dense(batch.y, batch, self.num_genes)
        mse = F.mse_loss(pred, target)
        logs = {"mse": float(mse.detach())}
        total = mse

        if self.corr_weight > 0.0:
            corr = _row_pearson(pred, target).mean()
            corr_loss = 1.0 - corr
            total = total + self.corr_weight * corr_loss
            logs["corr"] = float(corr.detach())
            logs["corr_loss"] = float(corr_loss.detach())

        logs["loss"] = float(total.detach())
        return total, logs


def _row_pearson(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-row Pearson correlation of ``[B, G]`` tensors -> ``[B]``."""
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    num = (p * t).sum(dim=-1)
    den = p.norm(dim=-1) * t.norm(dim=-1) + eps
    return num / den


__all__ = ["GraphSAGEPerturbation"]
