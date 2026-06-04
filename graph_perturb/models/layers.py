"""Reusable neural layers for the perturbation GNN.

The headline component is :class:`AttentionReadout`. Unlike a classic
graph-classification readout (which collapses a graph to a single vector), the
perturbation model predicts a **per-gene delta**, so we must keep one feature
vector per node. :class:`AttentionReadout` therefore implements attention as a
*gated refinement*: it computes scalar attention weights over the nodes of each
graph, pools them into a per-graph context vector, and then conditions every
node embedding on that context (concatenation + gating) before the output head.

This lets the model route global perturbation information (which node was
hit, the overall magnitude of the response) back to each gene while still
emitting an independent prediction per node.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

logger = logging.getLogger(__name__)


class AttentionReadout(nn.Module):
    """Attention-gated per-node refinement.

    Given node embeddings (either dense ``[B, G, H]`` or flat ``[N, H]`` plus a
    ``batch_index``), compute additive attention scores per node, normalise them
    within each graph to obtain attention weights, and form a per-graph context
    vector ``c_b = sum_i alpha_{b,i} * value_i``. Each node embedding is then
    refined by gating it with its graph context::

        gate_i   = sigmoid(W_g [h_i || c_{b(i)}])
        refined_i = ReLU(W_o [h_i || (gate_i * c_{b(i)})])

    The output has shape ``[N, out_dim]`` (defaults to ``in_dim``), one vector
    per node, ready for a per-node prediction head.

    Parameters
    ----------
    in_dim:
        Node embedding dimension ``H``.
    attn_dim:
        Hidden width of the additive-attention scorer. Defaults to ``in_dim``.
    heads:
        Number of independent attention heads. Their context vectors are
        concatenated and projected back to ``in_dim`` before gating, giving the
        layer multiple "views" of the global perturbation signal.
    out_dim:
        Output feature dimension per node. Defaults to ``in_dim``.
    dropout:
        Dropout applied to attention weights and to the refined features.
    """

    def __init__(
        self,
        in_dim: int,
        attn_dim: int | None = None,
        heads: int = 4,
        out_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        self.in_dim = in_dim
        self.heads = heads
        self.attn_dim = attn_dim or in_dim
        self.out_dim = out_dim or in_dim
        self.dropout = dropout

        # Additive attention scorer: project nodes then score per head.
        self.score_proj = nn.Linear(in_dim, self.attn_dim * heads)
        self.score_vec = nn.Parameter(torch.empty(heads, self.attn_dim))

        # Per-head value projection; contexts from all heads are concatenated.
        self.value_proj = nn.Linear(in_dim, in_dim * heads)
        # Collapse the multi-head context back to a single in_dim context vector.
        self.context_proj = nn.Linear(in_dim * heads, in_dim)

        # Gating + output refinement operate on [h_i || context].
        self.gate = nn.Linear(in_dim * 2, in_dim)
        self.out_proj = nn.Linear(in_dim * 2, self.out_dim)

        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.score_proj.weight)
        nn.init.zeros_(self.score_proj.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.score_vec)

    def forward(
        self,
        node_emb: torch.Tensor,
        batch_index: torch.Tensor | None,
        num_genes: int,
    ) -> torch.Tensor:
        """Refine node embeddings using per-graph attention context.

        Parameters
        ----------
        node_emb:
            Either flat ``[N, H]`` (PyG block-diagonal layout) or dense
            ``[B, G, H]``. ``H`` must equal ``in_dim``.
        batch_index:
            ``LongTensor`` ``[N]`` mapping each node to its graph id, as produced
            by PyG (``batch.batch``). Required when ``node_emb`` is flat; ignored
            (may be ``None``) when ``node_emb`` is already dense.
        num_genes:
            Number of nodes per graph ``G``. Used to derive the batch size when
            a flat input is supplied without a ``batch_index``.

        Returns
        -------
        torch.Tensor
            Refined node embeddings of shape ``[N, out_dim]`` (flat input) or
            ``[B * G, out_dim]`` flattened from a dense input. Always flat so the
            caller can apply a per-node head uniformly.
        """
        was_dense = node_emb.dim() == 3
        if was_dense:
            b, g, h = node_emb.shape
            node_emb = node_emb.reshape(b * g, h)
            batch_index = torch.arange(b, device=node_emb.device).repeat_interleave(g)
        else:
            if node_emb.dim() != 2:
                raise ValueError(
                    f"node_emb must be [N, H] or [B, G, H]; got {tuple(node_emb.shape)}"
                )
            if batch_index is None:
                # Infer a contiguous batching from num_genes.
                n = node_emb.shape[0]
                if num_genes <= 0 or n % num_genes != 0:
                    raise ValueError(
                        "batch_index is None and N is not a multiple of num_genes; "
                        f"N={n}, num_genes={num_genes}"
                    )
                b = n // num_genes
                batch_index = torch.arange(b, device=node_emb.device).repeat_interleave(
                    num_genes
                )

        n, h = node_emb.shape
        if h != self.in_dim:
            raise ValueError(f"expected in_dim={self.in_dim}, got {h}")
        num_graphs = int(batch_index.max().item()) + 1 if n else 1

        # --- additive attention scores per head: [N, heads] -----------------
        proj = self.score_proj(node_emb).view(n, self.heads, self.attn_dim)
        proj = torch.tanh(proj)
        # score_{i,head} = <score_vec_head, tanh(W proj h_i)>
        scores = (proj * self.score_vec.unsqueeze(0)).sum(dim=-1)  # [N, heads]

        # Normalise within each graph (segment softmax over nodes).
        alpha = softmax(scores, batch_index, num_nodes=num_graphs, dim=0)  # [N, heads]
        alpha = self.drop(alpha)

        # --- per-graph context via weighted sum of per-head values ----------
        values = self.value_proj(node_emb).view(n, self.heads, self.in_dim)
        weighted = values * alpha.unsqueeze(-1)  # [N, heads, in_dim]
        context = node_emb.new_zeros(num_graphs, self.heads, self.in_dim)
        context.index_add_(0, batch_index, weighted)  # [num_graphs, heads, in_dim]
        context = context.reshape(num_graphs, self.heads * self.in_dim)
        context = self.context_proj(context)  # [num_graphs, in_dim]

        # Scatter the per-graph context back onto each node.
        node_context = context.index_select(0, batch_index)  # [N, in_dim]

        # --- gated refinement ----------------------------------------------
        cat = torch.cat([node_emb, node_context], dim=-1)  # [N, 2*in_dim]
        gate = torch.sigmoid(self.gate(cat))  # [N, in_dim]
        gated_context = gate * node_context
        refined = F.relu(self.out_proj(torch.cat([node_emb, gated_context], dim=-1)))
        refined = self.drop(refined)
        return refined  # [N, out_dim]
