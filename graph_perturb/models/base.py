"""Common model interface + batching utilities.

Both the GNN and the conditional VAE predict a per-gene expression **delta**
(post-perturbation minus control mean) given a baseline cell state and a
one-hot perturbation flag. They share one interface so ``train.py`` /
``evaluate.py`` are model-agnostic.

Data convention
---------------
The dataset yields PyG :class:`torch_geometric.data.Data` objects, one per
"condition sample" (a cell or a pseudobulk replicate). Each ``Data`` has:

* ``x``        : ``[num_genes, 2]`` node features = ``[baseline_expr, pert_flag]``
* ``y``        : ``[num_genes]``   target delta
* ``edge_index``: ``[2, E]`` (identical graph topology for every sample)
* ``num_nodes``: ``num_genes``

A :class:`torch_geometric.data.Batch` of ``B`` such samples is a block-diagonal
graph with ``B * num_genes`` nodes; ``batch.batch`` maps node -> sample id.
:func:`batch_to_dense` recovers the ``[B, num_genes]`` dense view that the VAE
and the loss/metric code use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class PerturbationModel(nn.Module, ABC):
    """Predicts per-gene expression delta from baseline + perturbation flag."""

    #: Does this model perform message passing over the graph? (False for VAE.)
    requires_graph: bool = True

    def __init__(self, num_genes: int):
        super().__init__()
        self.num_genes = num_genes

    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        """Return predicted delta as a dense tensor of shape ``[B, num_genes]``.

        ``batch`` is a PyG ``Batch``. Implementations that ignore graph topology
        (e.g. the VAE) should still accept it and use :func:`batch_to_dense`.
        Models may return auxiliary tensors via :meth:`loss` instead; ``forward``
        itself must return only the prediction so evaluation stays uniform.
        """

    def loss(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(scalar_loss, logs)``.

        Default: MSE on the delta. Models with extra terms (e.g. VAE KL) override
        this and add their components to ``logs``.
        """
        pred = self.forward(batch)
        target = batch_to_dense(batch.y, batch, self.num_genes)
        mse = nn.functional.mse_loss(pred, target)
        return mse, {"mse": float(mse.detach())}


def batch_to_dense(values: torch.Tensor, batch, num_genes: int) -> torch.Tensor:
    """Reshape a flat ``[B*num_genes]`` (or ``[B*num_genes, d]``) tensor to dense.

    Relies on the fact that every sample contributes exactly ``num_genes`` nodes
    in the same canonical order, so a simple reshape recovers ``[B, num_genes]``
    (or ``[B, num_genes, d]``).
    """
    b = int(batch.batch.max().item()) + 1 if batch.batch.numel() else 1
    if values.dim() == 1:
        return values.view(b, num_genes)
    return values.view(b, num_genes, values.shape[-1])


def baseline_and_flag(batch, num_genes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``batch.x`` ``[N, 2]`` into dense baseline and pert-flag ``[B, num_genes]``."""
    dense = batch_to_dense(batch.x, batch, num_genes)  # [B, G, 2]
    return dense[..., 0], dense[..., 1]
