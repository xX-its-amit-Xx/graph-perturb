"""Conditional VAE baseline (CPA / scGen-flavored).

This baseline ignores the knowledge graph entirely (``requires_graph = False``).
It encodes a cell's baseline (control-mean) expression into a Gaussian latent,
adds a perturbation latent derived from the one-hot perturbation flag (CPA-style
additive composition in latent space), and decodes the composed latent back to a
predicted per-gene expression *delta*.

The model conforms to :class:`graph_perturb.models.base.PerturbationModel`:
``forward`` returns a dense ``[B, num_genes]`` delta prediction, and ``loss``
adds a KL term to the reconstruction MSE.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ..config import ModelConfig
from .base import PerturbationModel, baseline_and_flag, batch_to_dense

logger = logging.getLogger(__name__)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout: float) -> nn.Sequential:
    """Build an MLP with ``n_layers`` hidden blocks (Linear+ReLU+Dropout)."""
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(max(1, n_layers)):
        layers.append(nn.Linear(prev, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ConditionalVAE(PerturbationModel):
    """CPA/scGen-style conditional VAE predicting per-gene expression deltas.

    Encoder:    baseline expression ``[B, G]`` -> latent mean/logvar ``[B, L]``.
    Pert head:  one-hot pert flag ``[B, G]``   -> perturbation latent ``[B, L]``.
    Composition: ``z_composed = z_cell + z_pert`` (additive, CPA-style).
    Decoder:    composed latent ``[B, L]``      -> predicted delta ``[B, G]``.

    During training the cell latent is sampled via the reparameterization trick;
    at inference the latent mean is used (deterministic).
    """

    requires_graph: bool = False

    def __init__(self, num_genes: int, cfg: ModelConfig):
        super().__init__(num_genes=num_genes)
        self.cfg = cfg
        self.latent_dim = cfg.latent_dim
        self.kl_weight = cfg.kl_weight

        hidden = cfg.hidden_dim
        n_layers = cfg.n_layers
        dropout = cfg.dropout

        # Encoder: baseline expression -> shared hidden -> (mean, logvar).
        self.encoder = _mlp(num_genes, hidden, hidden, n_layers, dropout)
        self.fc_mu = nn.Linear(hidden, self.latent_dim)
        self.fc_logvar = nn.Linear(hidden, self.latent_dim)

        # Perturbation embedding: one-hot pert flag -> perturbation latent.
        self.pert_encoder = _mlp(num_genes, hidden, self.latent_dim, n_layers, dropout)

        # Decoder: composed latent -> predicted per-gene delta.
        self.decoder = _mlp(self.latent_dim, hidden, num_genes, n_layers, dropout)

    def encode(self, baseline: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map baseline expression ``[B, G]`` to latent ``(mean, logvar)``."""
        h = self.encoder(baseline)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample ``z ~ N(mu, exp(logvar))`` via the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map composed latent ``[B, L]`` to predicted delta ``[B, G]``."""
        return self.decoder(z)

    def _forward_components(
        self, batch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared forward producing ``(pred_delta, mu, logvar)``.

        Uses sampling when training, the latent mean otherwise.
        """
        baseline, flag = baseline_and_flag(batch, self.num_genes)  # both [B, G]
        mu, logvar = self.encode(baseline)
        z_cell = self.reparameterize(mu, logvar) if self.training else mu
        z_pert = self.pert_encoder(flag)
        z = z_cell + z_pert  # CPA-style additive composition
        pred = self.decode(z)
        return pred, mu, logvar

    def forward(self, batch) -> torch.Tensor:
        """Return predicted delta ``[B, num_genes]`` (latent mean at inference)."""
        pred, _, _ = self._forward_components(batch)
        return pred

    def loss(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        """Reconstruction MSE on the delta + ``kl_weight`` * KL divergence."""
        pred, mu, logvar = self._forward_components(batch)
        target = batch_to_dense(batch.y, batch, self.num_genes)

        mse = nn.functional.mse_loss(pred, target)
        # KL( N(mu, sigma^2) || N(0, I) ), averaged over the batch.
        kl = -0.5 * torch.mean(
            torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        )
        loss = mse + self.kl_weight * kl
        logs = {
            "mse": float(mse.detach()),
            "kl": float(kl.detach()),
            "loss": float(loss.detach()),
        }
        return loss, logs
