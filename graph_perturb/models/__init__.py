"""Models: GraphSAGE+attention GNN and a conditional VAE baseline."""

from __future__ import annotations

from .base import PerturbationModel, baseline_and_flag, batch_to_dense


def build_model(model_cfg, num_genes: int, graph=None) -> PerturbationModel:
    """Factory dispatching on ``model_cfg.name`` (``gnn`` | ``vae``)."""
    name = model_cfg.name.lower()
    if name == "gnn":
        from .gnn import GraphSAGEPerturbation

        return GraphSAGEPerturbation(num_genes=num_genes, cfg=model_cfg, graph=graph)
    if name == "vae":
        from .vae import ConditionalVAE

        return ConditionalVAE(num_genes=num_genes, cfg=model_cfg)
    raise KeyError(f"unknown model {model_cfg.name!r}; expected 'gnn' or 'vae'")


__all__ = [
    "PerturbationModel",
    "batch_to_dense",
    "baseline_and_flag",
    "build_model",
]
