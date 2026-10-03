"""Loss functions for soccer predictive and generative tasks."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    """VAE Reconstruction + KL divergence loss (TacticAI Eq. 11).

    L(theta, phi) = -E_{h_u}[log p_phi(x_u | h_u, o)] + KL(q_theta(h_u | X, o) || p(h_u | o))
    """

    def __init__(self, kl_weight: float = 1.0):
        super().__init__()
        self.kl_weight = kl_weight

    def forward(
        self,
        recon_x: torch.Tensor,
        target_x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss.

        Args:
            recon_x: Reconstructed player features (B, N, D)
            target_x: Ground truth player features (B, N, D)
            mu: Latent Gaussian mean (B, N, latent_dim)
            logvar: Latent Gaussian log-variance (B, N, latent_dim)
            mask: Optional (B, N) boolean mask

        Returns:
            Dictionary with 'loss', 'recon_loss', 'kl_loss'
        """
        # Reconstruction MSE loss per player
        diff = (recon_x - target_x) ** 2  # (B, N, D)
        recon_per_node = diff.sum(dim=-1)  # (B, N)

        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_per_node = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=-1
        )  # (B, N)

        if mask is not None:
            mask_float = mask.float()
            denom = mask_float.sum().clamp(min=1.0)
            recon_loss = (recon_per_node * mask_float).sum() / denom
            kl_loss = (kl_per_node * mask_float).sum() / denom
        else:
            recon_loss = recon_per_node.mean()
            kl_loss = kl_per_node.mean()

        total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


class SoccerMultiTaskLoss(nn.Module):
    """Combines receiver, target location, shot, creativity, and VAE losses."""

    def __init__(
        self,
        receiver_weight: float = 1.0,
        target_location_weight: float = 1.0,
        shot_weight: float = 1.0,
        creativity_weight: float = 1.0,
        vae_weight: float = 1.0,
        vae_kl_weight: float = 0.01,
    ):
        super().__init__()
        self.receiver_weight = receiver_weight
        self.target_location_weight = target_location_weight
        self.shot_weight = shot_weight
        self.creativity_weight = creativity_weight
        self.vae_weight = vae_weight

        self.receiver_criterion = nn.CrossEntropyLoss()
        self.shot_criterion = nn.BCEWithLogitsLoss()
        self.creativity_criterion = nn.MSELoss()
        self.vae_criterion = VAELoss(kl_weight=vae_kl_weight)

    def forward(
        self,
        model_outputs: Dict[str, Any],
        batch: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute composite multi-task loss."""
        total_loss = torch.tensor(0.0, device=model_outputs["node_embeddings"].device)
        losses: Dict[str, torch.Tensor] = {}

        # 1. Receiver prediction (Node classification)
        if "receiver_logits" in model_outputs and batch.receiver_label is not None:
            receiver_loss = self.receiver_criterion(
                model_outputs["receiver_logits"], batch.receiver_label
            )
            total_loss = total_loss + self.receiver_weight * receiver_loss
            losses["receiver_loss"] = receiver_loss

        # 2. Pass target location (pitch grid classification)
        target_zone_label = getattr(batch, "target_zone_label", None)
        if "target_zone_logits" in model_outputs and target_zone_label is not None:
            target_location_loss = self.receiver_criterion(
                model_outputs["target_zone_logits"], target_zone_label
            )
            total_loss = total_loss + self.target_location_weight * target_location_loss
            losses["target_location_loss"] = target_location_loss

        # 3. Shot prediction (Graph binary classification)
        if "shot_logits" in model_outputs and batch.shot_label is not None:
            shot_loss = self.shot_criterion(
                model_outputs["shot_logits"], batch.shot_label
            )
            total_loss = total_loss + self.shot_weight * shot_loss
            losses["shot_loss"] = shot_loss

        # 4. Player Creativity score (Node regression)
        if "creativity_scores" in model_outputs and batch.creativity_label is not None:
            creativity_loss = self.creativity_criterion(
                model_outputs["creativity_scores"], batch.creativity_label
            )
            total_loss = total_loss + self.creativity_weight * creativity_loss
            losses["creativity_loss"] = creativity_loss

        # 5. Generative VAE adjustment loss
        if "vae_recon" in model_outputs:
            target_xy_v = batch.node_features[
                ..., :4
            ]  # ground-truth coordinates & velocities
            vae_losses = self.vae_criterion(
                recon_x=model_outputs["vae_recon"],
                target_x=target_xy_v,
                mu=model_outputs["vae_mu"],
                logvar=model_outputs["vae_logvar"],
                mask=batch.player_mask,
            )
            total_loss = total_loss + self.vae_weight * vae_losses["loss"]
            losses["vae_loss"] = vae_losses["loss"]
            losses["vae_recon_loss"] = vae_losses["recon_loss"]
            losses["vae_kl_loss"] = vae_losses["kl_loss"]

        losses["total_loss"] = total_loss
        return losses
