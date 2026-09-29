"""End-to-end Soccer GATv2 model assembling D2 backbone and modular prediction heads."""

from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn

from soccer_ml.data.schema import SoccerGraphBatch
from soccer_ml.models.backbone import D2GATv2Backbone
from soccer_ml.models.heads import (
    NodeInvariantHead,
    GraphInvariantHead,
    PitchTargetGridHead,
    EquivariantVAEHead,
    frame_average_views,
)


class SoccerGATv2Model(nn.Module):
    """Full Soccer GATv2 Architecture with D2 Group Equivariance.

    Supports:
        1. Node classification (e.g., Receiver prediction, ball target player)
        2. Pitch target grid prediction (e.g., Target location discretized into M x N zones)
        3. Node regression (e.g., Player creativity index, expected threat per player)
        4. Graph classification / regression (e.g., Threatening shot likelihood, expected goals)
        5. Generative VAE refinement (e.g., Adjusting player positions and velocities conditioned on outcome)
    """

    def __init__(
        self,
        in_node_dim: int = 9,
        in_edge_dim: int = 1,
        in_graph_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 4,
        nb_heads: int = 8,
        num_zones: int = 96,
        use_mlp: bool = False,
        dropout: float = 0.0,
        enable_receiver_head: bool = True,
        enable_target_location_head: bool = True,
        enable_shot_head: bool = True,
        enable_creativity_head: bool = True,
        enable_vae_head: bool = True,
        vae_latent_dim: int = 16,
    ):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.in_edge_dim = in_edge_dim
        self.in_graph_dim = in_graph_dim
        self.hidden_dim = hidden_dim
        self.num_zones = num_zones

        # Core D2 GATv2 Backbone
        self.backbone = D2GATv2Backbone(
            in_node_dim=in_node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nb_heads=nb_heads,
            in_edge_dim=in_edge_dim,
            in_graph_dim=in_graph_dim,
            dropout=dropout,
        )

        # Invariant Node Head for Receiver Prediction (outputs 1 logit per player)
        self.receiver_head = (
            NodeInvariantHead(hidden_dim=hidden_dim, out_dim=1, dropout=dropout)
            if enable_receiver_head
            else None
        )

        # Invariant Head for Pitch Target Grid Location (outputs num_zones logits)
        self.target_location_head = (
            PitchTargetGridHead(
                hidden_dim=hidden_dim,
                num_zones=num_zones,
                use_mlp=use_mlp,
                dropout=dropout,
            )
            if enable_target_location_head
            else None
        )

        # Invariant Graph Head for Shot / Event Prediction (outputs 1 logit per graph)
        self.shot_head = (
            GraphInvariantHead(hidden_dim=hidden_dim, out_dim=1, dropout=dropout)
            if enable_shot_head
            else None
        )

        # Invariant Node Head for continuous Player Creativity Score (outputs 1 score per player)
        self.creativity_head = (
            NodeInvariantHead(hidden_dim=hidden_dim, out_dim=1, dropout=dropout)
            if enable_creativity_head
            else None
        )

        # Equivariant VAE Head for generating coordinate/velocity adjustments
        self.vae_head = (
            EquivariantVAEHead(
                hidden_dim=hidden_dim,
                out_channels=4,  # x, y, vx, vy
                condition_dim=1,  # conditioning on binary shot outcome
                latent_dim=vae_latent_dim,
            )
            if enable_vae_head
            else None
        )

    def forward(
        self,
        batch_or_node_fts: Union[SoccerGraphBatch, torch.Tensor],
        edge_fts: Optional[torch.Tensor] = None,
        graph_fts: Optional[torch.Tensor] = None,
        adj_mat: Optional[torch.Tensor] = None,
        player_mask: Optional[torch.Tensor] = None,
        condition_for_vae: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass through backbone and active heads.

        Can accept either a SoccerGraphBatch instance or individual tensors.
        """
        if isinstance(batch_or_node_fts, SoccerGraphBatch):
            batch = batch_or_node_fts
            node_fts = batch.node_features
            edge_fts = batch.edge_features
            graph_fts = batch.graph_features
            adj_mat = batch.adj_mat
            player_mask = batch.player_mask
            if condition_for_vae is None and batch.shot_label is not None:
                condition_for_vae = batch.shot_label
        else:
            node_fts = batch_or_node_fts

        # Backbone generates 4 D2 view representations
        views = self.backbone(
            node_fts=node_fts,
            edge_fts=edge_fts,
            graph_fts=graph_fts,
            adj_mat=adj_mat,
        )

        outputs: Dict[str, Any] = {
            "views": views,
            "node_embeddings": frame_average_views(views),  # (B, N, hidden_dim)
        }

        # Receiver Prediction Head
        if self.receiver_head is not None:
            outputs["receiver_logits"] = self.receiver_head(views, mask=player_mask)

        # Target Location Grid Prediction Head
        if self.target_location_head is not None:
            actor_mask = (node_fts[..., 6] > 0.5) if node_fts.shape[-1] > 6 else None
            outputs["target_zone_logits"] = self.target_location_head(
                views, mask=player_mask, actor_mask=actor_mask
            )

        # Shot Prediction Head
        if self.shot_head is not None:
            outputs["shot_logits"] = self.shot_head(views, mask=player_mask)

        # Creativity Head
        if self.creativity_head is not None:
            outputs["creativity_scores"] = self.creativity_head(views, mask=player_mask)

        # VAE Generative Head
        if self.vae_head is not None and condition_for_vae is not None:
            h_id = views[0]  # Identity view for equivariant generation
            recon, mu, logvar = self.vae_head(h_id, condition=condition_for_vae)
            outputs["vae_recon"] = recon
            outputs["vae_mu"] = mu
            outputs["vae_logvar"] = logvar

        return outputs
