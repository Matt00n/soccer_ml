"""Decoder heads for soccer tasks: Invariant Node/Graph heads and Equivariant VAE head."""

from typing import List, Tuple, Optional, Sequence
import torch
import torch.nn as nn

from soccer_ml.data.transforms import flip_grid


def frame_average_views(views: List[torch.Tensor]) -> torch.Tensor:
    """Computes frame-averaging over the 4 D2 views (TacticAI Eq. 9).

    H^{node} = 1/4 * (H_id + H_h + H_v + H_hv)

    Args:
        views: List of 4 tensors, each of shape (B, N, D)

    Returns:
        Frame-averaged tensor of shape (B, N, D)
    """
    return torch.stack(views, dim=0).mean(dim=0)


class NodeInvariantHead(nn.Module):
    """Invariant node-level prediction head (e.g. for receiver prediction or player creativity score).

    Performs D2 frame-averaging (Eq. 9) followed by an MLP/linear decoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int = 1,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        if mlp_hidden_dim is not None:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(mlp_hidden_dim, out_dim),
            )
        else:
            self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        views: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode node predictions from D2 views.

        Args:
            views: List of 4 view tensors, each of shape (B, N, hidden_dim)
            mask: Optional boolean mask (B, N)

        Returns:
            Logits or scores of shape (B, N) if out_dim == 1, else (B, N, out_dim)
        """
        # Step 1: Invariant frame averaging across the 4 reflection views
        h_node = frame_average_views(views)  # (B, N, hidden_dim)

        # Step 2: Decode per-player logits
        out = self.decoder(h_node)  # (B, N, out_dim)

        if self.out_dim == 1:
            out = out.squeeze(-1)  # (B, N)

        if mask is not None:
            # Mask out invalid/bench players with large negative value for softmax classification
            out = out.masked_fill(~mask, -1e9)

        return out


class GraphInvariantHead(nn.Module):
    """Invariant graph-level prediction head (e.g. for threatening shot or goal probability).

    Performs D2 frame-averaging, followed by mean pooling across players (Eq. 10), and an MLP/linear decoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int = 1,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        if mlp_hidden_dim is not None:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(mlp_hidden_dim, out_dim),
            )
        else:
            self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        views: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode graph-level prediction from D2 views.

        Args:
            views: List of 4 view tensors, each of shape (B, N, hidden_dim)
            mask: Optional (B, N) boolean mask

        Returns:
            Graph logits of shape (B, out_dim)
        """
        # Step 1: Invariant frame averaging across the 4 reflection views
        h_node = frame_average_views(views)  # (B, N, hidden_dim)

        # Step 2: Global pooling over on-pitch players (Eq. 10)
        if mask is not None:
            mask_float = mask.unsqueeze(-1).float()  # (B, N, 1)
            h_graph = (h_node * mask_float).sum(dim=1) / (mask_float.sum(dim=1).clamp(min=1.0))
        else:
            h_graph = h_node.mean(dim=1)  # (B, hidden_dim)

        # Step 3: Decode
        out = self.decoder(h_graph)  # (B, out_dim)
        return out


class EquivariantVAEHead(nn.Module):
    """Equivariant VAE generation head for player movement adjustments.

    Following TacticAI (arXiv:2310.10553 Methods Eq. 11):
        - Takes the identity view H_id (index 0) or outcome-conditioned view.
        - Predicts parameters (mu, log_var) of Gaussian distributions over player coordinates and velocities.
        - Applies the reparameterization trick for differentiable training.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int = 4,  # x, y, v_x, v_y
        condition_dim: int = 1,  # e.g. desired shot outcome (0 or 1)
        latent_dim: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        in_dim = hidden_dim + condition_dim

        # Encoder to latent Gaussian parameters
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # Decoder to reconstructed player movement
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) via reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(
        self,
        h_id: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate movement adjustments conditioned on target outcome.

        Args:
            h_id: Identity view node features of shape (B, N, hidden_dim)
            condition: Target outcome tensor of shape (B, condition_dim)

        Returns:
            Tuple of:
                - reconstructed adjustments (B, N, out_channels)
                - mu (B, N, latent_dim)
                - logvar (B, N, latent_dim)
        """
        b, n, _ = h_id.shape

        # Expand conditioning vector across all players: (B, 1, cond_dim) -> (B, N, cond_dim)
        cond_expanded = condition.unsqueeze(1).expand(-1, n, -1)
        h_cond = torch.cat([h_id, cond_expanded], dim=-1)

        mu = self.fc_mu(h_cond)  # (B, N, latent_dim)
        logvar = self.fc_logvar(h_cond)  # (B, N, latent_dim)
        z = self.reparameterize(mu, logvar)  # (B, N, latent_dim)

        z_cond = torch.cat([z, cond_expanded], dim=-1)
        reconstruction = self.decoder(z_cond)  # (B, N, out_channels)

        return reconstruction, mu, logvar


def pool_graph_readout(
    node_embeddings: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    actor_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Concatenate a masked mean pool with the passer's node embedding.

    Args:
        node_embeddings: (..., N, D)
        mask: Optional (..., N) valid-player mask.
        actor_mask: Optional (..., N) passer mask; frames without an actor fall
            back to the pooled embedding.

    Returns:
        (..., 2 * D) readout.
    """
    if mask is not None:
        w = mask.unsqueeze(-1).to(node_embeddings.dtype)
        pooled = (node_embeddings * w).sum(dim=-2) / w.sum(dim=-2).clamp_min(1.0)
    else:
        pooled = node_embeddings.mean(dim=-2)

    if actor_mask is not None:
        w_actor = actor_mask.unsqueeze(-1).to(node_embeddings.dtype)
        if mask is not None:
            w_actor = w_actor * mask.unsqueeze(-1).to(node_embeddings.dtype)
        actor_count = w_actor.sum(dim=-2)
        actor_emb = (node_embeddings * w_actor).sum(dim=-2) / actor_count.clamp_min(1.0)
        actor_emb = torch.where(actor_count > 0, actor_emb, pooled)
    else:
        actor_emb = pooled
    return torch.cat([pooled, actor_emb], dim=-1)


class PitchTargetGridHead(nn.Module):
    """Predicts pass destination logits over an M x N pitch grid, equivariantly.

    A location target moves with the pitch: reflecting the input must reflect
    the predicted distribution.  Averaging the views *before* decoding (as the
    invariant heads do) would make left- and right-wing situations
    indistinguishable.  Instead, every view is decoded in its own frame, its
    logit grid is mapped back to the canonical frame, and the aligned grids are
    averaged (frame averaging for an equivariant output):

        f(X) = 1/|G| sum_g  rho(g)^{-1} decode(H_g(X))

    Has configurable use_mlp (default False: single linear projection consistent with TacticAI).
    """

    def __init__(
        self,
        in_dim: int,
        grid_m: int,
        grid_n: int,
        group_actions: Sequence[int] = (0, 1, 2, 3),
        use_mlp: bool = False,
        mlp_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.num_zones = grid_m * grid_n
        self.group_actions = tuple(group_actions)
        self.use_mlp = use_mlp

        if use_mlp:
            mlp_dim = mlp_hidden_dim or in_dim
            self.decoder = nn.Sequential(
                nn.Linear(in_dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(mlp_dim, self.num_zones),
            )
        else:
            self.decoder = nn.Linear(in_dim, self.num_zones)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        """Decode pitch target zone logits.

        Args:
            view_features: (V, B, in_dim) per-view readouts, where view ``v`` was
                computed in the frame of ``group_actions[v]``.

        Returns:
            Logits of shape (B, grid_m * grid_n) in the canonical frame.
        """
        num_views, batch_size, _ = view_features.shape
        if num_views != len(self.group_actions):
            raise ValueError(f"Expected {len(self.group_actions)} views, got {num_views}.")
        logits = self.decoder(view_features).view(num_views, batch_size, self.grid_m, self.grid_n)
        aligned = [flip_grid(logits[index], action) for index, action in enumerate(self.group_actions)]
        return torch.stack(aligned, dim=0).mean(dim=0).flatten(1)
