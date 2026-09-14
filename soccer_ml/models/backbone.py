"""Multi-layer D2 GATv2 Backbone Encoder for soccer data."""

from typing import List, Optional
import torch
import torch.nn as nn

from soccer_ml.data.transforms import D2GroupTransform
from soccer_ml.models.group_conv import D2GroupConvLayer


class D2GATv2Backbone(nn.Module):
    """Backbone encoder composed of T layers of D2 group convolutions with GATv2.

    Following TacticAI (arXiv:2310.10553 Methods):
        - Takes input graphs and prepares the 4 D2-transformed views.
        - Applies 4 group convolutional layers (Equation 8).
        - Each layer has 8 attention heads.
        - Produces latent node representations H in R^{4 x B x N x D_hidden}.
    """

    def __init__(
        self,
        in_node_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        nb_heads: int = 8,
        in_edge_dim: int = 1,
        in_graph_dim: int = 4,
        dropout: float = 0.0,
        residual: bool = True,
        use_ln: bool = True,
    ):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nb_heads = nb_heads
        self.group_transform = D2GroupTransform()

        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            current_in_node_dim = in_node_dim if layer_idx == 0 else hidden_dim
            self.layers.append(
                D2GroupConvLayer(
                    in_node_dim=current_in_node_dim,
                    out_node_dim=hidden_dim,
                    nb_heads=nb_heads,
                    in_edge_dim=in_edge_dim,
                    in_graph_dim=in_graph_dim,
                    dropout=dropout,
                    residual=residual,
                    use_ln=use_ln,
                )
            )

    def forward(
        self,
        node_fts: torch.Tensor,
        edge_fts: torch.Tensor,
        graph_fts: torch.Tensor,
        adj_mat: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Encode input graph into 4 D2-equivariant view representations.

        Args:
            node_fts: (B, N, in_node_dim)
            edge_fts: (B, N, N, in_edge_dim)
            graph_fts: (B, in_graph_dim)
            adj_mat: Optional (B, N, N)

        Returns:
            List of 4 view representations [H_0, H_1, H_2, H_3], each of shape (B, N, hidden_dim)
        """
        # Step 1: Generate 4 D2-transformed views
        node_views, edge_views, graph_views = self.group_transform.generate_all_views(
            node_fts, edge_fts, graph_fts
        )

        # Step 2: Pass views through stacked D2 group convolutional layers
        current_node_views = node_views
        for layer in self.layers:
            current_node_views = layer(
                node_views=current_node_views,
                edge_views=edge_views,
                graph_views=graph_views,
                adj_mat=adj_mat,
            )

        return current_node_views
