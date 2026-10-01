"""Multi-layer D2 GATv2 Backbone Encoder for soccer data."""

from typing import List, Optional
import torch
import torch.nn as nn

from soccer_ml.data.transforms import D2GroupTransform, resolve_symmetry_group
from soccer_ml.models.group_conv import D2GroupConvLayer


class D2GATv2Backbone(nn.Module):
    """Backbone encoder composed of T layers of D2 group convolutions with GATv2.

    Following TacticAI (arXiv:2310.10553 Methods):
        - Takes input graphs and prepares the 4 D2-transformed views.
        - Applies 4 group convolutional layers (Equation 8).
        - Each layer has 8 attention heads.
        - Produces latent node representations H in R^{4 x B x N x D_hidden}.

    ``node_layout`` names the node feature layout (``"tracking"`` or
    ``"statsbomb"``) so reflections negate the right channels.
    ``symmetry`` selects the reflection group used for the views: ``"d2"``
    (TacticAI, 4 views), ``"y_flip"`` (wing reflection only, 2 views), or
    ``"none"`` (identity view only).  ``self.group_actions[i]`` is the D2 action
    that produced view ``i``.
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
        symmetry: str = "d2",
        node_layout: str = "tracking",
    ):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nb_heads = nb_heads
        self.symmetry = symmetry
        self.group_actions = resolve_symmetry_group(symmetry)
        self.node_layout = node_layout
        self.group_transform = D2GroupTransform.from_layout(node_layout)

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
                    group_actions=self.group_actions,
                )
            )

    def forward(
        self,
        node_fts: torch.Tensor,
        edge_fts: torch.Tensor,
        graph_fts: torch.Tensor,
        adj_mat: Optional[torch.Tensor] = None,
        graph_views: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """Encode input graph into D2-equivariant view representations.

        Args:
            node_fts: (B, N, in_node_dim)
            edge_fts: (B, N, N, in_edge_dim)
            graph_fts: (B, in_graph_dim); ignored when ``graph_views`` is given
            adj_mat: Optional (B, N, N)
            graph_views: Optional per-view graph features, one (B, in_graph_dim)
                tensor per group action.  Use this for graph context that is not
                reflection invariant (e.g. an encoding of event locations), with
                view ``i`` expressed in the frame of ``group_actions[i]``.

        Returns:
            List of V view representations, each of shape (B, N, hidden_dim)
        """
        # Step 1: Generate the transformed views
        if graph_views is not None:
            if len(graph_views) != len(self.group_actions):
                raise ValueError(f"Expected {len(self.group_actions)} graph views, got {len(graph_views)}.")
            graph_fts = graph_views[0]
        node_views, edge_views, default_graph_views = self.group_transform.generate_all_views(
            node_fts, edge_fts, graph_fts, actions=self.group_actions
        )
        if graph_views is None:
            graph_views = default_graph_views

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
