"""D2 Dihedral Group Convolution Layer matching TacticAI Eq. 8 and CLRS GATv2FullD2."""

from typing import List, Optional, Sequence
import torch
import torch.nn as nn

from soccer_ml.data.transforms import D2_MULTIPLY, D2_INVERSES
from soccer_ml.models.gatv2 import GATv2Layer


class D2GroupConvLayer(nn.Module):
    """Dihedral Group (D2) Convolution Layer with a shared GATv2 processor.

    Implements Equation 8 from TacticAI (arXiv:2310.10553) and GATv2FullD2 from DeepMind CLRS:
        H_g^{(t)} = 1/|G| sum_{h in G} g_G(H_g^{(t-1)} || H_{g^{-1}h}^{(t-1)},
                                           E_g || E_{g^{-1}h},
                                           g_g || g_{g^{-1}h})

    D2 contains 4 reflections:
        - g=0: id
        - g=1: horizontal flip
        - g=2: vertical flip
        - g=3: horizontal + vertical flip
    All elements of D2 are self-inverses (g^{-1} = g).

    ``group_actions`` may restrict the layer to a subgroup of D2, e.g. ``(0, 2)``
    for wing (y) reflections only.  The subgroup must be closed under D2
    multiplication; view ``i`` then corresponds to action ``group_actions[i]``.
    """

    def __init__(
        self,
        in_node_dim: int,
        out_node_dim: int,
        nb_heads: int = 8,
        in_edge_dim: int = 1,
        in_graph_dim: int = 4,
        mid_dim: Optional[int] = None,
        dropout: float = 0.0,
        residual: bool = True,
        use_ln: bool = True,
        group_actions: Sequence[int] = (0, 1, 2, 3),
    ):
        super().__init__()
        self.group_actions = tuple(group_actions)
        self.num_d2_actions = len(self.group_actions)
        self._view_index = {action: index for index, action in enumerate(self.group_actions)}
        for g in self.group_actions:
            for h in self.group_actions:
                if D2_MULTIPLY[g][h] not in self._view_index:
                    raise ValueError(f"group_actions {self.group_actions} is not closed under D2.")
        self.in_node_dim = in_node_dim
        self.out_node_dim = out_node_dim

        # Note: the input to the base processor is concatenated across pair (g, g^{-1}h),
        # so feature dimensions are doubled (2 * in_dim).
        self.gatv2 = GATv2Layer(
            in_node_dim=2 * in_node_dim,
            out_node_dim=out_node_dim,
            nb_heads=nb_heads,
            in_edge_dim=2 * in_edge_dim,
            in_graph_dim=2 * in_graph_dim,
            mid_dim=mid_dim,
            dropout=dropout,
            residual=residual,
            use_ln=use_ln,
        )

    def forward(
        self,
        node_views: List[torch.Tensor],
        edge_views: List[torch.Tensor],
        graph_views: List[torch.Tensor],
        adj_mat: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Perform D2 group convolution forward step.

        Args:
            node_views: List of V tensors, each of shape (B, N, in_node_dim)
            edge_views: List of V tensors, each of shape (B, N, N, in_edge_dim)
            graph_views: List of V tensors, each of shape (B, in_graph_dim)
            adj_mat: Optional (B, N, N) adjacency matrix
            hidden: Optional (B, N, D_h) hidden state

        Returns:
            List of V updated view tensors, each of shape (B, N, out_node_dim),
            where V = len(group_actions).
        """
        v = self.num_d2_actions
        assert len(node_views) == v, f"Expected {v} node views, got {len(node_views)}"
        assert len(edge_views) == v, f"Expected {v} edge views, got {len(edge_views)}"
        assert len(graph_views) == v, f"Expected {v} graph views, got {len(graph_views)}"

        ret_nodes: List[torch.Tensor] = []

        for g_action in self.group_actions:
            g = self._view_index[g_action]
            emb_values = []
            for h_action in self.group_actions:
                # Cayley multiplication: g^{-1}h. In D2, g^{-1} = g.
                gh = self._view_index[D2_MULTIPLY[D2_INVERSES[g_action]][h_action]]

                # Concatenate view g with view gh
                node_fts = torch.cat([node_views[g], node_views[gh]], dim=-1)
                edge_fts = torch.cat([edge_views[g], edge_views[gh]], dim=-1)
                graph_fts = torch.cat([graph_views[g], graph_views[gh]], dim=-1)

                cell_embedding = self.gatv2(
                    node_fts=node_fts,
                    edge_fts=edge_fts,
                    graph_fts=graph_fts,
                    adj_mat=adj_mat,
                    hidden=hidden,
                )
                emb_values.append(cell_embedding)

            # Average across all group members h in G (group convolution aggregation)
            ret_nodes.append(torch.stack(emb_values, dim=0).mean(dim=0))

        return ret_nodes
