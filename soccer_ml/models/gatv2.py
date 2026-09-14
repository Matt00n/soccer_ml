"""Graph Attention Network v2 (GATv2) layer faithful to Brody et al. (ICLR 2022) and DeepMind CLRS."""

from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """GATv2 Graph Attention Layer.

    Follows Brody et al. (ICLR 2022) and the DeepMind CLRS baseline implementation:
        - Incorporates node features, edge features, and graph-level features into the attention logits.
        - Supports multi-head attention with head-specific scoring vectors.
        - Includes residual skip connection, LayerNorm, and custom activation.
    """

    def __init__(
        self,
        in_node_dim: int,
        out_node_dim: int,
        nb_heads: int = 8,
        in_edge_dim: int = 1,
        in_graph_dim: int = 4,
        mid_dim: Optional[int] = None,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        negative_slope: float = 0.2,
        residual: bool = True,
        use_ln: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_node_dim = in_node_dim
        self.out_node_dim = out_node_dim
        self.nb_heads = nb_heads
        self.mid_dim = mid_dim if mid_dim is not None else out_node_dim
        self.activation = activation
        self.negative_slope = negative_slope
        self.residual = residual
        self.use_ln = use_ln
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if self.out_node_dim % self.nb_heads != 0:
            raise ValueError(
                f"out_node_dim ({out_node_dim}) must be divisible by nb_heads ({nb_heads})"
            )
        if self.mid_dim % self.nb_heads != 0:
            raise ValueError(
                f"mid_dim ({self.mid_dim}) must be divisible by nb_heads ({nb_heads})"
            )

        self.head_dim = self.out_node_dim // self.nb_heads
        self.mid_head_dim = self.mid_dim // self.nb_heads

        # Linear projections for values and skip connection
        self.linear_val = nn.Linear(in_node_dim, self.out_node_dim, bias=False)
        if self.residual:
            self.linear_skip = nn.Linear(in_node_dim, self.out_node_dim, bias=False)
        else:
            self.linear_skip = None

        # Pre-attention projections for receiver node (w1), sender node (w2), edges (we), globals (wg)
        self.w_1 = nn.Linear(in_node_dim, self.mid_dim, bias=False)
        self.w_2 = nn.Linear(in_node_dim, self.mid_dim, bias=False)
        self.w_e = nn.Linear(in_edge_dim, self.mid_dim, bias=False)
        self.w_g = nn.Linear(in_graph_dim, self.mid_dim, bias=False)

        # Attention scoring parameter vector 'a' per head: (nb_heads, mid_head_dim)
        self.a_heads = nn.Parameter(torch.empty(self.nb_heads, self.mid_head_dim))
        nn.init.xavier_uniform_(self.a_heads.unsqueeze(0))

        # Layer normalization
        if self.use_ln:
            self.layer_norm = nn.LayerNorm(self.out_node_dim)
        else:
            self.layer_norm = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_val.weight)
        if self.linear_skip is not None:
            nn.init.xavier_uniform_(self.linear_skip.weight)
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.xavier_uniform_(self.w_e.weight)
        nn.init.xavier_uniform_(self.w_g.weight)

    def forward(
        self,
        node_fts: torch.Tensor,
        edge_fts: torch.Tensor,
        graph_fts: torch.Tensor,
        adj_mat: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform GATv2 forward inference.

        Args:
            node_fts: (B, N, D_in) node feature tensor
            edge_fts: (B, N, N, D_e) edge feature tensor
            graph_fts: (B, D_g) graph-level feature tensor
            adj_mat: Optional (B, N, N) adjacency matrix (1.0 = connected, 0.0 = disconnected)
            hidden: Optional (B, N, D_h) additional hidden state to concatenate with node_fts

        Returns:
            Updated node representations of shape (B, N, out_node_dim)
        """
        b, n, _ = node_fts.shape

        if hidden is not None:
            z = torch.cat([node_fts, hidden], dim=-1)
        else:
            z = node_fts

        if adj_mat is None:
            adj_mat = torch.ones((b, n, n), dtype=node_fts.dtype, device=node_fts.device)

        # Compute values: (B, N, out_node_dim) -> (B, N, H, head_dim) -> (B, H, N, head_dim)
        values = self.linear_val(z)
        values = values.view(b, n, self.nb_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute pre-attention projections
        pre_att_1 = self.w_1(z)  # (B, N, mid_dim)
        pre_att_2 = self.w_2(z)  # (B, N, mid_dim)
        pre_att_e = self.w_e(edge_fts)  # (B, N, N, mid_dim)
        pre_att_g = self.w_g(graph_fts)  # (B, mid_dim)

        # Broadcast into (B, N, N, mid_dim):
        # pre_att_1 is receiver node (dim 1), pre_att_2 is sender node (dim 2)
        pre_att = (
            pre_att_1.unsqueeze(1)  # (B, 1, N, mid_dim)
            + pre_att_2.unsqueeze(2)  # (B, N, 1, mid_dim)
            + pre_att_e  # (B, N, N, mid_dim)
            + pre_att_g.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, mid_dim)
        )

        # Reshape pre_att to (B, H, N, N, mid_head_dim)
        pre_att = pre_att.view(b, n, n, self.nb_heads, self.mid_head_dim).permute(0, 3, 1, 2, 4)

        # Apply LeakyReLU non-linearity before attention vector scoring
        pre_att = F.leaky_relu(pre_att, negative_slope=self.negative_slope)

        # Compute attention logits: dot product with head scoring vectors 'a_heads'
        # a_heads is (H, mid_head_dim) -> (1, H, 1, 1, mid_head_dim)
        a_vec = self.a_heads.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        logits = (pre_att * a_vec).sum(dim=-1)  # (B, H, N, N)

        # Apply mask from adjacency matrix: disconnected edges get large negative penalty
        bias_mat = (adj_mat - 1.0) * 1e9  # (B, N, N)
        bias_mat = bias_mat.unsqueeze(1)  # (B, 1, N, N)

        attention_weights = F.softmax(logits + bias_mat, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Aggregate values: (B, H, N, N) x (B, H, N, head_dim) -> (B, H, N, head_dim)
        out = torch.matmul(attention_weights, values)

        # Reshape back to (B, N, H * head_dim) = (B, N, out_node_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, self.out_node_dim)

        # Residual connection
        if self.residual and self.linear_skip is not None:
            out = out + self.linear_skip(z)

        # Activation
        if self.activation is not None:
            out = self.activation(out)

        # Layer normalization
        out = self.layer_norm(out)

        return out
