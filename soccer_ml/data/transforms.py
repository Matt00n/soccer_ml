"""D2 Group symmetries, pitch transformations, and view generators for soccer graphs."""

from typing import List, Tuple
import torch


# Dihedral group D2 multiplication (Cayley) table
# Group elements: 0: id, 1: horizontal flip, 2: vertical flip, 3: horizontal + vertical flip
D2_INVERSES: List[int] = [0, 1, 2, 3]

D2_MULTIPLY: List[List[int]] = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
]


class D2GroupTransform:
    """Implements D2 dihedral reflection transformations on soccer graph features.

    Node features assume the following layout for spatial channels:
        - index 0: x coordinate
        - index 1: y coordinate
        - index 2: v_x velocity
        - index 3: v_y velocity
    Other channels (height, weight, possession, role) are scalar invariants.
    """

    def __init__(
        self,
        x_idx: int = 0,
        y_idx: int = 1,
        vx_idx: int = 2,
        vy_idx: int = 3,
    ):
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.vx_idx = vx_idx
        self.vy_idx = vy_idx

    def transform_node_features(self, node_fts: torch.Tensor, action: int) -> torch.Tensor:
        """Apply a D2 reflection action g in {0, 1, 2, 3} to node features.

        Args:
            node_fts: (B, N, D_v) tensor
            action: 0 (id), 1 (H-flip), 2 (V-flip), 3 (HV-flip)

        Returns:
            Transformed node features (B, N, D_v)
        """
        out = node_fts.clone()

        if action == 1:  # Horizontal flip: negate x and v_x
            out[..., self.x_idx] = -out[..., self.x_idx]
            out[..., self.vx_idx] = -out[..., self.vx_idx]
        elif action == 2:  # Vertical flip: negate y and v_y
            out[..., self.y_idx] = -out[..., self.y_idx]
            out[..., self.vy_idx] = -out[..., self.vy_idx]
        elif action == 3:  # Both flips: negate x, y, v_x, v_y
            out[..., self.x_idx] = -out[..., self.x_idx]
            out[..., self.y_idx] = -out[..., self.y_idx]
            out[..., self.vx_idx] = -out[..., self.vx_idx]
            out[..., self.vy_idx] = -out[..., self.vy_idx]

        return out

    def generate_all_views(
        self,
        node_fts: torch.Tensor,
        edge_fts: torch.Tensor,
        graph_fts: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Generate the 4 D2-transformed views for node, edge, and graph features.

        Args:
            node_fts: (B, N, D_v)
            edge_fts: (B, N, N, D_e)
            graph_fts: (B, D_g)

        Returns:
            Tuple of:
                - List of 4 node feature tensors [(B, N, D_v), ...]
                - List of 4 edge feature tensors [(B, N, N, D_e), ...]
                - List of 4 graph feature tensors [(B, D_g), ...]
        """
        node_views: List[torch.Tensor] = []
        edge_views: List[torch.Tensor] = []
        graph_views: List[torch.Tensor] = []

        for g in range(4):
            node_views.append(self.transform_node_features(node_fts, g))
            # Edge features and graph features in TacticAI are team/task indicators
            # and invariant under spatial reflections. Clone to preserve separate view references.
            edge_views.append(edge_fts.clone())
            graph_views.append(graph_fts.clone())

        return node_views, edge_views, graph_views


def normalize_pitch_coordinates(
    x: torch.Tensor,
    y: torch.Tensor,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    target_scale: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero-center coordinates and normalize onto a standardized target scale (default 10m x 10m).

    Following TacticAI (arXiv:2310.10553 Methods):
        "we zero-centered the positions of on-pitch players and normalised them
        onto a 10m x 10m pitch, and their velocities were re-scaled accordingly."
    """
    # Assuming input (0, 0) to (pitch_length, pitch_width)
    x_centered = x - (pitch_length / 2.0)
    y_centered = y - (pitch_width / 2.0)

    # Scale to [-target_scale/2, target_scale/2]
    x_norm = (x_centered / pitch_length) * target_scale
    y_norm = (y_centered / pitch_width) * target_scale

    return x_norm, y_norm
