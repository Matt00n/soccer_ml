"""D2 Group symmetries, pitch transformations, and view generators for soccer graphs."""

from typing import List, Optional, Sequence, Tuple
import torch

from soccer_ml.data.schema import NodeLayout, get_node_layout


# Dihedral group D2 multiplication (Cayley) table
# Group elements: 0: id, 1: horizontal flip, 2: vertical flip, 3: horizontal + vertical flip
D2_INVERSES: List[int] = [0, 1, 2, 3]

D2_MULTIPLY: List[List[int]] = [
    [0, 1, 2, 3],
    [1, 0, 3, 2],
    [2, 3, 0, 1],
    [3, 2, 1, 0],
]


# Subgroups of D2 that can be used for frame averaging.  StatsBomb always
# orients the team in possession left-to-right, so a flip of the x axis (the
# pitch length) reverses the attacking direction and is not a symmetry of
# pass destinations.  The y flip (swapping the wings) approximately is.
SYMMETRY_GROUPS = {
    "none": (0,),
    "y_flip": (0, 2),
    "d2": (0, 1, 2, 3),
}


def resolve_symmetry_group(symmetry: str) -> Tuple[int, ...]:
    """Return the D2 actions of a named symmetry group."""
    try:
        return SYMMETRY_GROUPS[symmetry]
    except KeyError as exc:
        supported = ", ".join(SYMMETRY_GROUPS)
        raise ValueError(f"Unknown symmetry '{symmetry}'. Choose one of: {supported}.") from exc


def flip_grid(grid: torch.Tensor, action: int) -> torch.Tensor:
    """Apply a D2 action to a spatial grid tensor of shape (..., M, N).

    Dimension -2 indexes bins along the pitch length (x), dimension -1 along
    the pitch width (y).  Every D2 action is self-inverse, so the same call
    maps a grid from a transformed view back to the canonical frame.
    """
    dims = {0: [], 1: [-2], 2: [-1], 3: [-2, -1]}[action]
    return grid.flip(dims) if dims else grid


class D2GroupTransform:
    """Implements D2 dihedral reflection transformations on soccer graph features.

    By default node features follow the tracking layout for spatial channels:
        - index 0: x coordinate
        - index 1: y coordinate
        - index 2: v_x velocity
        - index 3: v_y velocity
    Use :meth:`from_layout` for other layouts (StatsBomb 360 has no
    velocities).  All other channels are scalar invariants.
    """

    def __init__(
        self,
        x_idx: int = 0,
        y_idx: int = 1,
        vx_idx: Optional[int] = 2,
        vy_idx: Optional[int] = 3,
    ):
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.vx_idx = vx_idx
        self.vy_idx = vy_idx

    @classmethod
    def from_layout(cls, layout: "str | NodeLayout") -> "D2GroupTransform":
        """Build the transform for a node layout (velocities may be absent)."""
        layout = get_node_layout(layout)
        return cls(x_idx=layout.x, y_idx=layout.y, vx_idx=layout.vx, vy_idx=layout.vy)

    def transform_node_features(self, node_fts: torch.Tensor, action: int) -> torch.Tensor:
        """Apply a D2 reflection action g in {0, 1, 2, 3} to node features.

        Args:
            node_fts: (B, N, D_v) tensor
            action: 0 (id), 1 (H-flip), 2 (V-flip), 3 (HV-flip)

        Returns:
            Transformed node features (B, N, D_v)
        """
        out = node_fts.clone()
        x_channels = [self.x_idx] + ([self.vx_idx] if self.vx_idx is not None else [])
        y_channels = [self.y_idx] + ([self.vy_idx] if self.vy_idx is not None else [])

        if action in (1, 3):  # Horizontal flip: negate x (and v_x)
            out[..., x_channels] = -out[..., x_channels]
        if action in (2, 3):  # Vertical flip: negate y (and v_y)
            out[..., y_channels] = -out[..., y_channels]

        return out

    def generate_all_views(
        self,
        node_fts: torch.Tensor,
        edge_fts: torch.Tensor,
        graph_fts: torch.Tensor,
        actions: Sequence[int] = (0, 1, 2, 3),
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Generate the D2-transformed views for node, edge, and graph features.

        Args:
            node_fts: (B, N, D_v)
            edge_fts: (B, N, N, D_e)
            graph_fts: (B, D_g)
            actions: D2 actions (a subgroup) to generate views for; all 4 by default.

        Returns:
            Tuple of:
                - List of len(actions) node feature tensors [(B, N, D_v), ...]
                - List of len(actions) edge feature tensors [(B, N, N, D_e), ...]
                - List of len(actions) graph feature tensors [(B, D_g), ...]
        """
        node_views: List[torch.Tensor] = []
        edge_views: List[torch.Tensor] = []
        graph_views: List[torch.Tensor] = []

        for g in actions:
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
