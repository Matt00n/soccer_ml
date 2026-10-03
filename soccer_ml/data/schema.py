"""Data schemas and representations for soccer graph data."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass(frozen=True)
class NodeLayout:
    """Channel positions of the node feature tensor.

    ``vx``/``vy`` are ``None`` when the data has no velocities; reflections
    then only negate the coordinates.  ``teammate``/``keeper`` are ``None``
    when the layout lacks those flags.
    """

    name: str
    dim: int
    x: int
    y: int
    vx: Optional[int]
    vy: Optional[int]
    actor: int
    teammate: Optional[int]
    keeper: Optional[int]


# StatsBomb 360 freeze frames carry only positions and three flags.  They have
# no velocities or player attributes, so no constant placeholder channels.
STATSBOMB_NODE_LAYOUT = NodeLayout(
    name="statsbomb", dim=5, x=0, y=1, vx=None, vy=None, actor=2, teammate=3, keeper=4
)
# StatsBomb 360 with velocities estimated by matching players to the previous
# freeze frame (see NextActionDataset(velocities=True)); channel 7 flags
# whether a player could be matched (velocity known).
STATSBOMB_VELOCITY_NODE_LAYOUT = NodeLayout(
    name="statsbomb_velocity", dim=8, x=0, y=1, vx=5, vy=6, actor=2, teammate=3, keeper=4
)
# TacticAI-style tracking layout used by the synthetic dataset:
# x, y, vx, vy, height, weight, ball possession (+ optional extra channels).
TRACKING_NODE_LAYOUT = NodeLayout(
    name="tracking", dim=7, x=0, y=1, vx=2, vy=3, actor=6, teammate=None, keeper=None
)
NODE_LAYOUTS = {
    layout.name: layout for layout in (STATSBOMB_NODE_LAYOUT, STATSBOMB_VELOCITY_NODE_LAYOUT, TRACKING_NODE_LAYOUT)
}


def get_node_layout(layout: "str | NodeLayout") -> NodeLayout:
    if isinstance(layout, NodeLayout):
        return layout
    try:
        return NODE_LAYOUTS[layout]
    except KeyError as exc:
        raise ValueError(f"Unknown node layout '{layout}'. Choose one of: {', '.join(NODE_LAYOUTS)}.") from exc


@dataclass
class SoccerSample:
    """Represents a single soccer situation / frame graph.

    Attributes:
        node_features: (N, D_v) tensor representing on-pitch players, laid out
            as described by a :class:`NodeLayout`.  StatsBomb 360 data uses
            ``STATSBOMB_NODE_LAYOUT``:
                - index 0: x coordinate (normalized/zero-centered)
                - index 1: y coordinate (normalized/zero-centered)
                - index 2: actor flag (1.0 for the player performing the event)
                - index 3: is_teammate flag (1.0 if teammate of actor, 0.0 if opponent)
                - index 4: is_keeper flag (1.0 if goalkeeper, 0.0 otherwise)
            Synthetic tracking data uses ``TRACKING_NODE_LAYOUT``
            (x, y, v_x, v_y, height, weight, ball possession).
        edge_features: (N, N, D_e) tensor.
            Standard features include:
                - index 0: opposing team indicator (1.0 if opposing, 0.0 if same team)
        graph_features: (D_g,) tensor representing global game state, pass context, or conditioning tokens.
        adj_mat: (N, N) tensor representing adjacency connections (default: all 1s).
        receiver_label: Optional target node index (0 <= receiver < N) for receiver prediction.
        target_zone_label: Optional discrete zone index (0 <= zone < M * N) for pitch grid target prediction.
        pass_end_location: Optional continuous (2,) tensor with raw (x, y) pass destination coordinates.
        shot_label: Optional binary outcome float (1.0 for shot, 0.0 otherwise).
        creativity_label: Optional node-level (N,) or graph-level (1,) target value for creativity tasks.
        player_mask: (N,) boolean tensor indicating valid/tracked players (1 = valid, 0 = dummy/padded).
        metadata: Optional dictionary with extra context (e.g. match_id, frame_id, team_names).
    """

    node_features: torch.Tensor
    edge_features: torch.Tensor
    graph_features: torch.Tensor
    adj_mat: torch.Tensor
    receiver_label: Optional[torch.Tensor] = None
    target_zone_label: Optional[torch.Tensor] = None
    pass_end_location: Optional[torch.Tensor] = None
    shot_label: Optional[torch.Tensor] = None
    creativity_label: Optional[torch.Tensor] = None
    player_mask: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SoccerGraphBatch:
    """Represents a batch of soccer graph frames.

    Attributes:
        node_features: (B, N, D_v) float tensor
        edge_features: (B, N, N, D_e) float tensor
        graph_features: (B, D_g) float tensor
        adj_mat: (B, N, N) float tensor
        receiver_label: Optional (B,) long tensor
        target_zone_label: Optional (B,) long tensor
        pass_end_location: Optional (B, 2) float tensor
        shot_label: Optional (B, 1) float tensor
        creativity_label: Optional (B, N) or (B, 1) float tensor
        player_mask: Optional (B, N) bool tensor
    """

    node_features: torch.Tensor
    edge_features: torch.Tensor
    graph_features: torch.Tensor
    adj_mat: torch.Tensor
    receiver_label: Optional[torch.Tensor] = None
    target_zone_label: Optional[torch.Tensor] = None
    pass_end_location: Optional[torch.Tensor] = None
    shot_label: Optional[torch.Tensor] = None
    creativity_label: Optional[torch.Tensor] = None
    player_mask: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "SoccerGraphBatch":
        """Move all contained tensors to the specified device."""
        return SoccerGraphBatch(
            node_features=self.node_features.to(device),
            edge_features=self.edge_features.to(device),
            graph_features=self.graph_features.to(device),
            adj_mat=self.adj_mat.to(device),
            receiver_label=(
                self.receiver_label.to(device) if self.receiver_label is not None else None
            ),
            target_zone_label=(
                self.target_zone_label.to(device) if self.target_zone_label is not None else None
            ),
            pass_end_location=(
                self.pass_end_location.to(device) if self.pass_end_location is not None else None
            ),
            shot_label=(
                self.shot_label.to(device) if self.shot_label is not None else None
            ),
            creativity_label=(
                self.creativity_label.to(device) if self.creativity_label is not None else None
            ),
            player_mask=(
                self.player_mask.to(device) if self.player_mask is not None else None
            ),
        )

    @property
    def batch_size(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[1]

    @property
    def node_dim(self) -> int:
        return self.node_features.shape[2]

    @property
    def edge_dim(self) -> int:
        return self.edge_features.shape[3]

    @property
    def graph_dim(self) -> int:
        return self.graph_features.shape[1]
