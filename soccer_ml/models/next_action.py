"""Next-action model: event-sequence encoder conditioning a single-frame GATv2.

The encoder is the ``seq2spatial`` design (see
:class:`~soccer_ml.models.temporal_receiver.EventSeqToSpatialGATv2`): a masked
GRU reads the possession's event history plus a query token for the decision,
and its state conditions the reflection-view GATv2 over the decision's freeze
frame (as graph features) and the readout.

Three heads factorize the action likelihood:

    log P(action) = log P(type) + log P(zone | type) + log P(height | pass, zone)

* type: logits over :data:`ACTION_TYPES`, averaged over views (reflection
  invariant) and restricted to the actions possible at the decision
  (``batch.type_mask``);
* zone: one 26 x 17 grid per located type (pass, carry).  With visibility on,
  every zone uses one of two linear maps depending on whether it lies inside
  the 360 camera's visible area, so the model can treat unobserved regions
  differently;
* height: a ground/low/high distribution for every destination zone.

Zone and height grids are decoded per view, reflected back to the canonical
frame and averaged, so reflecting the input reflects them exactly.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from soccer_ml.data.next_action_dataset import (
    ACTION_TYPES,
    EDGE_DIM,
    GRAPH_DIM,
    IGNORE_INDEX,
    LOCATED_ACTIONS,
    NUM_EVENT_SUBTYPES,
    PASS_HEIGHTS,
    NextActionBatch,
)
from soccer_ml.data.pitch_grid import DEFAULT_GRID_M, DEFAULT_GRID_N
from soccer_ml.data.schema import STATSBOMB_NODE_LAYOUT, get_node_layout
from soccer_ml.data.transforms import flip_grid
from soccer_ml.models.backbone import D2GATv2Backbone
from soccer_ml.models.heads import pool_graph_readout
from soccer_ml.models.temporal_receiver import (
    EventContextEmbedding,
    _per_view_event_numeric,
    _repeat_views,
    _run_masked_gru,
    _time_embedding,
    _time_features,
)

# Located-action head index of each action type (-1: no end zone).
_LOCATED_INDEX = torch.tensor([LOCATED_ACTIONS.index(a) if a in LOCATED_ACTIONS else -1 for a in ACTION_TYPES])
_PASS = ACTION_TYPES.index("pass")
# Pitch normalization used by NextActionDataset (StatsBomb yards onto a 10 x 10 box).
_PITCH_LENGTH, _PITCH_WIDTH, _TARGET_SCALE = 120.0, 80.0, 10.0


def _linear_or_mlp(in_dim: int, out_dim: int, use_mlp: bool, dropout: float) -> nn.Module:
    if not use_mlp:
        return nn.Linear(in_dim, out_dim)
    return nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(in_dim, out_dim))


class NextActionModel(nn.Module):
    """Event-sequence-to-GATv2 next-action model (type, end zone, pass height)."""

    model_name = "next_action"

    def __init__(
        self,
        in_node_dim: int = STATSBOMB_NODE_LAYOUT.dim,
        in_edge_dim: int = EDGE_DIM,
        in_graph_dim: int = GRAPH_DIM,
        hidden_dim: int = 128,
        num_layers: int = 4,
        nb_heads: int = 8,
        temporal_layers: int = 2,
        max_sequence_length: int = 17,
        dropout: float = 0.1,
        grid_m: int = DEFAULT_GRID_M,
        grid_n: int = DEFAULT_GRID_N,
        symmetry: str = "y_flip",
        use_mlp: bool = False,
        use_history: bool = True,
        use_visibility: bool = True,
        node_layout: str = STATSBOMB_NODE_LAYOUT.name,
        edge_distance: bool = False,
    ):
        super().__init__()
        self.model_kwargs: Dict[str, Any] = {
            "in_node_dim": in_node_dim,
            "in_edge_dim": in_edge_dim,
            "in_graph_dim": in_graph_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "nb_heads": nb_heads,
            "temporal_layers": temporal_layers,
            "max_sequence_length": max_sequence_length,
            "dropout": dropout,
            "grid_m": grid_m,
            "grid_n": grid_n,
            "symmetry": symmetry,
            "use_mlp": use_mlp,
            "use_history": use_history,
            "use_visibility": use_visibility,
            "node_layout": node_layout,
            "edge_distance": edge_distance,
        }
        self.hidden_dim = hidden_dim
        self.grid_m, self.grid_n = grid_m, grid_n
        self.num_zones = grid_m * grid_n
        self.use_history = use_history
        self.use_visibility = use_visibility
        self.node_layout = get_node_layout(node_layout)
        self.edge_distance = edge_distance

        context_dim = hidden_dim if use_history else 0
        if use_history:
            if temporal_layers < 1:
                raise ValueError("temporal_layers must be at least 1.")
            self.event_embedding = EventContextEmbedding(hidden_dim)
            self.subtype_embedding = nn.Embedding(NUM_EVENT_SUBTYPES, hidden_dim)
            self.target_query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
            self.time_embedding = _time_embedding(hidden_dim)
            self.sequence_cells = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(temporal_layers)])
            self.temporal_norm = nn.LayerNorm(hidden_dim)

        self.backbone = D2GATv2Backbone(
            in_node_dim=in_node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nb_heads=nb_heads,
            in_edge_dim=in_edge_dim + int(edge_distance),
            in_graph_dim=in_graph_dim + context_dim,
            dropout=dropout,
            symmetry=symmetry,
            node_layout=self.node_layout.name,
        )
        self.group_actions = self.backbone.group_actions
        self.num_views = len(self.group_actions)

        readout_dim = 2 * hidden_dim + context_dim
        num_location_maps = 2 if use_visibility else 1
        self.type_head = _linear_or_mlp(readout_dim, len(ACTION_TYPES), use_mlp, dropout)
        self.zone_head = _linear_or_mlp(
            readout_dim, num_location_maps * len(LOCATED_ACTIONS) * self.num_zones, use_mlp, dropout
        )
        self.height_head = _linear_or_mlp(readout_dim, len(PASS_HEIGHTS) * self.num_zones, use_mlp, dropout)

    # -- encoder --------------------------------------------------------------

    def encode_history(self, batch: NextActionBatch) -> torch.Tensor:
        """Per-view temporal states ``(V * B, H)``."""
        views = self.num_views
        sequence_mask = _repeat_views(batch.sequence_mask, views)
        known = sequence_mask.clone()
        known[:, -1] = False
        tokens = self.event_embedding(
            _repeat_views(batch.event_type, views),
            _repeat_views(batch.event_outcome, views),
            _repeat_views(batch.event_body_part, views),
            _per_view_event_numeric(batch.event_numeric, self.group_actions),
            known,
        )
        tokens = tokens + self.subtype_embedding(_repeat_views(batch.event_subtype, views)) * known.unsqueeze(-1)
        query_slot = torch.zeros_like(sequence_mask)
        query_slot[:, -1] = True
        tokens = tokens + self.target_query * query_slot.unsqueeze(-1).to(tokens.dtype)
        tokens = tokens + self.time_embedding(_repeat_views(_time_features(batch.time_delta), views))
        tokens = tokens * sequence_mask.unsqueeze(-1).to(tokens.dtype)
        return self.temporal_norm(_run_masked_gru(self.sequence_cells, tokens, sequence_mask))

    @property
    def readout_dim(self) -> int:
        return 2 * self.hidden_dim + (self.hidden_dim if self.use_history else 0)

    def decode_grid(self, logits: torch.Tensor) -> torch.Tensor:
        """``(V, B, C, M, N)`` per-view grids -> canonical ``(B, C, M * N)``.

        Each view's grid is reflected back to the canonical frame before
        averaging, so a reflected input yields a reflected output.
        """
        aligned = [flip_grid(logits[view], action) for view, action in enumerate(self.group_actions)]
        return torch.stack(aligned).mean(dim=0).flatten(2)

    def visibility_views(self, batch: NextActionBatch) -> torch.Tensor:
        """The visible-zone mask in every view's frame, ``(V, B, M, N)``."""
        visible = batch.visible_zones.view(batch.batch_size, self.grid_m, self.grid_n)
        return torch.stack([flip_grid(visible, action) for action in self.group_actions])

    def pairwise_distance(self, batch: NextActionBatch) -> torch.Tensor:
        """Player distances in pitch units / 40, ``(B, N, N)``; zero for padded players.

        Coordinates are normalized differently along the length and width of
        the pitch, so they are mapped back to yards first.  Distances do not
        change under reflections, so the edge feature keeps equivariance.
        """
        xy = batch.node_features[..., [self.node_layout.x, self.node_layout.y]]
        xy = xy * xy.new_tensor([_PITCH_LENGTH / _TARGET_SCALE, _PITCH_WIDTH / _TARGET_SCALE])
        return torch.cdist(xy, xy) / 40.0 * batch.adj_mat

    def encode(self, batch: NextActionBatch) -> torch.Tensor:
        """Per-view decision readouts ``(V, B, readout_dim)``: [pooled nodes, actor node, history]."""
        views, batch_size = self.num_views, batch.batch_size
        static_graph = batch.graph_features
        if self.use_history:
            temporal = self.encode_history(batch).view(views, batch_size, -1)
            graph_views = [torch.cat([static_graph, temporal[view]], dim=-1) for view in range(views)]
        else:
            temporal = None
            graph_views = [static_graph] * views
        edges = batch.edge_features
        if self.edge_distance:
            edges = torch.cat([edges, self.pairwise_distance(batch).unsqueeze(-1)], dim=-1)
        node_views = self.backbone(batch.node_features, edges, graph_views[0], batch.adj_mat, graph_views=graph_views)
        actor_mask = batch.node_features[..., self.node_layout.actor] > 0.5
        readout = pool_graph_readout(torch.stack(node_views), batch.player_mask, actor_mask)  # (V, B, 2H)
        if temporal is not None:
            readout = torch.cat([readout, temporal], dim=-1)
        return readout

    def decode_policy(self, readout: torch.Tensor, batch: NextActionBatch) -> Dict[str, torch.Tensor]:
        views, batch_size = readout.shape[:2]
        # Actions that are structurally impossible at this decision (a carry right
        # after the player's own carry) get no probability mass.
        type_logits = self.type_head(readout).mean(dim=0).masked_fill(~batch.type_mask, -1e9)

        grid = (self.grid_m, self.grid_n)
        zone = self.zone_head(readout).view(views, batch_size, -1, len(LOCATED_ACTIONS), *grid)
        if self.use_visibility:
            zone = torch.where(self.visibility_views(batch)[:, :, None], zone[:, :, 0], zone[:, :, 1])
        else:
            zone = zone[:, :, 0]
        zone_logits = self.decode_grid(zone)  # (B, len(LOCATED_ACTIONS), Z)

        height = self.height_head(readout).view(views, batch_size, len(PASS_HEIGHTS), *grid)
        height_logits = self.decode_grid(height)  # (B, len(PASS_HEIGHTS), Z)
        return {"type_logits": type_logits, "zone_logits": zone_logits, "height_logits": height_logits}

    def forward(self, batch: NextActionBatch) -> Dict[str, torch.Tensor]:
        return self.decode_policy(self.encode(batch), batch)


def action_log_likelihood(outputs: Dict[str, torch.Tensor], batch: NextActionBatch) -> Dict[str, torch.Tensor]:
    """Per-sample negative log-likelihood of each factor (zero where a factor does not apply).

    Accepts any outputs in the model's format whose entries are logits or
    log-probabilities over the last (type/height: class, zone: zone) axis.
    """
    type_nll = F.cross_entropy(outputs["type_logits"], batch.action_type, reduction="none")

    located_index = _LOCATED_INDEX.to(batch.action_type.device)[batch.action_type]
    located = (located_index >= 0) & (batch.target_zone != IGNORE_INDEX)
    rows = torch.arange(batch.batch_size, device=batch.action_type.device)
    zone_logits = outputs["zone_logits"][rows, located_index.clamp_min(0)]  # (B, Z)
    zone_target = batch.target_zone.clamp_min(0)
    zone_nll = F.cross_entropy(zone_logits, zone_target, reduction="none") * located

    has_height = located & (batch.action_type == _PASS) & (batch.pass_height != IGNORE_INDEX)
    height_logits = outputs["height_logits"][rows, :, zone_target]  # (B, 3) at the true zone
    height_nll = F.cross_entropy(height_logits, batch.pass_height.clamp_min(0), reduction="none") * has_height

    return {
        "type_nll": type_nll,
        "zone_nll": zone_nll,
        "height_nll": height_nll,
        "joint_nll": type_nll + zone_nll + height_nll,
        "located": located,
        "has_height": has_height,
    }
