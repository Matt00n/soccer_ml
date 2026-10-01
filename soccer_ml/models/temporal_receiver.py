"""Temporal extensions of the D2 GATv2 model for pass target prediction.

Every model predicts either the pass target zone on an M x N pitch grid
(``task="target_zone"``, default) or the receiving player
(``task="receiver"``):

* ``StaticGATv2Receiver`` (``spatial_gatv2``) wraps the single-frame
  :class:`SoccerGATv2Model` on the target freeze frame only.
* ``GraphTemporalTransformerReceiver`` (``graph_transformer``) runs the GATv2
  on every event snapshot, pools each graph, and applies a causal Transformer
  over the graph latents.
* ``GraphGRUReceiver`` (``graph_gru``) is identical but uses a mask-aware
  stacked GRU instead of self-attention.
* ``TransportMemoryReceiver`` (``transport_memory``) propagates player-node
  memory through a differentiable, geometry- and team-aware Sinkhorn
  alignment.  This is useful for public 360 data, where node ordering changes
  and player IDs are absent.
* ``EventSeqToSpatialGATv2`` (``seq2spatial``) encodes only the *event*
  history with a sequence model and injects the resulting temporal state into
  a single GATv2 pass over the target freeze frame.  It avoids running the
  graph backbone on every history snapshot.

Reflection symmetry.  The backbone produces one representation per view of a
reflection group (``symmetry``: ``"y_flip"``, ``"d2"`` or ``"none"``).
Everything after the backbone runs per view with shared weights, with views
folded into the batch dimension (index ``v * B + b``), and non-invariant
inputs such as event locations are reflected to match each view.  The task
readout then frame-averages: receiver logits are averaged directly
(invariant), zone logit grids are first reflected back to the canonical frame
(equivariant).

None of the models consume event labels for the final/current pass.  Those
labels are present only for preceding snapshots in ``TemporalReceiverBatch``.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from soccer_ml.data.pitch_grid import DEFAULT_GRID_M, DEFAULT_GRID_N
from soccer_ml.data.schema import STATSBOMB_NODE_LAYOUT
from soccer_ml.data.temporal_receiver_dataset import (
    EVENT_NUMERIC_DIM,
    NUM_BODY_PARTS,
    NUM_EVENT_TYPES,
    NUM_OUTCOMES,
    ROLE_FEATURE_DIM,
    TemporalReceiverBatch,
    transform_event_numeric,
)
from soccer_ml.data.transforms import D2GroupTransform, resolve_symmetry_group
from soccer_ml.models.backbone import D2GATv2Backbone
from soccer_ml.models.heads import PitchTargetGridHead, pool_graph_readout
from soccer_ml.models.soccer_gat import SoccerGATv2Model

TASKS = ("target_zone", "receiver")
# Every temporal model consumes StatsBomb 360 graphs.
NODE_LAYOUT = STATSBOMB_NODE_LAYOUT
SEQUENCE_ENCODERS = ("gru", "transformer")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pool ``(B, ..., N, D)`` values over valid player nodes."""
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


def _repeat_views(tensor: torch.Tensor, num_views: int) -> torch.Tensor:
    """Tile a ``(B, ...)`` tensor to ``(V * B, ...)`` in view-major order."""
    if num_views == 1:
        return tensor
    return tensor.repeat(num_views, *([1] * (tensor.dim() - 1)))


def _per_view_event_numeric(event_numeric: torch.Tensor, group_actions: Sequence[int]) -> torch.Tensor:
    """Reflect ``(B, T, D)`` event features into every view: ``(V * B, T, D)``."""
    return torch.cat([transform_event_numeric(event_numeric, action) for action in group_actions], dim=0)


def _time_features(time_delta: torch.Tensor) -> torch.Tensor:
    """Log inter-event gap and log elapsed time, ``(B, T) -> (B, T, 2)``."""
    cumulative_time = torch.cumsum(time_delta, dim=-1)
    return torch.stack([torch.log1p(time_delta), torch.log1p(cumulative_time)], dim=-1)


def _time_embedding(hidden_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))


def _run_masked_gru(
    cells: nn.ModuleList, inputs: torch.Tensor, step_mask: torch.Tensor
) -> torch.Tensor:
    """Stacked GRU over ``(B, T, H)`` inputs that skips masked steps exactly.

    Unlike feeding zero vectors through ``nn.GRU``, left-padding leaves the
    state untouched (GRU biases would otherwise make a short context depend on
    its amount of padding).
    """
    batch_size, steps, hidden_dim = inputs.shape
    states = [inputs.new_zeros(batch_size, hidden_dim) for _ in cells]
    for step in range(steps):
        layer_input = inputs[:, step]
        valid_step = step_mask[:, step].unsqueeze(-1)
        for layer_index, cell in enumerate(cells):
            proposed_state = cell(layer_input, states[layer_index])
            states[layer_index] = torch.where(valid_step, proposed_state, states[layer_index])
            layer_input = states[layer_index]
    return states[-1]


class EventContextEmbedding(nn.Module):
    """Embed known historical event type, outcome, body part, and numerics."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.type_embedding = nn.Embedding(NUM_EVENT_TYPES, hidden_dim)
        self.outcome_embedding = nn.Embedding(NUM_OUTCOMES, hidden_dim)
        self.body_part_embedding = nn.Embedding(NUM_BODY_PARTS, hidden_dim)
        self.numeric = nn.Sequential(
            nn.Linear(EVENT_NUMERIC_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        event_type: torch.Tensor,
        event_outcome: torch.Tensor,
        event_body_part: torch.Tensor,
        event_numeric: torch.Tensor,
        known_event_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedding = (
            self.type_embedding(event_type)
            + self.outcome_embedding(event_outcome)
            + self.body_part_embedding(event_body_part)
            + self.numeric(event_numeric)
        )
        # Masking after LayerNorm makes the target's placeholder category
        # exactly zero, rather than a learnable "current outcome" shortcut.
        return self.norm(embedding) * known_event_mask.unsqueeze(-1).to(embedding.dtype)


class TaskReadout(nn.Module):
    """Per-view task decoder followed by frame averaging.

    ``target_zone``: [mean pool, passer node, context] -> equivariant
    :class:`PitchTargetGridHead`.  ``receiver``: per-node [node, context] ->
    scalar score, averaged over views.  Both decoders are a single linear layer
    unless ``use_mlp`` is set.
    """

    def __init__(
        self,
        task: str,
        node_dim: int,
        context_dim: int,
        group_actions: Sequence[int],
        grid_m: int,
        grid_n: int,
        use_mlp: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.task = task
        self.group_actions = tuple(group_actions)
        if task == "target_zone":
            self.grid_head = PitchTargetGridHead(
                in_dim=2 * node_dim + context_dim,
                grid_m=grid_m,
                grid_n=grid_n,
                group_actions=self.group_actions,
                use_mlp=use_mlp,
                dropout=dropout,
            )
        elif task == "receiver":
            in_dim = node_dim + context_dim
            self.receiver_scorer = (
                nn.Sequential(
                    nn.Linear(in_dim, node_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(node_dim, 1),
                )
                if use_mlp
                else nn.Linear(in_dim, 1)
            )
        else:
            raise ValueError(f"Unknown task '{task}'. Choose one of: {', '.join(TASKS)}.")

    def forward(
        self,
        nodes: torch.Tensor,
        context: torch.Tensor,
        player_mask: torch.Tensor,
        actor_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Decode task logits.

        Args:
            nodes: (V * B, N, node_dim) current-frame node embeddings per view.
            context: (V * B, context_dim) per-view temporal/graph context.
            player_mask: (B, N) valid players of the current frame.
            actor_mask: (B, N) passer of the current frame.
        """
        num_views = len(self.group_actions)
        batch_size, num_nodes = player_mask.shape
        nodes = nodes.view(num_views, batch_size, num_nodes, -1)
        context = context.view(num_views, batch_size, -1)
        if self.task == "target_zone":
            readout = torch.cat([pool_graph_readout(nodes, player_mask, actor_mask), context], dim=-1)
            return {"target_zone_logits": self.grid_head(readout)}
        shared = context.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        logits = self.receiver_scorer(torch.cat([nodes, shared], dim=-1)).squeeze(-1).mean(dim=0)
        return {"receiver_logits": logits.masked_fill(~player_mask, -1e9)}


class TemporalSpatialEncoder(nn.Module):
    """Encode every valid graph snapshot with the reflection-view GATv2.

    Only timesteps with ``sequence_mask`` set are passed through the backbone;
    left-padding stays zero.  Outputs keep one representation per view, folded
    into the batch dimension: node embeddings ``(V * B, T, N, H)``, graph
    latents ``(V * B, T, H)``, and event latents ``(V * B, T, H)``.

    Optional pitch Fourier features (``num_fourier_bands > 0``) are computed
    from each view's reflected coordinates and added to that view.
    """

    def __init__(
        self,
        in_node_dim: int,
        in_edge_dim: int,
        in_graph_dim: int,
        hidden_dim: int,
        num_layers: int,
        nb_heads: int,
        num_fourier_bands: int = 0,
        dropout: float = 0.0,
        symmetry: str = "y_flip",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone = D2GATv2Backbone(
            in_node_dim=in_node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nb_heads=nb_heads,
            in_edge_dim=in_edge_dim,
            in_graph_dim=in_graph_dim,
            dropout=dropout,
            symmetry=symmetry,
            node_layout=NODE_LAYOUT.name,
        )
        self.group_actions = self.backbone.group_actions
        self.num_views = len(self.group_actions)
        self.group_transform = D2GroupTransform.from_layout(NODE_LAYOUT)
        self.num_fourier_bands = num_fourier_bands
        if num_fourier_bands > 0:
            frequencies = (2.0 ** torch.arange(num_fourier_bands, dtype=torch.float32)) * torch.pi / 5.0
            self.register_buffer("fourier_frequencies", frequencies, persistent=False)
            self.position_projection = nn.Sequential(
                nn.Linear(4 * num_fourier_bands, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.event_embedding = EventContextEmbedding(hidden_dim)
        self.actor_event_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.recipient_event_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.role_projection = nn.Linear(ROLE_FEATURE_DIM, hidden_dim, bias=False)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.pool_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def _fourier_position(self, node_features: torch.Tensor) -> torch.Tensor:
        """2D sine/cosine features for a tensor of normalized coordinates."""
        xy = node_features[..., :2]
        phases = xy.unsqueeze(-1) * self.fourier_frequencies
        # (..., 2, bands) -> (..., 4 * bands), preserving x/y separately.
        return torch.cat(
            [phases[..., 0, :].sin(), phases[..., 0, :].cos(), phases[..., 1, :].sin(), phases[..., 1, :].cos()],
            dim=-1,
        )

    def forward(self, batch: TemporalReceiverBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, steps, nodes, _ = batch.node_features.shape
        num_views = self.num_views
        valid_steps = batch.sequence_mask.reshape(-1).nonzero(as_tuple=True)[0]
        flat_nodes = batch.node_features.reshape(batch_size * steps, nodes, -1)[valid_steps]
        flat_edges = batch.edge_features.reshape(batch_size * steps, nodes, nodes, -1)[valid_steps]
        flat_graphs = batch.graph_features.reshape(batch_size * steps, -1)[valid_steps]
        flat_adjacency = batch.adj_mat.reshape(batch_size * steps, nodes, nodes)[valid_steps]

        views = self.backbone(flat_nodes, flat_edges, flat_graphs, flat_adjacency)
        if self.num_fourier_bands > 0:
            views = [
                view
                + self.position_projection(
                    self._fourier_position(self.group_transform.transform_node_features(flat_nodes, action))
                )
                for action, view in zip(self.group_actions, views)
            ]
        encoded = torch.stack(views, dim=0)  # (V, K, N, H), K = number of valid steps
        spatial_nodes = encoded.new_zeros(num_views, batch_size * steps, nodes, self.hidden_dim)
        spatial_nodes[:, valid_steps] = encoded
        spatial_nodes = spatial_nodes.view(num_views * batch_size, steps, nodes, self.hidden_dim)

        sequence_mask = _repeat_views(batch.sequence_mask, num_views)
        player_mask = _repeat_views(batch.player_mask, num_views)
        role_features = _repeat_views(batch.role_features, num_views)
        known_event_mask = sequence_mask.clone()
        known_event_mask[:, -1] = False
        event_latent = self.event_embedding(
            _repeat_views(batch.event_type, num_views),
            _repeat_views(batch.event_outcome, num_views),
            _repeat_views(batch.event_body_part, num_views),
            _per_view_event_numeric(batch.event_numeric, self.group_actions),
            known_event_mask,
        )
        actor_gate = role_features[..., 0:1]
        recipient_gate = role_features[..., 1:2]
        event_to_actor = self.actor_event_projection(event_latent).unsqueeze(-2) * actor_gate
        event_to_recipient = self.recipient_event_projection(event_latent).unsqueeze(-2) * recipient_gate
        node_embeddings = self.node_norm(
            spatial_nodes + self.role_projection(role_features) + event_to_actor + event_to_recipient
        )
        node_embeddings = node_embeddings * player_mask.unsqueeze(-1).to(node_embeddings.dtype)

        attention_logits = self.pool_gate(node_embeddings).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~player_mask, -1e9)
        pool_weights = F.softmax(attention_logits, dim=-1)
        pool_weights = pool_weights * player_mask.to(pool_weights.dtype)
        pool_weights = pool_weights / pool_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = (pool_weights.unsqueeze(-1) * node_embeddings).sum(dim=-2)

        actor_weights = role_features[..., 0] * player_mask.to(role_features.dtype)
        actor_embedding = (node_embeddings * actor_weights.unsqueeze(-1)).sum(dim=-2)
        # Some 360 frames lack an actor.  Falling back to a pool avoids making
        # these valid snapshots appear as a zero/invalid event representation.
        has_actor = actor_weights.sum(dim=-1, keepdim=True) > 0
        actor_embedding = torch.where(has_actor, actor_embedding, pooled)
        graph_latent = self.graph_projection(torch.cat([pooled, actor_embedding + event_latent], dim=-1))
        graph_latent = graph_latent * sequence_mask.unsqueeze(-1).to(graph_latent.dtype)
        return node_embeddings, graph_latent, event_latent


class TemporalModelBase(nn.Module):
    """Shared configuration for all temporal benchmark models.

    Every model accepts the same keyword arguments so that one CLI
    configuration builds all of them and every checkpoint stores a complete,
    self-describing ``model_kwargs``.  ``sequence_encoder`` is only used by
    ``seq2spatial`` and ``sinkhorn_iterations`` only by ``transport_memory``.
    """

    model_name = ""

    def __init__(
        self,
        in_node_dim: int = NODE_LAYOUT.dim,
        in_edge_dim: int = 1,
        in_graph_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 4,
        nb_heads: int = 8,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        max_sequence_length: int = 9,
        num_fourier_bands: int = 0,
        dropout: float = 0.1,
        task: str = "target_zone",
        grid_m: int = DEFAULT_GRID_M,
        grid_n: int = DEFAULT_GRID_N,
        symmetry: str = "y_flip",
        use_mlp: bool = False,
        sequence_encoder: str = "gru",
        sinkhorn_iterations: int = 5,
    ):
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"Unknown task '{task}'. Choose one of: {', '.join(TASKS)}.")
        self.model_kwargs: Dict[str, Any] = {
            "in_node_dim": in_node_dim,
            "in_edge_dim": in_edge_dim,
            "in_graph_dim": in_graph_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "nb_heads": nb_heads,
            "temporal_layers": temporal_layers,
            "temporal_heads": temporal_heads,
            "max_sequence_length": max_sequence_length,
            "num_fourier_bands": num_fourier_bands,
            "dropout": dropout,
            "task": task,
            "grid_m": grid_m,
            "grid_n": grid_n,
            "symmetry": symmetry,
            "use_mlp": use_mlp,
            "sequence_encoder": sequence_encoder,
            "sinkhorn_iterations": sinkhorn_iterations,
        }
        self.task = task
        self.hidden_dim = hidden_dim
        self.group_actions = resolve_symmetry_group(symmetry)
        self.num_views = len(self.group_actions)

    def _spatial_encoder(self) -> TemporalSpatialEncoder:
        kwargs = self.model_kwargs
        return TemporalSpatialEncoder(
            in_node_dim=kwargs["in_node_dim"],
            in_edge_dim=kwargs["in_edge_dim"],
            in_graph_dim=kwargs["in_graph_dim"],
            hidden_dim=kwargs["hidden_dim"],
            num_layers=kwargs["num_layers"],
            nb_heads=kwargs["nb_heads"],
            num_fourier_bands=kwargs["num_fourier_bands"],
            dropout=kwargs["dropout"],
            symmetry=kwargs["symmetry"],
        )

    def _task_readout(self, context_dim: int) -> TaskReadout:
        kwargs = self.model_kwargs
        return TaskReadout(
            task=self.task,
            node_dim=self.hidden_dim,
            context_dim=context_dim,
            group_actions=self.group_actions,
            grid_m=kwargs["grid_m"],
            grid_n=kwargs["grid_n"],
            use_mlp=kwargs["use_mlp"],
            dropout=kwargs["dropout"],
        )

    @staticmethod
    def _current_actor_mask(batch: TemporalReceiverBatch) -> torch.Tensor:
        return batch.role_features[:, -1, :, 0] > 0.5


class StaticGATv2Receiver(TemporalModelBase):
    """Adapter exposing the single-frame GATv2 in this benchmark.

    It deliberately performs no temporal operation: it extracts the final
    target graph from a ``TemporalReceiverBatch`` and delegates to
    :class:`SoccerGATv2Model` with only the head for ``task`` enabled.  This
    makes its data split, loader order, loss, optimiser, and checkpoint format
    directly comparable to the temporal models.
    """

    model_name = "spatial_gatv2"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        config = self.model_kwargs
        self.gatv2 = SoccerGATv2Model(
            in_node_dim=config["in_node_dim"],
            in_edge_dim=config["in_edge_dim"],
            in_graph_dim=config["in_graph_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            nb_heads=config["nb_heads"],
            grid_m=config["grid_m"],
            grid_n=config["grid_n"],
            use_mlp=config["use_mlp"],
            symmetry=config["symmetry"],
            node_layout=NODE_LAYOUT.name,
            dropout=config["dropout"],
            enable_receiver_head=self.task == "receiver",
            enable_target_location_head=self.task == "target_zone",
            enable_shot_head=False,
            enable_creativity_head=False,
            enable_vae_head=False,
        )

    def forward(self, batch: TemporalReceiverBatch) -> Dict[str, Any]:
        return self.gatv2(batch.current_graph_batch())


class GraphTemporalTransformerReceiver(TemporalModelBase):
    """GATv2 per event graph followed by a causal graph-level Transformer."""

    model_name = "graph_transformer"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        config = self.model_kwargs
        hidden_dim = self.hidden_dim
        if hidden_dim % config["temporal_heads"] != 0:
            raise ValueError("hidden_dim must be divisible by temporal_heads.")
        self.spatial = self._spatial_encoder()
        self.position_embedding = nn.Embedding(config["max_sequence_length"], hidden_dim)
        self.time_embedding = _time_embedding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config["temporal_heads"],
            dim_feedforward=hidden_dim * 4,
            dropout=config["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        # PyTorch otherwise performs a nested-tensor left-alignment check when
        # ``src_key_padding_mask`` is supplied.  That operator is currently
        # unavailable on MPS, while the ordinary masked attention path is fully
        # supported and has the same semantics for these short sequences.
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config["temporal_layers"],
            enable_nested_tensor=False,
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.readout = self._task_readout(context_dim=2 * hidden_dim)

    def forward(self, batch: TemporalReceiverBatch) -> Dict[str, Any]:
        if batch.node_features.shape[1] > self.position_embedding.num_embeddings:
            raise ValueError(
                "Sequence length exceeds max_sequence_length configured for the temporal model."
            )
        nodes, graph_latents, _ = self.spatial(batch)
        steps = batch.node_features.shape[1]
        sequence_mask = _repeat_views(batch.sequence_mask, self.num_views)
        positions = torch.arange(steps, device=nodes.device)
        time_features = _repeat_views(_time_features(batch.time_delta), self.num_views)
        transformer_input = (
            graph_latents
            + self.position_embedding(positions).unsqueeze(0)
            + self.time_embedding(time_features)
        )
        transformer_input = transformer_input * sequence_mask.unsqueeze(-1).to(transformer_input.dtype)
        causal_mask = torch.triu(
            torch.ones(steps, steps, device=nodes.device, dtype=torch.bool), diagonal=1
        )
        temporal_states = self.temporal_encoder(
            transformer_input,
            mask=causal_mask,
            src_key_padding_mask=~sequence_mask,
        )
        context = torch.cat([self.context_norm(temporal_states[:, -1]), graph_latents[:, -1]], dim=-1)
        outputs: Dict[str, Any] = self.readout(
            nodes[:, -1], context, batch.player_mask[:, -1], self._current_actor_mask(batch)
        )
        outputs["temporal_context"] = context
        return outputs


class GraphGRUReceiver(TemporalModelBase):
    """GATv2 graph sequence encoder with a causal, padding-aware GRU.

    This is the recurrent counterpart to
    :class:`GraphTemporalTransformerReceiver`: both use the identical spatial
    GATv2 encoder, attention/actor graph readout, and observed-prior-event
    conditioning.  The sole temporal difference is a stacked GRU instead of
    causal self-attention.
    """

    model_name = "graph_gru"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if self.model_kwargs["temporal_layers"] < 1:
            raise ValueError("temporal_layers must be at least 1 for graph_gru.")
        hidden_dim = self.hidden_dim
        self.spatial = self._spatial_encoder()
        self.time_embedding = _time_embedding(hidden_dim)
        self.temporal_cells = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(self.model_kwargs["temporal_layers"])]
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.readout = self._task_readout(context_dim=2 * hidden_dim)

    def forward(self, batch: TemporalReceiverBatch) -> Dict[str, Any]:
        nodes, graph_latents, _ = self.spatial(batch)
        time_features = _repeat_views(_time_features(batch.time_delta), self.num_views)
        temporal_inputs = graph_latents + self.time_embedding(time_features)
        state = _run_masked_gru(
            self.temporal_cells, temporal_inputs, _repeat_views(batch.sequence_mask, self.num_views)
        )
        context = torch.cat([self.context_norm(state), graph_latents[:, -1]], dim=-1)
        outputs: Dict[str, Any] = self.readout(
            nodes[:, -1], context, batch.player_mask[:, -1], self._current_actor_mask(batch)
        )
        outputs["temporal_context"] = context
        return outputs


class SinkhornTemporalAlignment(nn.Module):
    """Softly align consecutive dynamic node sets without player identities."""

    def __init__(self, hidden_dim: int, iterations: int = 5):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.log_distance_scale = nn.Parameter(torch.tensor(0.0))
        self.log_team_penalty = nn.Parameter(torch.tensor(0.0))
        self.iterations = iterations

    def forward(
        self,
        current_nodes: torch.Tensor,
        previous_nodes: torch.Tensor,
        current_xy: torch.Tensor,
        previous_xy: torch.Tensor,
        current_team: torch.Tensor,
        previous_team: torch.Tensor,
        current_mask: torch.Tensor,
        previous_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return row-normalized transport and match confidence.

        Rows represent visible current players and columns previous players.
        Iterative row/column normalization discourages many current nodes from
        collapsing onto one nearby prior node while still allowing unequal
        visible-player counts.
        """
        scale = current_nodes.shape[-1] ** -0.5
        learned_similarity = torch.matmul(self.query(current_nodes), self.key(previous_nodes).transpose(-1, -2)) * scale
        squared_distance = (current_xy.unsqueeze(-2) - previous_xy.unsqueeze(-3)).pow(2).sum(dim=-1)
        team_mismatch = current_team.unsqueeze(-1) != previous_team.unsqueeze(-2)
        scores = learned_similarity - F.softplus(self.log_distance_scale) * squared_distance
        scores = scores - F.softplus(self.log_team_penalty) * team_mismatch.to(scores.dtype)
        valid_pairs = current_mask.unsqueeze(-1) & previous_mask.unsqueeze(-2)
        scores = scores.masked_fill(~valid_pairs, -30.0)
        transport = torch.exp(scores) * valid_pairs.to(scores.dtype)
        for _ in range(self.iterations):
            transport = transport / transport.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            transport = transport / transport.sum(dim=-2, keepdim=True).clamp_min(1e-8)
            transport = transport * valid_pairs.to(transport.dtype)
        transport = transport / transport.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        confidence = transport.max(dim=-1).values * current_mask.to(transport.dtype)
        return transport, confidence


class TransportMemoryReceiver(TemporalModelBase):
    """GATv2 spatial encoder plus OT-aligned causal player memory.

    Consecutive snapshots are joined by Sinkhorn transport over learned node
    similarity, pitch displacement, and team membership.  A GRU then updates
    one latent memory vector per *currently visible* node.  Nodes that newly
    enter a frame have low transport confidence and fall back to the GRU
    new-state path; padded/invisible nodes never become players.  Squared
    distances are reflection invariant, so every view aligns with the raw
    coordinates.
    """

    model_name = "transport_memory"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        hidden_dim = self.hidden_dim
        self.spatial = self._spatial_encoder()
        self.alignment = SinkhornTemporalAlignment(hidden_dim, self.model_kwargs["sinkhorn_iterations"])
        self.node_memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.time_decay = nn.Sequential(nn.Linear(1, hidden_dim), nn.Sigmoid())
        self.graph_memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.readout = self._task_readout(context_dim=2 * hidden_dim)

    def forward(self, batch: TemporalReceiverBatch) -> Dict[str, Any]:
        nodes, graph_latents, _ = self.spatial(batch)
        num_views = self.num_views
        batch_size, steps, num_nodes, hidden_dim = nodes.shape
        player_mask = _repeat_views(batch.player_mask, num_views)
        sequence_mask = _repeat_views(batch.sequence_mask, num_views)
        node_xy = _repeat_views(batch.node_features[..., :2], num_views)
        team = _repeat_views(batch.role_features[..., 2] > 0.5, num_views)
        time_delta = _repeat_views(batch.time_delta, num_views)

        memories = torch.zeros_like(nodes)
        previous_memory = nodes.new_zeros(batch_size, num_nodes, hidden_dim)
        previous_nodes = torch.zeros_like(previous_memory)
        previous_xy = nodes.new_zeros(batch_size, num_nodes, 2)
        previous_team = torch.zeros(batch_size, num_nodes, device=nodes.device, dtype=torch.bool)
        previous_mask = torch.zeros(batch_size, num_nodes, device=nodes.device, dtype=torch.bool)

        for step in range(steps):
            current_nodes = nodes[:, step]
            current_mask = player_mask[:, step] & sequence_mask[:, step].unsqueeze(-1)
            current_xy = node_xy[:, step]
            current_team = team[:, step]
            transport, confidence = self.alignment(
                current_nodes,
                previous_nodes,
                current_xy,
                previous_xy,
                current_team,
                previous_team,
                current_mask,
                previous_mask,
            )
            aligned_memory = torch.matmul(transport, previous_memory)
            delta_features = torch.log1p(time_delta[:, step]).unsqueeze(-1)
            decay = self.time_decay(delta_features).unsqueeze(1)
            carried_memory = aligned_memory * decay * confidence.unsqueeze(-1)
            updated = self.node_memory(
                current_nodes.reshape(-1, hidden_dim), carried_memory.reshape(-1, hidden_dim)
            ).reshape(batch_size, num_nodes, hidden_dim)
            updated = updated * current_mask.unsqueeze(-1).to(updated.dtype)
            memories[:, step] = updated
            previous_memory = updated
            previous_nodes = current_nodes
            previous_xy = current_xy
            previous_team = current_team
            previous_mask = current_mask

        graph_inputs = _masked_mean(memories, player_mask) + graph_latents
        graph_state = _run_masked_gru(nn.ModuleList([self.graph_memory]), graph_inputs, sequence_mask)
        context = torch.cat([self.memory_norm(graph_state), graph_latents[:, -1]], dim=-1)
        current_memory = self.memory_norm(memories[:, -1] + nodes[:, -1])
        outputs: Dict[str, Any] = self.readout(
            current_memory, context, batch.player_mask[:, -1], self._current_actor_mask(batch)
        )
        outputs["temporal_context"] = context
        return outputs


class EventSeqToSpatialGATv2(TemporalModelBase):
    """Event-sequence encoder whose state conditions a single-frame GATv2.

    1. The possession history ``[e_1, ..., e_{t-1}]`` (type, outcome, body
       part, time gaps, duration, start/end location, direction, pressure,
       team) is embedded per event.  The final slot is the decision point: its
       event is unknown, so it carries a learned query token plus the time
       since the previous event.
    2. A sequence model (masked GRU by default, or a Transformer) reads the
       sequence; its state at the query slot is ``h_temporal``.
    3. ``h_temporal`` is appended to the target frame's graph features, so it
       conditions every GATv2 attention logit.  The backbone runs on the
       target freeze frame only.
    4. Because GATv2 graph features only enter attention logits, not
       messages, ``h_temporal`` is also passed to the task readout.

    Event locations are reflected per view, so ``h_temporal`` is computed once
    per view and the model stays exactly equivariant.
    """

    model_name = "seq2spatial"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        config = self.model_kwargs
        hidden_dim = self.hidden_dim
        self.sequence_encoder = config["sequence_encoder"]
        if self.sequence_encoder not in SEQUENCE_ENCODERS:
            raise ValueError(
                f"Unknown sequence_encoder '{self.sequence_encoder}'. "
                f"Choose one of: {', '.join(SEQUENCE_ENCODERS)}."
            )
        if config["temporal_layers"] < 1:
            raise ValueError("temporal_layers must be at least 1 for seq2spatial.")

        self.event_embedding = EventContextEmbedding(hidden_dim)
        self.target_query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.time_embedding = _time_embedding(hidden_dim)
        if self.sequence_encoder == "gru":
            self.sequence_cells = nn.ModuleList(
                [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(config["temporal_layers"])]
            )
        else:
            if hidden_dim % config["temporal_heads"] != 0:
                raise ValueError("hidden_dim must be divisible by temporal_heads.")
            self.position_embedding = nn.Embedding(config["max_sequence_length"], hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=config["temporal_heads"],
                dim_feedforward=hidden_dim * 4,
                dropout=config["dropout"],
                activation="gelu",
                batch_first=True,
            )
            self.sequence_transformer = nn.TransformerEncoder(
                layer, num_layers=config["temporal_layers"], enable_nested_tensor=False
            )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.backbone = D2GATv2Backbone(
            in_node_dim=config["in_node_dim"],
            hidden_dim=hidden_dim,
            num_layers=config["num_layers"],
            nb_heads=config["nb_heads"],
            in_edge_dim=config["in_edge_dim"],
            in_graph_dim=config["in_graph_dim"] + hidden_dim,
            dropout=config["dropout"],
            symmetry=config["symmetry"],
            node_layout=NODE_LAYOUT.name,
        )
        self.readout = self._task_readout(context_dim=hidden_dim)

    def encode_history(self, batch: TemporalReceiverBatch) -> torch.Tensor:
        """Return per-view temporal states ``(V * B, H)``."""
        num_views = self.num_views
        steps = batch.event_type.shape[1]
        sequence_mask = _repeat_views(batch.sequence_mask, num_views)
        known_event_mask = sequence_mask.clone()
        known_event_mask[:, -1] = False
        tokens = self.event_embedding(
            _repeat_views(batch.event_type, num_views),
            _repeat_views(batch.event_outcome, num_views),
            _repeat_views(batch.event_body_part, num_views),
            _per_view_event_numeric(batch.event_numeric, self.group_actions),
            known_event_mask,
        )
        query_slot = torch.zeros_like(sequence_mask)
        query_slot[:, -1] = True
        tokens = tokens + self.target_query * query_slot.unsqueeze(-1).to(tokens.dtype)
        tokens = tokens + self.time_embedding(_repeat_views(_time_features(batch.time_delta), num_views))
        tokens = tokens * sequence_mask.unsqueeze(-1).to(tokens.dtype)

        if self.sequence_encoder == "gru":
            state = _run_masked_gru(self.sequence_cells, tokens, sequence_mask)
        else:
            if steps > self.position_embedding.num_embeddings:
                raise ValueError(
                    "Sequence length exceeds max_sequence_length configured for the temporal model."
                )
            positions = torch.arange(steps, device=tokens.device)
            tokens = tokens + self.position_embedding(positions).unsqueeze(0)
            # All tokens precede the decision, so no causal mask is needed; the
            # always-valid query slot guarantees a non-empty attention row.
            states = self.sequence_transformer(tokens, src_key_padding_mask=~sequence_mask)
            state = states[:, -1]
        return self.temporal_norm(state)

    def forward(self, batch: TemporalReceiverBatch) -> Dict[str, Any]:
        num_views = self.num_views
        batch_size = batch.batch_size
        temporal = self.encode_history(batch)  # (V * B, H)
        temporal_views = temporal.view(num_views, batch_size, -1)
        static_graph = batch.graph_features[:, -1]
        graph_views = [torch.cat([static_graph, temporal_views[view]], dim=-1) for view in range(num_views)]
        views = self.backbone(
            batch.node_features[:, -1],
            batch.edge_features[:, -1],
            graph_views[0],
            batch.adj_mat[:, -1],
            graph_views=graph_views,
        )
        nodes = torch.cat(views, dim=0)  # (V * B, N, H)
        outputs: Dict[str, Any] = self.readout(
            nodes, temporal, batch.player_mask[:, -1], self._current_actor_mask(batch)
        )
        outputs["temporal_context"] = temporal
        return outputs


TemporalModelName = Literal[
    "spatial_gatv2",
    "graph_transformer",
    "graph_gru",
    "transport_memory",
    "seq2spatial",
]

TEMPORAL_MODELS: Dict[str, Type[TemporalModelBase]] = {
    model.model_name: model
    for model in (
        StaticGATv2Receiver,
        GraphTemporalTransformerReceiver,
        GraphGRUReceiver,
        TransportMemoryReceiver,
        EventSeqToSpatialGATv2,
    )
}


def build_temporal_receiver_model(model_name: TemporalModelName | str, **kwargs: Any) -> TemporalModelBase:
    """Create a temporal model from the names used by the CLI."""
    try:
        model_class = TEMPORAL_MODELS[str(model_name)]
    except KeyError as exc:
        supported = ", ".join(sorted(TEMPORAL_MODELS))
        raise ValueError(f"Unknown temporal model '{model_name}'. Choose one of: {supported}.") from exc
    return model_class(**kwargs)
