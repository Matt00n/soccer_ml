"""Temporal extensions of the D2 GATv2 receiver model.

Both models in this module retain a D2 GATv2 spatial encoder for every
StatsBomb 360 snapshot.  They differ only in the causal temporal layer:

* ``GraphTemporalTransformerReceiver`` pools each graph, then uses a causal
  Transformer over event latents.
* ``GraphGRUReceiver`` uses the same graph latents and Fourier pitch features,
  but replaces self-attention with a mask-aware stacked GRU.
* ``TransportMemoryReceiver`` propagates player-node memory through a
  differentiable, geometry- and team-aware Sinkhorn alignment.  This is useful
  for public 360 data, where node ordering changes and player IDs are absent.

Neither model consumes event labels for the final/current pass.  Those labels
are present only for preceding snapshots in ``TemporalReceiverBatch``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Literal, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from soccer_ml.data.temporal_receiver_dataset import (
    EVENT_NUMERIC_DIM,
    NUM_BODY_PARTS,
    NUM_EVENT_TYPES,
    NUM_OUTCOMES,
    ROLE_FEATURE_DIM,
    TemporalReceiverBatch,
)
from soccer_ml.data.transforms import D2GroupTransform
from soccer_ml.models.backbone import D2GATv2Backbone
from soccer_ml.models.heads import frame_average_views
from soccer_ml.models.soccer_gat import SoccerGATv2Model


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pool ``(B, ..., N, D)`` values over valid player nodes."""
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


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


class D2TemporalSpatialEncoder(nn.Module):
    """Encode every graph snapshot while preserving D2 view averaging.

    The pitch Fourier features are computed separately under all four D2
    transformations and added to the corresponding backbone view *before*
    averaging.  This keeps their continuous-position benefit without breaking
    the existing reflection-invariant readout.
    """

    def __init__(
        self,
        in_node_dim: int,
        in_edge_dim: int,
        in_graph_dim: int,
        hidden_dim: int,
        num_layers: int,
        nb_heads: int,
        num_fourier_bands: int = 8,
        dropout: float = 0.0,
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
        )
        self.group_transform = D2GroupTransform()
        frequencies = (2.0 ** torch.arange(num_fourier_bands, dtype=torch.float32)) * math.pi / 5.0
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
        """Return node embeddings, graph latents, and event latents.

        Shapes are ``(B, T, N, H)``, ``(B, T, H)``, and ``(B, T, H)``.
        """
        batch_size, steps, nodes, _ = batch.node_features.shape
        flat_size = batch_size * steps
        flat_nodes = batch.node_features.reshape(flat_size, nodes, -1)
        flat_edges = batch.edge_features.reshape(flat_size, nodes, nodes, -1)
        flat_graphs = batch.graph_features.reshape(flat_size, -1)
        flat_adjacency = batch.adj_mat.reshape(flat_size, nodes, nodes)

        views = self.backbone(flat_nodes, flat_edges, flat_graphs, flat_adjacency)
        positional_views = []
        for action, view in enumerate(views):
            transformed_nodes = self.group_transform.transform_node_features(flat_nodes, action)
            positional_views.append(view + self.position_projection(self._fourier_position(transformed_nodes)))
        spatial_nodes = frame_average_views(positional_views).reshape(batch_size, steps, nodes, self.hidden_dim)

        known_event_mask = batch.sequence_mask.clone()
        known_event_mask[:, -1] = False
        event_latent = self.event_embedding(
            batch.event_type,
            batch.event_outcome,
            batch.event_body_part,
            batch.event_numeric,
            known_event_mask,
        )
        actor_gate = batch.role_features[..., 0:1]
        recipient_gate = batch.role_features[..., 1:2]
        event_to_actor = self.actor_event_projection(event_latent).unsqueeze(-2) * actor_gate
        event_to_recipient = self.recipient_event_projection(event_latent).unsqueeze(-2) * recipient_gate
        node_embeddings = self.node_norm(
            spatial_nodes + self.role_projection(batch.role_features) + event_to_actor + event_to_recipient
        )
        node_embeddings = node_embeddings * batch.player_mask.unsqueeze(-1).to(node_embeddings.dtype)

        attention_logits = self.pool_gate(node_embeddings).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~batch.player_mask, -1e9)
        pool_weights = F.softmax(attention_logits, dim=-1)
        pool_weights = pool_weights * batch.player_mask.to(pool_weights.dtype)
        pool_weights = pool_weights / pool_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled = (pool_weights.unsqueeze(-1) * node_embeddings).sum(dim=-2)

        actor_weights = batch.role_features[..., 0] * batch.player_mask.to(batch.role_features.dtype)
        actor_embedding = (node_embeddings * actor_weights.unsqueeze(-1)).sum(dim=-2)
        # Some 360 frames lack an actor.  Falling back to a pool avoids making
        # these valid snapshots appear as a zero/invalid event representation.
        has_actor = actor_weights.sum(dim=-1, keepdim=True) > 0
        actor_embedding = torch.where(has_actor, actor_embedding, pooled)
        graph_latent = self.graph_projection(torch.cat([pooled, actor_embedding + event_latent], dim=-1))
        graph_latent = graph_latent * batch.sequence_mask.unsqueeze(-1).to(graph_latent.dtype)
        return node_embeddings, graph_latent, event_latent


class ReceiverScorer(nn.Module):
    """Score candidate receivers from current node and temporal context."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        current_nodes: torch.Tensor,
        temporal_context: torch.Tensor,
        current_graph: torch.Tensor,
        current_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, nodes, _ = current_nodes.shape
        shared = torch.cat([temporal_context, current_graph], dim=-1).unsqueeze(1).expand(batch_size, nodes, -1)
        logits = self.scorer(torch.cat([current_nodes, shared], dim=-1)).squeeze(-1)
        return logits.masked_fill(~current_mask, -1e9)


class StaticGATv2Receiver(nn.Module):
    """Adapter exposing the unmodified single-frame GATv2 in this benchmark.

    It deliberately performs no temporal operation: it extracts the final
    target graph from a ``TemporalReceiverBatch`` and delegates to the existing
    :class:`SoccerGATv2Model`.  This makes its data split, loader order, loss,
    optimiser, and checkpoint format directly comparable to temporal models
    without changing ``scripts/train_receiver.py`` or the original model.
    """

    model_name = "spatial_gatv2"

    def __init__(
        self,
        in_node_dim: int = 7,
        in_edge_dim: int = 1,
        in_graph_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 4,
        nb_heads: int = 8,
        temporal_layers: int = 1,
        temporal_heads: int = 8,
        max_sequence_length: int = 17,
        num_fourier_bands: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Temporal arguments are recorded so all benchmark checkpoints have one
        # schema, but are intentionally not used by the spatial-only baseline.
        self.model_kwargs = {
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
        }
        self.gatv2 = SoccerGATv2Model(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            in_graph_dim=in_graph_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nb_heads=nb_heads,
            dropout=dropout,
            enable_receiver_head=True,
            enable_shot_head=False,
            enable_creativity_head=False,
            enable_vae_head=False,
        )

    def forward(self, batch: TemporalReceiverBatch) -> Dict[str, Any]:
        return self.gatv2(batch.current_graph_batch())


class GraphTemporalTransformerReceiver(nn.Module):
    """GATv2 per event graph followed by a causal graph-level Transformer."""

    model_name = "graph_transformer"

    def __init__(
        self,
        in_node_dim: int = 7,
        in_edge_dim: int = 1,
        in_graph_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 4,
        nb_heads: int = 8,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        max_sequence_length: int = 17,
        num_fourier_bands: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % temporal_heads != 0:
            raise ValueError("hidden_dim must be divisible by temporal_heads.")
        self.model_kwargs = {
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
        }
        self.spatial = D2TemporalSpatialEncoder(
            in_node_dim,
            in_edge_dim,
            in_graph_dim,
            hidden_dim,
            num_layers,
            nb_heads,
            num_fourier_bands,
            dropout,
        )
        self.position_embedding = nn.Embedding(max_sequence_length, hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=temporal_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
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
            num_layers=temporal_layers,
            enable_nested_tensor=False,
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.receiver_scorer = ReceiverScorer(hidden_dim, dropout)

    def forward(self, batch: TemporalReceiverBatch) -> Dict[str, Any]:
        if batch.node_features.shape[1] > self.position_embedding.num_embeddings:
            raise ValueError(
                "Sequence length exceeds max_sequence_length configured for the temporal model."
            )
        nodes, graph_latents, _ = self.spatial(batch)
        steps = batch.node_features.shape[1]
        positions = torch.arange(steps, device=nodes.device)
        cumulative_time = torch.cumsum(batch.time_delta, dim=-1)
        time_features = torch.stack(
            [torch.log1p(batch.time_delta), torch.log1p(cumulative_time)], dim=-1
        )
        transformer_input = graph_latents + self.position_embedding(positions).unsqueeze(0) + self.time_embedding(time_features)
        transformer_input = transformer_input * batch.sequence_mask.unsqueeze(-1).to(transformer_input.dtype)
        causal_mask = torch.triu(
            torch.ones(steps, steps, device=nodes.device, dtype=torch.bool), diagonal=1
        )
        temporal_states = self.temporal_encoder(
            transformer_input,
            mask=causal_mask,
            src_key_padding_mask=~batch.sequence_mask,
        )
        context = self.context_norm(temporal_states[:, -1])
        logits = self.receiver_scorer(
            nodes[:, -1], context, graph_latents[:, -1], batch.player_mask[:, -1]
        )
        return {
            "receiver_logits": logits,
            "node_embeddings": nodes[:, -1],
            "temporal_context": context,
            "graph_embeddings": graph_latents,
        }


class GraphGRUReceiver(nn.Module):
    """GATv2 graph sequence encoder with a causal, padding-aware GRU.

    This is the recurrent counterpart to
    :class:`GraphTemporalTransformerReceiver`: both use the identical spatial
    D2 GATv2 encoder, D2-view Fourier pitch features, attention/actor graph
    readout, and observed-prior-event conditioning.  The sole temporal
    difference is a stacked GRU instead of causal self-attention.
    """

    model_name = "graph_gru"

    def __init__(
        self,
        in_node_dim: int = 7,
        in_edge_dim: int = 1,
        in_graph_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 4,
        nb_heads: int = 8,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        max_sequence_length: int = 17,
        num_fourier_bands: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if temporal_layers < 1:
            raise ValueError("temporal_layers must be at least 1 for graph_gru.")
        # ``temporal_heads`` is retained only for a shared CLI/checkpoint schema
        # with the Transformer; a GRU has no attention heads.
        self.model_kwargs = {
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
        }
        self.spatial = D2TemporalSpatialEncoder(
            in_node_dim,
            in_edge_dim,
            in_graph_dim,
            hidden_dim,
            num_layers,
            nb_heads,
            num_fourier_bands,
            dropout,
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.temporal_cells = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(temporal_layers)]
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.receiver_scorer = ReceiverScorer(hidden_dim, dropout)

    def forward(self, batch: TemporalReceiverBatch) -> Dict[str, Any]:
        nodes, graph_latents, _ = self.spatial(batch)
        batch_size, steps, _, hidden_dim = nodes.shape
        cumulative_time = torch.cumsum(batch.time_delta, dim=-1)
        time_features = torch.stack(
            [torch.log1p(batch.time_delta), torch.log1p(cumulative_time)], dim=-1
        )
        temporal_inputs = graph_latents + self.time_embedding(time_features)
        states = [
            torch.zeros(batch_size, hidden_dim, device=nodes.device, dtype=nodes.dtype)
            for _ in self.temporal_cells
        ]
        for step in range(steps):
            layer_input = temporal_inputs[:, step]
            valid_step = batch.sequence_mask[:, step].unsqueeze(-1)
            for layer_index, cell in enumerate(self.temporal_cells):
                proposed_state = cell(layer_input, states[layer_index])
                # Leading missing history must not alter a recurrent state: GRU
                # biases would otherwise make context length affect the result.
                states[layer_index] = torch.where(valid_step, proposed_state, states[layer_index])
                layer_input = states[layer_index]
        context = self.context_norm(states[-1])
        logits = self.receiver_scorer(
            nodes[:, -1], context, graph_latents[:, -1], batch.player_mask[:, -1]
        )
        return {
            "receiver_logits": logits,
            "node_embeddings": nodes[:, -1],
            "temporal_context": context,
            "graph_embeddings": graph_latents,
        }


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


class TransportMemoryReceiver(nn.Module):
    """GATv2 spatial encoder plus OT-aligned causal player memory.

    Consecutive snapshots are joined by Sinkhorn transport over learned node
    similarity, pitch displacement, and attacking-team membership.  A GRU then
    updates one latent memory vector per *currently visible* node.  Nodes that
    newly enter a frame have low transport confidence and fall back to the GRU
    new-state path; padded/invisible nodes never become players.
    """

    model_name = "transport_memory"

    def __init__(
        self,
        in_node_dim: int = 7,
        in_edge_dim: int = 1,
        in_graph_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 4,
        nb_heads: int = 8,
        temporal_layers: int = 1,
        temporal_heads: int = 8,
        max_sequence_length: int = 17,
        num_fourier_bands: int = 8,
        dropout: float = 0.1,
        sinkhorn_iterations: int = 5,
    ):
        super().__init__()
        # temporal_layers/heads/max_sequence_length are accepted to give both
        # models one CLI/configuration surface.  This model uses a graph GRU,
        # so only temporal_layers=1 is meaningful and is recorded for parity.
        self.model_kwargs = {
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
            "sinkhorn_iterations": sinkhorn_iterations,
        }
        self.spatial = D2TemporalSpatialEncoder(
            in_node_dim,
            in_edge_dim,
            in_graph_dim,
            hidden_dim,
            num_layers,
            nb_heads,
            num_fourier_bands,
            dropout,
        )
        self.alignment = SinkhornTemporalAlignment(hidden_dim, sinkhorn_iterations)
        self.node_memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.time_decay = nn.Sequential(nn.Linear(1, hidden_dim), nn.Sigmoid())
        self.graph_memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.receiver_scorer = ReceiverScorer(hidden_dim, dropout)

    def forward(self, batch: TemporalReceiverBatch) -> Dict[str, Any]:
        nodes, graph_latents, _ = self.spatial(batch)
        batch_size, steps, num_nodes, hidden_dim = nodes.shape
        memories = torch.zeros_like(nodes)
        previous_memory = torch.zeros(batch_size, num_nodes, hidden_dim, device=nodes.device, dtype=nodes.dtype)
        previous_nodes = torch.zeros_like(previous_memory)
        previous_xy = torch.zeros(batch_size, num_nodes, 2, device=nodes.device, dtype=nodes.dtype)
        previous_team = torch.zeros(batch_size, num_nodes, device=nodes.device, dtype=torch.bool)
        previous_mask = torch.zeros(batch_size, num_nodes, device=nodes.device, dtype=torch.bool)

        for step in range(steps):
            current_nodes = nodes[:, step]
            current_mask = batch.player_mask[:, step] & batch.sequence_mask[:, step].unsqueeze(-1)
            current_xy = batch.node_features[:, step, :, :2]
            current_team = batch.role_features[:, step, :, 2] > 0.5
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
            delta_features = torch.log1p(batch.time_delta[:, step]).unsqueeze(-1)
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

        graph_inputs = _masked_mean(memories, batch.player_mask) + graph_latents
        graph_state = torch.zeros(batch_size, hidden_dim, device=nodes.device, dtype=nodes.dtype)
        for step in range(steps):
            proposed_state = self.graph_memory(graph_inputs[:, step], graph_state)
            is_valid_step = batch.sequence_mask[:, step].unsqueeze(-1)
            # Unlike feeding zero vectors through nn.GRU, this leaves the state
            # exactly unchanged for left-padding positions (GRU biases would
            # otherwise make a short context depend on its amount of padding).
            graph_state = torch.where(is_valid_step, proposed_state, graph_state)
        context = self.memory_norm(graph_state)
        current_memory = self.memory_norm(memories[:, -1] + nodes[:, -1])
        logits = self.receiver_scorer(
            current_memory, context, graph_latents[:, -1], batch.player_mask[:, -1]
        )
        return {
            "receiver_logits": logits,
            "node_embeddings": current_memory,
            "temporal_context": context,
            "transported_node_embeddings": memories,
            "graph_embeddings": graph_latents,
        }


TemporalModelName = Literal[
    "spatial_gatv2",
    "graph_transformer",
    "graph_gru",
    "transport_memory",
]


def build_temporal_receiver_model(model_name: TemporalModelName | str, **kwargs: Any) -> nn.Module:
    """Create a temporal receiver model from the names used by the CLI."""
    models: Dict[str, Type[nn.Module]] = {
        StaticGATv2Receiver.model_name: StaticGATv2Receiver,
        GraphTemporalTransformerReceiver.model_name: GraphTemporalTransformerReceiver,
        GraphGRUReceiver.model_name: GraphGRUReceiver,
        TransportMemoryReceiver.model_name: TransportMemoryReceiver,
    }
    try:
        return models[str(model_name)](**kwargs)
    except KeyError as exc:
        supported = ", ".join(sorted(models))
        raise ValueError(f"Unknown temporal receiver model '{model_name}'. Choose one of: {supported}.") from exc
