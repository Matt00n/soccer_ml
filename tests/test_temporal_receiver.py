"""Shape, masking, equivariance, and gradient tests for temporal pass-target models."""

import dataclasses

import pytest
import torch

from soccer_ml.data.temporal_receiver_dataset import (
    EVENT_NUMERIC_DIM,
    TemporalReceiverBatch,
    transform_event_numeric,
)
from soccer_ml.data.transforms import D2GroupTransform, flip_grid
from soccer_ml.models.temporal_receiver import TEMPORAL_MODELS, build_temporal_receiver_model

MODEL_NAMES = tuple(TEMPORAL_MODELS)
GRID_M, GRID_N = 26, 17


def _temporal_batch(node_dim: int = 9, graph_dim: int = 7) -> TemporalReceiverBatch:
    torch.manual_seed(0)
    batch_size, steps, players = 2, 3, 22
    player_mask = torch.zeros(batch_size, steps, players, dtype=torch.bool)
    player_mask[:, :, :15] = True
    sequence_mask = torch.tensor([[False, True, True], [True, True, True]])
    player_mask &= sequence_mask.unsqueeze(-1)
    role_features = torch.zeros(batch_size, steps, players, 3)
    role_features[:, :, 0, 0] = 1.0  # actor
    role_features[:, :, :8, 2] = 1.0  # teammates of the target passer
    role_features[:, :-1, 1, 1] = 1.0  # known prior recipient only
    node_features = torch.randn(batch_size, steps, players, node_dim)
    node_features[..., 6] = role_features[..., 0]
    node_features = node_features * player_mask.unsqueeze(-1)
    return TemporalReceiverBatch(
        node_features=node_features,
        edge_features=torch.zeros(batch_size, steps, players, players, 1),
        graph_features=torch.randn(batch_size, steps, graph_dim),
        adj_mat=player_mask.unsqueeze(-1).float() * player_mask.unsqueeze(-2).float(),
        player_mask=player_mask,
        role_features=role_features,
        event_type=torch.tensor([[0, 30, 0], [30, 42, 0]]),
        event_outcome=torch.tensor([[0, 1, 0], [1, 0, 0]]),
        event_body_part=torch.zeros(batch_size, steps, dtype=torch.long),
        event_numeric=torch.rand(batch_size, steps, EVENT_NUMERIC_DIM),
        time_delta=torch.tensor([[0.0, 1.2, 0.8], [0.0, 0.5, 1.0]]),
        sequence_mask=sequence_mask,
        receiver_label=torch.tensor([2, 3]),
        target_zone_label=torch.tensor([5, 400]),
        pass_end_location=torch.tensor([[10.0, 20.0], [100.0, 70.0]]),
    )


def _build(model_name: str, **overrides):
    kwargs = dict(
        hidden_dim=32,
        num_layers=2,
        nb_heads=4,
        temporal_layers=1,
        temporal_heads=4,
        max_sequence_length=3,
        dropout=0.0,
        grid_m=GRID_M,
        grid_n=GRID_N,
    )
    kwargs.update(overrides)
    return build_temporal_receiver_model(model_name, **kwargs)


def _reflect(batch: TemporalReceiverBatch, action: int) -> TemporalReceiverBatch:
    """Reflect the whole input: node coordinates/velocities and event locations."""
    transform = D2GroupTransform()
    return dataclasses.replace(
        batch,
        node_features=transform.transform_node_features(batch.node_features, action),
        event_numeric=transform_event_numeric(batch.event_numeric, action),
    )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("task", ["target_zone", "receiver"])
def test_models_forward_backward(model_name, task):
    batch = _temporal_batch()
    model = _build(model_name, task=task)
    outputs = model(batch)
    if task == "target_zone":
        logits = outputs["target_zone_logits"]
        assert logits.shape == (2, GRID_M * GRID_N)
        loss = torch.nn.functional.cross_entropy(logits, batch.target_zone_label)
    else:
        logits = outputs["receiver_logits"]
        assert logits.shape == (2, 22)
        assert torch.all(logits[:, 15:] < -1e8)
        loss = torch.nn.functional.cross_entropy(logits, batch.receiver_label)
    assert torch.isfinite(logits).any(dim=-1).all()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("symmetry,actions", [("y_flip", (2,)), ("d2", (1, 2, 3))])
def test_zone_predictions_are_reflection_equivariant(model_name, symmetry, actions):
    """Reflecting the input must reflect the predicted zone grid exactly."""
    model = _build(model_name, symmetry=symmetry).eval()
    batch = _temporal_batch()
    with torch.no_grad():
        logits = model(batch)["target_zone_logits"].view(-1, GRID_M, GRID_N)
        for action in actions:
            reflected = model(_reflect(batch, action))["target_zone_logits"].view(-1, GRID_M, GRID_N)
            torch.testing.assert_close(reflected, flip_grid(logits, action), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_zone_predictions_distinguish_wings(model_name):
    """Unlike an invariant head, mirroring the wings must change the prediction."""
    model = _build(model_name, symmetry="y_flip").eval()
    batch = _temporal_batch()
    with torch.no_grad():
        logits = model(batch)["target_zone_logits"]
        mirrored = model(_reflect(batch, 2))["target_zone_logits"]
    assert not torch.allclose(logits, mirrored, atol=1e-4)


@pytest.mark.parametrize("model_name", ["graph_transformer", "graph_gru", "seq2spatial"])
def test_current_event_labels_are_not_injected(model_name):
    """Changing target placeholders must not change a model's logits."""
    torch.manual_seed(7)
    model = _build(model_name).eval()
    batch = _temporal_batch()
    altered = dataclasses.replace(
        batch,
        event_type=batch.event_type.clone(),
        event_outcome=batch.event_outcome.clone(),
        event_body_part=batch.event_body_part.clone(),
        event_numeric=batch.event_numeric.clone(),
    )
    # Preserve all observed graph/history inputs, then alter only final event
    # labels.  They must be ignored by the causal encoder.
    altered.event_type[:, -1] = 63
    altered.event_outcome[:, -1] = 31
    altered.event_body_part[:, -1] = 15
    altered.event_numeric[:, -1] = 100.0
    with torch.no_grad():
        torch.testing.assert_close(
            model(batch)["target_zone_logits"], model(altered)["target_zone_logits"]
        )


@pytest.mark.parametrize("model_name", ["graph_transformer", "graph_gru", "transport_memory", "seq2spatial"])
def test_left_padding_does_not_change_predictions(model_name):
    """Extra all-padding history steps must not change the prediction."""
    model = _build(model_name, max_sequence_length=5).eval()
    batch = _temporal_batch()

    def pad(tensor: torch.Tensor) -> torch.Tensor:
        padding = torch.zeros(tensor.shape[0], 2, *tensor.shape[2:], dtype=tensor.dtype)
        return torch.cat([padding, tensor], dim=1)

    padded = dataclasses.replace(
        batch,
        **{
            name: pad(getattr(batch, name))
            for name in (
                "node_features", "edge_features", "graph_features", "adj_mat", "player_mask",
                "role_features", "event_type", "event_outcome", "event_body_part",
                "event_numeric", "time_delta", "sequence_mask",
            )
        },
    )
    with torch.no_grad():
        expected = model(batch)["target_zone_logits"]
        actual = model(padded)["target_zone_logits"]
    if model_name == "graph_transformer":
        # Absolute position embeddings shift with padding; only check finiteness.
        assert torch.isfinite(actual).all()
    else:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_seq2spatial_transformer_encoder_and_history_sensitivity():
    batch = _temporal_batch()
    model = _build("seq2spatial", sequence_encoder="transformer").eval()
    with torch.no_grad():
        logits = model(batch)["target_zone_logits"]
        changed = dataclasses.replace(batch, event_numeric=batch.event_numeric.clone())
        changed.event_numeric[:, 1] = torch.rand_like(changed.event_numeric[:, 1])
        assert not torch.allclose(logits, model(changed)["target_zone_logits"])
    assert torch.isfinite(logits).all()
