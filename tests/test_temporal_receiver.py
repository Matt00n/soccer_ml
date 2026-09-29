"""Shape, masking, and gradient tests for temporal receiver architectures."""

import torch

from soccer_ml.data.temporal_receiver_dataset import TemporalReceiverBatch
from soccer_ml.models.temporal_receiver import build_temporal_receiver_model


def _temporal_batch() -> TemporalReceiverBatch:
    batch_size, steps, players = 2, 3, 22
    player_mask = torch.zeros(batch_size, steps, players, dtype=torch.bool)
    player_mask[:, :, :15] = True
    sequence_mask = torch.tensor([[False, True, True], [True, True, True]])
    role_features = torch.zeros(batch_size, steps, players, 3)
    role_features[:, :, 0, 0] = 1.0  # actor
    role_features[:, :, :8, 2] = 1.0  # teammates of actor
    role_features[:, :-1, 1, 1] = 1.0  # known prior recipient only
    return TemporalReceiverBatch(
        node_features=torch.randn(batch_size, steps, players, 7),
        edge_features=torch.zeros(batch_size, steps, players, players, 1),
        graph_features=torch.randn(batch_size, steps, 4),
        adj_mat=player_mask.unsqueeze(-1).float() * player_mask.unsqueeze(-2).float(),
        player_mask=player_mask,
        role_features=role_features,
        event_type=torch.tensor([[0, 30, 43], [30, 42, 0]]),
        event_outcome=torch.tensor([[0, 1, 0], [1, 0, 0]]),
        event_body_part=torch.zeros(batch_size, steps, dtype=torch.long),
        event_numeric=torch.randn(batch_size, steps, 8),
        time_delta=torch.tensor([[0.0, 1.2, 0.8], [0.0, 0.5, 1.0]]),
        sequence_mask=sequence_mask,
        receiver_label=torch.tensor([2, 3]),
    )


def test_temporal_receiver_models_forward_and_masked_loss():
    batch = _temporal_batch()
    for model_name in (
        "spatial_gatv2",
        "graph_transformer",
        "graph_gru",
        "transport_memory",
    ):
        model = build_temporal_receiver_model(
            model_name,
            hidden_dim=32,
            num_layers=2,
            nb_heads=4,
            temporal_layers=1,
            temporal_heads=4,
            max_sequence_length=3,
            dropout=0.0,
        )
        outputs = model(batch)
        logits = outputs["receiver_logits"]
        assert logits.shape == (2, 22)
        assert torch.all(logits[:, 15:] < -1e8)
        loss = torch.nn.functional.cross_entropy(logits, batch.receiver_label)
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())


def test_current_event_labels_are_not_injected():
    """Changing target placeholders must not change a model's logits."""
    torch.manual_seed(7)
    model = build_temporal_receiver_model(
        "graph_transformer",
        hidden_dim=32,
        num_layers=2,
        nb_heads=4,
        temporal_layers=1,
        temporal_heads=4,
        max_sequence_length=3,
        dropout=0.0,
    ).eval()
    batch = _temporal_batch()
    altered = _temporal_batch()
    # Preserve all observed graph/history inputs, then alter only final event
    # labels.  They must be ignored by the causal encoder.
    for name in (
        "node_features",
        "edge_features",
        "graph_features",
        "adj_mat",
        "player_mask",
        "role_features",
        "event_type",
        "event_outcome",
        "event_body_part",
        "event_numeric",
        "time_delta",
        "sequence_mask",
        "receiver_label",
    ):
        setattr(altered, name, getattr(batch, name).clone())
    altered.event_type[:, -1] = 63
    altered.event_outcome[:, -1] = 31
    altered.event_body_part[:, -1] = 15
    altered.event_numeric[:, -1] = 100.0
    with torch.no_grad():
        assert torch.allclose(model(batch)["receiver_logits"], model(altered)["receiver_logits"])
