"""Tests for end-to-end SoccerGATv2Model and decoder heads."""

import dataclasses

import pytest
import torch
from soccer_ml.data.dataset import SyntheticSoccerDataset, collate_soccer_graphs
from soccer_ml.data.transforms import D2GroupTransform, flip_grid
from soccer_ml.models.soccer_gat import SoccerGATv2Model
from soccer_ml.training.losses import SoccerMultiTaskLoss


def test_soccer_gat_forward_and_backward():
    """Verify forward pass and backward gradients through the full model."""
    dataset = SyntheticSoccerDataset(num_samples=4, num_players=22, node_dim=7, edge_dim=1, graph_dim=4, seed=42)
    batch = collate_soccer_graphs([dataset[i] for i in range(4)])

    model = SoccerGATv2Model(
        in_node_dim=7,
        in_edge_dim=1,
        in_graph_dim=4,
        hidden_dim=32,
        num_layers=2,
        nb_heads=4,
        enable_receiver_head=True,
        enable_target_location_head=False,  # synthetic data has no zone labels
        enable_shot_head=True,
        enable_creativity_head=True,
        enable_vae_head=True,
        vae_latent_dim=8,
    )

    outputs = model(batch)

    assert "receiver_logits" in outputs
    assert outputs["receiver_logits"].shape == (4, 22)

    assert "shot_logits" in outputs
    assert outputs["shot_logits"].shape == (4, 1)

    assert "creativity_scores" in outputs
    assert outputs["creativity_scores"].shape == (4, 22)

    assert "vae_recon" in outputs
    assert outputs["vae_recon"].shape == (4, 22, 4)

    # Compute loss and check backward pass
    loss_fn = SoccerMultiTaskLoss()
    losses = loss_fn(outputs, batch)

    total_loss = losses["total_loss"]
    assert total_loss.item() > 0.0

    total_loss.backward()

    # Check gradients exist on backbone and heads
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


@pytest.mark.parametrize("symmetry,actions", [("d2", (1, 2, 3)), ("y_flip", (2,))])
def test_target_location_head_is_reflection_equivariant(symmetry, actions):
    """Reflecting the pitch must reflect the predicted target-zone grid."""
    dataset = SyntheticSoccerDataset(num_samples=3, num_players=22, node_dim=9, edge_dim=1, graph_dim=7, seed=3)
    batch = collate_soccer_graphs([dataset[i] for i in range(3)])
    model = SoccerGATv2Model(
        in_node_dim=9, in_graph_dim=7, hidden_dim=16, num_layers=2, nb_heads=4,
        grid_m=26, grid_n=17, symmetry=symmetry,
        enable_receiver_head=False, enable_shot_head=False, enable_creativity_head=False, enable_vae_head=False,
    ).eval()
    with torch.no_grad():
        logits = model(batch)["target_zone_logits"]
        assert logits.shape == (3, 26 * 17)
        grid = logits.view(3, 26, 17)
        for action in actions:
            reflected_batch = dataclasses.replace(
                batch, node_features=D2GroupTransform().transform_node_features(batch.node_features, action)
            )
            reflected = model(reflected_batch)["target_zone_logits"].view(3, 26, 17)
            torch.testing.assert_close(reflected, flip_grid(grid, action), atol=1e-4, rtol=1e-4)
