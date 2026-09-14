"""Tests for end-to-end SoccerGATv2Model and decoder heads."""

import torch
from soccer_ml.data.dataset import SyntheticSoccerDataset, collate_soccer_graphs
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
