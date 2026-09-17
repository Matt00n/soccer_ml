"""Tests for StatsBomb 360 data parsing and integration with Soccer GATv2."""

from pathlib import Path
import torch
from torch.utils.data import DataLoader

from soccer_ml.data.statsbomb import StatsBomb360Parser, StatsBomb360Dataset
from soccer_ml.data.dataset import collate_soccer_graphs
from soccer_ml.models.soccer_gat import SoccerGATv2Model
from soccer_ml.training.losses import SoccerMultiTaskLoss


def test_statsbomb_parser_and_dataset():
    """Verify StatsBomb360Dataset loads and produces valid SoccerGraphBatch."""
    three_sixty_path = Path("data_test/three_sixty_3764440.json")
    if not three_sixty_path.exists():
        return  # Skip if test file absent

    dataset = StatsBomb360Dataset(
        three_sixty_path=three_sixty_path,
        max_samples=16,
    )

    assert len(dataset) == 16
    sample = dataset[0]

    # Verify tensor shapes
    assert sample.node_features.shape == (22, 7)
    assert sample.edge_features.shape == (22, 22, 1)
    assert sample.graph_features.shape == (4,)
    assert sample.adj_mat.shape == (22, 22)
    assert sample.player_mask.shape == (22,)
    assert sample.player_mask.sum().item() > 0

    # Test batch collation
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_soccer_graphs)
    batch = next(iter(loader))

    assert batch.node_features.shape == (8, 22, 7)
    assert batch.edge_features.shape == (8, 22, 22, 1)
    assert batch.adj_mat.shape == (8, 22, 22)
    assert batch.player_mask.shape == (8, 22)


def test_soccer_gat_on_statsbomb_data():
    """Verify SoccerGATv2 model forward and backward pass on real StatsBomb 360 data."""
    three_sixty_path = Path("data_test/three_sixty_3764440.json")
    if not three_sixty_path.exists():
        return

    dataset = StatsBomb360Dataset(three_sixty_path=three_sixty_path, max_samples=8)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_soccer_graphs)
    batch = next(iter(loader))

    model = SoccerGATv2Model(
        in_node_dim=7,
        in_edge_dim=1,
        in_graph_dim=4,
        hidden_dim=32,
        num_layers=2,
        nb_heads=4,
    )

    outputs = model(batch)
    assert "receiver_logits" in outputs
    assert outputs["receiver_logits"].shape == (4, 22)
    assert "shot_logits" in outputs
    assert outputs["shot_logits"].shape == (4, 1)

    loss_fn = SoccerMultiTaskLoss()
    losses = loss_fn(outputs, batch)
    assert losses["total_loss"].item() > 0.0

    losses["total_loss"].backward()
