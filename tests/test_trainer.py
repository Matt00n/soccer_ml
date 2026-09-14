"""Tests for SoccerTrainer training loop and checkpointing."""

import tempfile
from torch.utils.data import DataLoader
from soccer_ml.data.dataset import SyntheticSoccerDataset, collate_soccer_graphs
from soccer_ml.models.soccer_gat import SoccerGATv2Model
from soccer_ml.training.trainer import SoccerTrainer
from soccer_ml.training.losses import SoccerMultiTaskLoss


def test_trainer_fit_and_checkpoint():
    """Verify trainer completes training loop and saves/loads checkpoints cleanly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_dataset = SyntheticSoccerDataset(num_samples=16, num_players=10, node_dim=7, edge_dim=1, graph_dim=4, seed=1)
        val_dataset = SyntheticSoccerDataset(num_samples=8, num_players=10, node_dim=7, edge_dim=1, graph_dim=4, seed=2)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_soccer_graphs)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_soccer_graphs)

        model = SoccerGATv2Model(
            in_node_dim=7,
            in_edge_dim=1,
            in_graph_dim=4,
            hidden_dim=16,
            num_layers=1,
            nb_heads=2,
        )

        trainer = SoccerTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=1e-3,
            checkpoint_dir=tmp_dir,
            loss_fn=SoccerMultiTaskLoss(),
        )

        history = trainer.fit(num_epochs=2)

        assert "total_loss" in history
        assert len(history["total_loss"]) == 2
        assert "val_total_loss" in history

        # Test loading saved best model
        best_ckpt = f"{tmp_dir}/best_model.pt"
        trainer.load_checkpoint(best_ckpt)
        assert trainer.current_epoch == 2
