"""CLI script to evaluate Soccer GATv2 model from a saved checkpoint."""

import argparse
from torch.utils.data import DataLoader

from soccer_ml.data.dataset import SyntheticSoccerDataset, collate_soccer_graphs
from soccer_ml.models.soccer_gat import SoccerGATv2Model
from soccer_ml.training.trainer import SoccerTrainer, get_default_device
from soccer_ml.training.losses import SoccerMultiTaskLoss
from soccer_ml.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Soccer GATv2 Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num-samples", type=int, default=128, help="Number of evaluation samples")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_default_device()

    print("=" * 60)
    print(f"Evaluating Soccer GATv2 Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print("=" * 60)

    dataset = SyntheticSoccerDataset(
        num_samples=args.num_samples,
        num_players=config.data.num_players,
        node_dim=config.data.node_dim,
        edge_dim=config.data.edge_dim,
        graph_dim=config.data.graph_dim,
        seed=9999,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_soccer_graphs,
    )

    model = SoccerGATv2Model(
        in_node_dim=config.data.node_dim,
        in_edge_dim=config.data.edge_dim,
        in_graph_dim=config.data.graph_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        nb_heads=config.model.nb_heads,
        dropout=config.model.dropout,
        enable_receiver_head=config.model.enable_receiver_head,
        enable_shot_head=config.model.enable_shot_head,
        enable_creativity_head=config.model.enable_creativity_head,
        enable_vae_head=config.model.enable_vae_head,
        vae_latent_dim=config.model.vae_latent_dim,
    )

    trainer = SoccerTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        device=device,
        loss_fn=SoccerMultiTaskLoss(),
    )

    trainer.load_checkpoint(args.checkpoint)
    metrics = trainer.evaluate(loader)

    print("\nEvaluation Results:")
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s}: {v:.4f}")


if __name__ == "__main__":
    main()
