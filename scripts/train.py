"""CLI script to train Soccer GATv2 model."""

import argparse
from torch.utils.data import DataLoader

from soccer_ml.data.dataset import SyntheticSoccerDataset, collate_soccer_graphs
from soccer_ml.models.soccer_gat import SoccerGATv2Model
from soccer_ml.training.losses import SoccerMultiTaskLoss
from soccer_ml.training.trainer import SoccerTrainer, get_default_device
from soccer_ml.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train Soccer GATv2 Model")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--num-samples", type=int, default=256, help="Number of synthetic samples (for testing)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    epochs = args.epochs if args.epochs is not None else config.training.epochs
    batch_size = args.batch_size if args.batch_size is not None else config.training.batch_size
    lr = args.lr if args.lr is not None else config.training.lr

    print("=" * 60)
    print("Starting Soccer GATv2 Training")
    print(f"Device: {get_default_device()}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Learning rate: {lr}")
    print("=" * 60)

    # Prepare datasets (using synthetic soccer dataset until raw StatsBomb data parser is plugged in)
    train_dataset = SyntheticSoccerDataset(
        num_samples=args.num_samples,
        num_players=config.data.num_players,
        node_dim=config.data.node_dim,
        edge_dim=config.data.edge_dim,
        graph_dim=config.data.graph_dim,
        seed=42,
    )
    val_dataset = SyntheticSoccerDataset(
        num_samples=max(32, args.num_samples // 4),
        num_players=config.data.num_players,
        node_dim=config.data.node_dim,
        edge_dim=config.data.edge_dim,
        graph_dim=config.data.graph_dim,
        seed=1337,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_soccer_graphs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_soccer_graphs,
    )

    # Initialize model
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

    loss_fn = SoccerMultiTaskLoss(
        receiver_weight=config.loss_weights.receiver_weight,
        shot_weight=config.loss_weights.shot_weight,
        creativity_weight=config.loss_weights.creativity_weight,
        vae_weight=config.loss_weights.vae_weight,
        vae_kl_weight=config.loss_weights.vae_kl_weight,
    )

    trainer = SoccerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        weight_decay=config.training.weight_decay,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.eps,
        clip_grad_norm=config.training.clip_grad_norm,
        checkpoint_dir=config.training.checkpoint_dir,
        loss_fn=loss_fn,
    )

    history = trainer.fit(num_epochs=epochs)
    print("Training finished successfully!")
    return history


if __name__ == "__main__":
    main()
