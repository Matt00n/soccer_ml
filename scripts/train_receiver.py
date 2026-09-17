"""CLI script to train Soccer GATv2 model on receiver prediction task."""

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ml.data.receiver_dataset import ReceiverDataset, collate_receiver_graphs
from soccer_ml.models.soccer_gat import SoccerGATv2Model
from soccer_ml.training.losses import SoccerMultiTaskLoss
from soccer_ml.training.trainer import SoccerTrainer, get_default_device
from soccer_ml.utils.config import load_config


def _match_id_from_filename(path: Path) -> str:
    """Normalize a file name to a match identifier.

    Works for files like:
      - 3764440.json
      - events_3764440.json
      - three_sixty_3764440.json
      - three-sixty/3764440.json
    """
    stem = path.stem
    for token in ["three_sixty_", "three-sixty_", "events_", "event_"]:
        if stem.startswith(token):
            stem = stem[len(token) :]
            break
    digits = "".join(ch for ch in stem if ch.isdigit())
    return digits or stem


def find_match_pairs(data_dir: str, sample_count: int | None = None, seed: int = 42):
    """Return matching (three-sixty, events) pairs for a data directory.

    Supported layouts:
      1) data_dir/three-sixty/*.json and data_dir/events/*.json
      2) data_dir/*.json, where files are named by match id or prefixed by events_/three_sixty_
    """
    base_dir = Path(data_dir)
    three_sixty_dir = base_dir / "three-sixty"
    events_dir = base_dir / "events"

    pairs = []

    if three_sixty_dir.exists() and events_dir.exists():
        for three_sixty_file in sorted(three_sixty_dir.glob("*.json")):
            events_file = events_dir / f"{three_sixty_file.stem}.json"
            if events_file.exists():
                pairs.append((str(three_sixty_file), str(events_file)))
    else:
        json_files = sorted(base_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {base_dir}. Expected either:\n"
                f"  - {base_dir}/three-sixty/*.json and {base_dir}/events/*.json\n"
                f"  - or all match files directly inside {base_dir}."
            )

        three_sixty_map = {}
        events_map = {}
        for file_path in json_files:
            match_id = _match_id_from_filename(file_path)
            if "event" in file_path.stem.lower() or "event" in file_path.name.lower():
                events_map[match_id] = str(file_path)
            elif "three" in file_path.stem.lower() or "sixty" in file_path.stem.lower():
                three_sixty_map[match_id] = str(file_path)
            else:
                # Heuristic: direct files are usually match ids, so map them to both sides
                # if an exact match exists. This keeps the behavior flexible for typical datasets.
                three_sixty_map.setdefault(match_id, str(file_path))
                events_map.setdefault(match_id, str(file_path))

        for match_id in sorted(set(three_sixty_map) & set(events_map)):
            pairs.append((three_sixty_map[match_id], events_map[match_id]))

    if not pairs:
        raise FileNotFoundError(
            f"Could not match any three-sixty/event pairs in {base_dir}.\n"
            "Expected one of these layouts:\n"
            "  - {train_dir}/three-sixty/*.json + {train_dir}/events/*.json\n"
            "  - {train_dir}/*.json with files like 3764440.json, events_3764440.json, three_sixty_3764440.json"
        )

    if sample_count is not None:
        if len(pairs) < sample_count:
            raise ValueError(
                f"Requested {sample_count} files but only {len(pairs)} match pairs exist."
            )
        rng = random.Random(seed)
        pairs = rng.sample(pairs, sample_count)

    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Soccer GATv2 on Receiver Prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Directory containing three-sixty/ and events/ for training",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Directory containing three-sixty/ and events/ for validation",
    )
    parser.add_argument(
        "--train-file-count",
        type=int,
        default=None,
        help="Number of match files to sample for training",
    )
    parser.add_argument(
        "--val-file-count",
        type=int,
        default=None,
        help="Number of match files to sample for validation",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when sampling match files",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--three-sixty",
        type=str,
        default=None,
        help="Path to a single StatsBomb 360 JSON file (legacy mode)",
    )
    parser.add_argument(
        "--events",
        type=str,
        default=None,
        help="Path to a single StatsBomb events JSON file (legacy mode)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max training samples"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=None, help="Override hidden dimension"
    )
    parser.add_argument(
        "--num-layers", type=int, default=None, help="Override number of layers"
    )
    return parser.parse_args()


def build_multimatch_dataset(file_pairs, max_samples=None):
    datasets = [
        ReceiverDataset(t360, events, max_samples=max_samples)
        for t360, events in file_pairs
    ]
    return ConcatDataset(datasets)


def compute_baseline_scores(data_loader):
    """Compute baseline receiver prediction accuracies for a dataset.
    
    Args:
        data_loader: DataLoader yielding SoccerGraphBatch instances
        
    Returns:
        dict with keys: closest_teammate_top_1, random_top_1, most_frequent_top_1
    """
    from soccer_ml.data.receiver_dataset import (
        closest_teammate_baseline,
        random_teammate_baseline,
        most_frequent_receiver_baseline,
    )
    from soccer_ml.training.metrics import compute_topk_accuracy

    if data_loader is None:
        return {}

    closest_correct = 0
    random_correct = 0
    most_frequent_correct = 0
    total = 0

    device = torch.device("cpu")

    for batch in data_loader:
        batch = batch.to(device)
        
        if batch.receiver_label is None:
            continue

        # Closest teammate baseline
        closest_preds = closest_teammate_baseline(batch)
        closest_correct += (closest_preds == batch.receiver_label).sum().item()

        # Random baseline
        random_preds = random_teammate_baseline(batch)
        random_correct += (random_preds == batch.receiver_label).sum().item()

        # Most frequent baseline (use random as fallback)
        mf_preds = most_frequent_receiver_baseline(batch)
        most_frequent_correct += (mf_preds == batch.receiver_label).sum().item()

        total += len(batch.receiver_label)

    return {
        "closest_teammate_top_1": closest_correct / max(total, 1),
        "random_top_1": random_correct / max(total, 1),
        "most_frequent_top_1": most_frequent_correct / max(total, 1),
    }


def _print_metric_summary(label: str, metrics: dict):
    """Print a compact metrics summary with loss and receiver accuracy metrics."""
    if not metrics:
        print(f"{label}: no metrics available")
        return

    print(f"{label}")
    keys = [
        "total_loss",
        "val_total_loss",
        "train_total_loss",
        "receiver_top_1_acc",
        "receiver_top_3_acc",
        "train_receiver_top_1_acc",
        "train_receiver_top_3_acc",
        "val_receiver_top_1_acc",
        "val_receiver_top_3_acc",
    ]
    for key in keys:
        if key in metrics:
            value = metrics[key]
            if "acc" in key:
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value:.4f}")

    for key, value in sorted(metrics.items()):
        if key in keys:
            continue
        if "acc" in key:
            print(f"  {key}: {value:.2%}")
        elif "loss" in key or "mse" in key or "mae" in key:
            print(f"  {key}: {value:.4f}")


def main():
    args = parse_args()
    config = load_config(args.config)

    epochs = args.epochs if args.epochs is not None else config.training.epochs
    batch_size = (
        args.batch_size if args.batch_size is not None else config.training.batch_size
    )
    lr = args.lr if args.lr is not None else config.training.lr
    hidden_dim = (
        args.hidden_dim if args.hidden_dim is not None else config.model.hidden_dim
    )
    num_layers = (
        args.num_layers if args.num_layers is not None else config.model.num_layers
    )

    print("=" * 60)
    print("Training Soccer GATv2 on Receiver Prediction Task")
    print(f"Device: {get_default_device()}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Learning rate: {lr}")
    print(f"Hidden dim: {hidden_dim} | Num layers: {num_layers}")
    print("=" * 60)

    # Build train/val datasets from file directories or single-file mode
    if args.train_dir is not None or args.val_dir is not None:
        if args.train_dir is None or args.val_dir is None:
            raise ValueError(
                "Both --train-dir and --val-dir must be provided together."
            )

        train_pairs = find_match_pairs(
            args.train_dir,
            sample_count=args.train_file_count,
            seed=args.sample_seed,
        )
        val_pairs = find_match_pairs(
            args.val_dir,
            sample_count=args.val_file_count,
            seed=args.sample_seed + 1,
        )

        train_dataset = build_multimatch_dataset(
            train_pairs, max_samples=args.max_samples
        )
        val_dataset = build_multimatch_dataset(val_pairs, max_samples=None)

        print(f"\nTraining files sampled: {len(train_pairs)}")
        print(f"Validation files sampled: {len(val_pairs)}")
        print(f"Train passes: {len(train_dataset)}")
        print(f"Val passes: {len(val_dataset)}")

    elif args.three_sixty is not None and args.events is not None:
        full_dataset = ReceiverDataset(
            three_sixty_path=args.three_sixty,
            events_path=args.events,
            max_samples=args.max_samples,
        )
        n_total = len(full_dataset)
        n_val = int(args.val_split * n_total)
        n_train = n_total - n_val

        train_dataset, val_dataset = random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.sample_seed),
        )

        print(f"\nSingle-file mode")
        print(f"Train passes: {len(train_dataset)}")
        print(f"Val passes: {len(val_dataset)}")

    else:
        raise ValueError(
            "Provide either (--train-dir and --val-dir) or (--three-sixty and --events)."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_receiver_graphs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_receiver_graphs,
    )

    # Initialize model (receiver head only)
    model = SoccerGATv2Model(
        in_node_dim=config.data.node_dim,
        in_edge_dim=config.data.edge_dim,
        in_graph_dim=config.data.graph_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        nb_heads=config.model.nb_heads,
        enable_receiver_head=True,
        enable_shot_head=False,
        enable_creativity_head=False,
        enable_vae_head=False,
    )

    # Create custom loss (receiver only)
    loss_fn = SoccerMultiTaskLoss(
        receiver_weight=1.0,
        shot_weight=0.0,
        creativity_weight=0.0,
        vae_weight=0.0,
    )

    # Initialize trainer
    trainer = SoccerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        weight_decay=config.training.weight_decay,
        checkpoint_dir="checkpoints/receiver",
        loss_fn=loss_fn,
    )

    # Train
    print("\nStarting training...")
    history = trainer.fit(num_epochs=epochs)

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")

    final_train_metrics = {
        k.replace("val_", "train_"): v
        for k, v in trainer.evaluate(data_loader=train_loader).items()
    }
    final_val_metrics = trainer.evaluate(data_loader=val_loader)

    _print_metric_summary("Final train metrics:", final_train_metrics)
    _print_metric_summary("Final validation metrics:", final_val_metrics)

    # Compute baseline scores on the actual train/val datasets
    print("\n" + "=" * 60)
    print("Baseline Receiver Prediction Accuracies (on this run's data)")
    print("=" * 60)
    
    train_baselines = compute_baseline_scores(train_loader)
    if train_baselines:
        print("\nTrain dataset baselines:")
        for key, val in train_baselines.items():
            print(f"  {key}: {val:.2%}")

    val_baselines = compute_baseline_scores(val_loader)
    if val_baselines:
        print("\nValidation dataset baselines:")
        for key, val in val_baselines.items():
            print(f"  {key}: {val:.2%}")

    if "train_total_loss" in history:
        print(
            f"\nFinal train-set loss from history: {history['train_total_loss'][-1]:.4f}"
        )
    if "train_receiver_top_1_acc" in history:
        print(
            "Final train-set receiver top-1 accuracy from history: "
            f"{history['train_receiver_top_1_acc'][-1]:.2%}"
        )
    if "val_receiver_top_1_acc" in history:
        print(
            "Final validation receiver top-1 accuracy from history: "
            f"{history['val_receiver_top_1_acc'][-1]:.2%}"
        )


if __name__ == "__main__":
    main()
