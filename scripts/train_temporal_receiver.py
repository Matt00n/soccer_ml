"""Train temporal receiver models on the same pass samples and baselines as GATv2.

This is intentionally parallel to ``train_receiver.py``: it does not alter the
pure single-freeze-frame GATv2 benchmark.  One invocation trains an adapter to
that original GATv2 plus both temporal architectures on exactly the same
sampled train/validation examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ml.data.receiver_dataset import (
    closest_teammate_baseline,
    most_frequent_receiver_baseline,
    random_teammate_baseline,
)
from soccer_ml.data.temporal_receiver_dataset import (
    TemporalReceiverDataset,
    collate_temporal_receiver_graphs,
)
from soccer_ml.models.temporal_receiver import build_temporal_receiver_model
from soccer_ml.training.temporal_receiver_trainer import TemporalReceiverTrainer
from soccer_ml.training.trainer import get_default_device
from soccer_ml.utils.config import load_config


def _match_id_from_filename(path: Path) -> str:
    stem = path.stem
    for prefix in ("three_sixty_", "three-sixty_", "events_", "event_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    digits = "".join(character for character in stem if character.isdigit())
    return digits or stem


def find_match_pairs(data_dir: str, sample_count: int | None, seed: int) -> list[tuple[str, str]]:
    """Use the same supported data layouts and sampling rule as train_receiver."""
    root = Path(data_dir)
    three_sixty_dir, events_dir = root / "three-sixty", root / "events"
    pairs: list[tuple[str, str]] = []
    if three_sixty_dir.exists() and events_dir.exists():
        for freeze_frame_path in sorted(three_sixty_dir.glob("*.json")):
            event_path = events_dir / f"{freeze_frame_path.stem}.json"
            if event_path.exists():
                pairs.append((str(freeze_frame_path), str(event_path)))
    else:
        three_sixty_map: dict[str, str] = {}
        events_map: dict[str, str] = {}
        for path in sorted(root.glob("*.json")):
            match_id = _match_id_from_filename(path)
            name = path.name.lower()
            if "event" in name:
                events_map[match_id] = str(path)
            elif "three" in name or "sixty" in name:
                three_sixty_map[match_id] = str(path)
            else:
                three_sixty_map.setdefault(match_id, str(path))
                events_map.setdefault(match_id, str(path))
        pairs = [(three_sixty_map[key], events_map[key]) for key in sorted(three_sixty_map.keys() & events_map.keys())]
    if not pairs:
        raise FileNotFoundError(f"Could not match three-sixty/events JSON files in {root}.")
    if sample_count is not None:
        if len(pairs) < sample_count:
            raise ValueError(f"Requested {sample_count} files but only {len(pairs)} match pairs exist.")
        pairs = random.Random(seed).sample(pairs, sample_count)
    return pairs


def sample_disjoint_train_val_pairs(
    train_dir: str,
    val_dir: str,
    train_file_count: int | None,
    val_file_count: int | None,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Sample validation first, then exclude its match IDs from training.

    This also protects against accidental overlap when different directories
    contain copied versions of the same StatsBomb match.
    """
    train_population = find_match_pairs(train_dir, sample_count=None, seed=seed)
    val_population = find_match_pairs(val_dir, sample_count=None, seed=seed + 1)

    if val_file_count is not None:
        if len(val_population) < val_file_count:
            raise ValueError(
                f"Requested {val_file_count} validation files but only "
                f"{len(val_population)} match pairs exist."
            )
        val_pairs = random.Random(seed + 1).sample(val_population, val_file_count)
    else:
        val_pairs = val_population

    validation_match_ids = {_match_id_from_filename(Path(pair[0])) for pair in val_pairs}
    disjoint_train_population = [
        pair
        for pair in train_population
        if _match_id_from_filename(Path(pair[0])) not in validation_match_ids
    ]
    if train_file_count is not None:
        if len(disjoint_train_population) < train_file_count:
            raise ValueError(
                f"Requested {train_file_count} training files, but only "
                f"{len(disjoint_train_population)} remain after excluding "
                f"{len(validation_match_ids)} validation match IDs."
            )
        train_pairs = random.Random(seed).sample(disjoint_train_population, train_file_count)
    else:
        train_pairs = disjoint_train_population

    return train_pairs, val_pairs


def _build_multimatch_dataset(
    pairs: Sequence[tuple[str, str]],
    context_length: int,
    context_scope: str,
    max_samples: int | None,
) -> Dataset:
    datasets = [
        TemporalReceiverDataset(
            freeze_frame_path,
            event_path,
            context_length=context_length,
            context_scope=context_scope,  # type: ignore[arg-type]
            max_samples=max_samples,
        )
        for freeze_frame_path, event_path in pairs
    ]
    return ConcatDataset(datasets)


def _sample_fingerprint(dataset: Dataset) -> str:
    """Hash target event IDs, making a benchmark split auditable and repeatable."""
    identifiers: list[str] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        metadata = getattr(sample, "metadata", None) or {}
        identifiers.append(str(metadata.get("event_uuid", index)))
    return hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()[:16]


def _baseline_scores(loader: DataLoader, seed: int) -> dict[str, float]:
    """Evaluate existing naive receivers on the target graph of every sequence."""
    torch.manual_seed(seed)
    correct = {"closest_teammate_top_1": 0, "random_top_1": 0, "most_frequent_top_1": 0}
    total = 0
    for temporal_batch in loader:
        batch = temporal_batch.current_graph_batch()
        targets = batch.receiver_label
        predictions = {
            "closest_teammate_top_1": closest_teammate_baseline(batch),
            "random_top_1": random_teammate_baseline(batch),
            "most_frequent_top_1": most_frequent_receiver_baseline(batch),
        }
        for name, values in predictions.items():
            correct[name] += int((values == targets).sum().item())
        total += len(targets)
    return {name: value / max(total, 1) for name, value in correct.items()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark temporal GATv2 receiver models.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--train-dir")
    parser.add_argument("--val-dir")
    parser.add_argument("--train-file-count", type=int)
    parser.add_argument("--val-file-count", type=int)
    parser.add_argument("--three-sixty", help="Single freeze-frame JSON file (single-file mode).")
    parser.add_argument("--events", help="Single event JSON file (single-file mode).")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=8, help="Number of earlier 360 events.")
    parser.add_argument(
        "--context-scope",
        choices=("possession", "period", "match"),
        default="possession",
        help="Reset history at a possession (default), period, or match boundary.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("spatial_gatv2", "graph_transformer", "graph_gru", "transport_memory"),
        default=("spatial_gatv2", "graph_transformer", "graph_gru", "transport_memory"),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--temporal-heads", type=int)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", default="checkpoints/temporal_receiver")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _set_seed(args.sample_seed)
    batch_size = args.batch_size or config.training.batch_size
    epochs = args.epochs or config.training.epochs
    learning_rate = args.lr or config.training.lr
    hidden_dim = args.hidden_dim or config.model.hidden_dim
    num_layers = args.num_layers or config.model.num_layers
    temporal_heads = args.temporal_heads or config.model.nb_heads

    if hidden_dim % temporal_heads:
        raise ValueError("--hidden-dim must be divisible by --temporal-heads.")

    if args.train_dir or args.val_dir:
        if not (args.train_dir and args.val_dir):
            raise ValueError("Provide both --train-dir and --val-dir together.")
        train_pairs, val_pairs = sample_disjoint_train_val_pairs(
            args.train_dir,
            args.val_dir,
            args.train_file_count,
            args.val_file_count,
            args.sample_seed,
        )
        train_dataset = _build_multimatch_dataset(
            train_pairs, args.context_length, args.context_scope, args.max_samples
        )
        val_dataset = _build_multimatch_dataset(
            val_pairs, args.context_length, args.context_scope, None
        )
        split_description = {
            "mode": "separate_match_sets",
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
        }
    elif args.three_sixty and args.events:
        full_dataset = TemporalReceiverDataset(
            args.three_sixty,
            args.events,
            context_length=args.context_length,
            context_scope=args.context_scope,
            max_samples=args.max_samples,
        )
        n_validation = int(args.val_split * len(full_dataset))
        n_training = len(full_dataset) - n_validation
        train_dataset, val_dataset = random_split(
            full_dataset,
            [n_training, n_validation],
            generator=torch.Generator().manual_seed(args.sample_seed),
        )
        split_description = {"mode": "single_match_random_pass_split", "three_sixty": args.three_sixty, "events": args.events}
    else:
        raise ValueError("Provide (--train-dir and --val-dir) or (--three-sixty and --events).")

    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Training and validation sets must both contain at least one valid pass.")

    print("=" * 72)
    print("Temporal Receiver Benchmark (single-frame GATv2 remains unchanged)")
    print(f"Device: {get_default_device()} | Context: {args.context_length} prior events, {args.context_scope} scope")
    print(f"Train targets: {len(train_dataset)} ({_sample_fingerprint(train_dataset)})")
    print(f"Validation targets: {len(val_dataset)} ({_sample_fingerprint(val_dataset)})")
    print(f"Models: {', '.join(args.models)}")
    print("=" * 72)

    # A non-shuffled loader defines baseline scores once.  It is the exact
    # current-target graph sequence subsequently used by every temporal model.
    baseline_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_receiver_graphs)
    baseline_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_receiver_graphs)
    baselines = {
        "train": _baseline_scores(baseline_train_loader, args.sample_seed),
        "validation": _baseline_scores(baseline_val_loader, args.sample_seed + 1),
    }
    print("Naive baselines on this exact split:")
    for split_name, scores in baselines.items():
        print(f"  {split_name}: " + ", ".join(f"{name}={score:.2%}" for name, score in scores.items()))

    base_model_kwargs = {
        "in_node_dim": config.data.node_dim,
        "in_edge_dim": config.data.edge_dim,
        "in_graph_dim": config.data.graph_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "nb_heads": config.model.nb_heads,
        "temporal_layers": args.temporal_layers,
        "temporal_heads": temporal_heads,
        "max_sequence_length": args.context_length + 1,
        "dropout": args.dropout,
    }
    run_summary: dict[str, object] = {
        "split": split_description,
        "sample_seed": args.sample_seed,
        "context_length": args.context_length,
        "context_scope": args.context_scope,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "train_fingerprint": _sample_fingerprint(train_dataset),
        "val_fingerprint": _sample_fingerprint(val_dataset),
        "baselines": baselines,
        "models": {},
    }

    for model_number, model_name in enumerate(args.models):
        print(f"\n{'=' * 72}\nTraining {model_name}\n{'=' * 72}")
        # A private DataLoader generator makes batch order identical for every
        # architecture, independent of its dropout/random-number consumption.
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.sample_seed),
            collate_fn=collate_temporal_receiver_graphs,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_receiver_graphs)
        _set_seed(args.sample_seed + model_number)
        model_kwargs = dict(base_model_kwargs)
        if model_name == "spatial_gatv2":
            # Match the original receiver script, whose SoccerGATv2Model uses
            # the configuration dropout (normally 0.0), not temporal dropout.
            model_kwargs["dropout"] = config.model.dropout
        model = build_temporal_receiver_model(model_name, **model_kwargs)
        checkpoint_dir = Path(args.checkpoint_dir) / model_name
        trainer = TemporalReceiverTrainer(
            model=model,
            model_name=model_name,
            model_kwargs=getattr(model, "model_kwargs", base_model_kwargs),
            train_loader=train_loader,
            val_loader=val_loader,
            lr=learning_rate,
            weight_decay=config.training.weight_decay,
            checkpoint_dir=checkpoint_dir,
        )
        history = trainer.fit(epochs)
        final_train = trainer.evaluate(baseline_train_loader)
        final_val = trainer.evaluate(baseline_val_loader)
        print(f"{model_name} final train top-1: {final_train.get('top_1_acc', 0.0):.2%}")
        print(f"{model_name} final val top-1:   {final_val.get('top_1_acc', 0.0):.2%}")
        with (checkpoint_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        run_summary["models"][model_name] = {
            "checkpoint": str(checkpoint_dir / "best_model.pt"),
            "best_val_loss": trainer.best_val_loss,
            "final_train": final_train,
            "final_validation": final_val,
        }

    output_path = Path(args.checkpoint_dir) / "benchmark_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(run_summary, file, indent=2)
    print(f"\nSaved comparable benchmark summary to {output_path}")


if __name__ == "__main__":
    main()
