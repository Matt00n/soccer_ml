"""Benchmark single-frame and temporal GATv2 models on pass target prediction.

By default every model predicts the pitch-grid zone (26 x 17) of the end
location of completed passes (``--task target_zone``); ``--task receiver``
reproduces the earlier receiver benchmark.  This is intentionally parallel to
``train_receiver.py``, which stays unchanged.  One invocation trains every
selected model and evaluates the naive baselines on exactly the same sampled
train/validation examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ml.data.pitch_grid import PitchGrid
from soccer_ml.data.schema import STATSBOMB_NODE_LAYOUT
from soccer_ml.data.splits import load_match_datasets, sample_disjoint_train_val_pairs, sample_fingerprint, set_seed
from soccer_ml.data.receiver_dataset import (
    closest_teammate_baseline,
    most_frequent_receiver_baseline,
    random_teammate_baseline,
)
from soccer_ml.data.temporal_receiver_dataset import (
    TemporalReceiverDataset,
    collate_temporal_receiver_graphs,
)
from soccer_ml.data.transforms import SYMMETRY_GROUPS
from soccer_ml.models.temporal_receiver import SEQUENCE_ENCODERS, TASKS, TEMPORAL_MODELS, build_temporal_receiver_model
from soccer_ml.training.temporal_receiver_trainer import (
    TemporalReceiverTrainer,
    evaluate_zone_baselines,
    fit_zone_baselines,
)
from soccer_ml.training.trainer import get_default_device
from soccer_ml.utils.config import load_config


def _build_multimatch_dataset(
    pairs: Sequence[tuple[str, str]],
    max_samples: int | None,
    **dataset_kwargs,
) -> tuple[Dataset, list[tuple[str, str]]]:
    return load_match_datasets(
        pairs,
        lambda three_sixty, events: TemporalReceiverDataset(three_sixty, events, max_samples=max_samples, **dataset_kwargs),
    )


_sample_fingerprint = sample_fingerprint


def _receiver_baseline_scores(loader: DataLoader, seed: int) -> dict[str, float]:
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


def _format_zone_metrics(metrics: dict[str, float]) -> str:
    nll = f"nll={metrics['nll']:.3f}" if "nll" in metrics else "nll=  -  "
    text = (
        f"{nll} top-1={metrics['top_1_acc']:.2%} top-3={metrics['top_3_acc']:.2%} "
        f"top-5={metrics['top_5_acc']:.2%} within-1={metrics['within_1_zone_acc']:.2%} "
        f"dist mean/median={metrics['mean_dist_yd']:.1f}/{metrics['median_dist_yd']:.1f}yd "
        f"({metrics['mean_dist_m']:.1f}/{metrics['median_dist_m']:.1f}m)"
    )
    if "receiver_acc" in metrics:
        text += f" receiver-acc={metrics['receiver_acc']:.2%}"
    return text


_set_seed = set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark temporal GATv2 pass target models.")
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
        choices=tuple(TEMPORAL_MODELS),
        default=tuple(TEMPORAL_MODELS),
    )
    parser.add_argument("--task", choices=TASKS, default="target_zone")
    parser.add_argument("--grid-m", type=int, default=26, help="Target grid bins along the pitch length.")
    parser.add_argument("--grid-n", type=int, default=17, help="Target grid bins along the pitch width.")
    parser.add_argument(
        "--symmetry",
        choices=tuple(SYMMETRY_GROUPS),
        default="y_flip",
        help="Reflection views for frame averaging (default y_flip: attacking direction is fixed).",
    )
    parser.add_argument("--use-mlp", action="store_true", help="Two-layer MLP heads instead of linear heads.")
    parser.add_argument("--sequence-encoder", choices=SEQUENCE_ENCODERS, default="gru", help="seq2spatial only.")
    parser.add_argument("--num-fourier-bands", type=int, default=0, help="Pitch Fourier features (0 = off).")
    parser.add_argument(
        "--zone-label-sigma",
        type=float,
        default=0.0,
        help="Gaussian zone label smoothing in pitch units (yards); 0 = hard cross-entropy.",
    )
    parser.add_argument(
        "--include-target-pass-attributes",
        action="store_true",
        help="Keep the target pass's height/body part in its graph features (leaks the decision).",
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
    # Line-buffered output, so logs written via nohup/redirection update immediately.
    sys.stdout.reconfigure(line_buffering=True)
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
    pitch_grid = PitchGrid(grid_m=args.grid_m, grid_n=args.grid_n)
    dataset_kwargs = {
        "context_length": args.context_length,
        "context_scope": args.context_scope,
        "grid_m": args.grid_m,
        "grid_n": args.grid_n,
        "include_target_pass_attributes": args.include_target_pass_attributes,
    }

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
        train_dataset, skipped_train = _build_multimatch_dataset(train_pairs, args.max_samples, **dataset_kwargs)
        val_dataset, skipped_val = _build_multimatch_dataset(val_pairs, None, **dataset_kwargs)
        split_description = {
            "mode": "separate_match_sets",
            "train_pairs": [pair for pair in train_pairs if pair not in skipped_train],
            "val_pairs": [pair for pair in val_pairs if pair not in skipped_val],
            "skipped_pairs": skipped_train + skipped_val,
        }
    elif args.three_sixty and args.events:
        full_dataset = TemporalReceiverDataset(
            args.three_sixty, args.events, max_samples=args.max_samples, **dataset_kwargs
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
    print(f"Temporal pass target benchmark | task={args.task}")
    print(f"Device: {get_default_device()} | Context: {args.context_length} prior events, {args.context_scope} scope")
    if args.task == "target_zone":
        print(
            f"Grid: {args.grid_m} x {args.grid_n} = {pitch_grid.num_zones} zones "
            f"({pitch_grid.cell_length:.2f} x {pitch_grid.cell_width:.2f} yd) | symmetry={args.symmetry}"
        )
    print(f"Train targets: {len(train_dataset)} ({_sample_fingerprint(train_dataset)})")
    print(f"Validation targets: {len(val_dataset)} ({_sample_fingerprint(val_dataset)})")
    print(f"Models: {', '.join(args.models)}")
    print("=" * 72)

    # A non-shuffled loader defines baseline scores once.  It is the exact
    # current-target graph sequence subsequently used by every temporal model.
    baseline_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_receiver_graphs)
    baseline_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_receiver_graphs)
    print("Naive baselines on this exact split:")
    if args.task == "target_zone":
        print("  (zone priors are fitted on the training split, so their train scores are in-sample)")
        # Zone priors are fitted on the training split only.
        zone_baselines = fit_zone_baselines(baseline_train_loader, pitch_grid)
        tau, sigma, eps = zone_baselines.teammate_kernel.params
        print(f"  (teammate_kernel fitted: tau={tau:g}yd sigma={sigma:g}yd eps={eps:g})")
        baselines = {
            "train": evaluate_zone_baselines(baseline_train_loader, pitch_grid, zone_baselines, args.sample_seed),
            "validation": evaluate_zone_baselines(
                baseline_val_loader, pitch_grid, zone_baselines, args.sample_seed + 1
            ),
        }
        for split_name, split_scores in baselines.items():
            print(f"  {split_name}:")
            for name, metrics in split_scores.items():
                print(f"    {name:>22}: {_format_zone_metrics(metrics)}")
    else:
        baselines = {
            "train": _receiver_baseline_scores(baseline_train_loader, args.sample_seed),
            "validation": _receiver_baseline_scores(baseline_val_loader, args.sample_seed + 1),
        }
        for split_name, scores in baselines.items():
            print(f"  {split_name}: " + ", ".join(f"{name}={score:.2%}" for name, score in scores.items()))

    base_model_kwargs = {
        "in_node_dim": STATSBOMB_NODE_LAYOUT.dim,
        "in_edge_dim": config.data.edge_dim,
        "in_graph_dim": config.data.graph_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "nb_heads": config.model.nb_heads,
        "temporal_layers": args.temporal_layers,
        "temporal_heads": temporal_heads,
        "max_sequence_length": args.context_length + 1,
        "num_fourier_bands": args.num_fourier_bands,
        "dropout": args.dropout,
        "task": args.task,
        "grid_m": args.grid_m,
        "grid_n": args.grid_n,
        "symmetry": args.symmetry,
        "use_mlp": args.use_mlp,
        "sequence_encoder": args.sequence_encoder,
    }
    run_summary: dict[str, object] = {
        "split": split_description,
        "sample_seed": args.sample_seed,
        "context_length": args.context_length,
        "context_scope": args.context_scope,
        "task": args.task,
        "grid": [args.grid_m, args.grid_n],
        "symmetry": args.symmetry,
        "zone_label_sigma": args.zone_label_sigma,
        "include_target_pass_attributes": args.include_target_pass_attributes,
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
            epochs=epochs,
            task=args.task,
            pitch_grid=pitch_grid,
            zone_label_sigma=args.zone_label_sigma,
        )
        history = trainer.fit(epochs)
        final_train = trainer.evaluate(baseline_train_loader)
        final_val = trainer.evaluate(baseline_val_loader)
        if args.task == "target_zone":
            print(f"{model_name} final train: {_format_zone_metrics(final_train)}")
            print(f"{model_name} final val:   {_format_zone_metrics(final_val)}")
        else:
            print(f"{model_name} final train top-1: {final_train.get('top_1_acc', 0.0):.2%}")
            print(f"{model_name} final val top-1:   {final_val.get('top_1_acc', 0.0):.2%}")
        with (checkpoint_dir / "history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        run_summary["models"][model_name] = {
            "checkpoint": str(checkpoint_dir / "best_model.pt"),
            "best_val_loss": trainer.best_val_loss,
            "selection_metric": trainer.selection_metric,
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
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
