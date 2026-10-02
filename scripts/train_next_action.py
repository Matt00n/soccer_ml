"""Train the next-action model (action type -> end zone -> pass height).

Samples are on-ball decisions with a StatsBomb 360 freeze frame: passes
(completed and failed), carries of at least one grid cell, shots, dribbles and
clearances.  See ``NEXT_ACTION_README.md``.

Example (all matches; the same 50 validation matches as the pass benchmark):

    python scripts/train_next_action.py \\
        --train-dir data/open-data/data --val-dir data/open-data/data \\
        --train-file-count 375 --val-file-count 50 --sample-seed 42 --epochs 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ml.data.next_action_dataset import ACTION_TYPES, DECISION_CONTEXTS, NextActionDataset, collate_next_action
from soccer_ml.data.pitch_grid import PitchGrid
from soccer_ml.data.splits import load_match_datasets, sample_disjoint_train_val_pairs, sample_fingerprint, set_seed
from soccer_ml.data.transforms import SYMMETRY_GROUPS
from soccer_ml.models.next_action import NextActionModel
from soccer_ml.training.next_action_trainer import (
    NextActionTrainer,
    evaluate_next_action_baselines,
    fit_next_action_priors,
)
from soccer_ml.training.trainer import get_default_device


def format_metrics(metrics: dict[str, float]) -> str:
    def get(key: str, percent: bool = False) -> str:
        if key not in metrics:
            return "-"
        return f"{metrics[key]:.1%}" if percent else f"{metrics[key]:.3f}"

    recalls = " ".join(f"{name}={get(f'recall_{name}', True)}" for name in ACTION_TYPES)
    contexts = " | ".join(
        f"{name} ({get(f'share_{name}', True)}): acc {get(f'type_acc_{name}', True)} nll {get(f'type_nll_{name}')}"
        for name in DECISION_CONTEXTS
    )
    return (
        f"joint nll={get('joint_nll')} [type {get('type_nll')} | pass zone {get('pass_nll')} | "
        f"carry zone {get('carry_nll')} | height {get('height_nll')}]\n"
        f"      type acc={get('type_acc', True)} macro recall={get('type_macro_recall', True)} ({recalls})\n"
        f"      by decision context: {contexts}\n"
        f"      pass top-1/3={get('pass_top_1_acc', True)}/{get('pass_top_3_acc', True)} "
        f"within-1={get('pass_within_1_zone_acc', True)} median dist={get('pass_median_dist_yd')}yd "
        f"receiver={get('pass_receiver_acc', True)} | carry top-1={get('carry_top_1_acc', True)} "
        f"median dist={get('carry_median_dist_yd')}yd | height acc={get('height_acc', True)} "
        f"lofted acc={get('lofted_acc', True)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the next-action model.")
    data = parser.add_argument_group("data")
    data.add_argument("--train-dir")
    data.add_argument("--val-dir")
    data.add_argument("--train-file-count", type=int)
    data.add_argument("--val-file-count", type=int)
    data.add_argument("--three-sixty", help="Single freeze-frame JSON file (single-match mode).")
    data.add_argument("--events", help="Single event JSON file (single-match mode).")
    data.add_argument("--val-split", type=float, default=0.2, help="Single-match mode only.")
    data.add_argument("--sample-seed", type=int, default=42)
    data.add_argument("--max-samples", type=int, help="Cap samples per training match (quick runs).")
    data.add_argument("--context-length", type=int, default=16, help="Previous events of the possession.")
    data.add_argument("--context-scope", choices=("possession", "period"), default="possession")
    data.add_argument("--carry-min-length", type=float, help="Shorter carries are not targets (default: one grid cell).")
    data.add_argument("--grid-m", type=int, default=26)
    data.add_argument("--grid-n", type=int, default=17)
    model = parser.add_argument_group("model")
    model.add_argument("--hidden-dim", type=int, default=128)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--nb-heads", type=int, default=8)
    model.add_argument("--temporal-layers", type=int, default=2)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--symmetry", choices=tuple(SYMMETRY_GROUPS), default="y_flip")
    model.add_argument("--use-mlp", action="store_true", help="Two-layer MLP heads instead of linear heads.")
    model.add_argument("--no-history", action="store_true", help="Ablation: drop the event-history encoder.")
    model.add_argument("--no-visibility", action="store_true", help="Ablation: one zone map regardless of visibility.")
    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=16)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--device", help="cpu, mps or cuda (default: best available).")
    training.add_argument("--final-train-eval", action="store_true", help="Also re-evaluate the full training set.")
    training.add_argument("--checkpoint-dir", default="checkpoints/next_action")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.sample_seed)
    device = torch.device(args.device) if args.device else get_default_device()
    pitch_grid = PitchGrid(args.grid_m, args.grid_n)
    data_config = {
        "context_length": args.context_length,
        "context_scope": args.context_scope,
        "grid_m": args.grid_m,
        "grid_n": args.grid_n,
        "carry_min_length": args.carry_min_length,
    }

    started = time.time()
    if args.train_dir or args.val_dir:
        if not (args.train_dir and args.val_dir):
            raise ValueError("Provide both --train-dir and --val-dir together.")
        train_pairs, val_pairs = sample_disjoint_train_val_pairs(
            args.train_dir, args.val_dir, args.train_file_count, args.val_file_count, args.sample_seed
        )
        train_dataset, skipped_train = load_match_datasets(
            train_pairs, lambda t, e: NextActionDataset(t, e, max_samples=args.max_samples, **data_config)
        )
        val_dataset, skipped_val = load_match_datasets(val_pairs, lambda t, e: NextActionDataset(t, e, **data_config))
        split = {
            "mode": "separate_match_sets",
            "train_pairs": [pair for pair in train_pairs if pair not in skipped_train],
            "val_pairs": [pair for pair in val_pairs if pair not in skipped_val],
            "skipped_pairs": skipped_train + skipped_val,
        }
    elif args.three_sixty and args.events:
        full = NextActionDataset(args.three_sixty, args.events, max_samples=args.max_samples, **data_config)
        n_val = int(args.val_split * len(full))
        train_dataset, val_dataset = random_split(
            full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(args.sample_seed)
        )
        split = {"mode": "single_match_random_split", "three_sixty": args.three_sixty, "events": args.events}
    else:
        raise ValueError("Provide (--train-dir and --val-dir) or (--three-sixty and --events).")

    train_samples = [train_dataset[i] for i in range(len(train_dataset))]
    train_counts = torch.bincount(torch.stack([s.action_type for s in train_samples]), minlength=len(ACTION_TYPES))
    context_counts = torch.bincount(
        torch.stack([s.decision_context for s in train_samples]), minlength=len(DECISION_CONTEXTS)
    )
    del train_samples
    print("=" * 96)
    print(f"Next-action prediction | device={device} | data loaded in {time.time() - started:.0f}s")
    print(f"Train: {len(train_dataset)} actions ({sample_fingerprint(train_dataset)}) | "
          f"validation: {len(val_dataset)} actions ({sample_fingerprint(val_dataset)})")
    print("Train action mix: " + ", ".join(
        f"{name} {count / max(len(train_dataset), 1):.1%}" for name, count in zip(ACTION_TYPES, train_counts.tolist())
    ))
    print("Decision contexts: " + ", ".join(
        f"{name} {count / max(len(train_dataset), 1):.1%}" for name, count in zip(DECISION_CONTEXTS, context_counts.tolist())
    ))
    print("=" * 96)

    baseline_train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, collate_fn=collate_next_action)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, collate_fn=collate_next_action)
    priors = fit_next_action_priors(baseline_train_loader, pitch_grid)
    baselines = evaluate_next_action_baselines(val_loader, pitch_grid, priors)
    print("Naive baselines on validation (fitted on train):")
    for name, metrics in baselines.items():
        print(f"  {name}: {format_metrics(metrics)}")

    model = NextActionModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nb_heads=args.nb_heads,
        temporal_layers=args.temporal_layers,
        max_sequence_length=args.context_length + 1,
        dropout=args.dropout,
        grid_m=args.grid_m,
        grid_n=args.grid_n,
        symmetry=args.symmetry,
        use_mlp=args.use_mlp,
        use_history=not args.no_history,
        use_visibility=not args.no_visibility,
    )
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"\nModel: {num_parameters:,} parameters | {model.model_kwargs}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.sample_seed),
        collate_fn=collate_next_action,
    )
    trainer = NextActionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        pitch_grid=pitch_grid,
        data_config=data_config,
        device=device,
    )
    history = trainer.fit(args.epochs)

    best = torch.load(Path(args.checkpoint_dir) / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    final_val = trainer.evaluate(val_loader)
    print(f"\nBest checkpoint (epoch {best['epoch']}) on validation:\n  model: {format_metrics(final_val)}")
    for name, metrics in baselines.items():
        print(f"  {name}: {format_metrics(metrics)}")
    final_train = trainer.evaluate(baseline_train_loader) if args.final_train_eval else {}
    if final_train:
        print(f"  model on train: {format_metrics(final_train)}")

    output_dir = Path(args.checkpoint_dir)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "split": split,
        "sample_seed": args.sample_seed,
        "data_config": data_config,
        "model_kwargs": model.model_kwargs,
        "num_parameters": num_parameters,
        "args": vars(args),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "train_fingerprint": sample_fingerprint(train_dataset),
        "val_fingerprint": sample_fingerprint(val_dataset),
        "train_action_counts": dict(zip(ACTION_TYPES, train_counts.tolist())),
        "train_context_counts": dict(zip(DECISION_CONTEXTS, context_counts.tolist())),
        "best_epoch": best["epoch"],
        "final_validation": final_val,
        "final_train": final_train,
        "baselines_validation": baselines,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved checkpoints, history.json and summary.json to {output_dir}")


if __name__ == "__main__":
    main()
