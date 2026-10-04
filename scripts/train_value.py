"""Train an action-value model on next-action decisions.

Configurations combine (see ``soccer_ml/training/value_trainer.py``):

    --target        mc | sarsa | expected_sarsa | iql | lambda_return (--lam)
    --head          direct | decomposed
    --distribution  expectation | quantile | categorical (expectation heads + C51)
    --training      warm  (value heads on a copy of a trained next-action
                           encoder; --freeze-encoder keeps it fixed)
                    joint (policy and value heads trained together; from
                           scratch, or initialized with --policy-checkpoint)

Example (warm start from the v2 next-action model):

    python scripts/train_value.py --policy-checkpoint checkpoints/next_action_v2/best_model.pt \\
        --train-dir data/open-data/data --val-dir data/open-data/data \\
        --train-file-count 375 --val-file-count 50 --sample-seed 42 \\
        --target mc --head decomposed --epochs 4 --checkpoint-dir checkpoints/value/mc_decomposed

At the end the best checkpoint is compared with VAEP, VAEP-xG and xT on the
identical validation passes (un-xPass Table 3 protocol), if
``outputs/value_baselines`` exists (``VALUE_BASELINES_README.md``).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ml.data.next_action_dataset import (
    VALUE_LABELS,
    NextActionDataset,
    TransitionDataset,
    collate_transitions,
    lambda_returns,
    load_value_labels,
)
from soccer_ml.data.splits import load_match_datasets, sample_disjoint_train_val_pairs, set_seed
from soccer_ml.models.next_action import NextActionModel
from soccer_ml.models.value import DISTRIBUTIONS, HEADS, ValueModel
from soccer_ml.training.metrics import probabilistic_binary_metrics
from soccer_ml.training.trainer import get_default_device
from soccer_ml.training.value_trainer import (
    TARGETS,
    TRAININGS,
    ValueConfig,
    ValueTrainer,
    validate_configuration,
)
from soccer_ml.value.expected_threat import ExpectedThreat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an action-value model.")
    value = parser.add_argument_group("value model")
    value.add_argument("--target", choices=TARGETS, default="mc")
    value.add_argument("--head", choices=HEADS, default="direct")
    value.add_argument("--distribution", choices=DISTRIBUTIONS, default="expectation")
    value.add_argument("--num-quantiles", type=int, default=5)
    value.add_argument("--training", choices=TRAININGS, default="warm")
    value.add_argument("--policy-checkpoint", help="Trained next-action model (required for warm training).")
    value.add_argument("--freeze-encoder", action="store_true", help="Warm training: train the value heads only.")
    value.add_argument(
        "--encoder-lr-scale", type=float,
        help="Learning-rate multiplier of the encoder (default 0.1 when warm-starting from a checkpoint, else 1).",
    )
    value.add_argument("--mc-label", choices=("xg", "goal"), default="xg")
    value.add_argument("--reward", choices=("xg", "goal"), default="xg")
    value.add_argument("--gamma", type=float, default=0.9, help="Discount per decision (TD and IQL).")
    value.add_argument("--expectile", type=float, default=0.7, help="IQL expectile of the score component.")
    value.add_argument("--lam", type=float, default=0.8, help="λ of lambda_return (1 = discounted Monte Carlo).")
    value.add_argument("--num-atoms", type=int, default=51, help="C51 atoms (categorical).")
    value.add_argument("--v-max", type=float, default=1.0, help="C51 support [-v_max, v_max] of the net return.")
    value.add_argument("--atom-density", type=float, default=5.0, help="C51 sinh spacing: larger = finer near 0.")
    value.add_argument("--distribution-weight", type=float, default=0.1, help="Weight of the (auxiliary) C51 loss.")
    value.add_argument("--target-tau", type=float, default=0.005, help="Polyak rate of the target network.")
    value.add_argument("--value-weight", type=float, default=1.0)
    value.add_argument("--policy-weight", type=float, default=1.0, help="Joint training only.")
    data = parser.add_argument_group("data")
    data.add_argument("--train-dir")
    data.add_argument("--val-dir")
    data.add_argument("--train-file-count", type=int)
    data.add_argument("--val-file-count", type=int)
    data.add_argument("--three-sixty", help="Single-match mode (train and validate on the same match).")
    data.add_argument("--events")
    data.add_argument("--sample-seed", type=int, default=42)
    data.add_argument("--max-samples", type=int)
    data.add_argument("--value-labels", default="outputs/value_baselines/value_labels_360.csv")
    data.add_argument("--baseline-dir", default="outputs/value_baselines")
    model = parser.add_argument_group("encoder (joint training from scratch)")
    model.add_argument("--hidden-dim", type=int, default=128)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--context-length", type=int, default=16)
    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=4)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--device")
    training.add_argument("--checkpoint-dir", default="checkpoints/value/run")
    return parser.parse_args()


def build_policy(args: argparse.Namespace) -> tuple[NextActionModel, dict, dict | None]:
    """The encoder (and policy heads) plus the data configuration it was trained with."""
    if args.policy_checkpoint:
        checkpoint = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=False)
        policy = NextActionModel(**checkpoint["model_kwargs"])
        policy.load_state_dict(checkpoint["model_state_dict"])
        return policy, dict(checkpoint["data_config"]), checkpoint
    if args.training == "warm":
        raise ValueError("--training warm needs --policy-checkpoint.")
    policy = NextActionModel(
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, max_sequence_length=args.context_length + 1
    )
    return policy, {"context_length": args.context_length, "context_scope": "possession", "grid_m": 26,
                    "grid_n": 17, "carry_min_length": None}, None


def compare_with_baselines(predictions: dict, event_ids: list[str], baseline_dir: Path) -> dict:
    """Protocol rows (completed open-play passes) scored by our model, VAEP, VAEP-xG and xT."""
    path = baseline_dir / "vaep_val_predictions.csv"
    if not path.exists():
        return {}
    vaep = {row["original_event_id"]: row for row in csv.DictReader(path.open())}
    protocol = (
        (predictions["valid"] > 0.5)
        & (predictions["action_type"] == 0)
        & (predictions["success"] == 1)
        & (predictions["open_play"] > 0.5)
    ).tolist()
    ours = predictions["v_success"] if "v_success" in predictions else predictions["q"]
    rows = [i for i, keep in enumerate(protocol) if keep and event_ids[i] in vaep]
    if not rows:
        return {}
    labels = predictions["labels"][rows]
    goals, soft = labels[:, VALUE_LABELS.index("scores")], labels[:, VALUE_LABELS.index("xg_scores")]
    candidates = {
        "ours": ours[rows][:, 0],
        "VAEP": torch.tensor([float(vaep[event_ids[i]]["vaep_p_scores"]) for i in rows]),
        "VAEP-xG": torch.tensor([float(vaep[event_ids[i]]["vaep_xg_p_scores"]) for i in rows]),
    }
    xt_path = baseline_dir / "xt_16x12.json"
    if xt_path.exists():
        xt = ExpectedThreat.load(xt_path)
        candidates["xT"] = torch.tensor([xt.value(*predictions["end_location"][i].tolist()) for i in rows])
    return {name: probabilistic_binary_metrics(values, goals, soft) for name, values in candidates.items()}


def main() -> None:
    # Line-buffered output, so logs written via nohup/redirection update immediately.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    config = ValueConfig(
        target=args.target,
        head=args.head,
        distribution=args.distribution,
        num_quantiles=args.num_quantiles,
        training=args.training,
        mc_label=args.mc_label,
        reward=args.reward,
        gamma=args.gamma,
        expectile=args.expectile,
        target_tau=args.target_tau,
        value_weight=args.value_weight,
        policy_weight=args.policy_weight,
        freeze_encoder=args.freeze_encoder,
        lam=args.lam,
        distribution_weight=args.distribution_weight,
        encoder_lr_scale=args.encoder_lr_scale
        if args.encoder_lr_scale is not None
        else (0.1 if args.policy_checkpoint else 1.0),
    )
    validate_configuration(config)
    set_seed(args.sample_seed)
    device = torch.device(args.device) if args.device else get_default_device()

    policy, data_config, checkpoint = build_policy(args)
    started = time.time()
    labels = load_value_labels(args.value_labels) if Path(args.value_labels).exists() else None
    if labels is None and config.target == "mc":
        raise FileNotFoundError(f"Monte Carlo targets need {args.value_labels} (run scripts/value/vaep_baseline.py).")
    factory_kwargs = {**data_config, "value_labels": labels}
    if args.train_dir:
        train_pairs, val_pairs = sample_disjoint_train_val_pairs(
            args.train_dir, args.val_dir, args.train_file_count, args.val_file_count, args.sample_seed
        )
        train_dataset, _ = load_match_datasets(
            train_pairs, lambda t, e: NextActionDataset(t, e, max_samples=args.max_samples, **factory_kwargs)
        )
        val_dataset, _ = load_match_datasets(val_pairs, lambda t, e: NextActionDataset(t, e, **factory_kwargs))
    else:
        train_dataset = val_dataset = NextActionDataset(args.three_sixty, args.events, **factory_kwargs)
    print(f"data: {len(train_dataset)} train / {len(val_dataset)} validation decisions "
          f"({time.time() - started:.0f}s) | config: {config}")

    # Every loader yields (decisions, next decisions, dataset indices).
    train_loader = DataLoader(
        TransitionDataset(train_dataset), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.sample_seed), collate_fn=collate_transitions,
    )
    val_loader = DataLoader(TransitionDataset(val_dataset), batch_size=512, shuffle=False, collate_fn=collate_transitions)
    # Realized discounted returns of the validation decisions: the estimand of TD, λ-return and C51 models.
    val_returns = lambda_returns(val_dataset, config.gamma, 1.0, reward_kind=config.reward)

    model = ValueModel(
        policy,
        head=config.head,
        distribution=config.distribution,
        num_quantiles=config.num_quantiles,
        state_value=config.target == "iql",
        joint=config.training == "joint",
        num_atoms=args.num_atoms,
        v_max=args.v_max,
        atom_density=args.atom_density,
    )
    if config.training == "warm":
        # Policy heads are unused; the encoder is fine-tuned unless frozen.
        frozen = [model.policy] if config.freeze_encoder else [
            model.policy.type_head, model.policy.zone_head, model.policy.height_head
        ]
        for module in frozen:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    frozen_policy = None
    if config.target == "expected_sarsa" and config.training == "warm":
        frozen_policy = NextActionModel(**checkpoint["model_kwargs"])
        frozen_policy.load_state_dict(checkpoint["model_state_dict"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"value model: {trainable:,} trainable parameters")

    payload = {
        "policy_kwargs": policy.model_kwargs,
        "value_kwargs": model.value_kwargs,
        "data_config": data_config,
        "policy_checkpoint": args.policy_checkpoint,
    }
    fit_started = time.time()
    trainer = ValueTrainer(
        model, config, train_loader, val_loader, args.lr, args.weight_decay, args.epochs,
        args.checkpoint_dir, payload, policy=frozen_policy, device=device,
        train_dataset=train_dataset, val_returns=val_returns,
    )
    history = trainer.fit(args.epochs)

    best = torch.load(Path(args.checkpoint_dir) / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    metrics, predictions = trainer.evaluate()
    event_ids = [val_dataset[i].metadata["event_uuid"] for i in range(len(val_dataset))]
    comparison = compare_with_baselines(predictions, event_ids, Path(args.baseline_dir))
    print(f"\nBest epoch {best['epoch']}: " + ", ".join(
        f"{key}={value:.4f}" for key, value in metrics.items() if not key.startswith("quantile")
    ))
    if comparison:
        print(f"\nProtocol comparison on identical passes (n={int(next(iter(comparison.values()))['n'])}):")
        print(f"{'model':<10} {'AUC':>6} {'log-loss':>9} {'xG log-loss':>12}")
        for name, result in sorted(comparison.items(), key=lambda item: -item[1]["auc"]):
            print(f"{name:<10} {result['auc']:6.3f} {result['logloss']:9.4f} {result['soft_logloss']:12.4f}")

    output = Path(args.checkpoint_dir)
    columns = []  # (prediction name, component index or None)
    for name in ("q", "p_success", "v_success", "state_value", "distribution_mean", "distribution_upside"):
        if name in predictions:
            columns += [(name, 0), (name, 1)] if predictions[name].dim() == 2 else [(name, None)]
    with (output / "val_predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["event_uuid"] + [name if c is None else f"{name}_{('score', 'concede')[c]}" for name, c in columns])
        for i, event_id in enumerate(event_ids):
            writer.writerow([event_id] + [
                f"{(predictions[name][i] if c is None else predictions[name][i, c]).item():.6f}" for name, c in columns
            ])
    (output / "summary.json").write_text(json.dumps({
        "config": config.__dict__, "args": vars(args), "best_epoch": best["epoch"], "metrics": metrics,
        "comparison": comparison, "train_size": len(train_dataset), "val_size": len(val_dataset),
        "train_seconds": time.time() - fit_started,
    }, indent=2))
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
