"""Evaluate a saved temporal checkpoint against the naive baselines.

The task (pass target zone or receiver), grid, and model configuration are
read from the self-describing checkpoint.  For the target-zone task the
zone-prior baselines need training data: pass ``--prior-three-sixty`` and
``--prior-events`` to fit them on a training match, otherwise they are fitted
on the evaluated data itself and reported as ``*_in_sample`` (optimistic).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ml.data.pitch_grid import PitchGrid, receiver_candidates
from soccer_ml.data.receiver_dataset import (
    closest_teammate_baseline,
    most_frequent_receiver_baseline,
    random_teammate_baseline,
)
from soccer_ml.data.temporal_receiver_dataset import (
    TemporalReceiverDataset,
    collate_temporal_receiver_graphs,
)
from soccer_ml.models.temporal_receiver import TASKS, build_temporal_receiver_model
from soccer_ml.training.metrics import ZoneMetricAccumulator
from soccer_ml.training.temporal_receiver_trainer import (
    FITTED_ZONE_BASELINES,
    evaluate_zone_baselines,
    fit_zone_baselines,
)
from soccer_ml.training.trainer import get_default_device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_target_zone(
    model: torch.nn.Module | None,
    loader: DataLoader,
    prior_loader: DataLoader,
    pitch_grid: PitchGrid,
    device: torch.device,
    seed: int,
    prior_suffix: str = "",
) -> dict[str, dict[str, float]]:
    """Evaluate a zone model and the zone baselines on one target set."""
    baselines = evaluate_zone_baselines(loader, pitch_grid, fit_zone_baselines(prior_loader, pitch_grid), seed)
    results: dict[str, dict[str, float]] = {}
    if model is not None:
        model.eval()
        accumulator = ZoneMetricAccumulator(pitch_grid)
        for temporal_batch in loader:
            temporal_batch = temporal_batch.to(device)
            log_probs = F.log_softmax(model(temporal_batch)["target_zone_logits"], dim=-1)
            accumulator.update(
                log_probs,
                temporal_batch.target_zone_label,
                temporal_batch.pass_end_location,
                log_probs=log_probs,
                receivers=receiver_candidates(temporal_batch.current_graph_batch(), pitch_grid),
            )
        results["model"] = accumulator.compute()
    for name, metrics in baselines.items():
        results[f"{name}{prior_suffix if name in FITTED_ZONE_BASELINES else ''}"] = metrics
    return results


@torch.no_grad()
def evaluate_receiver(
    model: torch.nn.Module | None,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    """Evaluate a receiver model and the three original naive baselines."""
    names = ["closest_teammate", "random", "most_frequent"]
    if model is not None:
        model.eval()
        names.insert(0, "model")
    correct_top1 = {name: 0 for name in names}
    correct_top3 = {name: 0 for name in names}
    distances: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    total = 0

    for temporal_batch in loader:
        temporal_batch = temporal_batch.to(device)
        current_batch = temporal_batch.current_graph_batch()
        predictions: dict[str, torch.Tensor] = {
            "closest_teammate": closest_teammate_baseline(current_batch),
            "random": random_teammate_baseline(current_batch),
            "most_frequent": most_frequent_receiver_baseline(current_batch),
        }
        if model is not None:
            logits = model(temporal_batch)["receiver_logits"]
            predictions["model"] = logits.argmax(dim=-1)
            top_indices = logits.topk(k=min(3, logits.shape[-1]), dim=-1).indices
            correct_top3["model"] += int(
                (top_indices == temporal_batch.receiver_label.unsqueeze(-1)).any(dim=-1).sum().item()
            )

        target = temporal_batch.receiver_label
        positions = temporal_batch.node_features[:, -1, :, :2]
        rows = torch.arange(len(target), device=device)
        for name, prediction in predictions.items():
            correct_top1[name] += int((prediction == target).sum().item())
            # Naive baselines expose a single hard prediction, so their
            # top-3 is intentionally not reported as a pseudo-confidence.
            distances[name].append(torch.norm(positions[rows, prediction] - positions[rows, target], dim=-1).cpu())
        total += temporal_batch.batch_size

    results: dict[str, dict[str, float | int]] = {}
    for name in names:
        values = torch.cat(distances[name]) if distances[name] else torch.empty(0)
        results[name] = {
            "top_1_acc": correct_top1[name] / max(total, 1),
            "mean_distance": float(values.mean().item()) if len(values) else 0.0,
            "median_distance": float(values.median().item()) if len(values) else 0.0,
            "num_samples": total,
        }
        if name == "model":
            results[name]["top_3_acc"] = correct_top3[name] / max(total, 1)
    return results


def _format(metrics: dict[str, Any]) -> str:
    if "within_1_zone_acc" in metrics:
        nll = f"nll={metrics['nll']:.3f}" if "nll" in metrics else "nll=  -  "
        text = (
            f"{nll} | top-1={metrics['top_1_acc']:.2%} top-3={metrics['top_3_acc']:.2%} "
            f"top-5={metrics['top_5_acc']:.2%} within-1={metrics['within_1_zone_acc']:.2%} "
            f"| dist mean/median={metrics['mean_dist_yd']:.1f}/{metrics['median_dist_yd']:.1f}yd "
            f"({metrics['mean_dist_m']:.1f}/{metrics['median_dist_m']:.1f}m)"
        )
        return text + (f" | receiver-acc={metrics['receiver_acc']:.2%}" if "receiver_acc" in metrics else "")
    suffix = f" | top-3={metrics['top_3_acc']:.2%}" if "top_3_acc" in metrics else ""
    return (
        f"top-1={metrics['top_1_acc']:.2%}{suffix} "
        f"| mean distance={metrics['mean_distance']:.3f} "
        f"| median distance={metrics['median_distance']:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a temporal pass target checkpoint.")
    parser.add_argument("--three-sixty", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--checkpoint", help="best_model.pt from train_temporal_receiver.py")
    parser.add_argument("--prior-three-sixty", help="Training freeze frames used to fit the zone priors.")
    parser.add_argument("--prior-events", help="Training events used to fit the zone priors.")
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--context-scope", choices=("possession", "period", "match"), default="possession")
    parser.add_argument("--task", choices=TASKS, default="target_zone", help="Only used with --skip-model.")
    parser.add_argument("--grid-m", type=int, default=26, help="Only used with --skip-model.")
    parser.add_argument("--grid-n", type=int, default=17, help="Only used with --skip-model.")
    parser.add_argument("--include-target-pass-attributes", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_model and not args.checkpoint:
        raise ValueError("Provide --checkpoint or use --skip-model for baseline-only evaluation.")
    if bool(args.prior_three_sixty) != bool(args.prior_events):
        raise ValueError("Provide both --prior-three-sixty and --prior-events, or neither.")
    _set_seed(args.seed)
    device = get_default_device()

    model = None
    task, grid_m, grid_n = args.task, args.grid_m, args.grid_n
    if not args.skip_model:
        checkpoint: dict[str, Any] = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model_name" not in checkpoint or "model_kwargs" not in checkpoint:
            raise ValueError("Checkpoint is not a self-describing temporal checkpoint.")
        model_kwargs = checkpoint["model_kwargs"]
        model = build_temporal_receiver_model(checkpoint["model_name"], **model_kwargs)
        expected_sequence_length = model_kwargs.get("max_sequence_length")
        if expected_sequence_length is not None and expected_sequence_length != args.context_length + 1:
            raise ValueError(
                "Checkpoint expects a sequence length of "
                f"{expected_sequence_length}; pass --context-length {expected_sequence_length - 1}."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        task = model_kwargs.get("task", "receiver")
        grid_m, grid_n = model_kwargs.get("grid_m", grid_m), model_kwargs.get("grid_n", grid_n)

    dataset_kwargs = {
        "context_length": args.context_length,
        "context_scope": args.context_scope,
        "grid_m": grid_m,
        "grid_n": grid_n,
        "include_target_pass_attributes": args.include_target_pass_attributes,
    }
    dataset = TemporalReceiverDataset(args.three_sixty, args.events, max_samples=args.max_samples, **dataset_kwargs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_temporal_receiver_graphs)

    print("=" * 72)
    print(f"Temporal {task} evaluation on {len(dataset)} pass targets | device={device}")
    print(f"Context: {args.context_length} prior events, {args.context_scope} scope")
    if task == "target_zone":
        pitch_grid = PitchGrid(grid_m=grid_m, grid_n=grid_n)
        print(f"Grid: {grid_m} x {grid_n} = {pitch_grid.num_zones} zones")
        if args.prior_three_sixty:
            prior_dataset = TemporalReceiverDataset(args.prior_three_sixty, args.prior_events, **dataset_kwargs)
            prior_suffix = ""
        else:
            prior_dataset, prior_suffix = dataset, "_in_sample"
        prior_loader = DataLoader(
            prior_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_temporal_receiver_graphs
        )
        results: dict[str, Any] = evaluate_target_zone(
            model, loader, prior_loader, pitch_grid, device, args.seed, prior_suffix
        )
    else:
        results = evaluate_receiver(model, loader, device)
    for name, metrics in results.items():
        print(f"{name:>32}: {_format(metrics)}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "task": task,
                    "grid": [grid_m, grid_n],
                    "context_length": args.context_length,
                    "context_scope": args.context_scope,
                    "results": results,
                },
                file,
                indent=2,
            )
        print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
