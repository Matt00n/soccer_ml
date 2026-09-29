"""Evaluate a saved temporal receiver checkpoint with the existing baselines."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

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
from soccer_ml.training.trainer import get_default_device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _distance_metrics(
    positions: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor
) -> tuple[float, float]:
    rows = torch.arange(len(targets), device=positions.device)
    distances = torch.norm(positions[rows, predictions] - positions[rows, targets], dim=-1)
    return float(distances.mean().item()), float(distances.median().item())


def evaluate(
    model: torch.nn.Module | None,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float | int]]:
    """Evaluate model and three original naive baselines on one target set."""
    names = ["closest_teammate", "random", "most_frequent"]
    if model is not None:
        model.eval()
        names.insert(0, "model")
    correct_top1 = {name: 0 for name in names}
    correct_top3 = {name: 0 for name in names}
    distances: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    total = 0

    with torch.no_grad():
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
            for name, prediction in predictions.items():
                correct_top1[name] += int((prediction == target).sum().item())
                # Naive baselines expose a single hard prediction, so their
                # top-3 is intentionally not reported as a pseudo-confidence.
                positions = temporal_batch.node_features[:, -1, :, :2]
                rows = torch.arange(len(target), device=device)
                distances[name].append(torch.norm(positions[rows, prediction] - positions[rows, target], dim=-1).cpu())
            total += temporal_batch.batch_size

    results: dict[str, dict[str, float | int]] = {}
    for name in names:
        values = torch.cat(distances[name]) if distances[name] else torch.empty(0)
        results[name] = {
            "top1_accuracy": correct_top1[name] / max(total, 1),
            "mean_distance": float(values.mean().item()) if len(values) else 0.0,
            "median_distance": float(values.median().item()) if len(values) else 0.0,
            "num_samples": total,
        }
        if name == "model":
            results[name]["top3_accuracy"] = correct_top3[name] / max(total, 1)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a temporal receiver checkpoint.")
    parser.add_argument("--three-sixty", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--checkpoint", help="best_model.pt from train_temporal_receiver.py")
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--context-scope", choices=("possession", "period", "match"), default="possession")
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
    _set_seed(args.seed)
    device = get_default_device()
    dataset = TemporalReceiverDataset(
        args.three_sixty,
        args.events,
        context_length=args.context_length,
        context_scope=args.context_scope,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_temporal_receiver_graphs)

    model = None
    checkpoint: dict[str, Any] | None = None
    if not args.skip_model:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model_name" not in checkpoint or "model_kwargs" not in checkpoint:
            raise ValueError("Checkpoint is not a self-describing temporal receiver checkpoint.")
        model = build_temporal_receiver_model(checkpoint["model_name"], **checkpoint["model_kwargs"])
        expected_sequence_length = checkpoint["model_kwargs"].get("max_sequence_length")
        if expected_sequence_length is not None and expected_sequence_length != args.context_length + 1:
            raise ValueError(
                "Checkpoint expects a sequence length of "
                f"{expected_sequence_length}; pass --context-length {expected_sequence_length - 1}."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

    print("=" * 72)
    print(f"Temporal receiver evaluation on {len(dataset)} pass targets | device={device}")
    print(f"Context: {args.context_length} prior events, {args.context_scope} scope")
    results = evaluate(model, loader, device)
    for name, metrics in results.items():
        suffix = f" | top-3={metrics['top3_accuracy']:.2%}" if "top3_accuracy" in metrics else ""
        print(
            f"{name:>17}: top-1={metrics['top1_accuracy']:.2%}{suffix} "
            f"| mean distance={metrics['mean_distance']:.3f} "
            f"| median distance={metrics['median_distance']:.3f}"
        )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
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
