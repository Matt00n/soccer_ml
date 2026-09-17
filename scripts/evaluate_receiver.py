"""Evaluation script for receiver prediction task with baseline comparisons."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ml.data.receiver_dataset import (
    ReceiverDataset,
    closest_teammate_baseline,
    collate_receiver_graphs,
    most_frequent_receiver_baseline,
    random_teammate_baseline,
)
from soccer_ml.models.soccer_gat import SoccerGATv2Model
from soccer_ml.training.trainer import get_default_device
from soccer_ml.utils.config import load_config


def topk_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, k: int = 1
) -> float:
    """Compute top-k accuracy."""
    if k == 1:
        correct = (predictions == targets).sum().item()
    else:
        # For top-k, we need confidence scores, not just argmax
        # For now, approximate with argmax (0 or 1)
        correct = (predictions == targets).sum().item()
    return 100.0 * correct / len(targets)


def distance_to_receiver(
    batch,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float]:
    """Compute Euclidean distance between predicted and true receiver positions.

    Returns:
        (mean_distance, median_distance) in normalized pitch coordinates
    """
    B = len(targets)
    distances = []

    for b in range(B):
        pred_idx = predictions[b].item()
        true_idx = targets[b].item()

        pred_pos = batch.node_features[b, pred_idx, :2]  # (x, y)
        true_pos = batch.node_features[b, true_idx, :2]

        dist = torch.norm(pred_pos - true_pos).item()
        distances.append(dist)

    distances = np.array(distances)
    return float(distances.mean()), float(np.median(distances))


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 22,
) -> np.ndarray:
    """Build confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, true in zip(predictions.cpu().numpy(), targets.cpu().numpy()):
        cm[true, pred] += 1
    return cm


def evaluate_receiver_prediction(
    model: Optional[SoccerGATv2Model],
    val_loader: DataLoader,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate model and baselines on receiver prediction task.

    Args:
        model: SoccerGATv2Model (can be None to skip model evaluation)
        val_loader: DataLoader with validation data
        device: torch device
        checkpoint_path: Path to load model weights

    Returns:
        Dict with metrics for model and baselines
    """
    results = {}

    # Load checkpoint if provided
    if model is not None and checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded checkpoint from {checkpoint_path}")

    all_predictions = {
        "model": [],
        "closest_teammate": [],
        "random": [],
        "most_frequent": [],
    }
    all_targets = []
    all_batches = []

    # Collect all predictions
    if model is not None:
        model.eval()

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            all_batches.append(batch)
            all_targets.append(batch.receiver_label.cpu())

            # Model prediction
            if model is not None:
                outputs = model(batch)
                receiver_logits = outputs["receiver_logits"]  # (B, N)
                model_preds = torch.argmax(receiver_logits, dim=1)
                all_predictions["model"].append(model_preds.cpu())

            # Baseline predictions
            closest_preds = closest_teammate_baseline(batch)
            all_predictions["closest_teammate"].append(closest_preds.cpu())

            random_preds = random_teammate_baseline(batch)
            all_predictions["random"].append(random_preds.cpu())

            freq_preds = most_frequent_receiver_baseline(batch)
            all_predictions["most_frequent"].append(freq_preds.cpu())

    # Concatenate all
    all_targets = torch.cat(all_targets, dim=0)

    for key in all_predictions:
        if all_predictions[key]:
            all_predictions[key] = torch.cat(all_predictions[key], dim=0)
        else:
            all_predictions[key] = None

    # Compute metrics
    full_batch = type(all_batches[0])(
        node_features=torch.cat([b.node_features for b in all_batches], dim=0),
        edge_features=torch.cat([b.edge_features for b in all_batches], dim=0),
        graph_features=torch.cat([b.graph_features for b in all_batches], dim=0),
        adj_mat=torch.cat([b.adj_mat for b in all_batches], dim=0),
        receiver_label=all_targets,
        shot_label=None,
        creativity_label=None,
        player_mask=(
            torch.cat([b.player_mask for b in all_batches], dim=0)
            if all_batches[0].player_mask is not None
            else None
        ),
    )

    for method_name, predictions in all_predictions.items():
        if predictions is None:
            continue

        acc1 = topk_accuracy(predictions, all_targets, k=1)
        mean_dist, median_dist = distance_to_receiver(
            full_batch, predictions, all_targets
        )

        cm = confusion_matrix(predictions, all_targets)

        results[method_name] = {
            "top1_accuracy": acc1,
            "mean_distance": mean_dist,
            "median_distance": median_dist,
            "confusion_matrix": cm.tolist(),
            "num_samples": len(all_targets),
        }

        print(f"\n{method_name.upper()}")
        print(f"  Top-1 Accuracy: {acc1:.1f}%")
        print(f"  Mean Distance:  {mean_dist:.3f}")
        print(f"  Median Distance: {median_dist:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Receiver Prediction Task")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--three-sixty",
        type=str,
        required=True,
        help="Path to StatsBomb 360 JSON file",
    )
    parser.add_argument(
        "--events",
        type=str,
        required=True,
        help="Path to StatsBomb events JSON file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to load (for quick testing)",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model evaluation (baselines only)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )

    args = parser.parse_args()

    device = get_default_device()
    config = load_config(args.config)

    print("=" * 60)
    print("Receiver Prediction Evaluation")
    print(f"Device: {device}")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading dataset from:")
    print(f"  360: {args.three_sixty}")
    print(f"  Events: {args.events}")

    dataset = ReceiverDataset(
        three_sixty_path=args.three_sixty,
        events_path=args.events,
        max_samples=args.max_samples,
    )

    print(f"Loaded {len(dataset)} pass frames")

    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_receiver_graphs,
    )

    # Initialize model (or skip if --skip-model)
    model = None
    if not args.skip_model:
        model = SoccerGATv2Model(
            in_node_dim=config.data.node_dim,
            in_edge_dim=config.data.edge_dim,
            in_graph_dim=config.data.graph_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            nb_heads=config.model.nb_heads,
            enable_receiver_head=True,
            enable_shot_head=False,
            enable_creativity_head=False,
            enable_vae_head=False,
        )
        model = model.to(device)

    # Evaluate
    results = evaluate_receiver_prediction(
        model=model,
        val_loader=val_loader,
        device=device,
        checkpoint_path=args.checkpoint,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for key, metrics in results.items():
            results_serializable[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }
        with open(output_path, "w") as f:
            json.dump(results_serializable, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
