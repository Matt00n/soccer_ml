"""Evaluation metrics for soccer tasks: Top-k accuracy, F1, precision, recall, and MSE."""

from typing import Dict, Any, List
import torch


def compute_topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k_list: List[int] = (1, 3),
) -> Dict[str, float]:
    """Computes Top-k accuracy for node classification (e.g. receiver prediction).

    Args:
        logits: (B, N)
        targets: (B,)
        k_list: Tuple of k values (e.g. (1, 3))

    Returns:
        Dictionary of top-k accuracy percentages (0.0 to 1.0)
    """
    with torch.no_grad():
        max_k = min(max(k_list), logits.shape[-1])
        _, top_indices = torch.topk(logits, k=max_k, dim=-1, largest=True, sorted=True)  # (B, max_k)

        targets_expanded = targets.view(-1, 1).expand_as(top_indices)
        correct = (top_indices == targets_expanded)

        metrics = {}
        for k in k_list:
            if k <= max_k:
                k_correct = correct[:, :k].any(dim=-1).float().mean().item()
                metrics[f"top_{k}_acc"] = k_correct

        return metrics


def compute_binary_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes binary classification metrics for graph predictions (e.g. shot likelihood).

    Args:
        logits: (B, 1) or (B,)
        targets: (B, 1) or (B,)
        threshold: Decision threshold for sigmoid probability

    Returns:
        Dictionary with 'accuracy', 'precision', 'recall', 'f1'
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits).view(-1)
        labels = targets.view(-1).float()

        preds = (probs >= threshold).float()

        tp = ((preds == 1.0) & (labels == 1.0)).sum().item()
        fp = ((preds == 1.0) & (labels == 0.0)).sum().item()
        fn = ((preds == 0.0) & (labels == 1.0)).sum().item()
        tn = ((preds == 0.0) & (labels == 0.0)).sum().item()

        total = tp + fp + fn + tn
        accuracy = (tp + tn) / max(total, 1)
        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = (
            (2.0 * precision * recall) / max(precision + recall, 1e-8)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Computes MSE and MAE for continuous player creativity ratings."""
    with torch.no_grad():
        diff = predictions.view(-1) - targets.view(-1)
        mse = (diff ** 2).mean().item()
        mae = diff.abs().mean().item()
        return {"mse": mse, "mae": mae}


class MetricTracker:
    """Accumulates batch metrics over an evaluation or training epoch."""

    def __init__(self):
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metric_dict: Dict[str, float], batch_size: int = 1):
        for k, v in metric_dict.items():
            self.totals[k] = self.totals.get(k, 0.0) + v * batch_size
            self.counts[k] = self.counts.get(k, 0) + batch_size

    def compute(self) -> Dict[str, float]:
        return {k: self.totals[k] / max(self.counts[k], 1) for k in self.totals}

    def reset(self):
        self.totals.clear()
        self.counts.clear()
