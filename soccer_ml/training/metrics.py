"""Evaluation metrics for soccer tasks: Top-k accuracy, F1, precision, recall, and MSE."""

from typing import Dict, Any, List, Optional
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


class ZoneMetricAccumulator:
    """Accumulates pass-target-zone metrics over an epoch.

    Works for any (B, M * N) score tensor, so learned models and ranking
    baselines are evaluated identically:

    * ``nll``: mean negative log-likelihood of the true zone (the headline
      metric: a proper scoring rule, and per pass exactly the "surprise").
      Only reported when ``log_probs`` are passed.
    * ``top_k_acc``: true zone among the k highest scores.
    * ``within_1_zone_acc``: top-1 zone is the true zone or one of its 8 neighbours.
    * ``mean/median_dist_yd``: distance from the top-1 zone center to the
      true pass end location in StatsBomb pitch units (yards).
    * ``mean/median_dist_m``: the same distance after rescaling the pitch to
      105 x 68 m.
    * ``receiver_acc``: un-xPass-style receiver accuracy.  The predicted
      receiver is the visible teammate closest to the top-1 zone center, the
      true receiver the visible teammate closest to the true end location.
      Only reported when ``receivers`` are passed; passes without a visible
      candidate teammate are skipped.
    """

    def __init__(self, pitch_grid, k_list=(1, 3, 5)):
        self.pitch_grid = pitch_grid
        self.k_list = tuple(k_list)
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.topk_correct = {k: 0 for k in self.k_list}
        self.neighbour_correct = 0
        self.nll_sum = 0.0
        self.nll_count = 0
        self.receiver_correct = 0
        self.receiver_count = 0
        self.distances_yd: List[torch.Tensor] = []
        self.distances_m: List[torch.Tensor] = []

    @torch.no_grad()
    def update(
        self,
        scores: torch.Tensor,
        target_zone: torch.Tensor,
        end_location: torch.Tensor,
        log_probs: Optional[torch.Tensor] = None,
        receivers: Optional[Any] = None,
    ) -> None:
        """Add a batch.

        Args:
            scores: (B, Z) ranking scores (log-probabilities for models).
            target_zone: (B,) true zones.
            end_location: (B, 2) true end locations in pitch units.
            log_probs: Optional (B, Z) normalized log-probabilities for NLL.
            receivers: Optional :class:`~soccer_ml.data.pitch_grid.ReceiverCandidates`.
        """
        scores = scores.detach().float()
        target_zone = target_zone.to(scores.device)
        end_location = end_location.to(scores.device).float()
        max_k = min(max(self.k_list), scores.shape[-1])
        top_indices = scores.topk(max_k, dim=-1).indices
        hits = top_indices == target_zone.unsqueeze(-1)
        for k in self.k_list:
            self.topk_correct[k] += int(hits[:, :k].any(dim=-1).sum().item())

        prediction = top_indices[:, 0]
        pred_x, pred_y = self.pitch_grid.zone_bins(prediction)
        true_x, true_y = self.pitch_grid.zone_bins(target_zone)
        neighbour = ((pred_x - true_x).abs() <= 1) & ((pred_y - true_y).abs() <= 1)
        self.neighbour_correct += int(neighbour.sum().item())

        self.distances_yd.append(self.pitch_grid.compute_distance(prediction, end_location).cpu())
        self.distances_m.append(self.pitch_grid.compute_distance(prediction, end_location, metric=True).cpu())
        if log_probs is not None:
            log_probs = log_probs.detach().float().to(scores.device)
            self.nll_sum += float(-log_probs.gather(1, target_zone.unsqueeze(-1)).sum().item())
            self.nll_count += len(target_zone)
        if receivers is not None:
            predicted_receiver = receivers.nearest(self.pitch_grid.batch_zones_to_coords(prediction))
            valid = receivers.has_candidate
            self.receiver_correct += int(((predicted_receiver == receivers.receiver) & valid).sum().item())
            self.receiver_count += int(valid.sum().item())
        self.count += len(target_zone)

    def compute(self) -> Dict[str, float]:
        total = max(self.count, 1)
        metrics: Dict[str, float] = {}
        if self.nll_count:
            metrics["nll"] = self.nll_sum / self.nll_count
        metrics.update({f"top_{k}_acc": correct / total for k, correct in self.topk_correct.items()})
        metrics["within_1_zone_acc"] = self.neighbour_correct / total
        for unit, values in (("yd", self.distances_yd), ("m", self.distances_m)):
            distances = torch.cat(values) if values else torch.zeros(1)
            metrics[f"mean_dist_{unit}"] = float(distances.mean().item())
            metrics[f"median_dist_{unit}"] = float(distances.median().item())
        if self.receiver_count:
            metrics["receiver_acc"] = self.receiver_correct / self.receiver_count
        return metrics


def probabilistic_binary_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    soft_labels: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """AUC (ties get average ranks), Brier score and log-loss of probabilities.

    With ``soft_labels`` (e.g. xG-sequence targets) also reports the
    cross-entropy against them, a lower-variance comparison than goals.
    """
    p = probabilities.double().clamp(eps, 1 - eps)
    y = labels.double()
    order = torch.argsort(p)
    sorted_p = p[order]
    ranks = torch.empty_like(p)
    _, inverse, counts = torch.unique_consecutive(sorted_p, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, dim=0).double()
    average_rank = ends - (counts.double() - 1) / 2  # 1-based average rank of each tie group
    ranks[order] = average_rank[inverse]
    positives, negatives = y.sum(), (1 - y).sum()
    auc = (ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    metrics = {
        "n": float(len(y)),
        "positives": float(positives),
        "auc": float(auc),
        "brier": float(((p - y) ** 2).mean()),
        "logloss": float(-(y * p.log() + (1 - y) * (1 - p).log()).mean()),
    }
    if soft_labels is not None:
        s = soft_labels.double()
        metrics["soft_logloss"] = float(-(s * p.log() + (1 - s) * (1 - p).log()).mean())
    return metrics
