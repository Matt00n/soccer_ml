"""Training and evaluation utilities for temporal pass-target benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from soccer_ml.data.pitch_grid import (
    PitchGrid,
    ZonePriorBaseline,
    closest_teammate_zone_scores,
    passer_zones,
    random_zone_scores,
)
from soccer_ml.training.metrics import MetricTracker, ZoneMetricAccumulator, compute_topk_accuracy
from soccer_ml.training.trainer import get_default_device


class TemporalReceiverTrainer:
    """Small, task-specific trainer with portable temporal-model checkpoints.

    ``task="target_zone"`` trains on the pitch-grid zone of the pass end
    location.  With ``zone_label_sigma > 0`` the one-hot target is replaced by
    a Gaussian over zone centers (sigma in pitch units); reported ``nll`` is
    always against the hard zone label, so runs stay comparable, and it is
    the checkpoint-selection criterion.  ``task="receiver"`` reproduces the
    earlier receiver benchmark.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        model_kwargs: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        lr: float,
        weight_decay: float,
        checkpoint_dir: str | Path,
        epochs: int,
        task: str = "target_zone",
        pitch_grid: Optional[PitchGrid] = None,
        zone_label_sigma: float = 0.0,
        device: Optional[torch.device] = None,
        clip_grad_norm: Optional[float] = 1.0,
    ):
        self.device = device or get_default_device()
        self.model = model.to(self.device)
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task = task
        self.pitch_grid = pitch_grid or PitchGrid(
            grid_m=model_kwargs.get("grid_m", 26), grid_n=model_kwargs.get("grid_n", 17)
        )
        self.zone_label_sigma = zone_label_sigma
        self.clip_grad_norm = clip_grad_norm
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(epochs, 1), eta_min=1e-6
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.selection_metric = "nll" if task == "target_zone" else "loss"

    def _loss(self, outputs: Dict[str, Any], batch: Any) -> torch.Tensor:
        if self.task == "receiver":
            return F.cross_entropy(outputs["receiver_logits"], batch.receiver_label)
        logits = outputs["target_zone_logits"]
        if self.zone_label_sigma > 0:
            soft_targets = self.pitch_grid.gaussian_soft_targets(batch.pass_end_location, self.zone_label_sigma)
            return -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        return F.cross_entropy(logits, batch.target_zone_label)

    def _run_epoch(self, loader: DataLoader, training: bool, label: str) -> Dict[str, float]:
        self.model.train(training)
        tracker = MetricTracker()
        zone_metrics = ZoneMetricAccumulator(self.pitch_grid)
        progress = tqdm(loader, desc=f"Epoch {self.current_epoch} [{label}]", leave=False)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in progress:
                batch = batch.to(self.device)
                if training:
                    self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = self._loss(outputs, batch)
                if training:
                    loss.backward()
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()

                metrics = {"loss": loss.item()}
                if self.task == "receiver":
                    metrics.update(compute_topk_accuracy(outputs["receiver_logits"], batch.receiver_label))
                else:
                    zone_metrics.update(
                        F.log_softmax(outputs["target_zone_logits"].detach(), dim=-1),
                        batch.target_zone_label,
                        batch.pass_end_location,
                        scores_are_log_probs=True,
                    )
                tracker.update(metrics, batch_size=batch.batch_size)
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
        results = tracker.compute()
        if self.task == "target_zone":
            results.update(zone_metrics.compute())
        return results

    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        selected_loader = loader or self.val_loader
        if selected_loader is None:
            return {}
        return self._run_epoch(selected_loader, training=False, label="Eval")

    def _save_checkpoint(self, filename: str) -> None:
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_name": self.model_name,
                "model_kwargs": self.model_kwargs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "selection_metric": self.selection_metric,
                "zone_label_sigma": self.zone_label_sigma,
            },
            self.checkpoint_dir / filename,
        )

    def fit(self, epochs: int) -> Dict[str, list[float]]:
        history: Dict[str, list[float]] = {}
        for _ in range(epochs):
            self.current_epoch += 1
            train_metrics = self._run_epoch(self.train_loader, training=True, label="Train")
            val_metrics = self.evaluate()
            self.scheduler.step()

            for source, prefix in ((train_metrics, "train"), (val_metrics, "val")):
                for name, value in source.items():
                    history.setdefault(f"{prefix}_{name}", []).append(value)

            comparison_loss = val_metrics.get(self.selection_metric, train_metrics[self.selection_metric])
            if comparison_loss < self.best_val_loss:
                self.best_val_loss = comparison_loss
                self._save_checkpoint("best_model.pt")
            self._save_checkpoint("last_model.pt")
            summary = (
                f"Epoch {self.current_epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics.get('loss', float('nan')):.4f} | "
                f"val_top1={val_metrics.get('top_1_acc', 0.0):.2%} | "
                f"val_top3={val_metrics.get('top_3_acc', 0.0):.2%}"
            )
            if self.task == "target_zone":
                summary += f" | val_median_dist={val_metrics.get('median_dist_yd', float('nan')):.1f}yd"
            print(summary)
        return history


ZONE_BASELINES = (
    "passer_zone_prior",
    "pass_vector_prior",
    "most_frequent_zone",
    "closest_teammate_zone",
    "random_zone",
)
# Baselines fitted on the training split (their training-split scores are in-sample).
FITTED_ZONE_BASELINES = ("passer_zone_prior", "pass_vector_prior", "most_frequent_zone")


def fit_zone_prior(loader: DataLoader, pitch_grid: PitchGrid, target_scale: float = 10.0) -> ZonePriorBaseline:
    """Fit marginal and passer-zone-conditional target priors on a training loader."""
    prior = ZonePriorBaseline(pitch_grid)
    for temporal_batch in loader:
        current = temporal_batch.current_graph_batch()
        prior.update(passer_zones(current, pitch_grid, target_scale), current.target_zone_label)
    return prior


@torch.no_grad()
def evaluate_zone_baselines(
    loader: DataLoader,
    pitch_grid: PitchGrid,
    prior: ZonePriorBaseline,
    seed: int,
    target_scale: float = 10.0,
) -> Dict[str, Dict[str, float]]:
    """Evaluate every naive zone baseline on the target frames of a loader."""
    generator = torch.Generator().manual_seed(seed)
    accumulators = {name: ZoneMetricAccumulator(pitch_grid) for name in ZONE_BASELINES}
    for temporal_batch in loader:
        current = temporal_batch.current_graph_batch()
        passer = passer_zones(current, pitch_grid, target_scale)
        scores = {
            "passer_zone_prior": (prior.conditional_scores(passer), True),
            "pass_vector_prior": (prior.pass_vector_scores(passer), True),
            "most_frequent_zone": (prior.marginal_scores(current.batch_size, current.node_features.device), True),
            "closest_teammate_zone": (closest_teammate_zone_scores(current, pitch_grid, target_scale), False),
            "random_zone": (random_zone_scores(current, pitch_grid, generator), False),
        }
        for name, (values, are_log_probs) in scores.items():
            accumulators[name].update(
                values, current.target_zone_label, current.pass_end_location, scores_are_log_probs=are_log_probs
            )
    return {name: accumulator.compute() for name, accumulator in accumulators.items()}
