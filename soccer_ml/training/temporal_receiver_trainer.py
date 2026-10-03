"""Training and evaluation utilities for temporal pass-target benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from soccer_ml.data.pitch_grid import (
    PitchGrid,
    TeammateKernelBaseline,
    ZonePriorBaseline,
    closest_teammate_zone_scores,
    passer_zones,
    random_zone_scores,
    receiver_candidates,
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
        progress = tqdm(loader, desc=f"Epoch {self.current_epoch} [{label}]", leave=False, disable=None)
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
                    log_probs = F.log_softmax(outputs["target_zone_logits"].detach(), dim=-1)
                    zone_metrics.update(
                        log_probs,
                        batch.target_zone_label,
                        batch.pass_end_location,
                        log_probs=log_probs,
                        receivers=receiver_candidates(batch.current_graph_batch(), self.pitch_grid),
                    )
                tracker.update(metrics, batch_size=batch.batch_size)
                progress.set_postfix({"loss": f"{loss.item():.4f}"}, refresh=False)
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
                + (f"val_nll={val_metrics['nll']:.4f} | " if "nll" in val_metrics else "")
                + f"val_top1={val_metrics.get('top_1_acc', 0.0):.2%} | "
                f"val_top3={val_metrics.get('top_3_acc', 0.0):.2%}"
            )
            if self.task == "target_zone":
                summary += f" | val_median_dist={val_metrics.get('median_dist_yd', float('nan')):.1f}yd"
            print(summary)
        return history


ZONE_BASELINES = (
    "teammate_kernel",
    "passer_zone_prior",
    "pass_vector_prior",
    "most_frequent_zone",
    "closest_teammate_zone",
    "random_zone",
)
# Baselines fitted on the training split (their training-split scores are in-sample).
FITTED_ZONE_BASELINES = ("teammate_kernel", "passer_zone_prior", "pass_vector_prior", "most_frequent_zone")


@dataclass
class ZoneBaselines:
    prior: ZonePriorBaseline
    teammate_kernel: TeammateKernelBaseline


def fit_zone_baselines(loader: DataLoader, pitch_grid: PitchGrid, target_scale: float = 10.0) -> ZoneBaselines:
    """Fit the zone priors and the teammate-kernel parameters on a training loader."""
    prior = ZonePriorBaseline(pitch_grid)
    teammate_kernel = TeammateKernelBaseline(pitch_grid, target_scale)
    for temporal_batch in loader:
        current = temporal_batch.current_graph_batch()
        prior.update(passer_zones(current, pitch_grid, target_scale), current.target_zone_label)
        teammate_kernel.update(current)
    teammate_kernel.finalize()
    return ZoneBaselines(prior=prior, teammate_kernel=teammate_kernel)


@torch.no_grad()
def evaluate_zone_baselines(
    loader: DataLoader,
    pitch_grid: PitchGrid,
    baselines: ZoneBaselines,
    seed: int,
    target_scale: float = 10.0,
) -> Dict[str, Dict[str, float]]:
    """Evaluate every naive zone baseline on the target frames of a loader."""
    generator = torch.Generator().manual_seed(seed)
    accumulators = {name: ZoneMetricAccumulator(pitch_grid) for name in ZONE_BASELINES}
    uniform = torch.full((pitch_grid.num_zones,), -math.log(pitch_grid.num_zones))
    for temporal_batch in loader:
        current = temporal_batch.current_graph_batch()
        passer = passer_zones(current, pitch_grid, target_scale)
        receivers = receiver_candidates(current, pitch_grid, target_scale)
        kernel_log_probs = baselines.teammate_kernel.log_probs(current)
        prior_log_probs = {
            "passer_zone_prior": baselines.prior.conditional_scores(passer),
            "pass_vector_prior": baselines.prior.pass_vector_scores(passer),
            "most_frequent_zone": baselines.prior.marginal_scores(current.batch_size, current.node_features.device),
        }
        # (ranking scores, log-probabilities or None)
        scores = {
            "teammate_kernel": (kernel_log_probs, kernel_log_probs),
            **{name: (values, values) for name, values in prior_log_probs.items()},
            "closest_teammate_zone": (closest_teammate_zone_scores(current, pitch_grid, target_scale), None),
            "random_zone": (
                random_zone_scores(current, pitch_grid, generator),
                uniform.expand(current.batch_size, -1),
            ),
        }
        for name, (ranking, log_probs) in scores.items():
            accumulators[name].update(
                ranking, current.target_zone_label, current.pass_end_location, log_probs=log_probs, receivers=receivers
            )
    return {name: accumulator.compute() for name, accumulator in accumulators.items()}
