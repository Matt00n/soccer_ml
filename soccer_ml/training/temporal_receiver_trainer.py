"""Training and evaluation utilities for temporal receiver benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from soccer_ml.training.metrics import MetricTracker, compute_topk_accuracy
from soccer_ml.training.trainer import get_default_device


class TemporalReceiverTrainer:
    """Small, task-specific trainer with portable temporal-model checkpoints."""

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
        device: Optional[torch.device] = None,
        clip_grad_norm: Optional[float] = 1.0,
    ):
        self.device = device or get_default_device()
        self.model = model.to(self.device)
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.clip_grad_norm = clip_grad_norm
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def _run_epoch(self, loader: DataLoader, training: bool, label: str) -> Dict[str, float]:
        self.model.train(training)
        tracker = MetricTracker()
        progress = tqdm(loader, desc=f"Epoch {self.current_epoch} [{label}]")
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in progress:
                batch = batch.to(self.device)
                if training:
                    self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = self.loss_fn(outputs["receiver_logits"], batch.receiver_label)
                if training:
                    loss.backward()
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()

                metrics = {"loss": loss.item()}
                metrics.update(compute_topk_accuracy(outputs["receiver_logits"], batch.receiver_label))
                tracker.update(metrics, batch_size=batch.batch_size)
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
        return tracker.compute()

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
            },
            self.checkpoint_dir / filename,
        )

    def fit(self, epochs: int) -> Dict[str, list[float]]:
        history: Dict[str, list[float]] = {
            "train_loss": [],
            "train_receiver_top_1_acc": [],
            "train_receiver_top_3_acc": [],
            "val_loss": [],
            "val_receiver_top_1_acc": [],
            "val_receiver_top_3_acc": [],
        }
        for _ in range(epochs):
            self.current_epoch += 1
            train_metrics = self._run_epoch(self.train_loader, training=True, label="Train")
            val_metrics = self.evaluate()
            self.scheduler.step()

            for source, prefix in ((train_metrics, "train"), (val_metrics, "val")):
                if not source:
                    continue
                history[f"{prefix}_loss"].append(source["loss"])
                history[f"{prefix}_receiver_top_1_acc"].append(source.get("top_1_acc", 0.0))
                history[f"{prefix}_receiver_top_3_acc"].append(source.get("top_3_acc", 0.0))

            comparison_loss = val_metrics.get("loss", train_metrics["loss"])
            if comparison_loss < self.best_val_loss:
                self.best_val_loss = comparison_loss
                self._save_checkpoint("best_model.pt")
            self._save_checkpoint("last_model.pt")
            print(
                f"Epoch {self.current_epoch:03d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_top1={train_metrics.get('top_1_acc', 0.0):.2%} | "
                f"val_loss={val_metrics.get('loss', float('nan')):.4f} | "
                f"val_top1={val_metrics.get('top_1_acc', 0.0):.2%}"
            )
        return history

