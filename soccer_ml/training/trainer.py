"""Full Training Pipeline for Soccer GATv2."""

from typing import Optional, Dict, Any
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from soccer_ml.models.soccer_gat import SoccerGATv2Model
from soccer_ml.training.losses import SoccerMultiTaskLoss
from soccer_ml.training.metrics import (
    compute_topk_accuracy,
    compute_binary_metrics,
    compute_regression_metrics,
    MetricTracker,
)


def get_default_device() -> torch.device:
    """Select the best available hardware accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SoccerTrainer:
    """Orchestrates model training, validation, evaluation, and checkpointing."""

    def __init__(
        self,
        model: SoccerGATv2Model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        clip_grad_norm: Optional[float] = 1.0,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = "checkpoints",
        loss_fn: Optional[SoccerMultiTaskLoss] = None,
    ):
        self.device = device or get_default_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.clip_grad_norm = clip_grad_norm

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = loss_fn or SoccerMultiTaskLoss()

        # Adam optimizer with parameters matching TacticAI S.Table 1
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        tracker = MetricTracker()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for batch in pbar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(batch)
            loss_dict = self.loss_fn(outputs, batch)

            total_loss = loss_dict["total_loss"]
            total_loss.backward()

            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

            self.optimizer.step()

            # Record batch losses
            batch_metrics = {k: v.item() for k, v in loss_dict.items()}
            b_size = batch.batch_size
            tracker.update(batch_metrics, batch_size=b_size)

            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        self.scheduler.step()
        return tracker.compute()

    @torch.no_grad()
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Run evaluation pass and compute domain metrics."""
        loader = data_loader or self.val_loader
        if loader is None:
            return {}

        self.model.eval()
        tracker = MetricTracker()

        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch + 1} [Eval]")
        for batch in pbar:
            batch = batch.to(self.device)
            outputs = self.model(batch)
            loss_dict = self.loss_fn(outputs, batch)

            batch_metrics = {f"val_{k}": v.item() for k, v in loss_dict.items()}

            # Task 1: Receiver prediction top-1 & top-3 accuracy
            if "receiver_logits" in outputs and batch.receiver_label is not None:
                topk = compute_topk_accuracy(outputs["receiver_logits"], batch.receiver_label)
                for k, v in topk.items():
                    batch_metrics[f"val_receiver_{k}"] = v

            # Task 2: Shot prediction F1 & accuracy
            if "shot_logits" in outputs and batch.shot_label is not None:
                shot_metrics = compute_binary_metrics(outputs["shot_logits"], batch.shot_label)
                for k, v in shot_metrics.items():
                    batch_metrics[f"val_shot_{k}"] = v

            # Task 3: Creativity score MSE
            if "creativity_scores" in outputs and batch.creativity_label is not None:
                reg_metrics = compute_regression_metrics(
                    outputs["creativity_scores"], batch.creativity_label
                )
                for k, v in reg_metrics.items():
                    batch_metrics[f"val_creativity_{k}"] = v

            tracker.update(batch_metrics, batch_size=batch.batch_size)

        return tracker.compute()

    def fit(self, num_epochs: int) -> Dict[str, Any]:
        """Train the model for a specified number of epochs."""
        history: Dict[str, list] = {}

        for _ in range(num_epochs):
            self.current_epoch += 1

            train_metrics = self.train_epoch()
            val_metrics = self.evaluate() if self.val_loader is not None else {}

            combined_metrics = {**train_metrics, **val_metrics}

            for k, v in combined_metrics.items():
                if k not in history:
                    history[k] = []
                history[k].append(v)

            # Checkpoint management
            if self.checkpoint_dir:
                val_loss = combined_metrics.get("val_total_loss", train_metrics.get("total_loss"))
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
                self.save_checkpoint("last_model.pt")

            print(
                f"Epoch {self.current_epoch:03d} | "
                f"Train Loss: {train_metrics.get('total_loss', 0.0):.4f} | "
                + (
                    f"Val Loss: {val_metrics.get('val_total_loss', 0.0):.4f}"
                    if val_metrics
                    else ""
                )
            )

        return history

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        if not self.checkpoint_dir:
            return
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            checkpoint_path,
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
