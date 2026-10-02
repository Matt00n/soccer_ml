"""Training, metrics and naive baselines for next-action prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from soccer_ml.data.next_action_dataset import (
    ACTION_TYPES,
    DECISION_CONTEXTS,
    IGNORE_INDEX,
    LOCATED_ACTIONS,
    PASS_HEIGHTS,
    NextActionBatch,
)
from soccer_ml.data.pitch_grid import PitchGrid, ReceiverCandidates, ZonePriorBaseline, passer_zones, receiver_candidates
from soccer_ml.models.next_action import action_log_likelihood
from soccer_ml.training.metrics import ZoneMetricAccumulator
from soccer_ml.training.trainer import get_default_device

_PASS = ACTION_TYPES.index("pass")


class NextActionMetrics:
    """Epoch metrics; the same code scores the model and the baselines.

    * ``joint_nll``: -log P(type, zone, height) per action (headline; checkpoint selection).
    * ``type_*``: NLL, accuracy, macro recall and per-class recall of the action type.
    * ``type_acc_<context>``, ``type_nll_<context>``, ``share_<context>``: the
      same per decision context, so a structural shortcut cannot hide behind the
      headline (``spell_start`` is where carry-vs-release is genuinely decided).
    * ``pass_*`` / ``carry_*``: zone metrics of :class:`ZoneMetricAccumulator`
      (NLL, top-k, within one zone, distance; receiver accuracy for passes)
      on actions of that type, given the true type.
    * ``height_*``: NLL and accuracy of ground/low/high at the true zone, and
      ``lofted_acc`` for the binary ground-vs-lofted decision.
    """

    def __init__(self, pitch_grid: PitchGrid):
        self.pitch_grid = pitch_grid
        num_types = len(ACTION_TYPES)
        self.count = 0
        self.joint_sum = self.type_sum = self.height_sum = 0.0
        self.height_count = self.height_correct = self.lofted_correct = 0
        self.confusion = torch.zeros(num_types, num_types, dtype=torch.long)
        self.context_count = torch.zeros(len(DECISION_CONTEXTS))
        self.context_correct = torch.zeros(len(DECISION_CONTEXTS))
        self.context_type_nll = torch.zeros(len(DECISION_CONTEXTS), dtype=torch.float64)
        self.zone = {name: ZoneMetricAccumulator(pitch_grid) for name in LOCATED_ACTIONS}

    @torch.no_grad()
    def update(self, outputs: Dict[str, torch.Tensor], batch: NextActionBatch, nll: Dict[str, torch.Tensor]) -> None:
        self.count += batch.batch_size
        self.joint_sum += float(nll["joint_nll"].sum())
        self.type_sum += float(nll["type_nll"].sum())
        predicted = outputs["type_logits"].argmax(dim=-1)
        flat = batch.action_type.cpu() * len(ACTION_TYPES) + predicted.cpu()
        self.confusion += torch.bincount(flat, minlength=self.confusion.numel()).view_as(self.confusion)
        context = batch.decision_context.cpu()
        correct = (predicted.cpu() == batch.action_type.cpu()).float()
        self.context_count += torch.bincount(context, minlength=len(DECISION_CONTEXTS)).float()
        self.context_correct += torch.bincount(context, weights=correct, minlength=len(DECISION_CONTEXTS))
        self.context_type_nll += torch.bincount(
            context, weights=nll["type_nll"].detach().cpu().double(), minlength=len(DECISION_CONTEXTS)
        )

        receivers = receiver_candidates(batch.current_graph_batch(), self.pitch_grid)
        for head, name in enumerate(LOCATED_ACTIONS):
            rows = nll["located"] & (batch.action_type == ACTION_TYPES.index(name))
            if not rows.any():
                continue
            log_probs = F.log_softmax(outputs["zone_logits"][rows, head].float(), dim=-1)
            self.zone[name].update(
                log_probs,
                batch.target_zone[rows],
                batch.end_location[rows],
                log_probs=log_probs,
                receivers=ReceiverCandidates(
                    receivers.xy[rows], receivers.candidates[rows], receivers.receiver[rows]
                )
                if name == "pass"
                else None,
            )

        rows = nll["has_height"]
        if rows.any():
            self.height_sum += float(nll["height_nll"][rows].sum())
            self.height_count += int(rows.sum())
            index = torch.arange(batch.batch_size, device=rows.device)[rows]
            height_logits = outputs["height_logits"][index, :, batch.target_zone[rows]]
            predicted_height = height_logits.argmax(dim=-1)
            true_height = batch.pass_height[rows]
            self.height_correct += int((predicted_height == true_height).sum())
            self.lofted_correct += int(((predicted_height > 0) == (true_height > 0)).sum())

    def compute(self) -> Dict[str, float]:
        total = max(self.count, 1)
        metrics: Dict[str, float] = {
            "joint_nll": self.joint_sum / total,
            "type_nll": self.type_sum / total,
            "type_acc": float(self.confusion.diag().sum()) / total,
        }
        support = self.confusion.sum(dim=1)
        recalls = []
        for index, name in enumerate(ACTION_TYPES):
            if support[index] > 0:
                recall = float(self.confusion[index, index]) / float(support[index])
                metrics[f"recall_{name}"] = recall
                recalls.append(recall)
        metrics["type_macro_recall"] = sum(recalls) / max(len(recalls), 1)
        for index, name in enumerate(DECISION_CONTEXTS):
            count = float(self.context_count[index])
            if count > 0:
                metrics[f"share_{name}"] = count / total
                metrics[f"type_acc_{name}"] = float(self.context_correct[index]) / count
                metrics[f"type_nll_{name}"] = float(self.context_type_nll[index]) / count
        for name, accumulator in self.zone.items():
            if accumulator.count:
                metrics.update({f"{name}_{key}": value for key, value in accumulator.compute().items()})
        if self.height_count:
            metrics["height_nll"] = self.height_sum / self.height_count
            metrics["height_acc"] = self.height_correct / self.height_count
            metrics["lofted_acc"] = self.lofted_correct / self.height_count
        return metrics


class NextActionPriors:
    """Naive baselines fitted on the training split, in the model's output format.

    * ``marginal``: training frequencies of type, of the end zone per located
      type, and of the pass height.
    * ``zone_prior``: P(type | actor zone), P(zone | actor zone, type) as a
      transition table smoothed towards the displacement prior, and
      P(height | pass displacement); each backed off to its marginal.
    """

    def __init__(self, pitch_grid: PitchGrid, backoff_weight: float = 20.0, alpha: float = 0.5):
        self.pitch_grid = pitch_grid
        self.backoff_weight = backoff_weight
        self.alpha = alpha
        z = pitch_grid.num_zones
        self.type_counts = torch.zeros(z, len(ACTION_TYPES))
        self.zone_priors = {name: ZonePriorBaseline(pitch_grid) for name in LOCATED_ACTIONS}
        offset_index = self.zone_priors[LOCATED_ACTIONS[0]]._offset_index
        self._offset_index = offset_index
        self.height_counts = torch.zeros(int(offset_index.max()) + 1, len(PASS_HEIGHTS))

    def update(self, batch: NextActionBatch) -> None:
        actor_zone = passer_zones(batch.current_graph_batch(), self.pitch_grid).cpu()
        action, zone, height = batch.action_type.cpu(), batch.target_zone.cpu(), batch.pass_height.cpu()
        self.type_counts.index_put_((actor_zone, action), torch.ones_like(action, dtype=torch.float), accumulate=True)
        for name in LOCATED_ACTIONS:
            rows = (action == ACTION_TYPES.index(name)) & (zone != IGNORE_INDEX)
            if rows.any():
                self.zone_priors[name].update(actor_zone[rows], zone[rows])
        rows = (action == _PASS) & (zone != IGNORE_INDEX) & (height != IGNORE_INDEX)
        offsets = self._offset_index[actor_zone[rows], zone[rows]]
        self.height_counts.index_put_((offsets, height[rows]), torch.ones(int(rows.sum())), accumulate=True)

    @staticmethod
    def _log_normalize(counts: torch.Tensor) -> torch.Tensor:
        return (counts / counts.sum(dim=-1, keepdim=True)).log()

    def predict(self, batch: NextActionBatch, kind: str) -> Dict[str, torch.Tensor]:
        device = batch.node_features.device
        actor_zone = passer_zones(batch.current_graph_batch(), self.pitch_grid).cpu()
        b, z = batch.batch_size, self.pitch_grid.num_zones
        type_marginal = self.type_counts.sum(dim=0) + self.alpha
        height_marginal = self.height_counts.sum(dim=0) + self.alpha
        mask = batch.type_mask.cpu()
        if kind == "marginal":
            type_logits = self._log_normalize(type_marginal).expand(b, -1)
            zone_logits = torch.stack(
                [self.zone_priors[name].marginal_log_probs().expand(b, -1) for name in LOCATED_ACTIONS], dim=1
            )
            height_logits = self._log_normalize(height_marginal)[None, :, None].expand(b, -1, z)
        elif kind == "zone_prior":
            type_marginal = type_marginal / type_marginal.sum()
            type_logits = self._log_normalize(self.type_counts[actor_zone] + self.backoff_weight * type_marginal)
            zone_logits = torch.stack(
                [self.zone_priors[name].conditional_scores(actor_zone) for name in LOCATED_ACTIONS], dim=1
            )
            height_marginal = height_marginal / height_marginal.sum()
            counts = self.height_counts[self._offset_index[actor_zone]]  # (B, Z, 3)
            height_logits = self._log_normalize(counts + self.backoff_weight * height_marginal).transpose(1, 2)
        else:
            raise ValueError(f"Unknown baseline '{kind}'.")
        # Restrict to the actions possible at each decision and renormalize.
        type_logits = torch.log_softmax(type_logits.masked_fill(~mask, -1e9), dim=-1)
        return {
            "type_logits": type_logits.to(device),
            "zone_logits": zone_logits.to(device),
            "height_logits": height_logits.contiguous().to(device),
        }


BASELINES = ("zone_prior", "marginal")


def fit_next_action_priors(loader: DataLoader, pitch_grid: PitchGrid) -> NextActionPriors:
    priors = NextActionPriors(pitch_grid)
    for batch in loader:
        priors.update(batch)
    return priors


@torch.no_grad()
def evaluate_next_action_baselines(
    loader: DataLoader, pitch_grid: PitchGrid, priors: NextActionPriors
) -> Dict[str, Dict[str, float]]:
    metrics = {name: NextActionMetrics(pitch_grid) for name in BASELINES}
    for batch in loader:
        for name in BASELINES:
            outputs = priors.predict(batch, name)
            metrics[name].update(outputs, batch, action_log_likelihood(outputs, batch))
    return {name: accumulator.compute() for name, accumulator in metrics.items()}


class NextActionTrainer:
    """Trains on the joint NLL; checkpoints are selected by validation joint NLL."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        lr: float,
        weight_decay: float,
        epochs: int,
        checkpoint_dir: str | Path,
        pitch_grid: PitchGrid,
        data_config: Dict[str, Any],
        device: Optional[torch.device] = None,
        clip_grad_norm: Optional[float] = 1.0,
    ):
        self.device = device or get_default_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pitch_grid = pitch_grid
        self.data_config = data_config
        self.clip_grad_norm = clip_grad_norm
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(epochs, 1), eta_min=1e-6)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_epoch = 0
        self.best_val_nll = float("inf")

    def _run_epoch(self, loader: DataLoader, training: bool, label: str) -> Dict[str, float]:
        self.model.train(training)
        metrics = NextActionMetrics(self.pitch_grid)
        progress = tqdm(loader, desc=f"Epoch {self.current_epoch} [{label}]", leave=False)
        with torch.enable_grad() if training else torch.no_grad():
            for batch in progress:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                nll = action_log_likelihood(outputs, batch)
                loss = nll["joint_nll"].mean()
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                metrics.update({key: value.detach() for key, value in outputs.items()}, batch, nll)
                progress.set_postfix({"nll": f"{loss.item():.3f}"})
        return metrics.compute()

    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        selected = loader or self.val_loader
        return self._run_epoch(selected, training=False, label="Eval") if selected is not None else {}

    def _save_checkpoint(self, filename: str) -> None:
        torch.save(
            {
                "epoch": self.current_epoch,
                "model_name": getattr(self.model, "model_name", "next_action"),
                "model_kwargs": self.model.model_kwargs,
                "data_config": self.data_config,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_joint_nll": self.best_val_nll,
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
            selection = val_metrics.get("joint_nll", train_metrics["joint_nll"])
            if selection < self.best_val_nll:
                self.best_val_nll = selection
                self._save_checkpoint("best_model.pt")
            self._save_checkpoint("last_model.pt")
            print(
                f"Epoch {self.current_epoch:03d} | train_nll={train_metrics['joint_nll']:.4f} | "
                f"val_nll={val_metrics.get('joint_nll', float('nan')):.4f} "
                f"(type {val_metrics.get('type_nll', float('nan')):.3f}, "
                f"pass zone {val_metrics.get('pass_nll', float('nan')):.3f}, "
                f"carry zone {val_metrics.get('carry_nll', float('nan')):.3f}, "
                f"height {val_metrics.get('height_nll', float('nan')):.3f}) | "
                f"type_acc={val_metrics.get('type_acc', 0.0):.2%} "
                f"(spell start {val_metrics.get('type_acc_spell_start', 0.0):.2%}) | "
                f"pass_top1={val_metrics.get('pass_top_1_acc', 0.0):.2%}"
            )
        return history
