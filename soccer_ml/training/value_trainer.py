"""Training and evaluation of action-value models.

Targets (``target``), all from the acting team's perspective with components
(score, concede):

* ``mc``: Monte Carlo labels, P(score / concede within the next 10 SPADL
  actions) with xG-sequence (default) or goal labels.  No bootstrapping.
* ``sarsa``: y = r + γ · Q_target(s', a'), a' the observed next action.
* ``expected_sarsa``: y = r + γ · Σ_a' π(a'|s') Q_target(s', a') with the
  next-action policy π (a frozen checkpoint, or the jointly trained heads).
* ``iql``: implicit Q-learning.  y = r + γ · V_target(s'); V(s) is fitted by
  expectile regression to Q_target(s, a): an upper expectile (τ > 0.5) of the
  score component (lower of the concede component) approaches the value of
  the best *in-distribution* action without querying unseen actions.
* ``lambda_return``: TD(λ) / GAE-style targets
  G_t = r_t + γ · [(1 - λ) · Q_target(s', a') + λ · G_{t+1}], recomputed over
  the whole training set at the start of every epoch.  λ = 1 is the
  discounted Monte Carlo return to the end of the period (computed once);
  dead balls are not episode ends, restarts carry their own learned value.

With ``distribution="categorical"`` a C51 head additionally learns the
distribution of the net return (score - concede) by distributional TD on the
observed next action (projected onto a fixed, symmetric support).

r is the xG (or goals) of shots between the action and the next decision;
components swap where the other team makes the next decision and the
bootstrap vanishes at the end of a period.

Losses: binary cross-entropy against soft targets in [0, 1] for expectation
heads (the decomposed head trains P(success) on outcomes and each branch on
its own outcomes); quantile-Huber loss for quantile heads, against the single
Monte Carlo label or the bootstrapped target quantiles (distributional TD).

Supported combinations: every target with every head for expectations;
quantiles with ``mc`` and ``sarsa`` and the direct head (see
:func:`validate_configuration`).
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from soccer_ml.data.next_action_dataset import (
    ACTION_TYPES,
    GRAPH_FEATURES,
    IGNORE_INDEX,
    REWARDS,
    VALUE_LABELS,
    NextActionBatch,
    collate_next_action,
    lambda_returns,
)
from soccer_ml.data.pitch_grid import PitchGrid
from soccer_ml.models.next_action import NextActionModel, action_log_likelihood
from soccer_ml.models.value import (
    ValueModel,
    distribution_logits,
    gather_action,
    project_categorical,
    is_located_known,
    observed_action_index,
    policy_action_log_probs,
    quantile_fractions,
    swap_components,
)
from soccer_ml.training.metrics import probabilistic_binary_metrics
from soccer_ml.training.trainer import get_default_device

TARGETS = ("mc", "sarsa", "expected_sarsa", "iql", "lambda_return")
TRAININGS = ("warm", "joint")


@dataclass
class ValueConfig:
    target: str = "mc"
    head: str = "direct"
    distribution: str = "expectation"
    num_quantiles: int = 5
    training: str = "warm"
    mc_label: str = "xg"  # xg | goal
    reward: str = "xg"  # xg | goal
    gamma: float = 0.9
    expectile: float = 0.7
    target_tau: float = 0.005  # Polyak rate of the target network
    huber_kappa: float = 0.01
    lam: float = 0.8  # lambda_return only
    distribution_weight: float = 0.1  # categorical only: the C51 head is auxiliary
    policy_weight: float = 1.0  # joint training only
    value_weight: float = 1.0
    freeze_encoder: bool = False
    encoder_lr_scale: float = 1.0  # learning-rate multiplier of the (pretrained) encoder and policy heads


def validate_configuration(config: ValueConfig) -> None:
    if config.target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}.")
    if config.training not in TRAININGS:
        raise ValueError(f"training must be one of {TRAININGS}.")
    if config.distribution == "quantile":
        if config.head != "direct":
            raise ValueError("Quantile values require the direct head.")
        if config.target == "expected_sarsa":
            raise ValueError(
                "Expected SARSA over quantile distributions needs the π-mixture of next-state "
                "distributions, which quantile regression cannot represent; use sarsa."
            )
        if config.target == "iql":
            raise ValueError("IQL fits expectiles of an expected Q; combine it with expectation values.")
        if config.target == "lambda_return":
            raise ValueError("λ-returns are implemented for expectation values (use categorical for distributions).")
    if not 0.0 <= config.lam <= 1.0:
        raise ValueError("lam must lie in [0, 1].")
    if config.freeze_encoder and config.training == "joint":
        raise ValueError("A frozen encoder cannot be trained jointly with the policy.")


def needs_target_network(config: ValueConfig) -> bool:
    if config.distribution == "categorical":
        return True
    if config.target == "lambda_return":
        return config.lam < 1.0
    return config.target != "mc"


def monte_carlo_targets(batch: NextActionBatch, label: str) -> torch.Tensor:
    columns = [VALUE_LABELS.index("xg_scores"), VALUE_LABELS.index("xg_concedes")] if label == "xg" else [
        VALUE_LABELS.index("scores"),
        VALUE_LABELS.index("concedes"),
    ]
    return batch.value_label[:, columns]


def td_rewards(batch: NextActionBatch, kind: str) -> torch.Tensor:
    names = ("our_xg", "their_xg") if kind == "xg" else ("our_goals", "their_goals")
    return batch.reward[:, [REWARDS.index(name) for name in names]]


def bootstrap(next_values: torch.Tensor, batch: NextActionBatch) -> torch.Tensor:
    """Express next-state values in the current actor's perspective; zero at the end of a period."""
    values = swap_components(next_values, batch.next_flip)
    shape = (-1,) + (1,) * (values.dim() - 1)
    return values * (~batch.done).view(shape).to(values.dtype)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    """Asymmetric squared loss; ``diff`` = target - prediction, (B, 2).

    The score component uses ``expectile``; concede uses 1 - expectile, so
    both lean towards the actor's better outcomes.
    """
    tau = torch.tensor([expectile, 1.0 - expectile], device=diff.device, dtype=diff.dtype)
    weight = torch.where(diff < 0, 1.0 - tau, tau)
    return (weight * diff.pow(2)).sum(dim=-1)


def quantile_huber_loss(prediction: torch.Tensor, target: torch.Tensor, kappa: float) -> torch.Tensor:
    """QR-DQN loss: ``prediction`` (B, C, K) quantiles, ``target`` (B, C, J) samples -> (B,)."""
    fractions = quantile_fractions(prediction.shape[-1], prediction.device)
    u = target.unsqueeze(2) - prediction.unsqueeze(3)  # (B, C, K, J)
    huber = torch.where(u.abs() <= kappa, 0.5 * u.pow(2), kappa * (u.abs() - 0.5 * kappa)) / kappa
    weight = (fractions.view(1, 1, -1, 1) - (u.detach() < 0).to(u.dtype)).abs()
    return (weight * huber).mean(dim=3).sum(dim=2).mean(dim=1)


class ValueLearner:
    """Computes targets and losses for one configuration (shared by training and evaluation)."""

    def __init__(self, model: ValueModel, config: ValueConfig, policy: Optional[NextActionModel] = None):
        validate_configuration(config)
        self.model = model
        self.config = config
        self.policy = policy  # frozen π for expected SARSA with warm training
        self.target_model: Optional[ValueModel] = None
        self.lambda_targets: Optional[torch.Tensor] = None  # (N, 2) for the training set
        if needs_target_network(config):
            self.target_model = copy.deepcopy(model).eval()
            for parameter in self.target_model.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def update_target(self) -> None:
        if self.target_model is None:
            return
        for target, online in zip(self.target_model.parameters(), self.model.parameters()):
            target.mul_(1.0 - self.config.target_tau).add_(online.detach(), alpha=self.config.target_tau)

    @torch.no_grad()
    def refresh_lambda_targets(self, dataset, device: torch.device) -> None:
        """Recompute λ-return targets of the whole training set (λ = 1 only once)."""
        config = self.config
        if config.lam >= 1.0 and self.lambda_targets is not None:
            return
        bootstrap = None
        if config.lam < 1.0:
            values = []
            loader = DataLoader(dataset, batch_size=512, shuffle=False, collate_fn=collate_next_action)
            for batch in tqdm(loader, desc="λ-return refresh", leave=False, disable=None):
                batch = batch.to(device)
                outputs = self.target_model(batch)
                values.append(gather_action(outputs["q"], observed_action_index(batch, self.model.num_zones)).cpu())
            bootstrap = torch.cat(values)
        returns = lambda_returns(dataset, config.gamma, config.lam, bootstrap, config.reward)
        self.lambda_targets = returns.clamp(0.0, 1.0).to(device)

    def _distribution_target(
        self, batch: NextActionBatch, next_batch: NextActionBatch, next_outputs: Dict[str, Any]
    ) -> torch.Tensor:
        """Projected distributional TD target of the net return, ``(B, K)``."""
        atoms = self.model.atoms
        next_action = observed_action_index(next_batch, self.model.num_zones)
        probabilities = torch.softmax(distribution_logits(self.target_model, next_outputs, next_action), dim=-1)
        # Symmetric support: the other team's net return is ours mirrored.
        probabilities = torch.where(batch.next_flip.unsqueeze(-1), probabilities.flip(-1), probabilities)
        rewards = td_rewards(batch, self.config.reward)
        net_reward = (rewards[:, 0] - rewards[:, 1]).unsqueeze(-1)
        continuing = (~batch.done).unsqueeze(-1).to(atoms.dtype)
        shifted = net_reward + self.config.gamma * continuing * atoms.unsqueeze(0)
        return project_categorical(probabilities, shifted, atoms)

    @torch.no_grad()
    def _next_policy_probs(self, next_batch: NextActionBatch, next_outputs: Dict[str, Any]) -> torch.Tensor:
        if self.config.training == "joint":
            policy_outputs = next_outputs["policy"]
        else:
            if self.policy is None:
                raise ValueError("Expected SARSA with warm training needs the frozen policy checkpoint.")
            policy_outputs = self.policy(next_batch)
        return policy_action_log_probs(policy_outputs).exp()

    @torch.no_grad()
    def targets(
        self, batch: NextActionBatch, next_batch: Optional[NextActionBatch], indices: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        config = self.config
        next_outputs = self.target_model(next_batch) if self.target_model is not None else None
        extra: Dict[str, torch.Tensor] = {}
        if config.distribution == "categorical":
            extra["distribution_target"] = self._distribution_target(batch, next_batch, next_outputs)
        if config.target == "mc":
            y = monte_carlo_targets(batch, config.mc_label)
            return {"y": y.unsqueeze(-1) if config.distribution == "quantile" else y, **extra}
        if config.target == "lambda_return":
            if self.lambda_targets is None or indices is None:
                raise RuntimeError("λ-return targets need refresh_lambda_targets() and sample indices.")
            return {"y": self.lambda_targets[indices.to(self.lambda_targets.device)], **extra}
        assert next_batch is not None and next_outputs is not None
        rewards = td_rewards(batch, config.reward)
        next_action = observed_action_index(next_batch, self.model.num_zones)
        if config.target == "sarsa":
            if config.distribution == "quantile":
                next_values = gather_action(next_outputs["quantiles"], next_action)  # (B, C, K)
                return {"y": rewards.unsqueeze(-1) + config.gamma * bootstrap(next_values, batch)}
            next_values = gather_action(next_outputs["q"], next_action)
        elif config.target == "expected_sarsa":
            probabilities = self._next_policy_probs(next_batch, next_outputs)  # (B, A)
            next_values = (next_outputs["q"] * probabilities.unsqueeze(1)).sum(dim=-1)
        else:  # iql
            next_values = next_outputs["state_value"]
        y = (rewards + config.gamma * bootstrap(next_values, batch)).clamp(0.0, 1.0)
        result = {"y": y, **extra}
        if config.target == "iql":
            current = self.target_model(batch)
            result["q_target_sa"] = gather_action(current["q"], observed_action_index(batch, self.model.num_zones))
        return result

    def loss(
        self, outputs: Dict[str, Any], batch: NextActionBatch, targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        config = self.config
        action = observed_action_index(batch, self.model.num_zones)
        valid = is_located_known(batch)
        if config.target == "mc":
            valid = valid & batch.has_value_label
        y = targets["y"].nan_to_num(0.0)  # rows without a label are masked out below
        terms: Dict[str, torch.Tensor] = {}
        if config.distribution == "quantile":
            prediction = gather_action(outputs["quantiles"], action)  # (B, C, K)
            per_sample = quantile_huber_loss(prediction, y, config.huber_kappa)
            terms["value"] = _masked_mean(per_sample, valid)
        elif config.head == "direct":
            logits = gather_action(outputs["q_logits"], action)  # (B, C)
            per_sample = F.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(-1)
            terms["value"] = _masked_mean(per_sample, valid)
        else:
            success = batch.success
            known = valid & (success != IGNORE_INDEX)
            success_logits = gather_action(outputs["success_logits"], action)
            per_sample = F.binary_cross_entropy_with_logits(
                success_logits, success.clamp_min(0).to(success_logits.dtype), reduction="none"
            )
            terms["success"] = _masked_mean(per_sample, known)
            target = y
            for branch, outcome in (("v_success_logits", 1), ("v_failure_logits", 0)):
                logits = gather_action(outputs[branch], action)
                per_sample = F.binary_cross_entropy_with_logits(logits, target, reduction="none").sum(-1)
                terms[branch.replace("_logits", "")] = _masked_mean(per_sample, known & (success == outcome))
            terms["value"] = terms["v_success"] + terms["v_failure"]
        if config.target == "iql":
            diff = targets["q_target_sa"] - outputs["state_value"]
            terms["state_value"] = _masked_mean(expectile_loss(diff, config.expectile), valid)
        if config.distribution == "categorical":
            log_probabilities = F.log_softmax(distribution_logits(self.model, outputs, action), dim=-1)
            per_sample = -(targets["distribution_target"] * log_probabilities).sum(dim=-1)
            terms["distribution"] = config.distribution_weight * _masked_mean(per_sample, is_located_known(batch))
        total = config.value_weight * sum(terms.values())
        if config.training == "joint":
            terms["policy"] = action_log_likelihood(outputs["policy"], batch)["joint_nll"].mean()
            total = total + config.policy_weight * terms["policy"]
        terms["total"] = total
        return terms


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class ValueEvaluation:
    """Collects per-sample predictions at the observed action and computes metrics.

    * ``score_*`` / ``concede_*``: ex-ante Q(s, a) of each component against
      the binary goal label (AUC, log-loss) and the xG-sequence label
      (``soft_logloss``); ``selection`` = sum of both soft log-losses.
    * ``protocol_*``: un-xPass Table 3 subset (completed open-play passes),
      using V(s, a | success) for the decomposed head and Q(s, a) otherwise.
    * ``success_auc_<type>``: decomposed head, P(success) per action type.
    * ``quantile_coverage_<i>``: share of xG-sequence labels below quantile i.
    * ``return_*``: with ``returns`` (the realized discounted return of every
      validation decision, :func:`lambda_returns` with λ = 1), Q(s, a) is also
      scored against it: the estimand of the TD, λ-return and C51 models.
    * ``distribution_*`` (categorical): CRPS and PIT calibration of the C51
      distribution against the realized net return (PIT shares below 0.1 ...
      0.9 should equal those levels), and the correlation of its mean with
      the expectation heads' net value.
    """

    def __init__(self, model: ValueModel, returns: Optional[torch.Tensor] = None):
        self.model = model
        self.returns = returns
        self.rows: Dict[str, List[torch.Tensor]] = {}

    def _add(self, name: str, values: torch.Tensor) -> None:
        self.rows.setdefault(name, []).append(values.detach().float().cpu())

    @torch.no_grad()
    def update(self, outputs: Dict[str, Any], batch: NextActionBatch, indices: Optional[torch.Tensor] = None) -> None:
        action = observed_action_index(batch, self.model.num_zones)
        q = gather_action(outputs["q"], action)
        self._add("q", q)
        if self.returns is not None and indices is not None:
            returns = self.returns[indices.cpu()]
            self._add("returns", returns)
            if "distribution_features" in outputs:
                probabilities = torch.softmax(distribution_logits(self.model, outputs, action), dim=-1).float().cpu()
                atoms = self.model.atoms.float().cpu()
                net = returns[:, 0] - returns[:, 1]
                cdf = probabilities.cumsum(dim=-1)
                below = (atoms.unsqueeze(0) <= net.unsqueeze(-1)).float()
                self._add("distribution_pit", (probabilities * below).sum(-1) - 0.5 * (probabilities * below * (
                    atoms.unsqueeze(0) == net.unsqueeze(-1)).float()).sum(-1))
                widths = atoms[1:] - atoms[:-1]
                step = (atoms[:-1].unsqueeze(0) >= net.unsqueeze(-1)).float()
                self._add("distribution_crps", ((cdf[:, :-1] - step) ** 2 * widths).sum(-1))
                self._add("distribution_mean", (probabilities * atoms).sum(-1))
                self._add("distribution_upside", (probabilities * (atoms > 0.1).float()).sum(-1))
        self._add("valid", (is_located_known(batch) & batch.has_value_label).float())
        self._add("labels", batch.value_label)
        self._add("action_type", batch.action_type.float())
        self._add("success", batch.success.float())
        self._add("open_play", batch.graph_features[:, GRAPH_FEATURES.index("open_play")])
        self._add("end_location", batch.end_location)
        if "success_logits" in outputs:
            self._add("p_success", torch.sigmoid(gather_action(outputs["success_logits"], action)))
            self._add("v_success", torch.sigmoid(gather_action(outputs["v_success_logits"], action)))
        if "quantiles" in outputs:
            self._add("quantiles", gather_action(outputs["quantiles"], action))
        if "state_value" in outputs:
            self._add("state_value", outputs["state_value"])

    def predictions(self) -> Dict[str, torch.Tensor]:
        return {name: torch.cat(values) for name, values in self.rows.items()}

    def compute(self) -> Dict[str, float]:
        data = self.predictions()
        valid = data["valid"] > 0.5
        labels, q = data["labels"][valid], data["q"][valid].clamp(1e-6, 1 - 1e-6)
        metrics: Dict[str, float] = {"num_samples": float(valid.sum())}
        selection = 0.0
        for component, (binary, soft) in enumerate((("scores", "xg_scores"), ("concedes", "xg_concedes"))):
            name = ("score", "concede")[component]
            result = probabilistic_binary_metrics(
                q[:, component], labels[:, VALUE_LABELS.index(binary)], labels[:, VALUE_LABELS.index(soft)]
            )
            for key in ("auc", "logloss", "brier", "soft_logloss"):
                metrics[f"{name}_{key}"] = result[key]
            selection += result["soft_logloss"]
        metrics["selection"] = selection

        protocol = (
            valid
            & (data["action_type"] == ACTION_TYPES.index("pass"))
            & (data["success"] == 1)
            & (data["open_play"] > 0.5)
        )
        protocol_values = data["v_success"] if "v_success" in data else data["q"]
        protocol_labels = data["labels"][protocol]
        if protocol_labels[:, VALUE_LABELS.index("scores")].sum() > 0:
            result = probabilistic_binary_metrics(
                protocol_values[protocol][:, 0].clamp(1e-6, 1 - 1e-6),
                protocol_labels[:, VALUE_LABELS.index("scores")],
                protocol_labels[:, VALUE_LABELS.index("xg_scores")],
            )
            metrics.update({f"protocol_{key}": value for key, value in result.items()})

        if "p_success" in data:
            known = data["success"] >= 0
            for index, name in enumerate(ACTION_TYPES):
                rows = known & (data["action_type"] == index)
                outcomes = data["success"][rows]
                if rows.sum() > 0 and 0 < outcomes.sum() < len(outcomes):
                    result = probabilistic_binary_metrics(data["p_success"][rows], outcomes)
                    metrics[f"success_auc_{name}"] = result["auc"]
                    metrics[f"success_brier_{name}"] = result["brier"]
        if "returns" in data:
            returns = data["returns"][valid].clamp(0.0, 1.0)
            total = 0.0
            for component, name in enumerate(("score", "concede")):
                target = returns[:, component]
                prediction = q[:, component]
                loss = -(target * prediction.log() + (1 - target) * (1 - prediction).log()).mean()
                metrics[f"return_{name}_soft_logloss"] = float(loss)
                total += float(loss)
            metrics["return_selection"] = total
        if "distribution_pit" in data:
            pit = data["distribution_pit"][valid]
            for level in (0.1, 0.3, 0.5, 0.7, 0.9):
                metrics[f"distribution_pit_below_{level}"] = float((pit <= level).float().mean())
            metrics["distribution_crps"] = float(data["distribution_crps"][valid].mean())
            net_expectation = q[:, 0] - q[:, 1]
            mean = data["distribution_mean"][valid]
            metrics["distribution_mean_corr"] = float(torch.corrcoef(torch.stack([mean, net_expectation]))[0, 1])
            metrics["distribution_mean_score_auc"] = probabilistic_binary_metrics(
                torch.sigmoid(mean * 20), labels[:, VALUE_LABELS.index("scores")]
            )["auc"]
        if "quantiles" in data:
            quantiles = data["quantiles"][valid][:, 0]  # score component (N, K)
            soft = labels[:, VALUE_LABELS.index("xg_scores")].unsqueeze(-1)
            coverage = (soft <= quantiles).float().mean(dim=0)
            for index, value in enumerate(coverage.tolist()):
                metrics[f"quantile_coverage_{index}"] = value
        return metrics


class ValueTrainer:
    def __init__(
        self,
        model: ValueModel,
        config: ValueConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float,
        weight_decay: float,
        epochs: int,
        checkpoint_dir: str | Path,
        save_payload: Dict[str, Any],
        policy: Optional[NextActionModel] = None,
        device: Optional[torch.device] = None,
        clip_grad_norm: Optional[float] = 1.0,
        train_dataset=None,
        val_returns: Optional[torch.Tensor] = None,
    ):
        self.device = device or get_default_device()
        self.train_dataset = train_dataset
        self.val_returns = val_returns
        self.model = model.to(self.device)
        if policy is not None:
            policy = policy.to(self.device).eval()
            for parameter in policy.parameters():
                parameter.requires_grad_(False)
        self.config = config
        self.learner = ValueLearner(self.model, config, policy)
        if self.learner.target_model is not None:
            self.learner.target_model.to(self.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        encoder_ids = {id(parameter) for parameter in self.model.policy.parameters()}
        encoder = [p for p in self.model.parameters() if p.requires_grad and id(p) in encoder_ids]
        heads = [p for p in self.model.parameters() if p.requires_grad and id(p) not in encoder_ids]
        groups = [{"params": heads, "lr": lr}]
        if encoder:
            groups.append({"params": encoder, "lr": lr * config.encoder_lr_scale})
        self.optimizer = torch.optim.Adam(groups, lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(epochs, 1), eta_min=1e-6)
        self.clip_grad_norm = clip_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_payload = save_payload
        self.current_epoch = 0
        self.best_selection = math.inf

    @staticmethod
    def _split(batch) -> tuple[NextActionBatch, Optional[NextActionBatch], Optional[torch.Tensor]]:
        """Loaders yield (batch, next batch, indices) triples or plain batches."""
        if isinstance(batch, tuple):
            return batch if len(batch) == 3 else (*batch, None)
        return batch, None, None

    def train_epoch(self) -> Dict[str, float]:
        if self.config.target == "lambda_return":
            if self.train_dataset is None:
                raise ValueError("λ-return training needs the training dataset.")
            self.learner.refresh_lambda_targets(self.train_dataset, self.device)
        self.model.train()
        if self.config.freeze_encoder:
            self.model.policy.eval()
        sums: Dict[str, float] = {}
        count = 0
        progress = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]", leave=False, disable=None)
        for raw in progress:
            batch, next_batch, indices = self._split(raw)
            batch = batch.to(self.device)
            next_batch = next_batch.to(self.device) if next_batch is not None else None
            targets = self.learner.targets(batch, next_batch, indices)
            terms = self.learner.loss(self.model(batch), batch, targets)
            self.optimizer.zero_grad()
            terms["total"].backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.learner.update_target()
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach())
            count += 1
            progress.set_postfix({"loss": f"{float(terms['total']):.4f}"}, refresh=False)
        return {f"loss_{name}": value / max(count, 1) for name, value in sums.items()}

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        self.model.eval()
        evaluation = ValueEvaluation(self.model, self.val_returns)
        policy_nll, count = 0.0, 0
        for raw in tqdm(loader or self.val_loader, desc=f"Epoch {self.current_epoch} [Eval]", leave=False, disable=None):
            batch, _, indices = self._split(raw)
            batch = batch.to(self.device)
            outputs = self.model(batch)
            evaluation.update(outputs, batch, indices)
            if "policy" in outputs:
                policy_nll += float(action_log_likelihood(outputs["policy"], batch)["joint_nll"].sum())
                count += batch.batch_size
        metrics = evaluation.compute()
        if count:
            metrics["policy_joint_nll"] = policy_nll / count
        return metrics, evaluation.predictions()

    def _save(self, filename: str) -> None:
        torch.save(
            {
                **self.save_payload,
                "epoch": self.current_epoch,
                "value_config": asdict(self.config),
                "model_state_dict": self.model.state_dict(),
                "best_selection": self.best_selection,
            },
            self.checkpoint_dir / filename,
        )

    def fit(self, epochs: int) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {}
        for _ in range(epochs):
            self.current_epoch += 1
            train_metrics = self.train_epoch()
            val_metrics, _ = self.evaluate()
            self.scheduler.step()
            for source, prefix in ((train_metrics, "train"), (val_metrics, "val")):
                for name, value in source.items():
                    history.setdefault(f"{prefix}_{name}", []).append(value)
            if val_metrics["selection"] < self.best_selection:
                self.best_selection = val_metrics["selection"]
                self._save("best_model.pt")
            self._save("last_model.pt")
            line = (
                f"Epoch {self.current_epoch:03d} | train loss {train_metrics['loss_total']:.4f} | "
                f"val selection {val_metrics['selection']:.4f} | score AUC {val_metrics['score_auc']:.3f} "
                f"concede AUC {val_metrics['concede_auc']:.3f} | protocol AUC {val_metrics.get('protocol_auc', float('nan')):.3f}"
            )
            if "return_selection" in val_metrics:
                line += f" | return selection {val_metrics['return_selection']:.4f}"
            if "distribution_crps" in val_metrics:
                line += f" | C51 CRPS {val_metrics['distribution_crps']:.4f}"
            if "policy_joint_nll" in val_metrics:
                line += f" | policy nll {val_metrics['policy_joint_nll']:.3f}"
            print(line, flush=True)
        (self.checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2))
        return history
