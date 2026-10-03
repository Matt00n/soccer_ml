"""Action-value models on top of the next-action encoder.

Actions live in one flat space shared with the policy, so Q(s, ·), π(·|s),
advantages and expectations are plain tensor operations:

    pass   (height h, zone z) -> h * Z + z                 3 * Z actions
    carry  (zone z)           -> 3 * Z + z                     Z actions
    shot, dribble, clearance  -> 4 * Z, 4 * Z + 1, 4 * Z + 2

Values have two components from the acting team's perspective (VAEP style):
``score`` and ``concede``; the net value is score - concede.  When the other
team makes the next decision, its components swap.

Heads (``head``):

* ``direct``: Q(s, a) for every action.
* ``decomposed``: P(success | s, a), V(s, a | success), V(s, a | failure) with
  Q = P * V_success + (1 - P) * V_failure (un-xPass style).  Pass values
  depend on the zone only, success probabilities also on the height.

Output distribution (``distribution``):

* ``expectation``: sigmoid probabilities.
* ``quantile``: ``num_quantiles`` quantiles of each component's return at the
  fractions (2i + 1) / (2K) (quantile regression, distributional RL).
* ``categorical``: the expectation heads plus a C51 head: a categorical
  distribution of the *net* discounted return (score - concede) over a fixed,
  symmetric support that is dense near zero.  Because the support is
  symmetric, the other team acting next is a reversal of the probability
  vector.  To stay affordable over 1,771 actions, the grid decoder emits a
  small feature per zone (and height) and one shared MLP maps it to atom
  logits, pointwise per action, so distributions stay reflection equivariant.

``state_value`` adds a V(s) head (used by IQL).  All grids are decoded per
reflection view and reflected back before averaging, exactly like the policy
heads, so value surfaces are reflection equivariant.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from soccer_ml.data.next_action_dataset import ACTION_TYPES, IGNORE_INDEX, PASS_HEIGHTS, NextActionBatch
from soccer_ml.models.next_action import NextActionModel

VALUE_COMPONENTS = ("score", "concede")
SCALAR_ACTIONS = ("shot", "dribble", "clearance")
HEADS = ("direct", "decomposed")
DISTRIBUTIONS = ("expectation", "quantile", "categorical")
INITIAL_VALUE_LOGIT = -4.0  # sigmoid(-4) ~ 0.018
INITIAL_DISTRIBUTION_SCALE = 0.02  # initial C51 logits -|z| / scale: Laplace-shaped around 0


def action_space_size(num_zones: int) -> int:
    return len(PASS_HEIGHTS) * num_zones + num_zones + len(SCALAR_ACTIONS)


def observed_action_index(batch: NextActionBatch, num_zones: int) -> torch.Tensor:
    """Flat index of every sample's observed action (passes without a height count as ground passes)."""
    action = batch.action_type
    zone = batch.target_zone.clamp_min(0)
    height = batch.pass_height.clamp_min(0)
    index = torch.empty_like(action)
    is_pass = action == ACTION_TYPES.index("pass")
    is_carry = action == ACTION_TYPES.index("carry")
    index[is_pass] = height[is_pass] * num_zones + zone[is_pass]
    index[is_carry] = len(PASS_HEIGHTS) * num_zones + zone[is_carry]
    for offset, name in enumerate(SCALAR_ACTIONS):
        rows = action == ACTION_TYPES.index(name)
        index[rows] = (len(PASS_HEIGHTS) + 1) * num_zones + offset
    return index


def policy_action_log_probs(policy_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """log π(a | s) over the flat action space from next-action model outputs, ``(B, A)``."""
    log_type = F.log_softmax(policy_outputs["type_logits"], dim=-1)  # (B, types)
    log_zone = F.log_softmax(policy_outputs["zone_logits"], dim=-1)  # (B, 2, Z)
    log_height = F.log_softmax(policy_outputs["height_logits"], dim=1)  # (B, 3, Z)
    passes = log_type[:, ACTION_TYPES.index("pass"), None, None] + log_zone[:, None, 0] + log_height
    carries = log_type[:, ACTION_TYPES.index("carry"), None] + log_zone[:, 1]
    scalars = log_type[:, [ACTION_TYPES.index(name) for name in SCALAR_ACTIONS]]
    return torch.cat([passes.flatten(1), carries, scalars], dim=-1)


def swap_components(values: torch.Tensor, flip: torch.Tensor) -> torch.Tensor:
    """Swap score/concede (dim 1) where the other team acts: ``values`` is (B, 2, ...)."""
    swapped = values.flip(1)
    shape = (-1,) + (1,) * (values.dim() - 1)
    return torch.where(flip.view(shape), swapped, values)


def quantile_fractions(num_quantiles: int, device: Optional[torch.device] = None) -> torch.Tensor:
    return (2 * torch.arange(num_quantiles, device=device, dtype=torch.float32) + 1) / (2 * num_quantiles)


def categorical_support(num_atoms: int = 51, v_max: float = 1.5, density: float = 3.0) -> torch.Tensor:
    """Symmetric atoms in [-v_max, v_max], denser near 0 (sinh spacing), with an exact 0 atom.

    Built as a positive half mirrored onto the negative axis, so the support is
    *exactly* symmetric (a team flip is an exact reversal) despite rounding.
    """
    if num_atoms % 2 == 0:
        raise ValueError("num_atoms must be odd (the support includes 0).")
    half = v_max * torch.sinh(density * torch.linspace(0.0, 1.0, num_atoms // 2 + 1)) / math.sinh(density)
    half[0] = 0.0
    return torch.cat([-half[1:].flip(0), half])


def project_categorical(
    probabilities: torch.Tensor, shifted_atoms: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    """Project a distribution with mass ``probabilities`` (B, K) at ``shifted_atoms`` (B, K) onto ``support`` (K,).

    Generalized C51 projection for a non-uniform support: every shifted atom
    splits its mass between the two neighbouring support atoms in inverse
    proportion to the distance, so total mass and (inside the support) the
    mean are preserved.
    """
    values = shifted_atoms.clamp(float(support[0]), float(support[-1]))
    upper = torch.searchsorted(support, values.contiguous()).clamp(1, len(support) - 1)
    lower = upper - 1
    width = support[upper] - support[lower]
    upper_weight = ((values - support[lower]) / width).clamp(0.0, 1.0)
    projected = torch.zeros_like(probabilities)
    projected.scatter_add_(-1, lower, probabilities * (1.0 - upper_weight))
    projected.scatter_add_(-1, upper, probabilities * upper_weight)
    return projected


class ValueModel(nn.Module):
    """Value heads on a :class:`NextActionModel` encoder.

    ``policy`` supplies the encoder (and, for joint training, the policy heads
    whose outputs are returned under ``"policy"``).
    """

    model_name = "value"

    def __init__(
        self,
        policy: NextActionModel,
        head: str = "direct",
        distribution: str = "expectation",
        num_quantiles: int = 5,
        state_value: bool = False,
        joint: bool = False,
        num_atoms: int = 51,
        v_max: float = 1.5,
        distribution_features: int = 16,
    ):
        super().__init__()
        if head not in HEADS or distribution not in DISTRIBUTIONS:
            raise ValueError(f"head must be one of {HEADS}, distribution one of {DISTRIBUTIONS}.")
        if head == "decomposed" and distribution == "quantile":
            raise ValueError(
                "Quantiles of the decomposed head would be quantiles of a success/failure mixture, "
                "which are not a function of the branch quantiles; use the direct head."
            )
        self.policy = policy
        self.head = head
        self.distribution = distribution
        self.num_quantiles = num_quantiles if distribution == "quantile" else 1
        self.joint = joint
        self.num_zones = policy.num_zones
        self.num_actions = action_space_size(self.num_zones)
        self.value_kwargs = {
            "head": head,
            "distribution": distribution,
            "num_quantiles": num_quantiles,
            "state_value": state_value,
            "joint": joint,
            "num_atoms": num_atoms,
            "v_max": v_max,
            "distribution_features": distribution_features,
        }
        d, z, c, q = policy.readout_dim, self.num_zones, len(VALUE_COMPONENTS), self.num_quantiles
        heights, scalars = len(PASS_HEIGHTS), len(SCALAR_ACTIONS)
        if head == "direct":
            self.pass_q = nn.Linear(d, c * q * heights * z)
            self.carry_q = nn.Linear(d, c * q * z)
            self.scalar_q = nn.Linear(d, c * q * scalars)
        else:
            self.pass_success = nn.Linear(d, heights * z)
            self.carry_success = nn.Linear(d, z)
            self.scalar_success = nn.Linear(d, scalars)
            # Two branches (success, failure) x components.
            self.pass_value = nn.Linear(d, 2 * c * z)
            self.carry_value = nn.Linear(d, 2 * c * z)
            self.scalar_value = nn.Linear(d, 2 * c * scalars)
        self.state_value = nn.Linear(d, c * q) if state_value else None
        if distribution == "categorical":
            f = distribution_features
            self.register_buffer("atoms", categorical_support(num_atoms, v_max))
            self.distribution_located = nn.Linear(d, f * (heights + 1) * z)  # pass heights + carry, per zone
            self.distribution_scalar = nn.Linear(d, f * scalars)
            self.distribution_state = nn.Linear(d, f)
            self.distribution_mlp = nn.Sequential(nn.GELU(), nn.Linear(f, 4 * f), nn.GELU(), nn.Linear(4 * f, num_atoms))
            self.distribution_features = f
            # Start from a distribution peaked at 0 (most returns are exactly 0), not uniform over the
            # atoms: a uniform start is badly calibrated and its large early gradients disturb the encoder.
            with torch.no_grad():
                self.distribution_mlp[-1].weight.mul_(0.1)
                self.distribution_mlp[-1].bias.copy_(-self.atoms.abs() / INITIAL_DISTRIBUTION_SCALE)
        if distribution in ("expectation", "categorical"):
            # Start value probabilities near the goal base rate (~2%), not at 0.5: large early
            # gradients would otherwise disturb a pretrained encoder.
            value_layers = [self.pass_q, self.carry_q, self.scalar_q] if head == "direct" else [
                self.pass_value, self.carry_value, self.scalar_value
            ]
            if self.state_value is not None:
                value_layers.append(self.state_value)
            for layer in value_layers:
                nn.init.constant_(layer.bias, INITIAL_VALUE_LOGIT)

    def _grid(self, layer: nn.Linear, readout: torch.Tensor, channels: int) -> torch.Tensor:
        views, batch_size = readout.shape[:2]
        logits = layer(readout).view(views, batch_size, channels, self.policy.grid_m, self.policy.grid_n)
        return self.policy.decode_grid(logits)  # (B, channels, Z)

    def _actions(self, readout: torch.Tensor, pass_layer, carry_layer, scalar_layer, channels: int, pass_heights: int):
        """Per-action outputs ``(B, channels, A)``; passes get ``pass_heights`` height channels (1 = shared)."""
        batch_size = readout.shape[1]
        z = self.num_zones
        passes = self._grid(pass_layer, readout, channels * pass_heights).view(batch_size, channels, pass_heights, z)
        passes = passes.expand(-1, -1, len(PASS_HEIGHTS), -1).flatten(2)
        carries = self._grid(carry_layer, readout, channels)
        scalars = scalar_layer(readout).mean(dim=0).view(batch_size, channels, len(SCALAR_ACTIONS))
        return torch.cat([passes, carries, scalars], dim=-1)

    def forward(self, batch: NextActionBatch, readout: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        readout = self.policy.encode(batch) if readout is None else readout
        batch_size = readout.shape[1]
        c, q = len(VALUE_COMPONENTS), self.num_quantiles
        outputs: Dict[str, Any] = {}
        if self.head == "direct":
            raw = self._actions(readout, self.pass_q, self.carry_q, self.scalar_q, c * q, len(PASS_HEIGHTS))
            if self.distribution == "quantile":
                outputs["quantiles"] = raw.view(batch_size, c, q, -1)  # (B, C, K, A)
                outputs["q"] = outputs["quantiles"].mean(dim=2)
            else:
                outputs["q_logits"] = raw  # (B, C, A)
                outputs["q"] = torch.sigmoid(raw)
        else:
            success_logits = self._actions(readout, self.pass_success, self.carry_success, self.scalar_success, 1, 3)
            values = self._actions(readout, self.pass_value, self.carry_value, self.scalar_value, 2 * c, 1)
            values = values.view(batch_size, 2, c, -1)
            outputs["success_logits"] = success_logits[:, 0]  # (B, A)
            outputs["v_success_logits"], outputs["v_failure_logits"] = values[:, 0], values[:, 1]  # (B, C, A)
            p = torch.sigmoid(outputs["success_logits"]).unsqueeze(1)
            outputs["q"] = p * torch.sigmoid(values[:, 0]) + (1 - p) * torch.sigmoid(values[:, 1])
        if self.distribution == "categorical":
            f, z = self.distribution_features, self.num_zones
            located = self._grid(self.distribution_located, readout, f * (len(PASS_HEIGHTS) + 1))
            located = located.view(batch_size, f, len(PASS_HEIGHTS) + 1, z).flatten(2)  # (B, F, 4Z)
            scalars = self.distribution_scalar(readout).mean(dim=0).view(batch_size, f, len(SCALAR_ACTIONS))
            outputs["distribution_features"] = torch.cat([located, scalars], dim=-1)  # (B, F, A)
            outputs["distribution_state"] = self.distribution_state(readout).mean(dim=0)  # (B, F)
        if self.state_value is not None:
            state = self.state_value(readout).mean(dim=0).view(batch_size, c, q)
            outputs["state_value_logits"] = state[:, :, 0] if q == 1 else None
            outputs["state_value"] = torch.sigmoid(state[:, :, 0]) if q == 1 else state.mean(dim=2)
        if self.joint:
            outputs["policy"] = self.policy.decode_policy(readout, batch)
        return outputs


def distribution_logits(model: "ValueModel", outputs: Dict[str, Any], action: Optional[torch.Tensor] = None) -> torch.Tensor:
    """C51 atom logits of the net return: at ``action`` (B,) -> (B, K), or for all actions -> (B, A, K)."""
    features = outputs["distribution_features"]  # (B, F, A)
    if action is not None:
        features = gather_action(features, action).unsqueeze(-1)  # (B, F, 1)
    combined = features + outputs["distribution_state"].unsqueeze(-1)  # (B, F, A')
    logits = model.distribution_mlp(combined.transpose(1, 2))  # (B, A', K)
    return logits.squeeze(1) if action is not None else logits


def gather_action(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Select the action ``index`` (B,) from ``values`` (B, ..., A)."""
    shape = values.shape[:-1] + (1,)
    gather_index = index.view((-1,) + (1,) * (values.dim() - 1)).expand(shape)
    return values.gather(-1, gather_index).squeeze(-1)


def is_located_known(batch: NextActionBatch) -> torch.Tensor:
    """Samples whose action can be placed in the flat action space."""
    action = batch.action_type
    located = (action == ACTION_TYPES.index("pass")) | (action == ACTION_TYPES.index("carry"))
    return ~located | (batch.target_zone != IGNORE_INDEX)
