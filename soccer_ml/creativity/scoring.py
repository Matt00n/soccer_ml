"""Per-decision originality and value statistics, the inputs of every creativity measure.

The policy π(·|s) (next-action model) and the action values Q(s, ·) (value
model) are evaluated over the flat action space of :mod:`soccer_ml.models.value`
(pass height x zone, carry zone, shot, dribble, clearance) and reduced to
scalars per decision, so that measures can be defined and compared cheaply
afterwards (:mod:`soccer_ml.creativity.measures`).

Originality of the chosen action a:

* ``surprise`` = -log π(a), with its type / zone / height factors;
* ``rarity`` = π(actions rated more likely than a) + π(a) / 2, a mid-PIT in
  [0, 1] that is independent of how spread out the options are; under a
  calibrated policy it is (close to) uniform for every kind of situation.
  ``rarity_random`` replaces the 1/2 by a uniform draw (exactly uniform under
  calibration; used to test calibration).

Values (net = score - concede, from the actor's perspective):

* ``q`` = Q(s, a), ``v`` = V(s) = E_π Q(s, ·), ``advantage`` = q - v;
* ``better_mass``: policy mass on actions with a higher Q than a;
* decomposed heads: success probability and branch values of a, and the
  ``upside`` E[(return - V)+] of the two-point success/failure mixture;
* C51 heads: mean, 10% / 90% quantiles, P(return > V) and upside of a, and
  ``c51_tail_advantage``: its ``tail_quantile`` quantile minus the state's.

Option sets.  For a set S of actions, ``<S>_mass`` = π(S), ``in_<S>`` =
whether a is in S, and (if it is) the rarity and surprise of a *within* S
(π restricted to S and renormalized):

* ``good``: positive advantage, Q(s, a') > V(s);
* ``tail``: the success branch beats the state value, V_success(a') > V(s)
  (an action with a positive tail; decomposed heads);
* ``c51_tail``: the action's ``tail_quantile`` return quantile beats that of the
  state's return distribution under π (a heavier right tail than the average
  option; C51).  Comparing with V(s) instead is uninformative: almost every
  action's 90% quantile exceeds the mean.
* ``pass``: passes (un-xPass's selection-within-passes view).

Calibration: ``calib_mass_<k>`` is the policy mass in log10-probability bin k
and ``calib_bin`` the bin of the chosen action; summed over decisions they give
expected vs. observed counts per probability band, down to the rare tail.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from soccer_ml.data.next_action_dataset import IGNORE_INDEX, NextActionBatch
from soccer_ml.data.schema import NodeLayout
from soccer_ml.models.next_action import action_log_likelihood
from soccer_ml.models.value import (
    PASS_HEIGHTS,
    ValueModel,
    distribution_logits,
    gather_action,
    observed_action_index,
    policy_action_log_probs,
)

# log10 π bin edges for the tail calibration table: bins (-inf, -6], (-6, -5], ..., (-0.5, 0].
CALIBRATION_EDGES = (-6.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5)
NUM_CALIBRATION_BINS = len(CALIBRATION_EDGES) + 1


def calibration_bin_labels() -> list[str]:
    edges = ("-inf",) + tuple(f"{edge:g}" for edge in CALIBRATION_EDGES) + ("0",)
    return [f"({low}, {high}]" for low, high in zip(edges, edges[1:])]


def set_rarity(
    pi: torch.Tensor, log_pi: torch.Tensor, member: Optional[torch.Tensor], observed: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Mass of a set S under π, and the chosen action's rarity and surprise within S.

    ``pi``/``log_pi``/``member``: (B, A); ``member`` None means all actions.
    Rarity and surprise are NaN where the chosen action is outside S.
    """
    p_obs = gather_action(pi, observed)
    log_p_obs = gather_action(log_pi, observed)
    weights = pi if member is None else pi * member
    mass = weights.sum(-1)
    above = (weights * (pi > p_obs.unsqueeze(-1))).sum(-1)
    inside = torch.ones_like(p_obs, dtype=torch.bool) if member is None else gather_action(member, observed)
    valid = inside & (mass > 0)
    nan = torch.full_like(p_obs, float("nan"))
    safe_mass = mass.clamp_min(1e-12)
    return {
        "mass": mass,
        "inside": inside.float(),
        "rarity": torch.where(valid, (above + 0.5 * p_obs) / safe_mass, nan),
        "surprise": torch.where(valid, safe_mass.log() - log_p_obs, nan),
        "above": above,
        "p_obs": p_obs,
    }


def _quantile(cdf: torch.Tensor, atoms: torch.Tensor, level: float) -> torch.Tensor:
    """Atom at which a categorical CDF (..., K) first reaches ``level``."""
    index = (cdf < level).sum(-1).clamp_max(atoms.numel() - 1)
    return atoms[index]


def _opponents_bypassed(batch: NextActionBatch, layout: NodeLayout, pitch_length: float, target_scale: float) -> torch.Tensor:
    """Visible outfield opponents between the actor and the action's end, along the attacking direction.

    A simple, model-free line-breaking count ("packing"); NaN for actions
    without an end location, 0 for backward actions.
    """
    nodes, mask = batch.node_features, batch.player_mask
    x = nodes[..., layout.x] / target_scale * pitch_length + pitch_length / 2
    actor = (nodes[..., layout.actor] > 0.5) & mask
    start_x = (x * actor).sum(-1) / actor.sum(-1).clamp_min(1)
    end_x = batch.end_location[:, 0]
    opponents = mask & (nodes[..., layout.teammate] < 0.5) & (nodes[..., layout.keeper] < 0.5)
    between = opponents & (x > start_x.unsqueeze(-1)) & (x < end_x.unsqueeze(-1))
    located = batch.target_zone != IGNORE_INDEX
    count = between.sum(-1).float()
    return torch.where(located, count, torch.full_like(count, float("nan")))


@torch.no_grad()
def decision_statistics(
    policy_outputs: Dict[str, torch.Tensor],
    value_outputs: Dict[str, torch.Tensor],
    batch: NextActionBatch,
    value_model: ValueModel,
    node_layout: NodeLayout,
    tail_quantile: float = 0.9,
    generator: Optional[torch.Generator] = None,
    pitch_length: float = 120.0,
    target_scale: float = 10.0,
) -> Dict[str, torch.Tensor]:
    """All per-decision statistics of one batch as (B,) tensors (see the module docstring)."""
    z = value_model.num_zones
    log_pi = policy_action_log_probs(policy_outputs)  # (B, A)
    pi = log_pi.exp()
    observed = observed_action_index(batch, z)
    stats: Dict[str, torch.Tensor] = {}

    # -- originality ------------------------------------------------------------
    nll = action_log_likelihood(policy_outputs, batch)
    full = set_rarity(pi, log_pi, None, observed)
    stats["surprise"] = full["surprise"]
    stats["surprise_type"] = nll["type_nll"]
    stats["surprise_zone"] = nll["zone_nll"]
    stats["surprise_height"] = nll["height_nll"]
    stats["probability"] = full["p_obs"]
    stats["entropy"] = -(pi * log_pi).sum(-1)
    stats["top_probability"] = pi.max(-1).values
    stats["rarity"] = full["rarity"]
    uniform = torch.rand(pi.shape[0], generator=generator).to(pi.device)
    stats["rarity_random"] = full["above"] + uniform * full["p_obs"]
    edges = torch.tensor(CALIBRATION_EDGES, device=pi.device) * math.log(10.0)
    bins = torch.bucketize(log_pi, edges)  # (B, A): bin k covers (edge[k-1], edge[k]]
    stats["calib_bin"] = gather_action(bins, observed).float()
    mass = torch.zeros(pi.shape[0], NUM_CALIBRATION_BINS, device=pi.device).scatter_add_(1, bins, pi)
    for k in range(NUM_CALIBRATION_BINS):
        stats[f"calib_mass_{k}"] = mass[:, k]

    # -- values -----------------------------------------------------------------
    q = value_outputs["q"]  # (B, 2, A)
    q_net = q[:, 0] - q[:, 1]
    v = (pi.unsqueeze(1) * q).sum(-1)  # (B, 2)
    v_net = v[:, 0] - v[:, 1]
    q_obs = gather_action(q, observed)  # (B, 2)
    stats["q_score"], stats["q_concede"] = q_obs[:, 0], q_obs[:, 1]
    stats["q"] = q_obs[:, 0] - q_obs[:, 1]
    stats["v_score"], stats["v_concede"] = v[:, 0], v[:, 1]
    stats["v"] = v_net
    stats["advantage"] = stats["q"] - v_net
    stats["better_mass"] = (pi * (q_net > stats["q"].unsqueeze(-1))).sum(-1)

    def add_set(name: str, member: torch.Tensor) -> None:
        result = set_rarity(pi, log_pi, member, observed)
        stats[f"{name}_mass"] = result["mass"]
        stats[f"in_{name}"] = result["inside"]
        stats[f"{name}_rarity"] = result["rarity"]
        stats[f"{name}_surprise"] = result["surprise"]

    add_set("good", q_net > v_net.unsqueeze(-1))
    is_pass = torch.arange(pi.shape[-1], device=pi.device) < len(PASS_HEIGHTS) * z
    add_set("pass", is_pass.unsqueeze(0).expand_as(pi))

    if "success_logits" in value_outputs:
        p = torch.sigmoid(value_outputs["success_logits"])  # (B, A)
        v_success = torch.sigmoid(value_outputs["v_success_logits"])  # (B, 2, A)
        v_failure = torch.sigmoid(value_outputs["v_failure_logits"])
        success_net = v_success[:, 0] - v_success[:, 1]
        failure_net = v_failure[:, 0] - v_failure[:, 1]
        excess_success = (success_net - v_net.unsqueeze(-1)).clamp_min(0)
        excess_failure = (failure_net - v_net.unsqueeze(-1)).clamp_min(0)
        upside = p * excess_success + (1 - p) * excess_failure
        stats["p_success"] = gather_action(p, observed)
        stats["v_success"] = gather_action(success_net, observed)
        stats["v_failure"] = gather_action(failure_net, observed)
        stats["upside"] = gather_action(upside, observed)
        stats["upside_policy"] = (pi * upside).sum(-1)
        add_set("tail", success_net > v_net.unsqueeze(-1))

    if "distribution_features" in value_outputs:
        atoms = value_model.atoms
        probabilities = torch.softmax(distribution_logits(value_model, value_outputs), dim=-1)  # (B, A, K)
        rows = torch.arange(pi.shape[0], device=pi.device)
        chosen = probabilities[rows, observed]  # (B, K)
        cdf = chosen.cumsum(-1)
        stats["c51_mean"] = (chosen * atoms).sum(-1)
        stats["c51_q10"] = _quantile(cdf, atoms, 0.1)
        stats["c51_q90"] = _quantile(cdf, atoms, 0.9)
        stats["c51_p_better"] = (chosen * (atoms > v_net.unsqueeze(-1))).sum(-1)
        stats["c51_upside"] = (chosen * (atoms - v_net.unsqueeze(-1)).clamp_min(0)).sum(-1)
        tail_value = _quantile(probabilities.cumsum(-1), atoms, tail_quantile)  # (B, A)
        mixture = (pi.unsqueeze(-1) * probabilities).sum(1)  # the state's return distribution under π, (B, K)
        state_tail = _quantile(mixture.cumsum(-1), atoms, tail_quantile)
        stats["c51_tail_advantage"] = gather_action(tail_value, observed) - state_tail
        add_set("c51_tail", tail_value > state_tail.unsqueeze(-1))

    # -- situation ----------------------------------------------------------------
    located = batch.target_zone != IGNORE_INDEX
    zone_visible = batch.visible_zones.float().gather(1, batch.target_zone.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    stats["end_zone_visible"] = torch.where(located, zone_visible, torch.full_like(zone_visible, float("nan")))
    stats["opponents_bypassed"] = _opponents_bypassed(batch, node_layout, pitch_length, target_scale)
    return {name: value.float().cpu() for name, value in stats.items()}


def action_name(index: int, num_zones: int) -> str:
    """Readable flat action: 'pass/High Pass/zone 123', 'carry/zone 40', 'shot', ..."""
    if index < len(PASS_HEIGHTS) * num_zones:
        return f"pass/{PASS_HEIGHTS[index // num_zones]}/zone {index % num_zones}"
    index -= len(PASS_HEIGHTS) * num_zones
    if index < num_zones:
        return f"carry/zone {index}"
    return ("shot", "dribble", "clearance")[index - num_zones]

