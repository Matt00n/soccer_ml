"""Candidate creativity measures, all computed from a decision table (``scripts/creativity/score_decisions.py``).

Creativity = originality x effectiveness (Runco & Jaeger's standard
definition; multiplicative as in Simonton 2012).  The candidates differ in
how originality is measured and how value enters:

baselines (one component only; a creativity measure must beat them)
    ``rarity``          originality only: mid-PIT rarity of the chosen action
    ``advantage``       value only: Q(s, a) - V(s)
    ``difficulty``      difficult *completed* actions: 1 - P(success), if successful

products (rarity x value)
    ``surprise_x_adv``  -log π(a) x max(A, 0)
    ``rarity_x_adv``    rarity x max(A, 0)
    ``rarity_x_adv_completed`` the same for completed actions only (0 if failed):
                        ex-post, closer to un-xPass, which scores successful passes
    ``rarity_x_td``     rarity x max(realized one-step advantage, 0): uses the
                        state actually reached, so it cannot exploit Q's
                        extrapolation to rare actions
    ``rarity_x_upside`` rarity x E[(return - V)+]: rewards a positive tail even
                        when the mean advantage is negative

rarity among valuable options (0 if the chosen action is not in the set)
    ``good_rarity``     rarity within the actions with positive advantage
    ``good_surprise``   -log π(a | positive-advantage actions)
    ``tail_rarity``     rarity within the actions whose success branch beats V(s)
    ``c51_tail_rarity`` rarity within the actions whose upper return quantile beats the
                        state's (a heavier right tail than the average option; C51 models)
    ``unxpass``         passes only: rarity among passes if A > 0 (un-xPass style)

Every measure is "higher = more creative" and NaN outside its scope (or when
the table lacks its inputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Measure:
    name: str
    family: str  # "baseline", "product" or "set"
    description: str
    requires: tuple[str, ...]
    compute: Callable[[pd.DataFrame], pd.Series]


def _positive(values: pd.Series) -> pd.Series:
    return values.clip(lower=0.0)


def _in_set(table: pd.DataFrame, name: str) -> pd.Series:
    return table[f"{name}_rarity"].where(table[f"in_{name}"] > 0.5, 0.0)


MEASURES: tuple[Measure, ...] = (
    Measure("rarity", "baseline", "originality only: rarity (mid-PIT) of the chosen action", ("rarity",),
            lambda t: t["rarity"]),
    Measure("advantage", "baseline", "value only: Q(s,a) - V(s)", ("advantage",), lambda t: t["advantage"]),
    Measure("difficulty", "baseline", "1 - P(success) of completed actions, 0 if failed", ("p_success", "success"),
            lambda t: (1.0 - t["p_success"]).where(t["success"] == 1, 0.0).where(t["success"].notna())),
    Measure("surprise_x_adv", "product", "-log π(a) x max(A, 0)", ("surprise", "advantage"),
            lambda t: t["surprise"] * _positive(t["advantage"])),
    Measure("rarity_x_adv", "product", "rarity x max(A, 0)", ("rarity", "advantage"),
            lambda t: t["rarity"] * _positive(t["advantage"])),
    Measure("rarity_x_adv_completed", "product", "rarity x max(A, 0), completed actions only", (
        "rarity", "advantage", "success"), lambda t: (t["rarity"] * _positive(t["advantage"])).where(
        t["success"] == 1, 0.0).where(t["success"].notna())),
    Measure("rarity_x_td", "product", "rarity x max(r + γV(s') - V(s), 0) (realized)", ("rarity", "td_advantage"),
            lambda t: t["rarity"] * _positive(t["td_advantage"])),
    Measure("rarity_x_upside", "product", "rarity x E[(return - V)+]", ("rarity", "upside"),
            lambda t: t["rarity"] * t["upside"]),
    Measure("good_rarity", "set", "rarity among positive-advantage actions", ("good_rarity", "in_good"),
            lambda t: _in_set(t, "good")),
    Measure("good_surprise", "set", "-log π(a | positive-advantage actions)", ("good_surprise", "in_good"),
            lambda t: t["good_surprise"].where(t["in_good"] > 0.5, 0.0)),
    Measure("tail_rarity", "set", "rarity among actions whose success branch beats V(s)", ("tail_rarity", "in_tail"),
            lambda t: _in_set(t, "tail")),
    Measure("c51_tail_rarity", "set", "rarity among actions with a heavier right tail than the state (C51)",
            ("c51_tail_rarity", "in_c51_tail"), lambda t: _in_set(t, "c51_tail")),
    Measure("unxpass", "set", "passes: rarity among passes if A > 0", ("pass_rarity", "advantage"),
            lambda t: t["pass_rarity"].where(t["advantage"] > 0, 0.0).where(t["action_type"] == "pass")),
)
MEASURE_NAMES = tuple(measure.name for measure in MEASURES)
CREATIVITY_MEASURES = tuple(measure.name for measure in MEASURES if measure.family != "baseline")


def available_measures(table: pd.DataFrame, names: Optional[tuple[str, ...]] = None) -> list[Measure]:
    """Measures whose inputs are all present (and not entirely missing) in ``table``."""
    chosen = [m for m in MEASURES if names is None or m.name in names]
    return [m for m in chosen if all(c in table and table[c].notna().any() for c in m.requires)]


def compute_measures(table: pd.DataFrame, names: Optional[tuple[str, ...]] = None) -> pd.DataFrame:
    """One column per available measure (``table``'s index)."""
    return pd.DataFrame({m.name: m.compute(table).astype(float) for m in available_measures(table, names)},
                        index=table.index)


def top_fraction_flags(scores: pd.DataFrame, fraction: float = 0.01) -> pd.DataFrame:
    """Per measure: is the decision in the measure's top ``fraction`` (among in-scope decisions)?"""
    flags = {}
    for name, values in scores.items():
        threshold = values.quantile(1.0 - fraction)
        flags[name] = (values >= threshold) & (values > 0) & values.notna()
    return pd.DataFrame(flags, index=scores.index)


def player_scores(table: pd.DataFrame, scores: pd.DataFrame, min_decisions: int = 200) -> pd.DataFrame:
    """Mean measure per player over in-scope decisions (players with at least ``min_decisions``)."""
    keys = ["player_id", "player"]
    counts = table.groupby(keys, sort=False).size().rename("decisions")
    means = scores.groupby([table[k] for k in keys], sort=False).mean()
    result = means.join(counts)
    positions = table.groupby(keys, sort=False)["position"].agg(lambda p: p.mode().iat[0] if len(p) else "")
    teams = table.groupby(keys, sort=False)["team"].agg(lambda p: p.mode().iat[0] if len(p) else "")
    result = result.join(positions.rename("position")).join(teams.rename("team"))
    return result[result["decisions"] >= min_decisions]


POSITION_GROUPS = {
    "Goalkeeper": "GK",
    "Center Back": "CB", "Left Center Back": "CB", "Right Center Back": "CB",
    "Left Back": "FB", "Right Back": "FB", "Left Wing Back": "FB", "Right Wing Back": "FB",
    "Center Defensive Midfield": "DM", "Left Defensive Midfield": "DM", "Right Defensive Midfield": "DM",
    "Center Midfield": "CM", "Left Center Midfield": "CM", "Right Center Midfield": "CM",
    "Center Attacking Midfield": "AM", "Left Attacking Midfield": "AM", "Right Attacking Midfield": "AM",
    "Left Midfield": "W", "Right Midfield": "W", "Left Wing": "W", "Right Wing": "W",
    "Center Forward": "FW", "Left Center Forward": "FW", "Right Center Forward": "FW", "Secondary Striker": "FW",
}


def position_group(position: str) -> str:
    return POSITION_GROUPS.get(position, "other")


def rank_percentile_within(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Percentile rank (0-1, higher = more creative) within each group."""
    return values.groupby(groups).rank(pct=True)


def zscore(values: pd.Series) -> pd.Series:
    spread = values.std()
    return (values - values.mean()) / spread if spread and np.isfinite(spread) else values * 0.0
