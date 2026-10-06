"""Second-order creativity analyses: sequences, game state, personnel, time, opponents, player profiles.

Inputs: a decision table (``score_decisions.py``) joined with the match context
(``context.py``) and the creativity measures as z-scores (decision level, in
scope).  Every analysis returns a dict of tables plus, where it tests a main
hypothesis, a :class:`Hypothesis` (estimate, 95% interval, p-value) on the
primary measure; ``run_second_order.py`` stores everything and corrects the
main hypotheses for multiple testing (Benjamini-Hochberg).

Statistics:

* intervals and p-values come from a match-clustered bootstrap (decisions,
  possessions and events of one match are dependent); p = 2 min(P(b <= 0), P(b >= 0));
* regressions with fixed effects demean within groups (strata, teams) once and
  bootstrap the matches' sufficient statistics;
* event studies compare creativity in a window after an event with the window
  before it, minus the same difference in placebo windows (same period, match
  clock within +-3 minutes, no such event nearby), which removes the normal
  within-match time profile.

Caveat for the interpretation: the policy and value models condition on goal
difference and match minute, so creativity is measured *relative to what is
typical in that score and time*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from soccer_ml.creativity.measures import position_group

Tables = Dict[str, pd.DataFrame]


def raw(measure: str) -> str:
    """Column of a measure's raw (unstandardized) values; measures themselves are z-scored ``z_<name>`` columns."""
    return "raw_" + (measure[2:] if measure.startswith("z_") else measure)


@dataclass
class Hypothesis:
    key: str
    description: str
    estimate: float
    low: float
    high: float
    p: float
    n: int
    unit: str


# -- statistics ---------------------------------------------------------------------------------------


def bootstrap_summary(estimate: float, draws: np.ndarray) -> tuple[float, float, float]:
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return float("nan"), float("nan"), float("nan")
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), float(min(p, 1.0))


def _cluster_weights(n_clusters: int, bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=bootstrap)  # (B, clusters)


def cluster_mean(values: pd.Series, clusters: pd.Series, bootstrap: int = 1000, seed: int = 0) -> tuple:
    """Mean with a cluster-bootstrap interval and p-value (H0: mean = 0)."""
    data = pd.DataFrame({"v": values, "c": clusters}).dropna()
    if data.empty:
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    sums = data.groupby("c")["v"].agg(["sum", "count"])
    w = _cluster_weights(len(sums), bootstrap, np.random.default_rng(seed))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (w @ sums["sum"].to_numpy()) / (w @ sums["count"].to_numpy())
    estimate = float(data["v"].mean())
    return (estimate, *bootstrap_summary(estimate, draws), len(data))


def demean(values: pd.DataFrame, groups) -> pd.DataFrame:
    return values - values.groupby(groups).transform("mean")


def fe_slope(y: pd.Series, x: pd.Series, strata, clusters: pd.Series, bootstrap: int = 1000, seed: int = 0) -> tuple:
    """Slope of y on x within strata (fixed effects), with a cluster bootstrap."""
    data = pd.DataFrame({"y": y, "x": x, "c": clusters, "s": strata}).dropna()
    tilde = demean(data[["y", "x"]], data["s"])
    stats_ = pd.DataFrame({"xy": tilde["x"] * tilde["y"], "xx": tilde["x"] ** 2, "c": data["c"]}).groupby("c").sum()
    w = _cluster_weights(len(stats_), bootstrap, np.random.default_rng(seed))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (w @ stats_["xy"].to_numpy()) / (w @ stats_["xx"].to_numpy())
    estimate = float(stats_["xy"].sum() / stats_["xx"].sum())
    return (estimate, *bootstrap_summary(estimate, draws), len(data))


def fe_regression(
    data: pd.DataFrame, outcome: str, regressors: List[str], groups: Optional[str], cluster: str,
    bootstrap: int = 1000, seed: int = 0,
) -> pd.DataFrame:
    """OLS of ``outcome`` on ``regressors`` with group fixed effects; match-cluster bootstrap per coefficient."""
    data = data.dropna(subset=[outcome] + regressors).reset_index(drop=True)
    if groups:
        sizes = data.groupby(groups)[outcome].transform("size")
        data = data[sizes >= 2].reset_index(drop=True)

    def fit(frame: pd.DataFrame) -> np.ndarray:
        values = frame[[outcome] + regressors].astype(float)
        values = demean(values, frame[groups]) if groups else values - values.mean()
        coef, *_ = np.linalg.lstsq(values[regressors].to_numpy(), values[outcome].to_numpy(), rcond=None)
        return coef

    estimate = fit(data)
    clusters = data[cluster].unique()
    index = data.groupby(cluster).indices
    rng = np.random.default_rng(seed)
    draws = np.array([fit(data.iloc[np.concatenate([index[c] for c in rng.choice(clusters, len(clusters))])]
                          .reset_index(drop=True)) for _ in range(bootstrap)])
    rows = []
    for k, name in enumerate(regressors):
        low, high, p = bootstrap_summary(estimate[k], draws[:, k])
        rows.append({"outcome": outcome, "regressor": name, "estimate": estimate[k], "low": low, "high": high, "p": p,
                     "n": len(data), "clusters": len(clusters)})
    return pd.DataFrame(rows)


def gini(values: np.ndarray) -> float:
    values = np.sort(np.clip(values, 0, None))
    if values.sum() <= 0:
        return float("nan")
    n = len(values)
    return float((2 * np.arange(1, n + 1) - n - 1) @ values / (n * values.sum()))


# -- event studies ------------------------------------------------------------------------------------


class Windows:
    """Fast window means of a decision-level value per (match, team, period)."""

    def __init__(self, d: pd.DataFrame, value: str):
        self.groups = {}
        for key, part in d[["match_id", "team", "period_ctx", "clock", "player_id", value]].dropna(
                subset=[value]).groupby(["match_id", "team", "period_ctx"]):
            part = part.sort_values("clock")
            self.groups[key] = (part["clock"].to_numpy(), part[value].to_numpy(), part["player_id"].to_numpy())
        self.keys = list(self.groups)

    def mean(self, key, start: float, end: float, exclude: tuple = ()) -> tuple[float, int]:
        if key not in self.groups:
            return float("nan"), 0
        clock, values, players = self.groups[key]
        lo, hi = np.searchsorted(clock, start, "left"), np.searchsorted(clock, end, "left")
        chosen = values[lo:hi]
        if exclude:
            chosen = chosen[~np.isin(players[lo:hi], exclude)]
        return (float(chosen.mean()) if len(chosen) else float("nan")), len(chosen)

    def change(self, key, t: float, width: float, exclude: tuple = (), minimum: int = 5) -> float:
        before, n_before = self.mean(key, t - width, t, exclude)
        after, n_after = self.mean(key, t, t + width, exclude)
        return after - before if n_before >= minimum and n_after >= minimum else float("nan")


def event_study(
    windows: Windows, events: pd.DataFrame, team_column: str, avoid: pd.DataFrame, width: float = 10.0,
    placebos: int = 20, exclude_column: Optional[str] = None, seed: int = 0,
) -> pd.DataFrame:
    """Per event: change after vs. before in the team's mean value, and the placebo change at the same clock.

    ``avoid`` (match_id, period, clock): events of the studied kind; placebo
    windows must not contain one within ``width`` minutes.
    """
    rng = np.random.default_rng(seed)
    avoid_times = {key: part["clock"].to_numpy() for key, part in avoid.groupby(["match_id", "period"])}
    by_period: Dict[int, list] = {}
    for key in windows.keys:
        by_period.setdefault(key[2], []).append(key)
    rows = []
    for event in events.itertuples(index=False):
        key = (event.match_id, getattr(event, team_column), event.period)
        exclude = tuple(x for x in (getattr(event, exclude_column) if exclude_column else ()) if x == x)
        delta = windows.change(key, event.clock, width, exclude)
        placebo_values = []
        candidates = by_period.get(event.period, [])
        for _ in range(placebos * 5):
            if len(placebo_values) >= placebos or not candidates:
                break
            other = candidates[rng.integers(len(candidates))]
            t = event.clock + rng.uniform(-3, 3)
            nearby = avoid_times.get((other[0], other[2]), np.empty(0))
            if other[0] == event.match_id or (np.abs(nearby - t) < width).any():
                continue
            value = windows.change(other, t, width)
            if value == value:
                placebo_values.append(value)
        placebo = float(np.mean(placebo_values)) if placebo_values else float("nan")
        rows.append({**event._asdict(), "delta": delta, "placebo": placebo, "effect": delta - placebo})
    return pd.DataFrame(rows)


def summarize_effect(per_event: pd.DataFrame, label: str, bootstrap: int = 1000, by: Optional[str] = None) -> pd.DataFrame:
    groups = [("all", per_event)] + (list(per_event.groupby(by)) if by else [])
    rows = []
    for name, part in groups:
        estimate, low, high, p, n = cluster_mean(part["effect"], part["match_id"], bootstrap)
        rows.append({"analysis": label, "group": name, "effect": estimate, "low": low, "high": high, "p": p,
                     "events": n, "raw_delta": part["delta"].mean(), "placebo_delta": part["placebo"].mean()})
    return pd.DataFrame(rows)


# -- 1. surprise premium ----------------------------------------------------------------------------


def surprise_premium(d: pd.DataFrame, bootstrap: int) -> tuple[Tables, List[Hypothesis]]:
    """Does originality yield more realized value than the value model predicts?

    Residuals: one-step (r + γV(s') - Q) and Monte Carlo (return - Q).  Their
    slope on rarity within Q deciles (fixed effects), for all in-scope
    decisions and for positive-advantage decisions (where creativity lives).
    """
    data = d[d["in_scope"]].copy()
    data["res_td"] = data["td_advantage"] - data["advantage"]
    data["res_mc"] = data["mc_return"] - data["q"]
    data["q_decile"] = pd.qcut(data["q"].rank(method="first"), 10, labels=False)
    data["rarity_decile"] = pd.qcut(data["rarity"].rank(method="first"), 10, labels=False)
    rows, hypotheses = [], []
    for subset, mask in (("all decisions", slice(None)), ("positive advantage", data["advantage"] > 0)):
        part = data.loc[mask] if not isinstance(mask, slice) else data
        for residual in ("res_td", "res_mc"):
            estimate, low, high, p, n = fe_slope(part[residual], part["rarity"], part["q_decile"], part["match_id"], bootstrap)
            rows.append({"subset": subset, "residual": residual, "slope_per_unit_rarity": estimate, "low": low,
                         "high": high, "p": p, "decisions": n})
            # Main test on all decisions: selecting on advantage > 0 selects on Q itself, which builds regression
            # to the mean into the residual (strongest for rare actions, whose Q is noisiest).
            if subset == "all decisions" and residual == "res_td":
                hypotheses.append(Hypothesis("H01_surprise_premium", "realized minus predicted value rises with rarity "
                                             "(all in-scope decisions, within Q deciles)", estimate, low, high, p, n,
                                             "goal-probability units per unit rarity"))
    within = demean(data[["res_td", "res_mc"]], data["q_decile"]).assign(rarity_decile=data["rarity_decile"],
                                                                         good=data["advantage"] > 0)
    by_decile = within.groupby(["good", "rarity_decile"])[["res_td", "res_mc"]].agg(["mean", "count"]).reset_index()
    by_decile.columns = ["_".join(c).strip("_") for c in by_decile.columns]
    return {"slopes": pd.DataFrame(rows), "residual_by_rarity_decile": by_decile}, hypotheses


# -- 2. sequences --------------------------------------------------------------------------------------


def possession_table(d: pd.DataFrame, possessions: pd.DataFrame, measure: str, threshold: float) -> pd.DataFrame:
    """One row per possession: creativity of the possession team's decisions (shots excluded) and outcomes."""
    meta = possessions.rename(columns={"possession_team": "team"})[["match_id", "possession", "team", "shots", "xg", "goal"]]
    data = d.merge(meta[["match_id", "possession", "team"]], on=["match_id", "possession", "team"])
    data = data.sort_values(["match_id", "possession", "period_ctx", "clock"])
    first = data.groupby(["match_id", "possession"]).first()[["v", "start_x", "play_pattern"]]
    build = data[(data["action_type"] != "shot") & data[measure].notna()]
    creativity = build.groupby(["match_id", "possession"])[raw(measure)]
    features = pd.DataFrame({
        "decisions": data.groupby(["match_id", "possession"]).size(),
        "max_creativity": creativity.max(), "sum_creativity": creativity.sum(),
        "creative_decisions": (build[raw(measure)] >= threshold).groupby([build["match_id"], build["possession"]]).sum(),
        "joint_surprise": data.groupby(["match_id", "possession"])["surprise"].sum(),
        "mean_rarity": data.groupby(["match_id", "possession"])["rarity"].mean(),
    }).join(first)
    table = features.reset_index().merge(meta, on=["match_id", "possession"])
    table["shot"] = table["shots"] > 0
    table["v_decile"] = pd.qcut(table["v"].rank(method="first"), 10, labels=False)
    table["third"] = pd.cut(table["start_x"], [-1, 40, 80, 121], labels=["own", "middle", "final"])
    table["length_bin"] = pd.cut(table["decisions"], [0, 1, 2, 4, 7, 1000], labels=["1", "2", "3-4", "5-7", "8+"])
    table["stratum"] = (table["v_decile"].astype(str) + "|" + table["third"].astype(str) + "|"
                        + table["play_pattern"].astype(str) + "|" + table["length_bin"].astype(str))
    return table


def sequences(d: pd.DataFrame, possessions: pd.DataFrame, measures: List[str], bootstrap: int) -> tuple[Tables, List[Hypothesis]]:
    tables: Tables = {}
    hypotheses: List[Hypothesis] = []
    rows = []
    for measure in measures:
        threshold = d[raw(measure)].quantile(0.9)
        table = possession_table(d, possessions, measure, threshold)
        if measure == measures[0]:
            tables["possessions"] = table
        for feature in ("max_creativity", "sum_creativity", "creative_decisions"):
            z = (table[feature] - table[feature].mean()) / table[feature].std()
            for outcome in ("shot", "goal", "xg"):
                estimate, low, high, p, n = fe_slope(table[outcome].astype(float), z, table["stratum"],
                                                     table["match_id"], bootstrap)
                rows.append({"measure": measure, "feature": feature, "outcome": outcome, "per_sd": estimate,
                             "low": low, "high": high, "p": p, "possessions": n,
                             "outcome_rate": float(table[outcome].astype(float).mean())})
                if measure == measures[0] and feature == "max_creativity" and outcome == "shot":
                    hypotheses.append(Hypothesis("H02_sequence_shots", "possessions with more build-up creativity end in "
                                                 "shots more often (same start value, zone, pattern, length)",
                                                 estimate, low, high, p, n, "shot probability per SD"))
                if measure == measures[0] and feature == "max_creativity" and outcome == "goal":
                    hypotheses.append(Hypothesis("H03_sequence_goals", "... and in goals", estimate, low, high, p, n,
                                                 "goal probability per SD"))
    tables["possession_outcomes"] = pd.DataFrame(rows)

    # Chain position: creativity by position from the end of the possession.
    primary = measures[0]
    meta = possessions.rename(columns={"possession_team": "team"})[["match_id", "possession", "team", "shots", "goal"]]
    data = d.merge(meta, on=["match_id", "possession", "team"]).sort_values(["match_id", "possession", "period_ctx", "clock"])
    data["from_end"] = data.groupby(["match_id", "possession"]).cumcount(ascending=False) + 1
    data["length"] = data.groupby(["match_id", "possession"])["clock"].transform("size")
    data = data[data["length"] >= 3]
    data["ending"] = np.where(data["goal"], "goal", np.where(data["shots"] > 0, "shot, no goal", "no shot"))
    data["position"] = data["from_end"].clip(upper=6).map({1: "last", 2: "2nd last", 3: "3rd last", 4: "4th last",
                                                           5: "5th last", 6: "earlier"})
    rows = []
    for (ending, position), part in data.groupby(["ending", "position"]):
        for measure in measures:
            estimate, low, high, p, n = cluster_mean(part[measure], part["match_id"], bootstrap)
            rows.append({"measure": measure, "ending": ending, "position": position, "mean_z": estimate, "low": low,
                         "high": high, "decisions": n})
    tables["chain_position"] = pd.DataFrame(rows)

    # Creative flow: lag-1 dependence within possessions vs. shuffling within possessions.
    rng = np.random.default_rng(0)
    seq = data[["match_id", "possession", primary]].dropna()  # sorted by match, possession, time
    group = seq.groupby(["match_id", "possession"], sort=False).ngroup().to_numpy()
    values = seq[primary].to_numpy()
    same = group[1:] == group[:-1]
    left, right = np.where(same)[0], np.where(same)[0] + 1
    observed = float(np.corrcoef(values[left], values[right])[0, 1])
    null = []
    for _ in range(200):  # shuffle decisions within each possession (groups are contiguous)
        shuffled = values[np.lexsort((rng.random(len(values)), group))]
        null.append(np.corrcoef(shuffled[left], shuffled[right])[0, 1])
    null = np.array(null)
    creative = values >= np.quantile(values, 0.9)
    lift = float(creative[right][creative[left]].mean() / creative[right].mean())
    tables["creative_flow"] = pd.DataFrame([{
        "lag1_correlation": observed, "null_mean": null.mean(), "null_low": np.percentile(null, 2.5),
        "null_high": np.percentile(null, 97.5), "excess": observed - null.mean(),
        "p_permutation": float((np.abs(null - null.mean()) >= abs(observed - null.mean())).mean()),
        "creative_after_creative_lift": lift, "pairs": len(left)}])
    hypotheses.append(Hypothesis("H04_creative_flow", "creative decisions follow creative decisions beyond the "
                                 "possession's overall level (lag-1 correlation minus within-possession shuffle)",
                                 observed - null.mean(), float(np.percentile(observed - null, 2.5)),
                                 float(np.percentile(observed - null, 97.5)),
                                 max(float((null >= observed).mean()) * 2, 1 / len(null)), len(left), "correlation"))
    return tables, hypotheses


# -- 3. team level, concentration and success ----------------------------------------------------------------


def team_success(d: pd.DataFrame, team_matches: pd.DataFrame, possessions: pd.DataFrame, primary: str,
                 bootstrap: int) -> tuple[Tables, List[Hypothesis]]:
    xg = possessions.groupby(["match_id", "possession_team", "period"])["xg"].sum().rename("xg").reset_index()
    xg_team = xg.groupby(["match_id", "possession_team"])["xg"].sum()
    tm = team_matches.copy()
    tm["xg_for"] = [xg_team.get((m, t), 0.0) for m, t in zip(tm["match_id"], tm["team"])]
    tm["xg_against"] = [xg_team.get((m, o), 0.0) for m, o in zip(tm["match_id"], tm["opponent"])]
    tm["xg_diff"], tm["goal_diff"] = tm["xg_for"] - tm["xg_against"], tm["goals_for"] - tm["goals_against"]
    second = xg[xg["period"] == 2].set_index(["match_id", "possession_team"])["xg"]
    first = xg[xg["period"] == 1].set_index(["match_id", "possession_team"])["xg"]
    for half, series in (("h1", first), ("h2", second)):
        tm[f"xg_diff_{half}"] = [series.get((m, t), 0.0) - series.get((m, o), 0.0)
                                 for m, t, o in zip(tm["match_id"], tm["team"], tm["opponent"])]
    tm["context"] = tm["team"] + " | " + tm["competition"].astype(str) + " " + tm["season"].astype(str)

    level = d[(d["goals_for"] == d["goals_against"]) & d[primary].notna()]
    agg = level.groupby(["match_id", "team"]).agg(level_mean=(primary, "mean"), level_decisions=(primary, "size"))
    shares = level.groupby(["match_id", "team", "player_id"])[raw(primary)].sum().reset_index()
    concentration = shares.groupby(["match_id", "team"])[raw(primary)].agg(
        gini=lambda v: gini(v.to_numpy()), top_share=lambda v: v.max() / v.sum() if v.sum() > 0 else np.nan,
        effective_creators=lambda v: 1 / ((v / v.sum()) ** 2).sum() if v.sum() > 0 else np.nan)
    h1 = d[(d["period_ctx"] == 1) & d[primary].notna()].groupby(["match_id", "team"])[primary].mean().rename("h1_mean")
    passes = d[d["action_type"] == "pass"]
    turnover = (passes["success"] == 0).groupby([passes["match_id"], passes["team"]]).mean().rename("failed_pass_rate")
    tm = tm.set_index(["match_id", "team"]).join(agg).join(concentration).join(h1).join(turnover).reset_index()
    strength = tm.groupby("context")["xg_diff"].agg(["sum", "count"])
    tm["strength"] = [(strength.at[c, "sum"] - x) / max(strength.at[c, "count"] - 1, 1) for c, x in zip(tm["context"], tm["xg_diff"])]
    opponent_strength = tm.set_index(["match_id", "team"])["strength"]
    tm["opponent_strength"] = [opponent_strength.get((m, o), np.nan) for m, o in zip(tm["match_id"], tm["opponent"])]
    tm = tm[tm["level_decisions"] >= 30]
    for column in ("level_mean", "gini", "h1_mean"):
        tm[f"{column}_z"] = (tm[column] - tm[column].mean()) / tm[column].std()

    results = [
        fe_regression(tm, "xg_diff", ["level_mean_z", "gini_z", "opponent_strength"], "context", "match_id", bootstrap),
        fe_regression(tm, "goal_diff", ["level_mean_z", "gini_z", "opponent_strength"], "context", "match_id", bootstrap),
        fe_regression(tm, "points", ["level_mean_z", "gini_z", "opponent_strength"], "context", "match_id", bootstrap),
        fe_regression(tm, "xg_diff_h2", ["h1_mean_z", "xg_diff_h1", "opponent_strength"], "context", "match_id", bootstrap),
        fe_regression(tm, "failed_pass_rate", ["level_mean_z"], "context", "match_id", bootstrap),
    ]
    regressions = pd.concat(results, ignore_index=True)
    hubs = tm.groupby("context").agg(matches=("match_id", "size"), mean_level_creativity=("level_mean", "mean"),
                                     gini=("gini", "mean"), top_share=("top_share", "mean"),
                                     effective_creators=("effective_creators", "mean"), xg_diff=("xg_diff", "mean"))
    top_player = (d[d[primary].notna()].assign(context=d["team"] + " | " + d["competition"].astype(str) + " " + d["season"].astype(str))
                  .groupby(["context", "player"])[raw(primary)].sum().reset_index()
                  .sort_values(raw(primary), ascending=False).drop_duplicates("context").set_index("context")["player"])
    hubs = hubs.join(top_player.rename("top_creator")).reset_index()
    hubs = hubs[hubs["matches"] >= 3].sort_values("top_share", ascending=False)

    def pick(outcome, regressor):
        row = regressions[(regressions["outcome"] == outcome) & (regressions["regressor"] == regressor)].iloc[0]
        return row
    a, b = pick("xg_diff", "level_mean_z"), pick("xg_diff", "gini_z")
    c = pick("xg_diff_h2", "h1_mean_z")
    hypotheses = [
        Hypothesis("H05_team_level", "team creativity at level score -> match xG difference (team FE)", a.estimate,
                   a.low, a.high, a.p, int(a.n), "xG difference per SD"),
        Hypothesis("H06_team_concentration", "concentration (Gini) of team creativity -> match xG difference",
                   b.estimate, b.low, b.high, b.p, int(b.n), "xG difference per SD"),
        Hypothesis("H07_first_half", "first-half creativity -> second-half xG difference (given first-half xG)",
                   c.estimate, c.low, c.high, c.p, int(c.n), "xG difference per SD"),
    ]
    return {"team_matches": tm, "regressions": regressions, "hubs": hubs}, hypotheses


# -- 4. game state -----------------------------------------------------------------------------------


def score_reactions(d: pd.DataFrame, goals: pd.DataFrame, team_matches: pd.DataFrame, primary: str,
                    measures: List[str], bootstrap: int, width: float) -> tuple[Tables, List[Hypothesis]]:
    tables: Tables = {}
    hypotheses: List[Hypothesis] = []
    goals = goals.copy()
    goals["state_after"] = np.sign(goals["score_for_after"] - goals["score_against_after"]).map(
        {1: "scorer leads", 0: "level", -1: "scorer still behind"})
    summaries = []
    for measure in measures:
        windows = Windows(d, measure)
        for side, column in (("conceding team", "conceding_team"), ("scoring team", "team")):
            per_event = event_study(windows, goals, column, goals, width)
            summary = summarize_effect(per_event, f"{side}", bootstrap, by="state_after").assign(measure=measure)
            summaries.append(summary)
            if measure == primary:
                tables[f"goals_{side.split()[0]}"] = per_event
                row = summary.iloc[0]
                key = "H08_after_conceding" if side == "conceding team" else "H09_after_scoring"
                hypotheses.append(Hypothesis(key, f"creativity change of the {side} after a goal (vs. placebo windows)",
                                             row.effect, row.low, row.high, row.p, int(row.events), "SD"))
    tables["goal_event_study"] = pd.concat(summaries, ignore_index=True)

    # Star involvement after conceding: share of the team's decisions taken by its top creator.
    contexts = (d["team"] + " | " + d["competition"].astype(str) + " " + d["season"].astype(str)).rename("context")
    outfield = d[primary].notna() & (d["position"] != "Goalkeeper")  # the "star" is the top outfield creator
    player_means = d[outfield].groupby([contexts[outfield], d.loc[outfield, "player_id"]])[primary].agg(["mean", "size"])
    player_means = player_means[player_means["size"] >= 150].reset_index()
    stars = player_means.sort_values("mean", ascending=False).drop_duplicates("context").set_index("context")["player_id"]
    is_star = (d["player_id"] == contexts.map(stars)).astype(float).where(d[primary].notna())
    star_windows = Windows(d.assign(star_share=is_star), "star_share")
    per_event = event_study(star_windows, goals, "conceding_team", goals, width)
    star = summarize_effect(per_event, "star share of decisions, conceding team", bootstrap, by="state_after")
    tables["star_share_after_conceding"] = star
    row = star.iloc[0]
    hypotheses.append(Hypothesis("H10_star_after_conceding", "share of the team's decisions taken by its top creator "
                                 "rises after conceding", row.effect, row.low, row.high, row.p, int(row.events),
                                 "share of decisions"))

    # Comebacks: creativity while trailing (relative to the team's level-state mean in that match).
    rows = []
    for (match, team), part in d[d[primary].notna()].groupby(["match_id", "team"]):
        trailing = part[part["goals_for"] < part["goals_against"]]
        if len(trailing) < 20:
            continue
        level = part[part["goals_for"] == part["goals_against"]]
        first = trailing.sort_values(["period_ctx", "clock"]).iloc[0]
        final = team_matches[(team_matches["match_id"] == match) & (team_matches["team"] == team)].iloc[0]
        rows.append({"match_id": match, "team": team, "trailing_mean": trailing[primary].mean(),
                     "level_mean": level[primary].mean() if len(level) >= 20 else np.nan,
                     "first_trailing_clock": first["clock"], "deficit_at_start": first["goals_against"] - first["goals_for"],
                     "comeback": float(final["goals_for"] >= final["goals_against"]), "trailing_decisions": len(trailing)})
    comebacks = pd.DataFrame(rows)
    comebacks["relative_trailing"] = comebacks["trailing_mean"] - comebacks["level_mean"]
    comebacks["minutes_left"] = (90 - comebacks["first_trailing_clock"]).clip(lower=0)  # match clock runs on in the 2nd half
    comebacks["trailing_mean_z"] = (comebacks["trailing_mean"] - comebacks["trailing_mean"].mean()) / comebacks["trailing_mean"].std()
    comebacks["group"] = "all"
    model = fe_regression(comebacks, "comeback", ["trailing_mean_z", "minutes_left", "deficit_at_start"], None,
                          "match_id", bootstrap)
    tables["comebacks"] = comebacks
    tables["comeback_regression"] = model
    row = model[model["regressor"] == "trailing_mean_z"].iloc[0]
    hypotheses.append(Hypothesis("H11_comeback", "creativity while trailing -> probability of drawing level or winning",
                                 row.estimate, row.low, row.high, row.p, int(row.n), "probability per SD"))

    # Game management: creativity by score state and time (within team-match).
    data = d[d[primary].notna()].copy()
    data["state"] = np.sign(data["goals_for"] - data["goals_against"]).map({1: "leading", 0: "level", -1: "trailing"})
    data["late"] = (data["period_ctx"] >= 2) & (data["clock"] >= 75)
    data["within"] = demean(data[[primary]], [data["match_id"], data["team"]])[primary]
    rows = []
    for (state, late), part in data.groupby(["state", "late"]):
        estimate, low, high, p, n = cluster_mean(part["within"], part["match_id"], bootstrap)
        rows.append({"state": state, "after_75": late, "within_team_match": estimate, "low": low, "high": high,
                     "decisions": n})
    tables["state_by_time"] = pd.DataFrame(rows)
    late = data[data["late"] & data["state"].isin(["leading", "level"])]
    per_match = late.groupby(["match_id", "team", "state"])["within"].mean().unstack().dropna()
    difference = (per_match["leading"] - per_match["level"]).reset_index()
    estimate, low, high, p, n = cluster_mean(difference[0], difference["match_id"], bootstrap)
    hypotheses.append(Hypothesis("H12_game_management", "late (75'+) creativity when leading minus when level, "
                                 "same team and match", estimate, low, high, p, n, "SD"))
    return tables, hypotheses


# -- 5. personnel -------------------------------------------------------------------------------------


def personnel(d: pd.DataFrame, subs: pd.DataFrame, reds: pd.DataFrame, shifts: pd.DataFrame, appearances: pd.DataFrame,
              primary: str, bootstrap: int, width: float, reference: Optional[pd.DataFrame] = None) -> tuple[Tables, List[Hypothesis]]:
    tables: Tables = {}
    hypotheses: List[Hypothesis] = []
    player_mean = d[d[primary].notna()].groupby("player_id")[primary].agg(["mean", "size"])
    player_mean = player_mean[player_mean["size"] >= 100]["mean"]
    windows = Windows(d, primary)

    subs = subs.copy()
    subs["on_creativity"] = subs["player_on_id"].map(player_mean)
    subs["off_creativity"] = subs["player_off_id"].map(player_mean)
    subs["swap"] = subs["on_creativity"] - subs["off_creativity"]
    subs["involved"] = list(zip(subs["player_on_id"], subs["player_off_id"]))
    team_level = event_study(windows, subs, "team", subs, width)
    teammates = event_study(windows, subs, "team", subs, width, exclude_column="involved")
    tables["substitutions_team"] = team_level
    tables["substitutions_teammates"] = teammates
    summaries = [summarize_effect(team_level, "team, after substitution", bootstrap, by="reason"),
                 summarize_effect(teammates, "teammates (players involved excluded)", bootstrap, by="reason")]
    swap = team_level.dropna(subset=["swap", "effect"]).assign(group="all")
    swap["swap_z"] = (swap["swap"] - swap["swap"].mean()) / swap["swap"].std()
    regression = fe_regression(swap, "effect", ["swap_z"], None, "match_id", bootstrap)
    tables["substitution_swap_regression"] = regression
    row = regression.iloc[0]
    hypotheses.append(Hypothesis("H13_substitution_swap", "team creativity rises more when the incoming player is more "
                                 "creative than the outgoing one", row.estimate, row.low, row.high, row.p, int(row.n),
                                 "SD per SD of creativity difference"))

    # On/off: teammates' creativity (time- and state-adjusted) while a player is on vs. off, same matches.
    adjusted = d[d[primary].notna()].copy()
    adjusted["cell"] = (adjusted["period_ctx"].astype(str) + "|" + (adjusted["clock"] // 15).astype(str) + "|"
                        + np.sign(adjusted["goals_for"] - adjusted["goals_against"]).astype(str))
    adjusted["resid"] = demean(adjusted[[primary]], adjusted["cell"])[primary]
    rows = []
    subbed = pd.concat([subs[["match_id", "team", "player_off_id", "period", "clock"]].rename(columns={"player_off_id": "player_id"}).assign(kind="off"),
                        subs[["match_id", "team", "player_on_id", "period", "clock"]].rename(columns={"player_on_id": "player_id"}).assign(kind="on")])
    team_groups = {key: part for key, part in adjusted.groupby(["match_id", "team"])}
    for row in subbed.dropna(subset=["player_id"]).itertuples():
        part = team_groups.get((row.match_id, row.team))
        if part is None:
            continue
        after = (part["period_ctx"] > row.period) | ((part["period_ctx"] == row.period) & (part["clock"] >= row.clock))
        others = part["player_id"] != row.player_id
        on_mask = (~after if row.kind == "off" else after) & others
        off_mask = (after if row.kind == "off" else ~after) & others
        if on_mask.sum() >= 30 and off_mask.sum() >= 30:
            rows.append({"match_id": row.match_id, "team": row.team, "player_id": int(row.player_id), "kind": row.kind,
                         "on_minus_off": part.loc[on_mask, "resid"].mean() - part.loc[off_mask, "resid"].mean()})
    on_off = pd.DataFrame(rows)
    per_player = on_off.groupby("player_id")["on_minus_off"].agg(["mean", "size"]).join(player_mean.rename("own_creativity"))
    per_player = per_player[per_player["size"] >= 3].dropna()
    tables["on_off_events"] = on_off
    tables["on_off_players"] = per_player.reset_index()
    data = on_off.assign(own=on_off["player_id"].map(player_mean)).dropna()
    data["own_z"] = (data["own"] - data["own"].mean()) / data["own"].std()
    data["group"] = "all"
    regression = fe_regression(data, "on_minus_off", ["own_z"], None, "match_id", bootstrap)
    tables["on_off_regression"] = regression
    row = regression.iloc[0]
    hypotheses.append(Hypothesis("H14_on_off", "teammates are more creative while a more creative player is on the pitch",
                                 row.estimate, row.low, row.high, row.p, int(row.n), "SD per SD of player creativity"))

    # Red cards and tactical shifts: event studies.
    for name, events, key in (("red_cards", reds, "H15_red_card"), ("tactical_shifts", shifts, "H16_tactical_shift")):
        events = events.dropna(subset=["team"])
        own = event_study(windows, events, "team", events, width)
        summaries.append(summarize_effect(own, f"{name}: team", bootstrap))
        if name == "red_cards":
            opponents = events.merge(d[["match_id", "team"]].drop_duplicates(), on="match_id", suffixes=("", "_other"))
            opponents = opponents[opponents["team_other"] != opponents["team"]]
            other = event_study(windows, opponents, "team_other", events, width)
            summaries.append(summarize_effect(other, f"{name}: opponent", bootstrap))
        tables[name] = own
        estimate, low, high, p, n = cluster_mean(own["effect"], own["match_id"], bootstrap)
        hypotheses.append(Hypothesis(key, f"creativity change of the team after {name.replace('_', ' ')}", estimate, low,
                                     high, p, n, "SD"))
    tables["event_studies"] = pd.concat(summaries, ignore_index=True)

    # Star absence (club seasons): teammates in matches with vs. without the star.
    rows = []
    for label, table in (("cross-fitted", d), ("reference", reference)):
        if table is None:
            continue
        table = table[table[primary].notna()]
        context = table["team"] + " | " + table["competition"].astype(str) + " " + table["season"].astype(str)
        for name, part in table.groupby(context):
            matches = part["match_id"].unique()
            if len(matches) < 15:
                continue
            means = part[part["position"] != "Goalkeeper"].groupby("player_id")[primary].agg(["mean", "size"])
            for star in means[means["size"] >= 300].sort_values("mean", ascending=False).index[:3]:
                played = set(appearances[(appearances["player_id"] == star)]["match_id"].astype(str))
                present = np.array([str(m) in played for m in matches])
                if present.sum() < 5 or (~present).sum() < 3:
                    continue
                mates = part[part["player_id"] != star]
                per_match = mates.groupby("match_id")[primary].mean()
                with_star = per_match[[str(m) in played for m in per_match.index]]
                without = per_match[[str(m) not in played for m in per_match.index]]
                rng = np.random.default_rng(0)
                draws = np.array([rng.choice(with_star, len(with_star)).mean() - rng.choice(without, len(without)).mean()
                                  for _ in range(bootstrap)])
                estimate = with_star.mean() - without.mean()
                low, high, p = bootstrap_summary(estimate, draws)
                rows.append({"yardstick": label, "context": name, "star": part.loc[part["player_id"] == star, "player"].iat[0],
                             "star_creativity": means.at[star, "mean"], "matches_with": len(with_star),
                             "matches_without": len(without), "teammates_with_minus_without": estimate,
                             "low": low, "high": high, "p": p})
    tables["star_absence"] = pd.DataFrame(rows)
    return tables, hypotheses


# -- 6. time --------------------------------------------------------------------------------------------


def time_profiles(d: pd.DataFrame, team_matches: pd.DataFrame, appearances: pd.DataFrame, primary: str,
                  measures: List[str], bootstrap: int) -> tuple[Tables, List[Hypothesis]]:
    tables: Tables = {}
    hypotheses: List[Hypothesis] = []
    data = d[d[primary].notna()].copy()
    data["bin"] = data["period_ctx"].astype(str) + ": " + (data["clock"] // 15 * 15).astype(int).astype(str) + "'"
    rows = []
    for name, part in data.groupby("bin"):
        for measure in measures:
            estimate, low, high, p, n = cluster_mean(part[measure], part["match_id"], bootstrap)
            rows.append({"measure": measure, "period_bin": name, "mean_z": estimate, "low": low, "high": high, "decisions": n})
    tables["minute_profile"] = pd.DataFrame(rows)

    starters = appearances[appearances["starter"]][["match_id", "team", "player_id"]].astype({"match_id": str})
    starters["starter"] = True
    late = data[(data["period_ctx"] == 2) & (data["clock"] >= 60)].astype({"match_id": str})
    late = late.merge(starters, on=["match_id", "team", "player_id"], how="left").fillna({"starter": False})
    per_match = late.groupby(["match_id", "team", "starter"])[primary].agg(["mean", "size"]).unstack()
    per_match = per_match[(per_match[("size", True)] >= 20) & (per_match[("size", False)] >= 10)]
    difference = (per_match[("mean", False)] - per_match[("mean", True)]).rename("subs_minus_starters").reset_index()
    estimate, low, high, p, n = cluster_mean(difference["subs_minus_starters"], difference["match_id"], bootstrap)
    tables["substitutes_vs_starters_late"] = difference
    hypotheses.append(Hypothesis("H17_fresh_legs", "after 60', substitutes are more creative than starters "
                                 "(same team and match)", estimate, low, high, p, n, "SD"))

    # Tournament pressure: knockout minus group stage, within team (tournaments only).
    tm = team_matches[team_matches["stage"].notna() & (team_matches["stage"] != "Regular Season")].copy()
    tm["knockout"] = tm["stage"] != "Group Stage"
    means = data.groupby(["match_id", "team"])[primary].mean().rename("mean").reset_index().astype({"match_id": str})
    tm = tm.astype({"match_id": str}).merge(means, on=["match_id", "team"])
    tm["context"] = tm["team"] + " | " + tm["competition"].astype(str) + " " + tm["season"].astype(str)
    tm["group"] = tm["context"]
    tm["knockout_flag"] = tm["knockout"].astype(float)
    regression = fe_regression(tm, "mean", ["knockout_flag"], "context", "match_id", bootstrap)
    tables["knockout_regression"] = regression
    row = regression.iloc[0]
    hypotheses.append(Hypothesis("H18_knockout", "team creativity in knockout vs. group matches (team FE)", row.estimate,
                                 row.low, row.high, row.p, int(row.n), "SD"))
    extra = data.assign(extra_time=data["period_ctx"] >= 3)
    tables["extra_time"] = extra.groupby("extra_time")[primary].agg(["mean", "size"]).reset_index()

    # Editions: players in consecutive editions of the same tournament family.
    families = {"men": ["UEFA Euro 2020", "FIFA World Cup 2022", "UEFA Euro 2024"],
                "women": ["UEFA Women's Euro 2022", "Women's World Cup 2023", "UEFA Women's Euro 2025"]}
    edition = (data["competition"].astype(str) + " " + data["season"].astype(str)).rename("edition")
    rows = []
    for family, editions in families.items():
        sub = data[edition.isin(editions)]
        means = sub.groupby([sub["player_id"], edition[edition.isin(editions)]])[primary].agg(["mean", "size"]).reset_index()
        means = means[means["size"] >= 150]
        wide = means.pivot(index="player_id", columns="edition", values="mean")
        for a, b in zip(editions, editions[1:]):
            if a in wide and b in wide:
                pair = wide[[a, b]].dropna()
                for pid, (x, y) in pair.iterrows():
                    rows.append({"family": family, "from": a, "to": b, "player_id": pid, "change": y - x})
    changes = pd.DataFrame(rows)
    tables["edition_changes"] = changes
    if len(changes):
        changes["cluster"] = changes["player_id"]
        estimate, low, high, p, n = cluster_mean(changes["change"], changes["cluster"], bootstrap)
        hypotheses.append(Hypothesis("H19_editions", "player creativity change between consecutive tournaments",
                                     estimate, low, high, p, n, "SD"))
    return tables, hypotheses


# -- 7. opponents and context -----------------------------------------------------------------------------


def opponents(d: pd.DataFrame, team_matches: pd.DataFrame, primary: str, bootstrap: int) -> tuple[Tables, List[Hypothesis]]:
    tables: Tables = {}
    hypotheses: List[Hypothesis] = []
    data = d[d[primary].notna()].copy()
    data["context"] = data["team"] + " | " + data["competition"].astype(str) + " " + data["season"].astype(str)
    data["opp_adjusted"] = data[primary] - data.groupby("context")[primary].transform("mean")
    allowed = data.groupby(["opponent", "competition", "season"]).agg(
        allowed=("opp_adjusted", "mean"), matches=("match_id", "nunique"), decisions=("opp_adjusted", "size")).reset_index()
    tables["creativity_allowed"] = allowed[allowed["matches"] >= 5].sort_values("allowed")

    tm = team_matches.astype({"match_id": str}).copy()
    means = data.astype({"match_id": str}).groupby(["match_id", "team"])[primary].mean().rename("mean").reset_index()
    tm = tm.merge(means, on=["match_id", "team"])
    tm["context"] = tm["team"] + " | " + tm["competition"].astype(str) + " " + tm["season"].astype(str)
    tm["goal_diff"] = tm["goals_for"] - tm["goals_against"]
    stats_ = tm.groupby("context")["goal_diff"].agg(["sum", "count"])
    tm["strength"] = [(stats_.at[c, "sum"] - g) / max(stats_.at[c, "count"] - 1, 1) for c, g in zip(tm["context"], tm["goal_diff"])]
    lookup = tm.set_index(["match_id", "team"])["strength"]
    tm["opponent_strength"] = [lookup.get((m, o), np.nan) for m, o in zip(tm["match_id"], tm["opponent"])]
    regression = fe_regression(tm, "mean", ["opponent_strength"], "context", "match_id", bootstrap)
    tables["opponent_strength_regression"] = regression
    row = regression.iloc[0]
    hypotheses.append(Hypothesis("H20_opponent_strength", "team creativity vs. opponent strength (leave-one-out goal "
                                 "difference per match; team FE)", row.estimate, row.low, row.high, row.p, int(row.n),
                                 "SD per goal of opponent strength"))

    pressure = data.groupby(["player_id", "player", "under_pressure"])[primary].agg(["mean", "size"]).unstack()
    pressure = pressure[(pressure[("size", True)] >= 50) & (pressure[("size", False)] >= 50)]
    difference = (pressure[("mean", True)] - pressure[("mean", False)]).rename("pressure_minus_free").reset_index()
    difference["under_pressure_creativity"] = pressure[("mean", True)].to_numpy()
    difference["decisions_under_pressure"] = pressure[("size", True)].to_numpy()
    tables["pressure_players"] = difference.sort_values("under_pressure_creativity", ascending=False)
    estimate, low, high, p, n = cluster_mean(difference["pressure_minus_free"], difference["player_id"], bootstrap)
    hypotheses.append(Hypothesis("H21_pressure", "within-player creativity under pressure minus without", estimate, low,
                                 high, p, n, "SD"))

    data["cell_x"] = (data["start_x"] // 10).clip(0, 11).astype(int)
    data["cell_y"] = (data["start_y"] // 10).clip(0, 7).astype(int)
    tables["pitch_map"] = data.groupby(["cell_x", "cell_y"])[primary].agg(["mean", "size"]).reset_index()
    tables["pitch_map_by_team"] = data.groupby(["context", "cell_x", "cell_y"])[primary].agg(["mean", "size"]).reset_index()

    # Novelty wears off: repeated creative patterns within a team-match.
    data["end_cell"] = np.where(data["end_x"].notna(), (data["end_x"] // 20).astype("Int64").astype(str) + "-"
                                + (data["end_y"] // 20).astype("Int64").astype(str), "none")
    data["pattern"] = (data["action_type"] + "|" + (data["start_x"] // 20).astype(int).astype(str) + "-"
                       + (data["start_y"] // 20).astype(int).astype(str) + "|" + data["end_cell"])
    data = data.sort_values(["match_id", "team", "period_ctx", "clock"])
    data["creative"] = data[primary] >= data[primary].quantile(0.9)
    data["occurrence"] = data.groupby(["match_id", "team", "creative", "pattern"]).cumcount() + 1
    data["repeat"] = data["occurrence"] > 1
    data["realized"] = data["td_advantage"] - data["advantage"]
    cells = data.groupby(["match_id", "creative", "repeat"])["realized"].mean().unstack(["creative", "repeat"])
    did = ((cells[(True, True)] - cells[(True, False)]) - (cells[(False, True)] - cells[(False, False)])).dropna()
    did = did.rename("did").reset_index()
    estimate, low, high, p, n = cluster_mean(did["did"], did["match_id"], bootstrap)
    tables["novelty"] = data.groupby(["creative", "repeat"])["realized"].agg(["mean", "size"]).reset_index()
    hypotheses.append(Hypothesis("H22_novelty_wears_off", "realized minus predicted value of repeated vs. first creative "
                                 "patterns in a match, minus the same for non-creative patterns", estimate, low, high, p, n,
                                 "goal-probability units"))
    return tables, hypotheses


# -- 8. players -------------------------------------------------------------------------------------------


def player_profiles(d: pd.DataFrame, primary: str, seed: int = 0) -> Tables:
    data = d[d[primary].notna()].copy()
    creative = data[primary] >= data[primary].quantile(0.9)
    grouped = data.groupby(["player_id", "player"])
    features = pd.DataFrame({
        "decisions": grouped.size(), "creativity": grouped[primary].mean(), "originality": grouped["rarity"].mean(),
        "value": grouped["advantage"].mean(), "risk": 1 - grouped["p_success"].mean(),
        "position": grouped["position"].agg(lambda p: p.mode().iat[0] if p.notna().any() else ""),
        "team": grouped["team"].agg(lambda p: p.mode().iat[0] if p.notna().any() else ""),
    })
    for kind in ("pass", "carry", "dribble", "shot"):
        features[f"creative_share_{kind}"] = (creative & (data["action_type"] == kind)).groupby(
            [data["player_id"], data["player"]]).sum() / creative.groupby([data["player_id"], data["player"]]).sum().clip(lower=1)
    features = features[features["decisions"] >= 300].reset_index()
    columns = ["creativity", "originality", "value", "risk", "creative_share_pass", "creative_share_carry",
               "creative_share_dribble"]
    x = features[columns].to_numpy(dtype=float)
    x = (x - x.mean(0)) / x.std(0)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(10):
        centers = x[rng.choice(len(x), 5, replace=False)]
        for _ in range(100):
            labels = ((x[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
            centers = np.stack([x[labels == k].mean(0) if (labels == k).any() else centers[k] for k in range(5)])
        inertia = ((x - centers[labels]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, labels, centers)
    features["cluster"] = best[1]
    features["group"] = features["position"].map(position_group)
    centroids = pd.DataFrame(best[2], columns=columns).assign(
        players=np.bincount(best[1], minlength=5),
        examples=[", ".join(features[features["cluster"] == k].sort_values("decisions", ascending=False)["player"].head(6))
                  for k in range(5)])

    match_means = data.groupby(["player_id", "match_id"])[primary].agg(["mean", "var", "size"])
    match_means = match_means[match_means["size"] >= 30]
    consistency = match_means.groupby("player_id").agg(
        matches=("mean", "size"), between_var=("mean", "var"),
        sampling_var=("var", "mean"), mean_size=("size", "mean"))
    consistency["excess_variance"] = consistency["between_var"] - consistency["sampling_var"] / consistency["mean_size"]
    consistency = consistency[consistency["matches"] >= 8].join(features.set_index("player_id")[["player", "creativity", "team"]])

    passes = data[(data["action_type"] == "pass") & data["recipient"].notna()]
    team_pass_mean = passes.groupby("team")[primary].transform("mean")
    passes = passes.assign(relative=passes[primary] - team_pass_mean)
    enablers = passes.groupby(["recipient", "team"]).agg(received=("relative", "size"), enabler_score=("relative", "mean"),
                                                        mean_received=(primary, "mean")).reset_index()
    enablers = enablers[enablers["received"] >= 100].sort_values("enabler_score", ascending=False)
    passer_mean = passes.groupby("player")[primary].transform("mean")
    pairs = passes.assign(excess=passes[primary] - passer_mean).groupby(["player", "recipient", "team"]).agg(
        passes=("excess", "size"), pair_excess=("excess", "mean"), pair_mean=(primary, "mean")).reset_index()
    pairs = pairs[pairs["passes"] >= 30].sort_values("pair_excess", ascending=False)
    return {"profiles": features, "profile_clusters": centroids, "consistency": consistency.reset_index(),
            "enablers": enablers, "duos": pairs}
