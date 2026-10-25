"""Step-2 analyses: robustness of the proposal's results, creative opportunity and its use, enabling.

Builds on :mod:`soccer_ml.creativity.second_order` (same inputs: decision tables joined with match
context, measures z-scored over intended-scope decisions as ``z_<measure>``).

* B1 build-up creativity -> shots and goals: excluding the last actions, team / match fixed
  effects, field-progression control, originality and value jointly, alternative measures,
  reference yardstick;
* B2 hidden contributions: creativity by role in goal and shot possessions (credited vs.
  uncredited), player creativity vs. goals + assists;
* B3-B5 event studies (scoring, conceding, red cards, tactical shifts, tactical substitutions):
  window lengths, pre-trends, kick-off exclusion, splits, pressure-adjusted creativity, and the
  decomposition into creative opportunity and its use (C2);
* B6 power of the team-level nulls; possession-level distribution of creativity;
* B8 surprise premium: finer strata, passes only, reference yardstick;
* C1 creative opportunity: descriptives and player-level decomposition into opportunity and use;
* C3 enabling: does a creative action raise the next teammate's opportunity and space?

Opportunity (per decision, independent of the choice; ``scoring.py``):
``opportunity_expected`` = E_pi[good_surprise] = pi(S) H(pi | S) with S the positive-advantage
actions; ``opportunity_quadrant`` = share of the unusual options (least likely 20% of policy
mass) that are useful.  Use = realized good_surprise - opportunity_expected.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from soccer_ml.creativity.second_order import (
    Windows,
    bootstrap_summary,
    cluster_mean,
    demean,
    raw,
)

Tables = Dict[str, pd.DataFrame]


# -- statistics -----------------------------------------------------------------------------------------


def demean_multi(values: pd.DataFrame, groups: Sequence[pd.Series], iterations: int = 30) -> pd.DataFrame:
    """Remove several sets of fixed effects by alternating projections."""
    out = values.astype(float).copy()
    if len(groups) == 1:
        return demean(out, groups[0])
    for _ in range(iterations):
        for group in groups:
            out = out - out.groupby(group).transform("mean")
    return out


def fe_ols(data: pd.DataFrame, outcome: str, regressors: List[str], fixed_effects: List[str], cluster: str,
           bootstrap: int = 1000, seed: int = 0) -> pd.DataFrame:
    """OLS with fixed effects (demeaned once) and a cluster bootstrap of the sufficient statistics."""
    data = data.dropna(subset=[outcome] + regressors + fixed_effects).reset_index(drop=True)
    values = data[[outcome] + regressors].astype(float)
    tilde = demean_multi(values, [data[c] for c in fixed_effects]) if fixed_effects else values - values.mean()
    x, y = tilde[regressors].to_numpy(), tilde[outcome].to_numpy()
    k = len(regressors)
    codes, clusters = pd.factorize(data[cluster])
    xtx = np.zeros((len(clusters), k, k))
    xty = np.zeros((len(clusters), k))
    np.add.at(xtx, codes, x[:, :, None] * x[:, None, :])
    np.add.at(xty, codes, x * y[:, None])
    estimate = np.linalg.solve(xtx.sum(0), xty.sum(0))
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=bootstrap)
    draws = np.array([np.linalg.solve(np.tensordot(wi, xtx, 1), wi @ xty) for wi in w])
    rows = []
    for j, name in enumerate(regressors):
        low, high, p = bootstrap_summary(estimate[j], draws[:, j])
        rows.append({"outcome": outcome, "regressor": name, "estimate": estimate[j], "low": low, "high": high, "p": p,
                     "n": len(data), "clusters": len(clusters), "fixed_effects": "+".join(fixed_effects)})
    return pd.DataFrame(rows)


def zscore(values: pd.Series) -> pd.Series:
    return (values - values.mean()) / values.std()


# -- event studies -------------------------------------------------------------------------------------------


def window_change(windows: Windows, key, t: float, before: tuple, after: tuple, exclude: tuple = (), minimum: int = 5) -> float:
    b, nb = windows.mean(key, t + before[0], t + before[1], exclude)
    a, na = windows.mean(key, t + after[0], t + after[1], exclude)
    return a - b if nb >= minimum and na >= minimum else float("nan")


def event_effects(windows: Windows, events: pd.DataFrame, team_column: str, avoid: pd.DataFrame,
                  before: tuple = (-10, 0), after: tuple = (0, 10), placebos: int = 20, seed: int = 0) -> pd.DataFrame:
    """Per event: after-minus-before change of the team's mean, minus the same change in placebo windows
    (other matches, same period, clock within +-3 min, no event of this kind within the window span)."""
    rng = np.random.default_rng(seed)
    span = max(abs(before[0]), abs(after[1]))
    avoid_times = {key: part["clock"].to_numpy() for key, part in avoid.groupby(["match_id", "period"])}
    by_period: Dict[int, list] = {}
    for key in windows.keys:
        by_period.setdefault(key[2], []).append(key)
    rows = []
    for event in events.itertuples(index=False):
        key = (event.match_id, getattr(event, team_column), event.period)
        delta = window_change(windows, key, event.clock, before, after)
        values = []
        candidates = by_period.get(event.period, [])
        for _ in range(placebos * 5):
            if len(values) >= placebos or not candidates:
                break
            other = candidates[rng.integers(len(candidates))]
            t = event.clock + rng.uniform(-3, 3)
            nearby = avoid_times.get((other[0], other[2]), np.empty(0))
            if other[0] == event.match_id or (np.abs(nearby - t) < span).any():
                continue
            value = window_change(windows, other, t, before, after)
            if value == value:
                values.append(value)
        placebo = float(np.mean(values)) if values else float("nan")
        rows.append({**event._asdict(), "delta": delta, "placebo": placebo, "effect": delta - placebo})
    return pd.DataFrame(rows)


def summarize(per_event: pd.DataFrame, labels: dict, bootstrap: int, by: Optional[str] = None) -> pd.DataFrame:
    groups = [("all", per_event)] + (list(per_event.groupby(by)) if by else [])
    rows = []
    for name, part in groups:
        estimate, low, high, p, n = cluster_mean(part["effect"], part["match_id"], bootstrap)
        rows.append({**labels, "group": name, "effect": estimate, "low": low, "high": high, "p": p, "events": n})
    return pd.DataFrame(rows)


# -- B1 build-up creativity -> shots / goals ---------------------------------------------------------------------


def possession_features(d: pd.DataFrame, possessions: pd.DataFrame, measures: List[str], exclude_last: int = 0) -> pd.DataFrame:
    meta = possessions.rename(columns={"possession_team": "team"})[["match_id", "possession", "team", "shots", "xg", "goal"]]
    data = d.merge(meta[["match_id", "possession", "team"]], on=["match_id", "possession", "team"])
    data = data.sort_values(["match_id", "possession", "period_ctx", "clock"])
    data["from_end"] = data.groupby(["match_id", "possession"]).cumcount(ascending=False) + 1
    key = ["match_id", "possession"]
    first = data.groupby(key).first()[["v", "start_x", "play_pattern", "competition", "season", "team"]]
    build = data[(data["action_type"] != "shot") & (data["from_end"] > exclude_last) & data["in_scope"]]
    features = pd.DataFrame({"decisions": data.groupby(key).size(), "max_x": data.groupby(key)["start_x"].max()})
    for measure in measures:
        features[f"max_{measure}"] = build.groupby(key)[raw(measure)].max()
    features["max_raw_rarity"] = build.groupby(key)["rarity"].max()
    features["max_raw_advantage"] = build.groupby(key)["advantage"].max()
    table = features.join(first).reset_index().merge(meta.drop(columns="team"), on=key)
    table["shot"] = (table["shots"] > 0).astype(float)
    table["goal"] = table["goal"].astype(float)
    table["v_decile"] = pd.qcut(table["v"].rank(method="first"), 10, labels=False)
    table["third"] = pd.cut(table["start_x"], [-1, 40, 80, 121], labels=["own", "middle", "final"]).astype(str)
    table["length_bin"] = pd.cut(table["decisions"], [0, 1, 2, 4, 7, 1000], labels=["1", "2", "3-4", "5-7", "8+"]).astype(str)
    table["stratum"] = (table["v_decile"].astype(str) + "|" + table["third"] + "|" + table["play_pattern"].astype(str)
                        + "|" + table["length_bin"])
    table["context"] = table["team"] + " | " + table["competition"].astype(str) + " " + table["season"].astype(str)
    return table


def buildup(d: pd.DataFrame, possessions: pd.DataFrame, measures: List[str], bootstrap: int,
            reference: Optional[pd.DataFrame] = None) -> Tables:
    primary = measures[0]
    rows = []

    def run(table: pd.DataFrame, spec: str, regressors: List[str], fixed: List[str], sample: str = "cross-fitted"):
        table = table.copy()
        for column in regressors:
            if column.startswith("max_") or column == "max_x":
                table[column] = zscore(table[column])
        for outcome in ("shot", "goal", "xg"):
            result = fe_ols(table, outcome, regressors, fixed, "match_id", bootstrap)
            rows.append(result.assign(spec=spec, sample=sample, base_rate=float(table[outcome].mean())))

    base = possession_features(d, possessions, measures)
    for measure in measures:
        run(base, f"base: {measure}", [f"max_{measure}"], ["stratum"])
    run(base, "base + team-season FE", [f"max_{primary}"], ["stratum", "context"])
    run(base, "base + match FE", [f"max_{primary}"], ["stratum", "match_id"])
    run(base, "base + field progression", [f"max_{primary}", "max_x"], ["stratum"])
    run(base, "originality and value jointly", ["max_raw_rarity", "max_raw_advantage"], ["stratum"])
    for k in (1, 2, 3):
        trimmed = possession_features(d, possessions, measures, exclude_last=k)
        trimmed = trimmed[trimmed["decisions"] > k + 1]
        run(trimmed, f"excluding the last {k} decision(s)", [f"max_{primary}"], ["stratum"])
        if k == 2:
            run(trimmed, "excluding last 2, originality and value jointly", ["max_raw_rarity", "max_raw_advantage"], ["stratum"])
            run(trimmed, "excluding last 2 + match FE", [f"max_{primary}"], ["stratum", "match_id"])
    if reference is not None:
        ref = possession_features(reference, possessions, [primary])
        run(ref, "base, reference yardstick (club and Argentina matches)", [f"max_{primary}"], ["stratum"], "reference")
        same = base[base["match_id"].isin(ref["match_id"].unique())]
        run(same, "base, cross-fitted on the same matches", [f"max_{primary}"], ["stratum"], "cross-fitted (same matches)")
    return {"buildup_specifications": pd.concat(rows, ignore_index=True)}


# -- B2 hidden contributions ------------------------------------------------------------------------------------------


def hidden_contributions(d: pd.DataFrame, possessions: pd.DataFrame, goals: pd.DataFrame, appearances: pd.DataFrame,
                         primary: str, bootstrap: int) -> Tables:
    meta = possessions.rename(columns={"possession_team": "team"})[["match_id", "possession", "team", "shots", "goal"]]
    data = d.merge(meta, on=["match_id", "possession", "team"]).sort_values(["match_id", "possession", "period_ctx", "clock"])
    data = data[data["in_scope"]].copy()
    threshold = d.loc[d["in_scope"], raw(primary)].quantile(0.9)
    data["creative"] = data[raw(primary)] >= threshold
    tables: Tables = {}
    for label, sample, credit in (("goal possessions", data[data["goal"]], "goal_assist"),
                                  ("shot possessions", data[data["shots"] > 0], "shot_assist")):
        sample = sample.copy()
        sample["order"] = sample.groupby(["match_id", "possession"]).cumcount()
        credited = sample[credit].astype(bool)
        key_order = sample[credited].groupby(["match_id", "possession"])["order"].max().rename("credit_order")
        sample = sample.join(key_order, on=["match_id", "possession"])
        sample["role"] = np.select(
            [sample["action_type"] == "shot", credited, sample["order"] == sample["credit_order"] - 1,
             sample["credit_order"].notna() & (sample["order"] < sample["credit_order"] - 1)],
            ["shot", "assist" if credit == "goal_assist" else "key pass", "pre-assist" if credit == "goal_assist" else "pre-key pass",
             "earlier"], default="other (no credited pass)")
        summary = sample.groupby("role").agg(decisions=(primary, "size"), mean_z=(primary, "mean"),
                                             creative_rate=("creative", "mean"), creative=("creative", "sum"),
                                             creativity_sum=(raw(primary), "sum")).reset_index()
        summary["share_of_creative"] = summary["creative"] / summary["creative"].sum()
        summary["share_of_creativity"] = summary["creativity_sum"] / summary["creativity_sum"].sum()
        tables[f"roles_{label.split()[0]}"] = summary
        uncredited = sample[~sample["role"].isin(["shot", "assist", "key pass"])]
        estimate = float(uncredited["creative"].sum() / sample["creative"].sum())
        per_match = sample.groupby("match_id").agg(total=("creative", "sum"))
        per_match["uncredited"] = uncredited.groupby("match_id")["creative"].sum()
        per_match = per_match.fillna(0)
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(bootstrap):
            s = per_match.iloc[rng.integers(0, len(per_match), len(per_match))]
            draws.append(s["uncredited"].sum() / max(s["total"].sum(), 1))
        tables.setdefault("uncredited_share", pd.DataFrame())
        tables["uncredited_share"] = pd.concat([tables["uncredited_share"], pd.DataFrame([{
            "possessions": label, "creative_decisions": int(sample["creative"].sum()),
            "uncredited_share": estimate, "low": float(np.percentile(draws, 2.5)), "high": float(np.percentile(draws, 97.5)),
            "uncredited_share_of_creativity": float(uncredited[raw(primary)].sum() / sample[raw(primary)].sum())}])])

    # Player level: creativity per decision vs. goals + assists per 90 minutes.
    app = appearances.copy()
    app["minutes"] = (app["end_clock"] - app["start_clock"]).clip(lower=0)
    minutes = app.groupby("player_id")["minutes"].sum()
    scorers = goals[~goals["own_goal"]].groupby("player").size()
    assists = d[d["goal_assist"].astype(bool)].groupby("player").size()
    players = d[d["in_scope"]].groupby(["player_id", "player"]).agg(
        creativity=(primary, "mean"), decisions=(primary, "size"),
        position=("position", lambda p: p.mode().iat[0] if p.notna().any() else ""),
        team=("team", lambda p: p.mode().iat[0] if p.notna().any() else ""),
        women=("women", lambda w: str(w.iat[0]) == "True")).reset_index()
    players["minutes"] = players["player_id"].map(minutes)
    players["goals"] = players["player"].map(scorers).fillna(0)
    players["assists"] = players["player"].map(assists).fillna(0)
    players["ga_per90"] = (players["goals"] + players["assists"]) / players["minutes"] * 90
    players = players[(players["minutes"] >= 600) & (players["decisions"] >= 300)].copy()
    from soccer_ml.creativity.measures import position_group
    players["group"] = players["position"].map(position_group)
    players = players[players["group"] != "GK"]
    players["creativity_pct"] = players.groupby(["women", "group"])["creativity"].rank(pct=True)
    players["ga_pct"] = players.groupby(["women", "group"])["ga_per90"].rank(pct=True)
    rho = float(players[["creativity_pct", "ga_pct"]].corr(method="spearman").iloc[0, 1])
    rng = np.random.default_rng(1)
    draws = [players.sample(len(players), replace=True, random_state=int(rng.integers(1e9)))[["creativity_pct", "ga_pct"]]
             .corr(method="spearman").iloc[0, 1] for _ in range(bootstrap // 2)]
    hidden = players[(players["creativity_pct"] >= 0.75) & (players["ga_pct"] <= 0.5)].sort_values("creativity_pct", ascending=False)
    tables["player_creativity_vs_ga"] = pd.DataFrame([{
        "players": len(players), "spearman_within_position": rho, "low": float(np.percentile(draws, 2.5)),
        "high": float(np.percentile(draws, 97.5)),
        "share_top_quartile_creators_with_below_median_ga": float(
            ((players["creativity_pct"] >= 0.75) & (players["ga_pct"] <= 0.5)).sum() / (players["creativity_pct"] >= 0.75).sum())}])
    tables["hidden_creators"] = hidden[["player", "team", "position", "women", "decisions", "minutes", "goals", "assists",
                                        "ga_per90", "creativity", "creativity_pct", "ga_pct"]]
    tables["players_creativity_ga"] = players
    return tables


# -- B3-B5 + C2 event studies -------------------------------------------------------------------------------------------


def event_robustness(d: pd.DataFrame, goals: pd.DataFrame, reds: pd.DataFrame, shifts: pd.DataFrame, subs: pd.DataFrame,
                     values: List[str], bootstrap: int, placebos: int = 20) -> Tables:
    """Event studies for all event types, window variants and outcome variables (creativity, opportunity, use, pressure)."""
    goals = goals.copy()
    goals["state_after"] = np.sign(goals["score_for_after"] - goals["score_against_after"]).map(
        {1: "scorer leads", 0: "level", -1: "scorer behind"})
    goals["time"] = np.where(goals["clock"] < 60, "before 60'", "60' or later")
    reds = reds.dropna(subset=["team"]).copy()
    opponents_of = d[["match_id", "team"]].drop_duplicates()
    reds_opp = reds.merge(opponents_of, on="match_id", suffixes=("", "_other"))
    reds_opp = reds_opp[reds_opp["team_other"] != reds_opp["team"]]
    shifts = shifts.dropna(subset=["team"]).copy()
    near = subs.merge(shifts[["match_id", "team", "period", "clock"]], on=["match_id", "team", "period"], suffixes=("", "_shift"))
    near = near[(near["clock"] - near["clock_shift"]).abs() <= 1][["match_id", "team", "period", "clock_shift"]]
    shifts = shifts.merge(near.drop_duplicates().rename(columns={"clock_shift": "clock"}).assign(with_substitution=True),
                          on=["match_id", "team", "period", "clock"], how="left").fillna({"with_substitution": False})
    shifts["state"] = np.sign(shifts["goals_for"] - shifts["goals_against"]).map({1: "leading", 0: "level", -1: "trailing"})
    tactical_subs = subs[subs["reason"] == "Tactical"]
    event_sets = [("goal: scoring team", goals, "team", goals, ["state_after", "time"]),
                  ("goal: conceding team", goals, "conceding_team", goals, ["state_after", "time"]),
                  ("red card: team sent off", reds, "team", reds, []),
                  ("red card: opponent", reds_opp, "team_other", reds, []),
                  ("tactical shift", shifts, "team", shifts, ["with_substitution", "state"]),
                  ("tactical substitution", tactical_subs, "team", subs, [])]
    variants = [("10 min", (-10, 0), (0, 10)), ("5 min", (-5, 0), (0, 5)), ("15 min", (-15, 0), (0, 15)),
                ("pre-trend (-20..-10 vs -10..0)", (-20, -10), (-10, 0)), ("skip 2 min after", (-10, 0), (2, 12))]
    rows, per_event_main = [], {}
    for value in values:
        windows = Windows(d, value)
        for name, events, column, avoid, splits in event_sets:
            for variant, before, after in variants:
                if variant.startswith("skip") and not name.startswith("goal"):
                    continue
                if value not in values[:1] and variant not in ("10 min", "pre-trend (-20..-10 vs -10..0)"):
                    continue
                per_event = event_effects(windows, events, column, avoid, before, after, placebos)
                labels = {"event": name, "value": value, "variant": variant}
                rows.append(summarize(per_event, labels, bootstrap))
                if variant == "10 min":
                    for split in splits:
                        rows.append(summarize(per_event, {**labels, "split": split}, bootstrap, by=split).iloc[1:])
                    if value == values[0]:
                        per_event_main[name] = per_event
    table = pd.concat(rows, ignore_index=True)
    slug = lambda name: name.replace(":", "").replace(" ", "_")  # noqa: E731
    return {"event_studies": table, **{f"events_{slug(k)}": v for k, v in per_event_main.items()}}


def fresh_legs_by_position(d: pd.DataFrame, appearances: pd.DataFrame, primary: str, bootstrap: int) -> Tables:
    from soccer_ml.creativity.measures import position_group
    starters = appearances[appearances["starter"]][["match_id", "team", "player_id"]].assign(starter=True)
    late = d[(d["period_ctx"] == 2) & (d["clock"] >= 60) & d[primary].notna()]
    late = late.merge(starters, on=["match_id", "team", "player_id"], how="left").fillna({"starter": False})
    late["group"] = late["position"].map(position_group)
    cell = late.groupby(["match_id", "team", "group", "starter"])[primary].agg(["mean", "size"]).unstack()
    cell = cell[(cell[("size", True)] >= 10) & (cell[("size", False)] >= 10)]
    diff = (cell[("mean", False)] - cell[("mean", True)]).rename("subs_minus_starters").reset_index()
    estimate, low, high, p, n = cluster_mean(diff["subs_minus_starters"], diff["match_id"], bootstrap)
    by_group = [{"group": g, **dict(zip(["estimate", "low", "high", "p", "n"], cluster_mean(part["subs_minus_starters"], part["match_id"], bootstrap)))}
                for g, part in diff.groupby("group")]
    return {"fresh_legs_within_position": pd.DataFrame([{"group": "all (within team-match x position)", "estimate": estimate,
                                                         "low": low, "high": high, "p": p, "n": n}] + by_group)}


# -- B6 power and possession-level distribution --------------------------------------------------------------------------


def power_and_distribution(d: pd.DataFrame, possessions: pd.DataFrame, team_regressions: pd.DataFrame, primary: str,
                           bootstrap: int) -> Tables:
    rows = []
    for _, r in team_regressions.iterrows():
        se = (r["high"] - r["low"]) / 3.92
        rows.append({"outcome": r["outcome"], "regressor": r["regressor"], "estimate": r["estimate"], "se": se,
                     "mde_80_power": 2.8 * se, "n": r["n"]})
    power = pd.DataFrame(rows)
    table = possession_features(d, possessions, [primary])
    meta = possessions.rename(columns={"possession_team": "team"})[["match_id", "possession", "team"]]
    data = d.merge(meta, on=["match_id", "possession", "team"])
    data = data[data["in_scope"] & (data["action_type"] != "shot")]
    threshold = d.loc[d["in_scope"], raw(primary)].quantile(0.9)
    creative = data[data[raw(primary)] >= threshold]
    contributors = creative.groupby(["match_id", "possession"])["player_id"].nunique().rename("creative_contributors")
    sums = data.groupby(["match_id", "possession", "player_id"])[raw(primary)].sum().reset_index()
    top_share = sums.groupby(["match_id", "possession"])[raw(primary)].agg(lambda v: v.max() / v.sum() if v.sum() > 0 else np.nan)
    table = table.join(contributors, on=["match_id", "possession"]).join(top_share.rename("top_share"), on=["match_id", "possession"])
    table["creative_contributors"] = table["creative_contributors"].fillna(0)
    sample = table[table["decisions"] >= 3].copy()
    sample["two_plus"] = (sample["creative_contributors"] >= 2).astype(float)
    for c in (f"max_{primary}", "top_share"):
        sample[c] = zscore(sample[c])
    results = []
    for outcome in ("shot", "goal"):
        results.append(fe_ols(sample, outcome, ["two_plus", f"max_{primary}"], ["stratum"], "match_id", bootstrap)
                       .assign(spec="two or more creative contributors, given max creativity"))
        results.append(fe_ols(sample.dropna(subset=["top_share"]), outcome, ["top_share", f"max_{primary}"], ["stratum"],
                              "match_id", bootstrap).assign(spec="concentration (top player's share), given max creativity"))
    return {"team_level_power": power, "possession_distribution": pd.concat(results, ignore_index=True)}


# -- B8 surprise premium -----------------------------------------------------------------------------------------------------


def premium_robustness(d: pd.DataFrame, bootstrap: int, reference: Optional[pd.DataFrame] = None) -> Tables:
    from soccer_ml.creativity.second_order import fe_slope
    rows = []
    for label, table in (("cross-fitted", d), ("reference", reference)):
        if table is None:
            continue
        data = table[table["in_scope"]].copy()
        data["res_td"] = data["td_advantage"] - data["advantage"]
        data["res_mc"] = data["mc_return"] - data["q"]
        data["q_decile"] = pd.qcut(data["q"].rank(method="first"), 10, labels=False)
        data["fine"] = (data["q_decile"].astype(str) + "|" + data["action_type"] + "|"
                        + pd.cut(data["start_x"], [-1, 40, 80, 121]).astype(str))
        specs = [("Q deciles", data, "q_decile"), ("Q deciles x action type x third", data, "fine"),
                 ("passes only, Q deciles", data[data["action_type"] == "pass"], "q_decile"),
                 ("carries only, Q deciles", data[data["action_type"] == "carry"], "q_decile"),
                 ("targets in view, fine strata", data[data["end_zone_visible"] != 0], "fine")]
        for name, part, strata in specs:
            for residual in ("res_td", "res_mc"):
                estimate, low, high, p, n = fe_slope(part[residual], part["rarity"], part[strata], part["match_id"], bootstrap)
                rows.append({"sample": label, "spec": name, "residual": residual, "slope": estimate, "low": low,
                             "high": high, "p": p, "decisions": n, "mean_q": float(part["q"].mean()),
                             "slope_relative_to_mean_q": estimate / float(part["q"].mean())})
    return {"premium_specifications": pd.DataFrame(rows)}


# -- C1 opportunity and use ----------------------------------------------------------------------------------------------------


def opportunity(d: pd.DataFrame, primary: str) -> Tables:
    from soccer_ml.creativity.measures import position_group
    data = d[d["in_scope"]].copy()
    data["use"] = data[raw(primary)] - data["opportunity_expected"]
    data["group"] = data["position"].map(position_group)
    data["third"] = pd.cut(data["start_x"], [-1, 40, 80, 121], labels=["own", "middle", "final"]).astype(str)
    data["state"] = np.sign(data["goals_for"] - data["goals_against"]).map({1: "leading", 0: "level", -1: "trailing"})
    cols = {"opportunity_expected": "mean", "opportunity_quadrant": "mean", "chose_quadrant": "mean",
            "use": "mean", raw(primary): "mean", "good_mass": "mean"}
    by = {f"by_{name}": data.groupby(name).agg(cols).assign(decisions=data.groupby(name).size()).reset_index()
          for name in ("group", "third", "state", "action_type", "decision_context")}
    players = data.groupby(["player_id", "player"]).agg(
        creativity=(raw(primary), "mean"), opportunity=("opportunity_expected", "mean"), use=("use", "mean"),
        decisions=("use", "size"), position=("position", lambda p: p.mode().iat[0] if p.notna().any() else ""),
        team=("team", lambda p: p.mode().iat[0] if p.notna().any() else "")).reset_index()
    players = players[players["decisions"] >= 200]
    total = players["creativity"].var()
    decomposition = pd.DataFrame([{
        "players": len(players), "var_creativity": total, "var_opportunity": players["opportunity"].var(),
        "var_use": players["use"].var(), "cov_opportunity_use": players[["opportunity", "use"]].cov().iloc[0, 1],
        "share_opportunity": players["opportunity"].var() / total, "share_use": players["use"].var() / total,
        "share_covariance": 2 * players[["opportunity", "use"]].cov().iloc[0, 1] / total,
        "corr_opportunity_use": players[["opportunity", "use"]].corr().iloc[0, 1]}])
    players["group"] = players["position"].map(position_group)
    return {**by, "player_decomposition": decomposition, "players_opportunity_use": players}


# -- C3 enabling ------------------------------------------------------------------------------------------------------------------


def enabling(d: pd.DataFrame, primary: str, bootstrap: int) -> Tables:
    """Does a creative action raise the opportunity (and space) of the next decision by a teammate?

    Pairs each in-scope decision with the next decision if it is by a different player of the same
    team.  Strata: the action's end cell (10 x 10 yd; start cell for actions without a target) x action
    type x value decile; fixed effects for match.
    """
    data = d.reset_index(drop=True)
    nxt = data["next_row"].to_numpy()
    valid = (nxt >= 0) & data["in_scope"].to_numpy()  # next_row: global row of the next decision (-1: none)
    pairs = data[valid].copy()
    follow = data.iloc[nxt[valid]].reset_index(drop=True)
    pairs = pairs.reset_index(drop=True)
    keep = (follow["player_id"].to_numpy() != pairs["player_id"].to_numpy()) & (follow["team"].to_numpy() == pairs["team"].to_numpy())
    pairs, follow = pairs[keep].reset_index(drop=True), follow[keep].reset_index(drop=True)
    for column in ("opportunity_expected", "opportunity_quadrant", "nearest_opponent", "opponents_within_5", raw(primary), "v"):
        pairs[f"next_{column}"] = follow[column].to_numpy()
    x = np.where(pairs["end_x"].notna(), pairs["end_x"], pairs["start_x"])
    y = np.where(pairs["end_y"].notna(), pairs["end_y"], pairs["start_y"])
    pairs["cell"] = (np.floor(x / 10).astype(int).astype(str) + "-" + np.floor(y / 10).astype(int).astype(str))
    pairs["v_decile"] = pd.qcut(pairs["v"].rank(method="first"), 10, labels=False)
    pairs["stratum"] = pairs["cell"] + "|" + pairs["action_type"] + "|" + pairs["v_decile"].astype(str)
    pairs["creativity_z"] = zscore(pairs[raw(primary)])
    rows = []
    for outcome in ("next_opportunity_expected", "next_opportunity_quadrant", "next_nearest_opponent",
                    "next_opponents_within_5", f"next_{raw(primary)}", "next_v"):
        for fixed in (["stratum"], ["stratum", "match_id"]):
            rows.append(fe_ols(pairs, outcome, ["creativity_z"], fixed, "match_id", bootstrap)
                        .assign(outcome_mean=float(pairs[outcome].mean()), outcome_sd=float(pairs[outcome].std())))
    effects = pd.concat(rows, ignore_index=True)
    effects["estimate_in_outcome_sd"] = effects["estimate"] / effects["outcome_sd"]
    # Player enabling score: mean residual next-teammate opportunity after the player's actions.
    pairs["resid_next_opportunity"] = demean(pairs[["next_opportunity_expected"]], pairs["stratum"])["next_opportunity_expected"]
    players = pairs.groupby(["player_id", "player"]).agg(
        enabling=("resid_next_opportunity", "mean"), actions=("resid_next_opportunity", "size"),
        creativity=(raw(primary), "mean"), team=("team", lambda p: p.mode().iat[0]),
        position=("position", lambda p: p.mode().iat[0] if p.notna().any() else "")).reset_index()
    players = players[players["actions"] >= 200].sort_values("enabling", ascending=False)
    corr = float(players[["enabling", "creativity"]].corr(method="spearman").iloc[0, 1])
    return {"enabling_effects": effects, "enabling_players": players,
            "enabling_summary": pd.DataFrame([{"pairs": len(pairs), "players": len(players),
                                               "spearman_enabling_vs_own_creativity": corr}])}
