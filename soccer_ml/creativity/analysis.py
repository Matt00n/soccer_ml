"""Higher-order creativity analyses: teams, players across contexts, change over a season.

All intervals come from a match bootstrap (decisions of one match are not
independent).  Differences between a player and their teammates resample the
same matches for both, so the comparison is paired and shared match effects
(opponent, score, weather) cancel.

These analyses are only meaningful on a common yardstick: on decisions scored
by a model that never trained on the teams compared (the reference model),
because a cross-fitted model learns a single-team club's style from its other
matches and then finds its players predictable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _match_sums(scores: pd.DataFrame, rows: pd.Series, matches: pd.Series, index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Per match (``index``) and measure: sum and count of in-scope scores of the selected rows."""
    part = scores[rows]
    sums = part.groupby(matches[rows]).sum(min_count=1).reindex(index).fillna(0.0).to_numpy()
    counts = part.notna().groupby(matches[rows]).sum().reindex(index).fillna(0.0).to_numpy()
    return sums, counts


def _bootstrap_weights(n_matches: int, bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multinomial(n_matches, np.full(n_matches, 1 / n_matches), size=bootstrap).T  # (matches, B)


def compare_groups(
    scores: pd.DataFrame, matches: pd.Series, first: pd.Series, second: pd.Series,
    bootstrap: int = 1000, seed: int = 0,
) -> pd.DataFrame:
    """Mean of each measure for two row groups observed in the same matches, and their difference.

    Returns one row per measure: ``first``, ``second``, ``difference`` and its
    95% interval from resampling the matches that contain either group.
    """
    index = pd.Index(sorted(set(matches[first | second])))
    sums_a, counts_a = _match_sums(scores, first, matches, index)
    sums_b, counts_b = _match_sums(scores, second, matches, index)
    weights = _bootstrap_weights(len(index), bootstrap, np.random.default_rng(seed))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a, mean_b = sums_a.sum(0) / counts_a.sum(0), sums_b.sum(0) / counts_b.sum(0)
        boot = (weights.T @ sums_a) / (weights.T @ counts_a) - (weights.T @ sums_b) / (weights.T @ counts_b)
    rows = []
    for k, name in enumerate(scores.columns):
        rows.append({"measure": name, "first": mean_a[k], "second": mean_b[k], "difference": mean_a[k] - mean_b[k],
                     "low": float(np.nanpercentile(boot[:, k], 2.5)), "high": float(np.nanpercentile(boot[:, k], 97.5)),
                     "matches": len(index), "first_decisions": int(counts_a[:, k].sum()),
                     "second_decisions": int(counts_b[:, k].sum())})
    return pd.DataFrame(rows)


def contexts(table: pd.DataFrame) -> pd.Series:
    """'team | competition season' of every decision."""
    return table["team"].astype(str) + " | " + table["competition"].astype(str) + " " + table["season"].astype(str)


def team_summary(table: pd.DataFrame, scores: pd.DataFrame, min_matches: int = 5, bootstrap: int = 1000) -> pd.DataFrame:
    """Each team-season with at least ``min_matches``: team vs. its opponents in the same matches."""
    context = contexts(table)
    played = table.groupby(context)["match_id"].nunique()
    rows = []
    for name in played.index[played >= min_matches]:
        own = context == name
        in_matches = table["match_id"].isin(table.loc[own, "match_id"].unique())
        result = compare_groups(scores, table["match_id"], own, in_matches & ~own, bootstrap)
        rows.append(result.assign(context=name).rename(columns={"first": "team", "second": "opponents"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def player_in_contexts(
    table: pd.DataFrame, scores: pd.DataFrame, player_id: int, min_decisions: int = 100, bootstrap: int = 1000,
) -> pd.DataFrame:
    """A player in every team-season: vs. teammates in the same matches, and percentile among those teammates.

    The difference to teammates removes what is shared by the team (style,
    opponents, match situations), so contexts can be compared even when the
    team's level differs.
    """
    context = contexts(table)
    is_player = table["player_id"] == player_id
    rows = []
    for name in context[is_player].unique():
        team_rows = context == name
        own = team_rows & is_player
        matches = table.loc[own, "match_id"].unique()
        teammates = team_rows & ~is_player & table["match_id"].isin(matches)
        result = compare_groups(scores, table["match_id"], own, teammates, bootstrap)
        team_scores = scores[team_rows & table["match_id"].isin(matches)]
        players = team_scores.groupby(table.loc[team_scores.index, "player_id"])
        means = players.mean()[players.count().max(axis=1) >= min_decisions]
        percentile = means.rank(pct=True)
        result["percentile_in_team"] = [percentile.at[player_id, m] if player_id in percentile.index else np.nan
                                        for m in result["measure"]]
        result["teammates_ranked"] = len(means) - int(player_id in means.index)
        rows.append(result.assign(context=name).rename(columns={"first": "player", "second": "teammates"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS slopes of each row of ``y`` (R, n) on ``x`` (n,)."""
    centered = x - x.mean()
    return (y - y.mean(axis=-1, keepdims=True)) @ centered / (centered @ centered)


def season_trend(
    table: pd.DataFrame, scores: pd.DataFrame, context_name: str, permutations: int = 10000,
    bootstrap: int = 2000, seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Change of a team's creativity over a season (matches in date order).

    Per match: the team's mean and its opponents' mean.  Per measure: OLS slope
    per match with a permutation p-value (match order shuffled), the
    second-half minus first-half difference with a bootstrap interval, and the
    same for the team-minus-opponents difference (which removes changes shared
    by both sides, e.g. a drift of the yardstick or the weather).
    """
    context = contexts(table)
    own = context == context_name
    match_ids = table.loc[own, ["match_id", "match_date"]].drop_duplicates().sort_values(["match_date", "match_id"])
    order = match_ids["match_id"].to_numpy()
    opponent_rows = table["match_id"].isin(order) & ~own
    team_means = scores[own].groupby(table.loc[own, "match_id"]).mean().reindex(order)
    opponent_means = scores[opponent_rows].groupby(table.loc[opponent_rows, "match_id"]).mean().reindex(order)
    per_match = team_means.add_prefix("team_").join(opponent_means.add_prefix("opponents_"))
    per_match.insert(0, "match_date", match_ids["match_date"].to_numpy())
    per_match.insert(1, "matchday", np.arange(1, len(order) + 1))
    rng = np.random.default_rng(seed)
    rows = []
    for name in scores.columns:
        for label, series in (("team", team_means[name]), ("team_minus_opponents", team_means[name] - opponent_means[name])):
            y = series.to_numpy(dtype=float)
            keep = ~np.isnan(y)
            n = int(keep.sum())
            if n < 6:
                continue
            x, y = np.arange(len(y), dtype=float)[keep], y[keep]
            slope = float(_slopes(x, y[None])[0])
            shuffled = np.stack([y[rng.permutation(n)] for _ in range(permutations)])
            first, second = y[: n // 2], y[n // 2:]
            boot = (rng.choice(second, (bootstrap, len(second))).mean(1) - rng.choice(first, (bootstrap, len(first))).mean(1))
            rows.append({
                "measure": name, "series": label, "matches": n, "season_mean": float(y.mean()),
                "slope_per_match": slope, "change_over_season": slope * (x[-1] - x[0]),
                "permutation_p": float((np.abs(_slopes(x, shuffled)) >= abs(slope)).mean()),
                "second_minus_first_half": float(second.mean() - first.mean()),
                "half_low": float(np.percentile(boot, 2.5)), "half_high": float(np.percentile(boot, 97.5)),
            })
    return per_match.reset_index(drop=True), pd.DataFrame(rows)
