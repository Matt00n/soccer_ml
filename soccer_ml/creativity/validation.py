"""Automated validity checks of creativity measures (no human input needed).

Each function returns a tidy DataFrame; ``scripts/creativity/validate_measures.py``
assembles them into a report.  The checks follow the validation plan:

components in the tail
    :func:`policy_calibration`, :func:`pit_uniformity` (is surprise calibrated,
    also for rare actions and in every context?), :func:`value_calibration_by_rarity`
    and :func:`winners_curse` (are values of rare / top-ranked actions inflated?)
artefacts
    :func:`artefact_rates` (out-of-view targets, miscommunications, deflections,
    restarts among each measure's top decisions)
convergent / discriminant evidence
    :func:`annotation_controls` (StatsBomb annotations as independent positive
    and negative controls), :func:`line_breaking`, :func:`discriminant`
reliability and robustness
    :func:`split_half_reliability`, :func:`cross_competition_stability`,
    :func:`model_agreement`
players and external criteria
    :func:`player_rankings` (match-bootstrap rank intervals), :func:`known_groups`,
    :func:`external_ratings`, :func:`famous_moments`
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from soccer_ml.creativity.measures import player_scores, position_group, top_fraction_flags
from soccer_ml.creativity.scoring import NUM_CALIBRATION_BINS, calibration_bin_labels

# -- helpers -------------------------------------------------------------------------


def auc(scores: np.ndarray, positive: np.ndarray) -> tuple[float, float, int]:
    """Rank AUC of ``scores`` for ``positive`` (ties count half) with its Hanley-McNeil standard error."""
    keep = ~np.isnan(scores)
    scores, positive = scores[keep], positive[keep].astype(bool)
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan"), n1
    ranks = stats.rankdata(scores)
    value = (ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    q1, q2 = value / (2 - value), 2 * value**2 / (1 + value)
    variance = (value * (1 - value) + (n1 - 1) * (q1 - value**2) + (n0 - 1) * (q2 - value**2)) / (n1 * n0)
    return float(value), float(math.sqrt(max(variance, 0.0))), n1


def spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    if keep.sum() < 3 or np.ptp(a[keep]) == 0 or np.ptp(b[keep]) == 0:
        return float("nan")
    return float(stats.spearmanr(a[keep], b[keep]).statistic)


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    values = values.dropna()
    if len(values) < 2:
        return float(values.mean()) if len(values) else float("nan"), float("nan")
    return float(values.mean()), float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))


def markdown(frame: pd.DataFrame, digits: int = 3) -> str:
    """A DataFrame as a GitHub table (no tabulate dependency)."""
    if frame.empty:
        return "_(no rows)_\n"
    def cell(value) -> str:
        if isinstance(value, float):
            return "" if value != value else f"{value:.{digits}f}"
        return str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


# -- components in the tail ----------------------------------------------------------------


def policy_calibration(table: pd.DataFrame, by: Optional[str] = None) -> pd.DataFrame:
    """Expected (Σ policy mass) vs. observed choices per log10-probability band, optionally per group.

    Under calibration observed / expected = 1 in every band, in particular in
    the rare bands that drive surprise-based measures.
    """
    groups = [("all", table)] if by is None else list(table.groupby(by, sort=True))
    rows = []
    labels = calibration_bin_labels()
    for name, part in groups:
        observed = np.bincount(part["calib_bin"].astype(int), minlength=NUM_CALIBRATION_BINS)
        for k in range(NUM_CALIBRATION_BINS):
            expected = float(part[f"calib_mass_{k}"].sum())
            if expected < 1e-9 and observed[k] == 0:
                continue
            low, high = stats.chi2.ppf(0.025, 2 * observed[k]) / 2 if observed[k] else 0.0, \
                stats.chi2.ppf(0.975, 2 * (observed[k] + 1)) / 2
            rows.append({
                "group": name, "log10_p_band": labels[k], "expected": expected, "observed": int(observed[k]),
                "ratio": observed[k] / expected if expected > 0 else float("nan"),
                "ratio_low": low / expected if expected > 0 else float("nan"),
                "ratio_high": high / expected if expected > 0 else float("nan"),
            })
    return pd.DataFrame(rows)


def pit_uniformity(table: pd.DataFrame, by: Optional[str] = None) -> pd.DataFrame:
    """Rarity should be uniform under a calibrated policy: mean 0.5, KS distance ~ 0, flat deciles."""
    groups = [("all", table)] if by is None else list(table.groupby(by, sort=True))
    rows = []
    for name, part in groups:
        u = part["rarity_random"].dropna().to_numpy()
        if len(u) < 50:
            continue
        deciles = np.histogram(u, bins=10, range=(0, 1))[0] / len(u)
        rows.append({
            "group": name, "n": len(u), "mean_rarity": float(part["rarity"].mean()),
            "ks_distance": float(stats.kstest(u, "uniform").statistic),
            "share_top_decile": float(deciles[-1]), "share_bottom_decile": float(deciles[0]),
            "max_decile_deviation": float(np.abs(deciles - 0.1).max()),
        })
    return pd.DataFrame(rows)


def value_calibration_by_rarity(table: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Predicted vs. realized value per rarity decile: are rare actions' values inflated?

    ``q_score`` vs. the Monte Carlo label the MC models were trained on (xG of
    the own team within 10 actions), and Q vs. the realized discounted return.
    """
    data = table.assign(rarity_bin=pd.cut(table["rarity"], np.linspace(0, 1, bins + 1), include_lowest=True))
    rows = []
    for interval, part in data.groupby("rarity_bin", observed=True):
        row = {"rarity": str(interval), "n": len(part)}
        if "label_xg_scores" in part:
            labeled = part[part["label_xg_scores"].notna()]
            row["pred_q_score"] = float(labeled["q_score"].mean())
            row["label_xg_scores"] = float(labeled["label_xg_scores"].mean())
            row["score_ratio"] = row["pred_q_score"] / row["label_xg_scores"] if row["label_xg_scores"] > 0 else float("nan")
        row["pred_q_net"] = float(part["q"].mean())
        row["realized_return_net"] = float(part["mc_return"].mean())
        bias, half_width = _mean_ci(part["q"] - part["mc_return"])
        row["bias"], row["bias_ci"] = bias, half_width
        rows.append(row)
    return pd.DataFrame(rows)


def winners_curse(table: pd.DataFrame, scores: pd.DataFrame, fraction: float = 0.01) -> pd.DataFrame:
    """Predicted vs. realized advantage among each measure's top decisions.

    Ranking on noisy value estimates selects overestimates; if the predicted
    advantage of the top decisions far exceeds the realized one, the measure
    exploits value errors rather than finding valuable actions.
    """
    flags = top_fraction_flags(scores, fraction)
    rows = []
    for name in scores:
        top = table[flags[name]]
        predicted, realized_mc = top["advantage"].mean(), top["mc_advantage"].mean()
        mc_mean, mc_ci = _mean_ci(top["mc_advantage"])
        row = {
            "measure": name, "n": len(top), "pred_advantage": predicted,
            "realized_td_advantage": top["td_advantage"].mean(),
            "realized_mc_advantage": realized_mc, "realized_mc_ci": mc_ci,
            "pred_q": top["q"].mean(), "realized_return": top["mc_return"].mean(),
        }
        if "label_xg_scores" in top:
            row["pred_q_score"] = top["q_score"].mean()
            row["label_xg_scores"] = top["label_xg_scores"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# -- artefacts --------------------------------------------------------------------


ARTEFACTS = {
    "end_out_of_view": lambda t: t["end_zone_visible"] == 0,
    "restart": lambda t: t["restart"].fillna("") != "",
    "miscommunication": lambda t: t["miscommunication"].astype(bool),
    "deflected": lambda t: t["deflected"].astype(bool),
    "failed": lambda t: t["success"] == 0,
    "clearance": lambda t: t["action_type"] == "clearance",
    "goalkeeper": lambda t: t["position"] == "Goalkeeper",
}


def artefact_rates(table: pd.DataFrame, scores: pd.DataFrame, fraction: float = 0.01) -> pd.DataFrame:
    """Share of suspicious decisions among each measure's top decisions vs. all decisions (lift)."""
    flags = top_fraction_flags(scores, fraction)
    masks = {name: rule(table).fillna(False).astype(bool) for name, rule in ARTEFACTS.items()}
    rows = [{"measure": "all decisions", "n": len(table), **{k: float(m.mean()) for k, m in masks.items()}}]
    for name in scores:
        top = flags[name]
        rows.append({"measure": name, "n": int(top.sum()), **{k: float(m[top].mean()) for k, m in masks.items()}})
    return pd.DataFrame(rows)


# -- convergent / discriminant evidence -----------------------------------------------------------


@dataclass(frozen=True)
class Control:
    name: str
    scope: str  # action type compared within ("pass", "dribble", "shot")
    expected: str  # "+" creative, "-" not creative, "?" exploratory
    description: str


CONTROLS = (
    Control("through_ball", "pass", "+", "through balls (StatsBomb technique)"),
    Control("no_touch", "pass", "+", "dummies: body part 'No Touch'"),
    Control("nutmeg", "dribble", "+", "nutmegs"),
    Control("flair_shot", "shot", "+", "backheel, lob or overhead-kick shots"),
    Control("cut_back", "pass", "?", "cut-backs"),
    Control("switch", "pass", "?", "switches of play"),
    Control("first_time", "pass", "?", "first-time passes"),
    Control("miscommunication", "pass", "-", "miscommunication passes"),
    Control("deflected", "pass", "-", "deflected passes (end location not intended)"),
    Control("goal_kick", "pass", "-", "goal kicks"),
    Control("throw_in", "pass", "-", "throw-ins"),
)


def _control_mask(table: pd.DataFrame, name: str) -> pd.Series:
    if name == "goal_kick":
        return table["restart"] == "Goal Kick"
    if name == "throw_in":
        return table["restart"] == "Throw-in"
    return table[name].astype(bool)


def annotation_controls(table: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """AUC of every measure for each StatsBomb annotation within its action type.

    AUC > 0.5 means annotated actions score higher.  Positive controls should
    exceed 0.5, negative controls fall below it; '?' rows are exploratory.
    """
    rows = []
    for control in CONTROLS:
        if control.name not in table and control.name not in ("goal_kick", "throw_in"):
            continue
        scope = (table["action_type"] == control.scope).to_numpy()
        positive = _control_mask(table, control.name).fillna(False).to_numpy()[scope]
        for name in scores:
            value, se, n1 = auc(scores[name].to_numpy()[scope], positive)
            rows.append({"control": control.name, "expected": control.expected, "scope": control.scope,
                         "n_annotated": n1, "measure": name, "auc": value, "auc_se": se})
    return pd.DataFrame(rows)


def line_breaking(table: pd.DataFrame, scores: pd.DataFrame, fraction: float = 0.01) -> pd.DataFrame:
    """Opponents bypassed (a model-free line-breaking count) vs. each measure, forward passes."""
    forward = ((table["action_type"] == "pass") & (table["end_x"] > table["start_x"])).to_numpy()
    flags = top_fraction_flags(scores, fraction)
    bypassed = table["opponents_bypassed"].to_numpy()
    rows = []
    for name in scores:
        top = flags[name].to_numpy() & forward
        rows.append({
            "measure": name,
            "spearman_forward_passes": spearman(scores[name].to_numpy()[forward], bypassed[forward]),
            "mean_bypassed_top": float(np.nanmean(bypassed[top])) if top.any() else float("nan"),
            "mean_bypassed_all_forward": float(np.nanmean(bypassed[forward])),
        })
    return pd.DataFrame(rows)


def discriminant(table: pd.DataFrame, scores: pd.DataFrame, min_decisions: int = 200) -> pd.DataFrame:
    """Is a measure just position, volume, risk or team? Correlations and variance shares.

    Action level: Spearman with pitch position (start x), success probability
    and advantage.  Player level: Spearman with decisions (volume), mean
    advantage, mean success probability, final-third share, and the share of
    between-player variance explained by team and by position group (eta²).
    """
    extra = table[["player_id", "player"]].assign(
        final_third=(table["start_x"] >= 80).astype(float),
        p_success_mean=table.get("p_success", pd.Series(np.nan, index=table.index)),
        advantage_mean=table["advantage"],
    )
    players = player_scores(table, scores, min_decisions)
    context = extra.groupby(["player_id", "player"])[["final_third", "p_success_mean", "advantage_mean"]].mean()
    players = players.join(context)
    players["group"] = players["position"].map(position_group)
    rows = []
    for name in scores:
        row = {"measure": name}
        values = scores[name].to_numpy()
        row["action_vs_start_x"] = spearman(values, table["start_x"])
        if "p_success" in table:
            row["action_vs_p_success"] = spearman(values, table["p_success"])
        row["action_vs_advantage"] = spearman(values, table["advantage"])
        row["player_vs_decisions"] = spearman(players[name], players["decisions"])
        row["player_vs_mean_advantage"] = spearman(players[name], players["advantage_mean"])
        row["player_vs_final_third"] = spearman(players[name], players["final_third"])
        if players["p_success_mean"].notna().any():
            row["player_vs_mean_p_success"] = spearman(players[name], players["p_success_mean"])
        for factor in ("team", "group"):
            row[f"eta2_{factor}"] = _eta_squared(players[name], players[factor])
        rows.append(row)
    return pd.DataFrame(rows)


def _eta_squared(values: pd.Series, groups: pd.Series) -> float:
    data = pd.DataFrame({"v": values, "g": groups}).dropna()
    if data["g"].nunique() < 2:
        return float("nan")
    total = ((data["v"] - data["v"].mean()) ** 2).sum()
    between = data.groupby("g")["v"].agg(lambda v: len(v) * (v.mean() - data["v"].mean()) ** 2).sum()
    return float(between / total) if total > 0 else float("nan")


# -- reliability and robustness -------------------------------------------------------------------


def split_half_reliability(table: pd.DataFrame, scores: pd.DataFrame, min_decisions: int = 200) -> pd.DataFrame:
    """Player scores from alternating matches (odd vs. even per player), Spearman-Brown corrected."""
    order = table[["player_id", "match_date", "match_id"]].drop_duplicates().sort_values(["player_id", "match_date", "match_id"])
    order["half"] = order.groupby("player_id").cumcount() % 2
    half = table[["player_id", "match_id"]].merge(order, on=["player_id", "match_id"], how="left")["half"].to_numpy()
    counts = pd.crosstab(table["player_id"], half).reindex(columns=[0, 1], fill_value=0)
    eligible = counts.index[(counts.min(axis=1) >= min_decisions // 2)]
    rows = []
    for name in scores:
        means = scores[name].groupby([table["player_id"], half]).mean().unstack().reindex(columns=[0, 1])
        means = means.loc[means.index.intersection(eligible)].dropna()
        r = spearman(means[0], means[1]) if len(means) > 3 else float("nan")
        rows.append({"measure": name, "players": len(means), "split_half_spearman": r,
                     "spearman_brown": 2 * r / (1 + r) if r == r and r > -1 else float("nan")})
    return pd.DataFrame(rows)


def cross_competition_stability(
    table: pd.DataFrame, scores: pd.DataFrame, min_decisions: int = 150, min_players: int = 10
) -> pd.DataFrame:
    """Player scores in different competition-seasons (e.g. Euro 2020 vs. Euro 2024): trait stability."""
    key = table["competition"].astype(str) + " " + table["season"].astype(str)
    counts = table.groupby([table["player_id"], key]).size().unstack()
    means = {name: scores[name].groupby([table["player_id"], key]).mean().unstack() for name in scores}
    rows = []
    editions = sorted(key.unique())
    for i, first in enumerate(editions):
        for second in editions[i + 1:]:
            both = counts[[first, second]].dropna()
            both = both[(both >= min_decisions).all(axis=1)]
            if len(both) < min_players:
                continue
            for name in scores:
                pair = means[name].loc[both.index, [first, second]]
                rows.append({"first": first, "second": second, "players": len(both), "measure": name,
                             "spearman": spearman(pair[first], pair[second])})
    return pd.DataFrame(rows)


def model_agreement(
    table_a: pd.DataFrame, scores_a: pd.DataFrame, table_b: pd.DataFrame, scores_b: pd.DataFrame,
    fraction: float = 0.01, min_decisions: int = 100,
) -> pd.DataFrame:
    """Two independently trained model pairs on the same decisions: action- and player-level agreement."""
    left = pd.concat([table_a[["event_uuid", "player_id"]], scores_a.add_prefix("a_")], axis=1)
    right = pd.concat([table_b[["event_uuid"]], scores_b.add_prefix("b_")], axis=1)
    both = left.merge(right, on="event_uuid")
    rows = []
    for name in scores_a.columns.intersection(scores_b.columns):
        a, b = both[f"a_{name}"], both[f"b_{name}"]
        top_a, top_b = a >= a.quantile(1 - fraction), b >= b.quantile(1 - fraction)
        players = both.groupby("player_id")[[f"a_{name}", f"b_{name}"]].agg(["mean", "size"])
        players = players[players[(f"a_{name}", "size")] >= min_decisions]
        rows.append({
            "measure": name, "decisions": len(both), "action_spearman": spearman(a, b),
            "top_jaccard": float((top_a & top_b).sum() / max((top_a | top_b).sum(), 1)),
            "players": len(players),
            "player_spearman": spearman(players[(f"a_{name}", "mean")], players[(f"b_{name}", "mean")]),
        })
    return pd.DataFrame(rows)


# -- players and external criteria ----------------------------------------------------------------


def player_rankings(
    table: pd.DataFrame, scores: pd.DataFrame, min_decisions: int = 200, top_n: int = 25,
    bootstrap: int = 200, seed: int = 0,
) -> pd.DataFrame:
    """Top players per measure with 90% rank intervals from a match bootstrap."""
    players = player_scores(table, scores, min_decisions)
    keep = table["player_id"].isin(players.index.get_level_values("player_id"))
    data = table.loc[keep, ["player_id", "match_id"]]
    player_codes, player_index = pd.factorize(data["player_id"])
    match_codes, match_index = pd.factorize(data["match_id"])
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(match_index), np.full(len(match_index), 1 / len(match_index)), size=bootstrap).T
    rows = []
    for name in scores:
        values = scores.loc[keep, name].to_numpy()
        valid = ~np.isnan(values)
        sums = np.zeros((len(player_index), len(match_index)))
        counts = np.zeros_like(sums)
        np.add.at(sums, (player_codes[valid], match_codes[valid]), values[valid])
        np.add.at(counts, (player_codes[valid], match_codes[valid]), 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (sums @ weights) / (counts @ weights)  # (players, bootstrap)
        ranks = pd.DataFrame(boot).rank(ascending=False, axis=0).to_numpy()
        point = players[name].droplevel("player").reindex(player_index)
        order = point.rank(ascending=False, method="min")
        for i in np.argsort(order.to_numpy())[:top_n]:
            pid = player_index[i]
            info = players.xs(pid, level="player_id").iloc[0]
            rows.append({
                "measure": name, "rank": int(order.iat[i]), "player": info.name, "team": info["team"],
                "position": info["position"], "decisions": int(info["decisions"]), "score": float(point.iat[i]),
                "rank_p05": float(np.nanpercentile(ranks[i], 5)), "rank_p95": float(np.nanpercentile(ranks[i], 95)),
            })
    return pd.DataFrame(rows)


def match_player_names(players: pd.DataFrame, names: pd.Series, nicknames: Optional[dict] = None) -> pd.Series:
    """Player id for each listed name: all name tokens must occur in the StatsBomb name or nickname.

    Returns NaN for unmatched names and for ambiguous ones (several players).
    """
    candidates = players.reset_index()[["player_id", "player"]].drop_duplicates("player_id")
    tokens = {
        row.player_id: set(normalize_name(row.player).split()) | set(normalize_name((nicknames or {}).get(row.player_id, "")).split())
        for row in candidates.itertuples()
    }
    result = {}
    for listed in names:
        wanted = set(normalize_name(listed).split())
        hits = [pid for pid, have in tokens.items() if wanted and wanted <= have]
        result[listed] = hits[0] if len(hits) == 1 else (float("nan") if not hits else "ambiguous")
    return pd.Series(result)


def known_groups(
    players: pd.DataFrame, known_ids: set, measures: list[str], baselines: tuple[str, ...] = ("advantage", "rarity"),
    bootstrap: int = 2000, seed: int = 0,
) -> pd.DataFrame:
    """Known creative players vs. all others, compared within position groups.

    AUC of within-group percentiles with a 95% bootstrap interval (players
    resampled within the known and the other group), and the paired AUC
    difference to each baseline measure (same resamples), so a measure's
    added value over value-only or originality-only scores is tested directly.
    """
    data = players.reset_index()
    data["group"] = data["position"].map(position_group)
    known = data["player_id"].isin(known_ids).to_numpy()
    percentiles = {name: data.groupby("group")[name].rank(pct=True).to_numpy() for name in measures}
    rng = np.random.default_rng(seed)
    positive, negative = np.where(known)[0], np.where(~known)[0]
    samples = [np.concatenate([rng.choice(positive, len(positive)), rng.choice(negative, len(negative))])
               for _ in range(bootstrap if len(positive) else 0)]
    boot = {name: np.array([auc(percentiles[name][i], known[i])[0] for i in samples]) for name in measures}
    rows = []
    for name in measures:
        value, se, n1 = auc(percentiles[name], known)
        row = {
            "measure": name, "known_players": n1, "auc_within_position": value, "auc_se": se,
            "auc_low": float(np.nanpercentile(boot[name], 2.5)) if samples else float("nan"),
            "auc_high": float(np.nanpercentile(boot[name], 97.5)) if samples else float("nan"),
            "mean_known_percentile": float(np.nanmean(percentiles[name][known])) if n1 else float("nan"),
            "known_in_top_decile": float((percentiles[name][known] >= 0.9).mean()) if n1 else float("nan"),
        }
        for baseline in baselines:
            if baseline in boot and baseline != name and samples:
                difference = boot[name] - boot[baseline]
                row[f"diff_vs_{baseline}"] = float(np.nanmean(difference))
                row[f"diff_vs_{baseline}_low"] = float(np.nanpercentile(difference, 2.5))
                row[f"diff_vs_{baseline}_high"] = float(np.nanpercentile(difference, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def position_rankings(players: pd.DataFrame, measures: list[str], top: int = 5, bottom: int = 3) -> pd.DataFrame:
    """Top and bottom players per measure within gender x position group (within-group percentiles)."""
    data = players.reset_index()
    data["group"] = data["position"].map(position_group)
    data["gender"] = data["women"].map({True: "women", False: "men"}) if "women" in data else "all"
    rows = []
    for name in measures:
        data["percentile"] = data.groupby(["gender", "group"])[name].rank(pct=True)
        for (gender, group), part in data.groupby(["gender", "group"]):
            ordered = part.sort_values(name, ascending=False)
            picks = [("top", ordered.head(top))] + ([("bottom", ordered.tail(bottom).iloc[::-1])] if len(part) > top else [])
            for side, chosen in picks:
                for rank, row in enumerate(chosen.itertuples(), start=1):
                    rows.append({"measure": name, "gender": gender, "group": group, "players": len(part), "side": side,
                                 "rank": rank, "player": row.player, "team": row.team, "decisions": int(row.decisions),
                                 "score": float(getattr(row, name)), "percentile": float(row.percentile)})
    return pd.DataFrame(rows)


def external_ratings(players: pd.DataFrame, ratings: pd.DataFrame, measures: list[str]) -> pd.DataFrame:
    """Spearman of player scores with external ratings, raw and within position (group percentiles)."""
    data = players.reset_index().merge(ratings, on="player_id")
    data["group"] = data["position"].map(position_group)
    attributes = [c for c in ratings.columns if c not in ("player_id", "player_name", "source", "notes")
                  and pd.api.types.is_numeric_dtype(ratings[c])]
    rows = []
    for attribute in attributes:
        rated = data[data[attribute].notna()]
        attribute_pct = rated.groupby("group")[attribute].rank(pct=True)
        for name in measures:
            rows.append({
                "attribute": attribute, "measure": name, "players": len(rated),
                "spearman": spearman(rated[name], rated[attribute]),
                "spearman_within_position": spearman(rated.groupby("group")[name].rank(pct=True), attribute_pct),
            })
    return pd.DataFrame(rows)


def famous_moments(table: pd.DataFrame, scores: pd.DataFrame, moments: pd.DataFrame) -> pd.DataFrame:
    """Percentile of listed famous creative moments under each measure (recall check).

    A moment is located by team names (both must appear in the match), the
    match minute (±1) and the player name (token match); ``action_type`` narrows
    it further if given.  The latest matching decision before the listed
    outcome is used when several remain (an assist's pass, not the receipt).
    """
    percentiles = scores.rank(pct=True)
    teams = (table["home_team"].map(normalize_name) + " | " + table["away_team"].map(normalize_name))
    players = table["player"].map(normalize_name)
    rows = []
    for moment in moments.itertuples():
        mask = pd.Series(True, index=table.index)
        for team in (getattr(moment, "team_a", ""), getattr(moment, "team_b", "")):
            if isinstance(team, str) and team.strip():
                mask &= teams.str.contains(normalize_name(team), regex=False)
        minute = getattr(moment, "minute", float("nan"))
        if minute == minute:
            mask &= (table["minute"] - float(minute)).abs() <= 1
        wanted = set(normalize_name(getattr(moment, "player", "")).split())
        if wanted:
            mask &= players.map(lambda name: wanted <= set(name.split()))
        action = getattr(moment, "action_type", "")
        if isinstance(action, str) and action.strip():
            mask &= table["action_type"] == action.strip().lower()
        hits = table.index[mask]
        row = {"moment": getattr(moment, "description", ""), "matches_found": len(hits)}
        if len(hits):
            chosen = table.loc[hits].sort_values(["period", "minute", "second"]).index[-1]
            row.update(event_uuid=table.at[chosen, "event_uuid"], action_type=table.at[chosen, "action_type"],
                       **{f"pct_{name}": float(percentiles.at[chosen, name]) for name in scores})
        rows.append(row)
    return pd.DataFrame(rows)
