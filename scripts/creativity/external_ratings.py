"""External validation with FIFA / EA Sports FC player ratings (sofifa-derived, step 2 A1).

Ratings: the public "FIFA 23 complete player dataset" (sofifa scrape; mirror on Hugging Face
``jsulz/FIFA23``): men FIFA 15-23 at launch (September of each year), women FIFA 16-23 with
all in-game updates.  Downloaded to ``data/external/ratings/``.

Each StatsBomb player-context (player x competition-season, >= ``--min-decisions`` in-scope
decisions) is matched to the rating version current at the competition:

    contemporaneous: Euro 2020 -> FIFA 21, La Liga 20/21 -> 21, Ligue 1 21/22 -> 22,
                     Ligue 1 22/23 / WC 2022 / MLS 2023 -> 23, Women's Euro 2022 -> 22 (last update
                     before the tournament), Women's World Cup 2023 -> 23
    lagged (no FC 24/25 data): Euro 2024, Bundesliga 23/24 -> FIFA 23; Women's Euro 2025 -> FIFA 23

Matching: candidates are rating rows of the same team (national team: nationality; club: club
name), scored by name-token overlap with the StatsBomb name and nickname; a match needs a
unique best candidate above a threshold.

Analyses (within competition-season x position group, so neither the yardstick nor position
drives them):

* Spearman correlation of each creativity measure with each attribute (convergent: vision,
  passing, ball control, dribbling; discriminant: strength, jumping, stamina, sprint speed,
  tackling), raw and partial on overall rating, with player-clustered bootstrap intervals;
* paired differences of the vision correlation between measures;
* known groups from independent labels: "Playmaker (AI)" trait / #Playmaker tag, "Flair" trait.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from soccer_ml.creativity.events import load_nicknames
from soccer_ml.creativity.measures import compute_measures, position_group
from soccer_ml.creativity.validation import auc, markdown, spearman

MEASURES = ("good_surprise", "good_rarity", "c51_tail_rarity", "unxpass", "rarity", "advantage", "rarity_x_adv")
ATTRIBUTES = {
    "mentality_vision": "convergent", "attacking_short_passing": "convergent", "skill_long_passing": "convergent",
    "skill_ball_control": "convergent", "skill_dribbling": "convergent", "skill_curve": "convergent",
    "mentality_composure": "convergent", "overall": "quality",
    "power_strength": "discriminant", "power_jumping": "discriminant", "power_stamina": "discriminant",
    "movement_sprint_speed": "discriminant", "defending_standing_tackle": "discriminant",
}
# competition-season -> (gender, rating version, lagged?, cutoff date for women's in-game updates)
VERSIONS = {
    "UEFA Euro 2020": ("male", 21, False, None), "La Liga 2020/2021": ("male", 21, False, None),
    "Ligue 1 2021/2022": ("male", 22, False, None), "Ligue 1 2022/2023": ("male", 23, False, None),
    "FIFA World Cup 2022": ("male", 23, False, None), "Major League Soccer 2023": ("male", 23, False, None),
    "UEFA Euro 2024": ("male", 23, True, None), "1. Bundesliga 2023/2024": ("male", 23, True, None),
    "UEFA Women's Euro 2022": ("female", 22, False, "2022-07-06"),
    "Women's World Cup 2023": ("female", 23, False, "2023-07-20"),
    "UEFA Women's Euro 2025": ("female", 23, True, "2025-07-02"),
}
ALIASES = {
    "iran islamic republic of": "iran", "ir iran": "iran", "korea south": "korea republic", "south korea": "korea republic",
    "macedonia republic of": "north macedonia", "united states of america": "united states", "usa": "united states",
    "venezuela bolivarian republic": "venezuela", "ireland": "republic of ireland", "china": "china pr",
    "cape verde": "cape verde islands", "congo kinshasa": "congo dr", "dr congo": "congo dr", "guinea bissau": "guinea bissau",
    "turkiye": "turkey", "czechia": "czech republic", "cote d ivoire": "cote d ivoire", "ivory coast": "cote d ivoire",
    "paris saint germain": "paris saint germain", "bayer 04 leverkusen": "bayer leverkusen", "fc barcelona": "barcelona",
    "inter miami cf": "inter miami", "northern ireland": "northern ireland", "wnt finland": "finland",
}


def norm(text) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def team_key(name) -> str:
    key = norm(name).replace(" women s", "").replace(" women", "").strip()
    return ALIASES.get(key, key)


def name_tokens(*names) -> set:
    return {t for n in names if isinstance(n, str) for t in norm(n).split() if len(t) > 1}


def load_ratings(directory: Path) -> pd.DataFrame:
    columns = ["player_id", "fifa_version", "fifa_update_date", "short_name", "long_name", "nationality_name",
               "club_name", "player_traits", "player_tags", "player_positions", *ATTRIBUTES]
    male = pd.read_csv(directory / "male_players_legacy.csv", usecols=columns, low_memory=False).assign(gender="male")
    female = pd.read_csv(directory / "female_players.csv", usecols=columns, low_memory=False).assign(gender="female")
    ratings = pd.concat([male, female], ignore_index=True)
    ratings["nation_key"] = ratings["nationality_name"].map(team_key)
    ratings["club_key"] = ratings["club_name"].map(team_key)
    return ratings


def version_rows(ratings: pd.DataFrame, gender: str, version: int, cutoff) -> pd.DataFrame:
    rows = ratings[(ratings["gender"] == gender) & (ratings["fifa_version"] == version)]
    if cutoff:
        rows = rows[rows["fifa_update_date"] <= cutoff]
    return rows.sort_values("fifa_update_date").drop_duplicates("player_id", keep="last")


def match_players(contexts: pd.DataFrame, ratings: pd.DataFrame, nicknames: dict) -> pd.DataFrame:
    """Best unique rating row per player-context (team-restricted, name-token overlap)."""
    rows = []
    cache = {}
    for ctx in contexts.itertuples():
        spec = VERSIONS.get(ctx.competition_season)
        if spec is None:
            continue
        gender, version, lagged, cutoff = spec
        if (gender, version, cutoff) not in cache:
            table = version_rows(ratings, gender, version, cutoff)
            table = table.assign(tokens=[name_tokens(a, b) for a, b in zip(table["long_name"], table["short_name"])])
            cache[(gender, version, cutoff)] = table
        table = cache[(gender, version, cutoff)]
        team = team_key(ctx.team)
        candidates = table[(table["nation_key"] == team) | table["club_key"].str.contains(team, regex=False, na=False)]
        wanted = name_tokens(ctx.player, nicknames.get(ctx.player_id, ""))
        best, score, second = None, 0.0, 0.0
        for cand in candidates.itertuples():
            overlap = len(wanted & cand.tokens)
            if not overlap:
                continue
            value = overlap / len(cand.tokens | wanted) + overlap * 0.1
            if value > score:
                best, second, score = cand, score, value
            elif value > second:
                second = value
        status = "matched" if best is not None and score >= 0.25 and score - second >= 0.05 else (
            "ambiguous" if best is not None and score >= 0.25 else "not found")
        row = {"player_id": ctx.player_id, "competition_season": ctx.competition_season, "status": status,
               "version": version, "lagged": lagged}
        if status == "matched":
            row.update({"fifa_id": best.player_id, "fifa_name": best.long_name, "match_score": score,
                        "traits": best.player_traits, "tags": best.player_tags,
                        **{a: getattr(best, a) for a in ATTRIBUTES}})
        rows.append(row)
    return pd.DataFrame(rows)


def within_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby(["competition_season", "group"])[column].rank(pct=True)


def resamples(data: pd.DataFrame, bootstrap: int, rng) -> list:
    """Player-clustered bootstrap samples (row positions)."""
    players = data["player_id"].unique()
    index = data.groupby("player_id").indices
    return [np.concatenate([index[p] for p in rng.choice(players, len(players))]) for _ in range(bootstrap)]


def _corr_arrays(x: np.ndarray, y: np.ndarray, z: np.ndarray | None) -> float:
    keep = ~(np.isnan(x) | np.isnan(y) | (np.isnan(z) if z is not None else False))
    x, y = x[keep], y[keep]
    if z is not None:
        z = z[keep]
        x = x - np.polyval(np.polyfit(z, x, 1), z)
        y = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 3 else float("nan")


def _corr(data: pd.DataFrame, a: str, b: str, partial: str | None) -> float:
    return _corr_arrays(data[a].to_numpy(float), data[b].to_numpy(float),
                        data[partial].to_numpy(float) if partial else None)


def bootstrap_corr(data: pd.DataFrame, a: str, b: str, samples: list, partial: str | None = None) -> np.ndarray:
    x, y = data[a].to_numpy(float), data[b].to_numpy(float)
    z = data[partial].to_numpy(float) if partial else None
    return np.array([_corr_arrays(x[i], y[i], z[i] if z is not None else None) for i in samples])


def main() -> None:
    parser = argparse.ArgumentParser(description="External validation with FIFA ratings.")
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--ratings-dir", default="data/external/ratings")
    parser.add_argument("--min-decisions", type=int, default=150)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/creativity/step2/A1_external_ratings")
    args = parser.parse_args()
    started = time.time()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    table = pd.concat([pd.read_csv(p, low_memory=False) for p in args.tables], ignore_index=True)
    scores = compute_measures(table, MEASURES, scope="intended")
    table["competition_season"] = table["competition"].astype(str) + " " + table["season"].astype(str)
    keys = [table["player_id"], table["player"], table["team"], table["competition_season"]]
    contexts = scores.groupby(keys).mean()
    contexts["decisions"] = scores.notna().any(axis=1).groupby(keys).sum()
    contexts["position"] = table["position"].groupby(keys).agg(lambda p: p.mode().iat[0] if p.notna().any() else "")
    contexts = contexts[contexts["decisions"] >= args.min_decisions].reset_index()
    contexts.columns = ["player_id", "player", "team", "competition_season", *contexts.columns[4:]]
    contexts["group"] = contexts["position"].map(position_group)
    contexts = contexts[contexts["group"] != "GK"]
    nicknames = load_nicknames(Path("data/open-data/data/lineups"), table["match_id"].unique())
    ratings = load_ratings(Path(args.ratings_dir))
    matched = match_players(contexts, ratings, nicknames)
    coverage = matched.merge(contexts[["player_id", "competition_season", "team"]], on=["player_id", "competition_season"])
    coverage = coverage.groupby(["competition_season", "status"]).size().unstack(fill_value=0).reset_index()
    data = contexts.merge(matched[matched["status"] == "matched"], on=["player_id", "competition_season"])
    data.to_csv(output / "matched_player_contexts.csv", index=False)
    print(f"{len(contexts)} player-contexts, {len(data)} matched ({time.time() - started:.0f}s)")

    measures = [m for m in MEASURES if m in data]
    for column in measures + list(ATTRIBUTES):
        data[f"r_{column}"] = within_rank(data, column)
    rng = np.random.default_rng(0)
    results, paired = [], []
    for subset, frame in (("contemporaneous", data[~data["lagged"]]), ("all incl. lagged", data)):
        frame = frame[frame.groupby(["competition_season", "group"])["player_id"].transform("size") >= 4].reset_index(drop=True)
        samples = resamples(frame, args.bootstrap, rng)
        for attribute, role in ATTRIBUTES.items():
            draws_by_measure = {}
            for measure in measures:
                for partial in (None, "r_overall"):
                    if partial and attribute == "overall":
                        continue
                    estimate = _corr(frame, f"r_{measure}", f"r_{attribute}", partial)
                    draws = bootstrap_corr(frame, f"r_{measure}", f"r_{attribute}", samples, partial)
                    results.append({"subset": subset, "attribute": attribute, "role": role, "measure": measure,
                                    "partial_on_overall": bool(partial), "rho": estimate,
                                    "low": float(np.nanpercentile(draws, 2.5)), "high": float(np.nanpercentile(draws, 97.5)),
                                    "player_contexts": len(frame), "players": frame["player_id"].nunique()})
                    if partial is None:
                        draws_by_measure[measure] = draws
            if attribute == "mentality_vision" and subset == "contemporaneous":
                for measure in measures:
                    for baseline in ("advantage", "rarity", "rarity_x_adv"):
                        if measure != baseline:
                            diff = draws_by_measure[measure] - draws_by_measure[baseline]
                            paired.append({"measure": measure, "baseline": baseline, "difference": float(np.nanmean(diff)),
                                           "low": float(np.nanpercentile(diff, 2.5)), "high": float(np.nanpercentile(diff, 97.5))})
    correlations = pd.DataFrame(results)
    correlations.to_csv(output / "correlations.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "vision_paired_differences.csv", index=False)

    groups = []
    frame = data[~data["lagged"]]
    labels = {
        "playmaker": frame["traits"].fillna("").str.contains("Playmaker") | frame["tags"].fillna("").str.contains("#Playmaker"),
        "flair": frame["traits"].fillna("").str.contains("Flair"),
        "long_passer": frame["traits"].fillna("").str.contains("Long Passer"),
        "technical_dribbler": frame["traits"].fillna("").str.contains("Technical Dribbler"),
    }
    for label, flag in labels.items():
        for measure in measures:
            value, se, n1 = auc(frame[f"r_{measure}"].to_numpy(float), flag.to_numpy())
            groups.append({"label": label, "measure": measure, "auc": value, "low": value - 1.96 * se,
                           "high": value + 1.96 * se, "labelled": n1, "player_contexts": len(frame)})
    known = pd.DataFrame(groups)
    known.to_csv(output / "known_groups_fifa_labels.csv", index=False)
    coverage.to_csv(output / "coverage.csv", index=False)

    main_table = correlations[(correlations["subset"] == "contemporaneous") & (~correlations["partial_on_overall"])]
    pivot = main_table.pivot_table(index=["role", "attribute"], columns="measure", values="rho").reset_index()
    partial_table = correlations[(correlations["subset"] == "contemporaneous") & correlations["partial_on_overall"]]
    pivot_partial = partial_table.pivot_table(index=["role", "attribute"], columns="measure", values="rho").reset_index()
    lines = ["# External validation with FIFA ratings", "",
             f"{len(contexts)} outfield player-contexts with >= {args.min_decisions} in-scope decisions; {len(data)} matched "
             f"({(~data['lagged']).sum()} contemporaneous). Correlations are Spearman of within (competition-season x "
             "position group) percentiles; 95% player-clustered bootstrap intervals in correlations.csv.", "",
             "## Coverage", "", markdown(coverage), "",
             "## Correlations (contemporaneous ratings)", "", markdown(pivot, 3), "",
             "## Partial correlations, controlling for overall rating", "", markdown(pivot_partial, 3), "",
             "## Vision: paired differences between measures", "", markdown(pd.DataFrame(paired), 3), "",
             "## Known groups from FIFA labels (AUC of within-group percentiles)", "",
             markdown(known.pivot_table(index=["label", "labelled"], columns="measure", values="auc").reset_index(), 3), ""]
    (output / "report.md").write_text("\n".join(lines))
    (output / "manifest.json").write_text(json.dumps({
        "created": time.strftime("%Y-%m-%d %H:%M"), "tables": args.tables, "ratings": "huggingface jsulz/FIFA23 "
        "(male_players (legacy).csv, female_players.csv)", "versions": {k: list(v) for k, v in VERSIONS.items()},
        "min_decisions": args.min_decisions, "bootstrap": args.bootstrap, "seconds": round(time.time() - started)}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
