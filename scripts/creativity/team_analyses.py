"""Team, player-context and season-trend analyses on a common yardstick (the reference model).

    python scripts/creativity/team_analyses.py --tables outputs/creativity/reference_holdout.csv.gz \\
        --contrast outputs/creativity/crossfit_fold0.csv.gz outputs/creativity/crossfit_fold1.csv.gz

* teams vs. their opponents in the same matches (team-seasons with enough matches);
* players across contexts (e.g. Messi at Barcelona, PSG, Argentina, Inter Miami): player minus
  teammates in the same matches, and percentile among those teammates;
* change over a season (e.g. Leverkusen 2023/24): slope with a permutation test, second minus
  first half, both for the team and for team minus opponents.

``--contrast`` repeats the player analysis with cross-fitted tables on the same matches, which
shows how much the yardstick matters (a cross-fitted model learned each club's style).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from soccer_ml.creativity.analysis import player_in_contexts, season_trend, team_summary
from soccer_ml.creativity.events import load_nicknames
from soccer_ml.creativity.measures import SCOPES, compute_measures, player_scores
from soccer_ml.creativity.validation import markdown, match_player_names

DEFAULT_PLAYERS = ["Lionel Messi", "Neymar", "Ángel Di María", "Marco Verratti", "Kylian Mbappé", "Florian Wirtz",
                   "Pedri", "Antoine Griezmann", "Alejandro Grimaldo"]
DEFAULT_SEASONS = ["Bayer Leverkusen | 1. Bundesliga 2023/2024", "Paris Saint-Germain | Ligue 1 2021/2022",
                   "Paris Saint-Germain | Ligue 1 2022/2023", "Barcelona | La Liga 2020/2021"]
DEFAULT_MEASURES = ["good_surprise", "good_rarity", "c51_tail_rarity", "rarity_x_adv", "rarity", "advantage"]


def read(paths: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(path, low_memory=False) for path in paths], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Team and change analyses of creativity.")
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--contrast", nargs="*", default=[], help="Cross-fitted tables for the same matches.")
    parser.add_argument("--players", nargs="*", default=DEFAULT_PLAYERS)
    parser.add_argument("--seasons", nargs="*", default=DEFAULT_SEASONS, help="'team | competition season' contexts.")
    parser.add_argument("--measures", nargs="*", default=DEFAULT_MEASURES)
    parser.add_argument("--scope", choices=SCOPES, default="intended")
    parser.add_argument("--min-matches", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--lineups-dir", default="data/open-data/data/lineups")
    parser.add_argument("--output-dir", default="outputs/creativity/team_analyses")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    table = read(args.tables)
    scores = compute_measures(table, tuple(args.measures), scope=args.scope)
    measures = list(scores.columns)
    lines = ["# Team and change analyses", "", f"Tables: {', '.join(args.tables)}; scope '{args.scope}'; "
             f"{len(table)} decisions in {table['match_id'].nunique()} matches; measures: {', '.join(measures)}.",
             "Intervals: 95% match bootstrap. Differences are paired by match.", ""]

    teams = team_summary(table, scores, args.min_matches, args.bootstrap)
    teams.to_csv(output / "team_summary.csv", index=False)
    lines += ["## Teams vs. their opponents (same matches)", "",
              markdown(teams[["context", "measure", "team", "opponents", "difference", "low", "high", "matches"]], 4), ""]

    players = player_scores(table, scores, 1)
    nicknames = load_nicknames(Path(args.lineups_dir), table["match_id"].unique())
    ids = match_player_names(players, pd.Series(args.players), nicknames)
    contrast = read(args.contrast) if args.contrast else None
    if contrast is not None:
        contrast = contrast[contrast["match_id"].isin(table["match_id"].unique())].reset_index(drop=True)
        contrast_scores = compute_measures(contrast, tuple(measures), scope=args.scope)
    results = []
    for name, pid in ids.items():
        if not isinstance(pid, (int, float)) or pid != pid:
            lines += [f"_{name}: {'ambiguous' if pid == 'ambiguous' else 'not found'}_", ""]
            continue
        result = player_in_contexts(table, scores, int(pid), bootstrap=args.bootstrap).assign(name=name)
        if contrast is not None:
            other = player_in_contexts(contrast, contrast_scores, int(pid), bootstrap=args.bootstrap)
            if len(other):
                other = other[["context", "measure", "difference", "percentile_in_team"]].rename(
                    columns={"difference": "crossfit_difference", "percentile_in_team": "crossfit_percentile"})
                result = result.merge(other, on=["context", "measure"], how="left")
        results.append(result)
    if results:
        contexts = pd.concat(results, ignore_index=True)
        contexts.to_csv(output / "player_contexts.csv", index=False)
        columns = ["name", "context", "measure", "player", "teammates", "difference", "low", "high",
                   "percentile_in_team", "teammates_ranked", "first_decisions", "matches"]
        columns += [c for c in ("crossfit_difference", "crossfit_percentile") if c in contexts]
        lines += ["## Players across contexts (player minus teammates in the same matches)", "",
                  "percentile_in_team: among teammates with at least 100 in-scope decisions in that context.", ""]
        for name, part in contexts.groupby("name", sort=False):
            lines += [f"### {name}", "", markdown(part[columns].drop(columns="name"), 4), ""]

    lines += ["## Change over a season", ""]
    for context in args.seasons:
        per_match, trend = season_trend(table, scores, context, bootstrap=args.bootstrap)
        if trend.empty:
            lines += [f"_{context}: not enough matches_", ""]
            continue
        slug = context.replace(" | ", "_").replace(" ", "_").replace("/", "-")
        per_match.to_csv(output / f"season_{slug}.csv", index=False)
        trend.to_csv(output / f"trend_{slug}.csv", index=False)
        lines += [f"### {context} ({len(per_match)} matches)", "", markdown(trend, 4), ""]
    (output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
