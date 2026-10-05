"""Automated validation report for all creativity measures on scored decision tables.

Reads one or more tables from ``score_decisions.py`` (e.g. both cross-fitting
folds), computes every available measure (``soccer_ml/creativity/measures.py``)
and writes ``report.md`` plus one CSV per check to ``--output-dir``.

Optional inputs (templates in ``validation/``; fill them *before* looking at
any ranking and keep them under version control, so they are pre-registered):

    --known-players    known creative players (known-groups validity)
    --external-ratings player attributes from an external source, e.g. EA FC Vision
    --famous-moments   famous creative moments (recall)
    --compare          tables of an independently trained model on overlapping
                       matches (robustness across model instances)

Example:
    python scripts/creativity/validate_measures.py \\
        --tables outputs/creativity/crossfit_fold0.csv.gz outputs/creativity/crossfit_fold1.csv.gz \\
        --compare outputs/creativity/main_val_c51.csv.gz --output-dir outputs/creativity/validation
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from soccer_ml.creativity.events import load_nicknames
from soccer_ml.creativity.measures import MEASURES, compute_measures, player_scores
from soccer_ml.creativity.validation import (
    CONTROLS,
    annotation_controls,
    artefact_rates,
    cross_competition_stability,
    discriminant,
    external_ratings,
    famous_moments,
    known_groups,
    line_breaking,
    markdown,
    match_player_names,
    model_agreement,
    pit_uniformity,
    player_rankings,
    policy_calibration,
    split_half_reliability,
    value_calibration_by_rarity,
    winners_curse,
)


def read_tables(paths: list[str]) -> pd.DataFrame:
    tables = [pd.read_csv(path, low_memory=False) for path in paths]
    table = pd.concat(tables, ignore_index=True)
    duplicated = table["event_uuid"].duplicated().sum()
    if duplicated:
        raise ValueError(f"{duplicated} decisions appear in several tables (overlapping folds?).")
    return table


def _file_record(path: str) -> str:
    data = Path(path).read_bytes()
    return f"`{path}` (sha256 {hashlib.sha256(data).hexdigest()[:12]}, modified {time.ctime(Path(path).stat().st_mtime)})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate creativity measures on scored decision tables.")
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--compare", nargs="*", default=[], help="Tables of another model on overlapping matches.")
    parser.add_argument("--known-players", help="CSV: player_name[, player_id, source, notes].")
    parser.add_argument("--external-ratings", help="CSV: player_name[, player_id], numeric attribute columns.")
    parser.add_argument("--famous-moments", help="CSV: description, team_a, team_b, minute, player[, action_type].")
    parser.add_argument("--min-decisions", type=int, default=200, help="Player-level checks: minimum decisions.")
    parser.add_argument("--top-fraction", type=float, default=0.01, help="'Top decisions' = this fraction per measure.")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--lineups-dir", default="data/open-data/data/lineups")
    parser.add_argument("--output-dir", default="outputs/creativity/validation")
    return parser.parse_args()


def resolve_players(listed: pd.DataFrame, players: pd.DataFrame, nicknames: dict) -> tuple[pd.Series, pd.DataFrame]:
    """Player ids for a user list, aligned with its rows: an explicit player_id wins, else name matching."""
    names = listed["player_name"].astype(str)
    matched = match_player_names(players, names, nicknames)
    explicit = pd.to_numeric(listed["player_id"], errors="coerce") if "player_id" in listed else pd.Series(
        float("nan"), index=listed.index)
    ids, report = [], []
    for name, pid in zip(names, explicit):
        if pid == pid:
            status = "ok (player_id given)"
        else:
            found = matched.get(name, float("nan"))
            if isinstance(found, str):
                status = "ambiguous: add a player_id"
            elif found != found:
                status = "not found (or fewer than --min-decisions decisions)"
            else:
                pid, status = float(found), "ok"
        ids.append(pid if status.startswith("ok") else float("nan"))
        report.append({"player_name": name, "player_id": ids[-1], "status": status})
    return pd.Series(ids, index=listed.index, dtype=float), pd.DataFrame(report)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    table = read_tables(args.tables)
    scores = compute_measures(table)
    measures = list(scores.columns)
    top = args.top_fraction
    print(f"{len(table)} decisions, {table['match_id'].nunique()} matches, measures: {', '.join(measures)}")

    sections: list[tuple[str, str, pd.DataFrame]] = []

    def add(key: str, title: str, frame: pd.DataFrame, note: str = "") -> None:
        frame.to_csv(output / f"{key}.csv", index=False)
        sections.append((title, note, frame))
        print(f"  {title} ({time.time() - started:.0f}s)")

    descriptions = pd.DataFrame([{"measure": m.name, "family": m.family, "definition": m.description,
                                  "in_scope": int(scores[m.name].notna().sum()) if m.name in scores else 0,
                                  "available": m.name in scores} for m in MEASURES])
    add("measures", "Measures", descriptions)

    # 1. Components in the tail.
    add("policy_calibration", "Policy calibration by probability band", policy_calibration(table),
        "Observed / expected choices per log10 π band. 1 = calibrated; the rare bands drive surprise-based measures.")
    table["gender"] = table["women"].map({True: "women", False: "men"}).fillna("unknown")
    table["competition_season"] = table["competition"].astype(str) + " " + table["season"].astype(str)
    add("policy_calibration_by_gender", "Policy calibration by gender", policy_calibration(table, "gender"))
    pit = pd.concat([pit_uniformity(table, by).assign(by=by) for by in
                     (None, "gender", "competition_season", "decision_context", "action_type")], ignore_index=True)
    add("pit_uniformity", "Rarity uniformity (PIT) by context", pit,
        "Under calibration mean rarity = 0.5 and KS distance ~ 0 in every group (measurement invariance).")
    add("value_calibration_by_rarity", "Value calibration by rarity decile", value_calibration_by_rarity(table),
        "Is Q inflated for rare actions? bias = mean(Q - realized return).")
    add("winners_curse", "Predicted vs. realized advantage of each measure's top decisions",
        winners_curse(table, scores, top), f"Top {top:.0%} per measure.")

    # 2. Artefacts.
    add("artefacts", "Artefact rates among top decisions", artefact_rates(table, scores, top),
        "Shares among each measure's top decisions vs. all decisions. High lifts flag measures driven by data artefacts.")

    # 3. Convergent / discriminant evidence.
    controls = annotation_controls(table, scores)
    add("annotation_controls", "StatsBomb annotations as controls (AUC within action type)",
        controls.pivot_table(index=["control", "expected", "scope", "n_annotated"], columns="measure",
                             values="auc").reset_index(),
        "Positive (+) controls should exceed 0.5, negative (-) fall below; '?' exploratory. "
        + "; ".join(f"{c.name}: {c.description}" for c in CONTROLS))
    add("line_breaking", "Line-breaking (opponents bypassed) vs. measures", line_breaking(table, scores, top))
    add("discriminant", "Discriminant checks (is it just position, volume, risk, team?)",
        discriminant(table, scores, args.min_decisions))

    # 4. Reliability and robustness.
    add("split_half", "Split-half reliability (alternating matches per player)",
        split_half_reliability(table, scores, args.min_decisions))
    add("cross_competition", "Stability across competition-seasons",
        cross_competition_stability(table, scores, args.min_decisions // 2))
    if args.compare:
        other = read_tables(args.compare)
        add("model_agreement", "Agreement with an independently trained model",
            model_agreement(table, scores, other, compute_measures(other), top, args.min_decisions // 2))

    # 5. Players and external criteria.
    players = player_scores(table, scores, args.min_decisions)
    players.reset_index().to_csv(output / "player_scores.csv", index=False)
    add("player_rankings", "Top players per measure (90% match-bootstrap rank intervals)",
        player_rankings(table, scores, args.min_decisions, 25, args.bootstrap))
    nicknames = load_nicknames(Path(args.lineups_dir), table["match_id"].unique())
    if args.known_players:
        listed = pd.read_csv(args.known_players, comment="#")
        ids, resolution = resolve_players(listed, players, nicknames)
        add("known_players_resolution", f"Known creative players: name resolution ({_file_record(args.known_players)})",
            resolution)
        add("known_groups", "Known creative players vs. others (within position groups)",
            known_groups(players, set(ids.dropna().astype(int)), measures))
    if args.external_ratings:
        ratings = pd.read_csv(args.external_ratings, comment="#")
        ids, resolution = resolve_players(ratings, players, nicknames)
        add("external_ratings_resolution", f"External ratings: name resolution ({_file_record(args.external_ratings)})",
            resolution)
        ratings = ratings.assign(player_id=ids).dropna(subset=["player_id"]).astype({"player_id": int})
        add("external_ratings", "Correlation with external ratings", external_ratings(players, ratings, measures))
    if args.famous_moments:
        moments = pd.read_csv(args.famous_moments, comment="#")
        add("famous_moments", f"Famous creative moments: percentile per measure ({_file_record(args.famous_moments)})",
            famous_moments(table, scores, moments))

    lines = ["# Creativity measure validation", "",
             f"Tables: {', '.join(_file_record(p) for p in args.tables)}", "",
             f"{len(table)} decisions in {table['match_id'].nunique()} matches; player checks use players with "
             f"at least {args.min_decisions} decisions; 'top' = top {top:.0%} per measure.", ""]
    for title, note, frame in sections:
        lines += [f"## {title}", ""] + ([note, ""] if note else []) + [markdown(frame), ""]
    (output / "report.md").write_text("\n".join(lines))
    print(f"wrote {output / 'report.md'} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
