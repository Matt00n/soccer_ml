"""Run all second-order creativity analyses and store every result table.

    python scripts/creativity/run_second_order.py \\
        --tables outputs/creativity/crossfit_c51_fold0.csv.gz outputs/creativity/crossfit_c51_fold1.csv.gz \\
        --reference outputs/creativity/reference_holdout.csv.gz --context-dir outputs/creativity/context

Output (``--output-dir``, default ``outputs/creativity/second_order``):

* one folder per section with a CSV per table and a ``summary.md``;
* ``hypotheses.csv``: the main hypothesis of each analysis on the primary
  measure, with Benjamini-Hochberg q-values across all of them;
* ``index.md`` and ``manifest.json`` (inputs with hashes, settings, git commit).

Measures are z-scored over in-scope decisions (scope ``intended``); the
primary measure is ``good_surprise``, the others are robustness checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from soccer_ml.creativity.measures import compute_measures
from soccer_ml.creativity.second_order import (
    opponents,
    personnel,
    player_profiles,
    score_reactions,
    sequences,
    surprise_premium,
    team_success,
    time_profiles,
)
from soccer_ml.creativity.validation import markdown

MEASURES = ["good_surprise", "c51_tail_rarity", "good_rarity", "rarity", "advantage"]


def load(paths: list[str], context: pd.DataFrame | None, measures: list[str]) -> pd.DataFrame:
    table = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)
    scores = compute_measures(table, tuple(measures), scope="intended")
    columns = {"in_scope": scores.notna().any(axis=1)}
    for name in scores:
        raw = scores[name]
        columns[f"raw_{name}"] = raw
        columns[f"z_{name}"] = (raw - raw.mean()) / raw.std()  # raw statistic columns (rarity, advantage) stay intact
    table = pd.concat([table.drop(columns=[c for c in columns if c in table]), pd.DataFrame(columns)], axis=1)
    if context is not None:
        table = table.merge(context, on="event_uuid", how="left")
    return table


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    order = p.sort_values()
    ranked = order * len(order) / np.arange(1, len(order) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(upper=1.0)
    return q.reindex(p.index)


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Second-order creativity analyses.")
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--reference", nargs="*", default=[], help="Reference-model tables (star absence on a common yardstick).")
    parser.add_argument("--context-dir", default="outputs/creativity/context")
    parser.add_argument("--output-dir", default="outputs/creativity/second_order")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--window", type=float, default=10.0, help="Event-study window (minutes).")
    args = parser.parse_args()
    started = time.time()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ctx = Path(args.context_dir)
    context = pd.read_csv(ctx / "decisions.csv.gz")
    frames = {name: pd.read_csv(ctx / f"{name}.csv") for name in
              ("goals", "substitutions", "red_cards", "tactical_shifts", "possessions", "team_matches", "appearances")}
    d = load(args.tables, context, MEASURES)
    measures = [f"z_{m}" for m in MEASURES if f"z_{m}" in d]
    primary = measures[0]
    reference = load(args.reference, None, MEASURES) if args.reference else None
    print(f"{len(d)} decisions ({int(d['in_scope'].sum())} in scope), {d['match_id'].nunique()} matches; "
          f"measures {measures}; primary {primary}")

    hypotheses = []
    sections = []

    def store(section: str, title: str, tables: dict, notes: str = "") -> None:
        folder = output / section
        folder.mkdir(exist_ok=True)
        lines = [f"# {title}", "", notes, ""] if notes else [f"# {title}", ""]
        for name, frame in tables.items():
            frame.to_csv(folder / f"{name}.csv", index=False)
            shown = frame if len(frame) <= 40 else frame.head(40)
            lines += [f"## {name} ({len(frame)} rows{', first 40 shown' if len(frame) > 40 else ''})", "",
                      markdown(shown, 4), ""]
        (folder / "summary.md").write_text("\n".join(lines))
        sections.append((section, title, list(tables)))
        print(f"  {title}: {len(tables)} tables ({time.time() - started:.0f}s)")

    runs = [
        ("01_surprise_premium", "Does originality pay beyond the predicted value?",
         lambda: surprise_premium(d, args.bootstrap),
         "Residual = realized minus predicted value (one-step TD and Monte Carlo), slope on rarity within Q deciles."),
        ("02_sequences", "Sequence creativity: possessions, chain position, creative flow",
         lambda: sequences(d, frames["possessions"], measures, args.bootstrap),
         "Possession creativity from the possession team's in-scope decisions, shots excluded; strata = start value "
         "decile x third x play pattern x length."),
        ("03_team_success", "Team creativity level, concentration and success",
         lambda: team_success(d, frames["team_matches"], frames["possessions"], primary, args.bootstrap),
         "Creativity measured at level score (reverse causality); team-context fixed effects; opponent strength = "
         "leave-one-out xG difference."),
        ("04_score_reactions", "Reactions to goals, comebacks and game management",
         lambda: score_reactions(d, frames["goals"], frames["team_matches"], primary, measures, args.bootstrap, args.window),
         f"Event studies: {args.window:.0f}-minute windows after vs. before, minus placebo windows at the same clock. "
         "The models condition on goal difference and minute: effects are relative to what is typical in that state."),
        ("05_personnel", "Substitutions, on/off splits, red cards, tactical shifts, star absence",
         lambda: personnel(d, frames["substitutions"], frames["red_cards"], frames["tactical_shifts"],
                           frames["appearances"], primary, args.bootstrap, args.window, reference),
         "On/off: teammates' creativity adjusted for period x 15-minute bin x score state."),
        ("06_time", "Time: within-match profile, fresh legs, tournament pressure, editions",
         lambda: time_profiles(d, frames["team_matches"], frames["appearances"], primary, measures, args.bootstrap), ""),
        ("07_opponents", "Opponents and context: creativity allowed, opponent strength, pressure, pitch maps, novelty",
         lambda: opponents(d, frames["team_matches"], primary, args.bootstrap), ""),
    ]
    for section, title, run, notes in runs:
        tables, found = run()
        hypotheses += found
        store(section, title, tables, notes)
    store("08_players", "Player profiles, consistency, enablers and duos", player_profiles(d, primary),
          "Profiles: k-means (k = 5) on standardized creativity, originality, value, risk and creative-action shares.")

    table = pd.DataFrame([asdict(h) for h in hypotheses])
    table["q_bh"] = benjamini_hochberg(table["p"])
    table["significant_q05"] = table["q_bh"] < 0.05
    table.to_csv(output / "hypotheses.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M"), "git_commit": commit, "seconds": round(time.time() - started),
        "tables": {p: sha(p) for p in args.tables}, "reference": {p: sha(p) for p in args.reference},
        "context_dir": args.context_dir, "measures": measures, "primary": primary, "scope": "intended",
        "bootstrap": args.bootstrap, "window_minutes": args.window, "decisions": len(d),
        "decisions_in_scope": int(d["in_scope"].sum()), "matches": int(d["match_id"].nunique()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lines = ["# Second-order creativity analyses", "",
             f"{manifest['decisions']} decisions ({manifest['decisions_in_scope']} in scope) in {manifest['matches']} "
             f"matches; primary measure `{primary}` (z-scored); robustness: {', '.join(measures[1:])}. "
             f"95% match-clustered bootstrap intervals ({args.bootstrap} draws); q = Benjamini-Hochberg over the "
             f"{len(table)} main hypotheses.", "",
             "## Main hypotheses", "", markdown(table.drop(columns=["significant_q05"]), 4), "", "## Sections", ""]
    lines += [f"- [{title}]({section}/summary.md): {', '.join(names)}" for section, title, names in sections]
    (output / "index.md").write_text("\n".join(lines))
    print(f"done in {time.time() - started:.0f}s -> {output}")


if __name__ == "__main__":
    main()
