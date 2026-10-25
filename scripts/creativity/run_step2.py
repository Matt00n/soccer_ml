"""Run the step-2 analyses (robustness, opportunity and use, enabling) and store every result.

    python scripts/creativity/run_step2.py \\
        --tables outputs/creativity/step2/scored/crossfit_c51_fold0.csv.gz outputs/creativity/step2/scored/crossfit_c51_fold1.csv.gz \\
        --reference outputs/creativity/step2/scored/reference_holdout.csv.gz

Output (``outputs/creativity/step2/``): one folder per analysis (CSV per table + summary.md),
``claims.csv`` (the proposal's claims with Benjamini-Hochberg q-values), ``index.md``, ``manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from soccer_ml.creativity.measures import compute_measures
from soccer_ml.creativity.second_order import demean
from soccer_ml.creativity.step2 import (
    buildup,
    enabling,
    event_robustness,
    fresh_legs_by_position,
    hidden_contributions,
    opportunity,
    power_and_distribution,
    premium_robustness,
)
from soccer_ml.creativity.validation import markdown

MEASURES = ["good_surprise", "good_rarity", "c51_tail_rarity", "rarity", "advantage"]


def load(paths: list[str], context: pd.DataFrame) -> pd.DataFrame:
    parts, offset = [], 0
    for path in paths:
        part = pd.read_csv(path, low_memory=False)
        part["next_row"] = np.where(part["next_row"] >= 0, part["next_row"] + offset, -1)  # rows are per file
        offset += len(part)
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    scores = compute_measures(table, tuple(MEASURES), scope="intended")
    in_scope = scores.notna().any(axis=1)
    columns = {"in_scope": in_scope}
    for name in scores:
        columns[f"raw_{name}"] = scores[name]
        columns[f"z_{name}"] = (scores[name] - scores[name].mean()) / scores[name].std()
    use = (scores["good_surprise"] - table["opportunity_expected"]).where(in_scope)
    opp = table["opportunity_expected"].where(in_scope)
    columns.update({
        "use": use, "z_use": (use - use.mean()) / use.std(), "z_opportunity": (opp - opp.mean()) / opp.std(),
        "quadrant_share": table["opportunity_quadrant"].where(in_scope), "chose_quadrant_s": table["chose_quadrant"].where(in_scope),
        "pressure_5yd": table["opponents_within_5"].where(in_scope),
    })
    table = pd.concat([table.drop(columns=[c for c in columns if c in table]), pd.DataFrame(columns)], axis=1)
    cell = (table["opponents_within_5"].clip(upper=3).astype(str) + "|"
            + pd.cut(table["nearest_opponent"], [-1, 2, 4, 7, 12, 1e9]).astype(str))
    table["z_gs_pressure_adjusted"] = demean(table[["z_good_surprise"]], cell)["z_good_surprise"].where(in_scope)
    return table.merge(context, on="event_uuid", how="left")


def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    order = p.sort_values()
    ranked = order * len(order) / np.arange(1, len(order) + 1)
    return np.minimum.accumulate(ranked[::-1])[::-1].clip(upper=1.0).reindex(p.index)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Step-2 creativity analyses.")
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--reference", nargs="*", default=[])
    parser.add_argument("--context-dir", default="outputs/creativity/context")
    parser.add_argument("--second-order-dir", default="outputs/creativity/second_order")
    parser.add_argument("--external-dir", default="outputs/creativity/step2/A1_external_ratings")
    parser.add_argument("--output-dir", default="outputs/creativity/step2")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    started = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = Path(args.context_dir)
    context = pd.read_csv(ctx / "decisions.csv.gz")
    frames = {n: pd.read_csv(ctx / f"{n}.csv") for n in
              ("goals", "substitutions", "red_cards", "tactical_shifts", "possessions", "appearances")}
    d = load(args.tables, context)
    reference = load(args.reference, context) if args.reference else None
    measures = [f"z_{m}" for m in MEASURES]
    primary = measures[0]
    print(f"{len(d)} decisions ({int(d['in_scope'].sum())} in scope); reference: {0 if reference is None else len(reference)}")
    sections = []

    def store(folder: str, title: str, tables: dict, notes: str = "") -> dict:
        path = out / folder
        path.mkdir(exist_ok=True)
        lines = [f"# {title}", ""] + ([notes, ""] if notes else [])
        for name, frame in tables.items():
            frame.to_csv(path / f"{name}.csv", index=False)
            shown = frame if len(frame) <= 60 else frame.head(60)
            lines += [f"## {name} ({len(frame)} rows{', first 60 shown' if len(frame) > 60 else ''})", "", markdown(shown, 4), ""]
        (path / "summary.md").write_text("\n".join(lines))
        sections.append((folder, title, list(tables)))
        print(f"  {title} ({time.time() - started:.0f}s)")
        return tables

    b1 = store("B1_buildup", "Build-up creativity -> shots and goals: robustness",
               buildup(d, frames["possessions"], measures, args.bootstrap, reference),
               "Linear probability models; possession creativity = max over the possession team's in-scope, non-shot "
               "decisions (z across possessions); strata = start value decile x third x play pattern x length; "
               "95% match-cluster bootstrap.")
    b2 = store("B2_hidden_contributions", "Hidden contributions: creativity by role, credited vs. uncredited",
               hidden_contributions(d, frames["possessions"], frames["goals"], frames["appearances"], primary, args.bootstrap),
               "Creative = top 10% of good_surprise among in-scope decisions. Credited = shot, assist (key pass).")
    values = [primary, "z_good_rarity", "z_c51_tail_rarity", "z_opportunity", "z_use", "quadrant_share",
              "chose_quadrant_s", "pressure_5yd", "z_gs_pressure_adjusted"]
    b3 = store("B3_event_studies", "Event studies: windows, pre-trends, splits, opportunity vs. use (B3-B5, C2)",
               event_robustness(d, frames["goals"], frames["red_cards"], frames["tactical_shifts"], frames["substitutions"],
                                values, args.bootstrap),
               "Effect = change after vs. before the event minus the same change in 20 placebo windows per event "
               "(other matches, same period, clock +-3 min). Values: z-scores (SD) except quadrant_share, "
               "chose_quadrant_s (rates) and pressure_5yd (opponents within 5 yd).")
    fresh = store("B4_fresh_legs", "Substitutes vs. starters after 60', within position group",
                  fresh_legs_by_position(d, frames["appearances"], primary, args.bootstrap))
    team_regressions = pd.read_csv(Path(args.second_order_dir) / "03_team_success" / "regressions.csv")
    b6 = store("B6_power_distribution", "Power of team-level nulls; distribution of creativity within possessions",
               power_and_distribution(d, frames["possessions"], team_regressions, primary, args.bootstrap))
    b8 = store("B8_surprise_premium", "Surprise premium: robustness", premium_robustness(d, args.bootstrap, reference),
               "Slope of realized minus predicted value on rarity within strata (goal-probability units per unit rarity).")
    c1 = store("C1_opportunity", "Creative opportunity and its use", opportunity(d, primary),
               "opportunity_expected = E_pi[good_surprise]; opportunity_quadrant = share of unusual options that are "
               "useful; use = good_surprise - opportunity_expected.")
    c3 = store("C3_enabling", "Enabling: effect of a creative action on the next teammate's opportunity and space",
               enabling(d, primary, args.bootstrap),
               "Pairs: in-scope decision -> next decision by a different player of the same team. Strata: end cell "
               "(10 x 10 yd) x action type x value decile. estimate = change of the outcome per SD of creativity.")

    # Claims used in the proposal, with BH correction.
    claims = []

    def claim(name, row, estimate="estimate", unit=""):
        claims.append({"claim": name, "estimate": row[estimate], "low": row["low"], "high": row["high"], "p": row["p"], "unit": unit})

    spec = b1["buildup_specifications"]
    for s, o in (("base: z_good_surprise", "shot"), ("base: z_good_surprise", "goal"), ("excluding the last 2 decision(s)", "shot"),
                 ("excluding the last 2 decision(s)", "goal"), ("excluding last 2, originality and value jointly", "shot")):
        row = spec[(spec["spec"] == s) & (spec["outcome"] == o)].iloc[0]
        claim(f"B1 {s} -> {o}" + (" (rarity coefficient)" if "jointly" in s else ""), row, unit="probability per SD")
    ev = b3["event_studies"]
    pick = lambda e, v, var="10 min": ev[(ev["event"] == e) & (ev["value"] == v) & (ev["variant"] == var) & (ev["group"] == "all") & (ev.get("split").isna() if "split" in ev else True)].iloc[0]  # noqa: E731
    for e in ("goal: scoring team", "red card: team sent off", "tactical shift", "tactical substitution"):
        row = pick(e, primary)
        claims.append({"claim": f"B3 {e}: creativity change", "estimate": row["effect"], "low": row["low"], "high": row["high"],
                       "p": row["p"], "unit": "SD"})
    for v in ("z_opportunity", "z_use"):
        row = pick("goal: scoring team", v)
        claims.append({"claim": f"C2 goal: scoring team: {v}", "estimate": row["effect"], "low": row["low"], "high": row["high"],
                       "p": row["p"], "unit": "SD"})
    row = fresh["fresh_legs_within_position"].iloc[0]
    claim("B4 substitutes minus starters after 60' (within position)", row, unit="SD")
    prem = b8["premium_specifications"]
    row = prem[(prem["sample"] == "cross-fitted") & (prem["spec"] == "Q deciles x action type x third") & (prem["residual"] == "res_td")].iloc[0]
    claim("B8 surprise premium (fine strata)", row, estimate="slope", unit="goal probability per unit rarity")
    eff = c3["enabling_effects"]
    row = eff[(eff["outcome"] == "next_opportunity_expected") & (eff["fixed_effects"] == "stratum+match_id")].iloc[0]
    claim("C3 creative action -> next teammate's opportunity", row, unit="nats per SD")
    row = eff[(eff["outcome"] == "next_nearest_opponent") & (eff["fixed_effects"] == "stratum+match_id")].iloc[0]
    claim("C3 creative action -> next teammate's space (nearest opponent)", row, unit="yd per SD")
    external = Path(args.external_dir) / "correlations.csv"
    if external.exists():
        corr = pd.read_csv(external)
        row = corr[(corr["subset"] == "contemporaneous") & (corr["attribute"] == "mentality_vision")
                   & (corr["measure"] == "good_surprise") & (~corr["partial_on_overall"])].iloc[0]
        claims.append({"claim": "A1 good_surprise vs. FIFA vision (within context x position)", "estimate": row["rho"],
                       "low": row["low"], "high": row["high"], "p": float("nan"), "unit": "Spearman rho"})
    claims = pd.DataFrame(claims)
    tested = claims["p"].notna()
    claims.loc[tested, "q_bh"] = benjamini_hochberg(claims.loc[tested, "p"])
    claims.to_csv(out / "claims.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M"), "git_commit": commit, "seconds": round(time.time() - started),
                "tables": {p: sha(p) for p in args.tables}, "reference": {p: sha(p) for p in args.reference},
                "bootstrap": args.bootstrap, "scope": "intended", "primary": primary,
                "decisions": len(d), "decisions_in_scope": int(d["in_scope"].sum())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    lines = ["# Step 2: robustness, creative opportunity, enabling", "",
             f"{manifest['decisions']} decisions ({manifest['decisions_in_scope']} in scope); primary measure `{primary}`; "
             f"95% match-cluster bootstrap ({args.bootstrap}); q = Benjamini-Hochberg over the tested claims.", "",
             "## Claims", "", markdown(claims, 4), "", "## Sections", ""]
    lines += [f"- [{title}]({folder}/summary.md)" for folder, title, _ in sections]
    lines += ["- [A1 external ratings](A1_external_ratings/report.md)"]
    (out / "index.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
