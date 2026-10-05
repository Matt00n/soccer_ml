"""Join expert ratings with a review kit's key and evaluate every creativity measure against them.

    python scripts/creativity/score_review.py --kit-dir outputs/creativity/review_kit1 \\
        --ratings ~/Downloads/ratings_review_kit1-....csv

Reports (``review_report.md`` in the kit directory):

* mean creativity rating per selection group (each measure's top clips,
  controls, random clips);
* per measure over all rated clips: Spearman with the creativity rating and
  AUC for "rated creative" (>= 4); precision of its own top clips;
* paired contrasts: flagged clips vs. their value-matched low-originality
  controls (does originality matter?) and their originality-matched
  low-value controls (does value matter?);
* component checks: rarity vs. the originality rating, advantage vs. the
  effectiveness rating;
* data issues among each measure's top clips; 2D vs. video ratings.

With a few dozen clips from one rater this is a first verification, not a
confirmatory test: read effect sizes and their spread, not p-values alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from soccer_ml.creativity.validation import auc, markdown, spearman


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate creativity measures against expert ratings.")
    parser.add_argument("--kit-dir", required=True)
    parser.add_argument("--ratings", nargs="+", required=True, help="Ratings CSV(s); several raters are averaged.")
    parser.add_argument("--creative-threshold", type=float, default=4.0)
    return parser.parse_args()


def _mean_ci(values: pd.Series) -> tuple[float, float, int]:
    values = values.dropna()
    if len(values) < 2:
        return (float(values.mean()) if len(values) else float("nan")), float("nan"), len(values)
    return float(values.mean()), float(1.96 * values.std(ddof=1) / np.sqrt(len(values))), len(values)


def paired(data: pd.DataFrame, reason: str, rating: str) -> dict:
    """Flagged clip minus its paired control on ``rating`` (sign test)."""
    controls = data[data["reason"] == reason]
    flagged = data.set_index("clip_id")
    differences = []
    for control in controls.itertuples():
        if control.pair_clip in flagged.index:
            differences.append(flagged.at[control.pair_clip, rating] - getattr(control, rating))
    differences = pd.Series(differences, dtype=float).dropna()
    wins, losses = int((differences > 0).sum()), int((differences < 0).sum())
    p = stats.binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue if wins + losses else float("nan")
    mean, ci, n = _mean_ci(differences)
    return {"contrast": reason, "rating": rating, "pairs": n, "mean_difference": mean, "ci95": ci,
            "flagged_higher": wins, "control_higher": losses, "sign_test_p": p}


def main() -> None:
    args = parse_args()
    kit = Path(args.kit_dir)
    key = pd.read_csv(kit / "key_DO_NOT_OPEN.csv")
    raters = []
    for number, path in enumerate(args.ratings):
        ratings = pd.read_csv(path)
        for column in ("creative", "original", "effective", "creative_video"):
            ratings[column] = pd.to_numeric(ratings.get(column), errors="coerce")
        raters.append(ratings.assign(rater=number))
    all_ratings = pd.concat(raters, ignore_index=True)
    numeric = all_ratings.groupby("clip_id")[["creative", "original", "effective", "creative_video"]].mean()
    issues = all_ratings.groupby("clip_id")["issue"].agg(lambda v: v.dropna().mode().iat[0] if v.notna().any() else "")
    data = key.merge(numeric.reset_index(), on="clip_id", how="left").merge(issues.rename("issue").reset_index(),
                                                                         on="clip_id", how="left")
    rated = data[data["creative"].notna()]
    measures = [c[4:] for c in key.columns if c.startswith("pct_")]
    sections = []

    sections.append(("Coverage", pd.DataFrame([{
        "clips": len(data), "rated": len(rated), "raters": len(raters),
        "issues_flagged": int((~rated["issue"].fillna("").isin(["", "none"])).sum()),
        "creative_rate": float((rated["creative"] >= args.creative_threshold).mean()),
    }])))

    groups = []
    for reason, part in rated.groupby("reason"):
        mean, ci, n = _mean_ci(part["creative"])
        groups.append({"selection": reason, "clips": n, "creative_mean": mean, "ci95": ci,
                       "original_mean": part["original"].mean(), "effective_mean": part["effective"].mean(),
                       "rated_creative": float((part["creative"] >= args.creative_threshold).mean())})
    sections.append(("Ratings by selection group", pd.DataFrame(groups).sort_values("creative_mean", ascending=False)))

    rows = []
    positive = (rated["creative"] >= args.creative_threshold).to_numpy()
    for name in measures:
        own = rated[rated["reason"] == f"top:{name}"]
        value, se, _ = auc(rated[f"score_{name}"].to_numpy(dtype=float), positive)
        rows.append({
            "measure": name, "spearman_creative": spearman(rated[f"score_{name}"], rated["creative"]),
            "auc_rated_creative": value, "auc_se": se,
            "own_top_clips": len(own), "own_top_precision": float((own["creative"] >= args.creative_threshold).mean())
            if len(own) else float("nan"),
        })
    sections.append(("Measures vs. creativity ratings (all rated clips)",
                     pd.DataFrame(rows).sort_values("spearman_creative", ascending=False)))

    contrasts = [paired(rated, "originality_control", r) for r in ("creative", "original", "effective")]
    contrasts += [paired(rated, "value_control", r) for r in ("creative", "original", "effective")]
    sections.append(("Paired contrasts (flagged minus control)", pd.DataFrame(contrasts)))

    components = [
        {"component": "rarity vs. originality rating", "spearman": spearman(rated["rarity"], rated["original"])},
        {"component": "surprise vs. originality rating", "spearman": spearman(rated["surprise"], rated["original"])},
        {"component": "advantage vs. effectiveness rating", "spearman": spearman(rated["advantage"], rated["effective"])},
        {"component": "realized TD advantage vs. effectiveness", "spearman": spearman(rated["td_advantage"], rated["effective"])},
        {"component": "originality x effectiveness ratings vs. creativity rating",
         "spearman": spearman(rated["original"] * rated["effective"], rated["creative"])},
    ]
    sections.append(("Component checks", pd.DataFrame(components)))

    issue_rows = []
    for reason, part in rated.groupby("reason"):
        counts = part["issue"].fillna("").replace("", "none").value_counts(normalize=True)
        issue_rows.append({"selection": reason, "clips": len(part), **counts.to_dict()})
    sections.append(("Data issues by selection group", pd.DataFrame(issue_rows).fillna(0.0)))

    both = rated.dropna(subset=["creative_video"])
    sections.append(("2D vs. video ratings", pd.DataFrame([{
        "clips_with_video": len(both), "spearman": spearman(both["creative"], both["creative_video"]),
        "mean_difference_video_minus_2d": float((both["creative_video"] - both["creative"]).mean()) if len(both) else float("nan"),
    }])))

    lines = ["# Expert review", "", f"Kit: `{kit}`; ratings: {', '.join(f'`{p}`' for p in args.ratings)}", "",
             "First verification with few clips: read effect sizes and intervals, not p-values alone.", ""]
    for title, frame in sections:
        lines += [f"## {title}", "", markdown(frame), ""]
    (kit / "review_report.md").write_text("\n".join(lines))
    data.to_csv(kit / "rated_clips.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
