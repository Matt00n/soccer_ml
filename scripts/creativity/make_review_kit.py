"""Build a blinded review kit: a rating page (HTML) and a separate key, from scored decision tables.

    python scripts/creativity/make_review_kit.py \\
        --tables outputs/creativity/crossfit_fold0.csv.gz outputs/creativity/crossfit_fold1.csv.gz \\
        --output-dir outputs/creativity/review_kit1

Open ``review.html`` in a browser, rate every clip (ratings are kept in the
browser), download the ratings CSV and run ``score_review.py``.  Do not open
``key_DO_NOT_OPEN.csv`` before you have finished rating.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from soccer_ml.creativity.measures import compute_measures
from soccer_ml.creativity.review import build_key, clip_payload, review_page, sample_clips


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a blinded creativity review kit.")
    parser.add_argument("--tables", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data/open-data/data")
    parser.add_argument("--per-measure", type=int, default=3, help="Clips drawn from each measure's top decisions.")
    parser.add_argument("--pool-fraction", type=float, default=0.01, help="'Top decisions' = this fraction.")
    parser.add_argument("--originality-controls", type=int, default=10)
    parser.add_argument("--value-controls", type=int, default=8)
    parser.add_argument("--random-clips", type=int, default=8)
    parser.add_argument("--include-restarts", action="store_true", help="Also sample set pieces and other restarts.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if (output / "key_DO_NOT_OPEN.csv").exists():
        raise FileExistsError(f"{output} already holds a kit; use a new --output-dir (ratings are tied to a kit).")
    output.mkdir(parents=True, exist_ok=True)
    table = pd.concat([pd.read_csv(path, low_memory=False) for path in args.tables], ignore_index=True)
    scores = compute_measures(table)
    clips = sample_clips(
        table, scores, args.per_measure, args.pool_fraction, args.originality_controls, args.value_controls,
        args.random_clips, args.seed, exclude_restarts=not args.include_restarts,
    )
    payloads = []
    for clip in clips.itertuples():
        row = table.loc[clip.row]
        data_dir = Path(args.data_dir)
        payload = clip_payload(row, data_dir / "events" / f"{row['match_id']}.json",
                               data_dir / "three-sixty" / f"{row['match_id']}.json")
        payloads.append({"clip_id": clip.clip_id, **payload})
    kit_id = f"{output.name}-{time.strftime('%Y%m%d%H%M')}"
    (output / "review.html").write_text(review_page(payloads, kit_id))
    build_key(table, scores, clips).to_csv(output / "key_DO_NOT_OPEN.csv", index=False)
    (output / "kit.json").write_text(json.dumps({
        "kit_id": kit_id, "tables": args.tables, "clips": len(clips), "reasons": clips["reason"].value_counts().to_dict(),
        **{key: value for key, value in vars(args).items() if key not in ("tables", "output_dir")},
    }, indent=2))
    print(f"{len(clips)} clips -> {output / 'review.html'} (key: {output / 'key_DO_NOT_OPEN.csv'})")


if __name__ == "__main__":
    main()
