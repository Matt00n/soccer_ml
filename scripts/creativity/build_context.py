"""Build the match-context tables for second-order analyses (``soccer_ml/creativity/context.py``).

    python scripts/creativity/build_context.py --tables outputs/creativity/crossfit_c51_fold0.csv.gz \\
        outputs/creativity/crossfit_c51_fold1.csv.gz --output-dir outputs/creativity/context
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from soccer_ml.creativity.context import build_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Build match-context tables.")
    parser.add_argument("--tables", nargs="+", required=True, help="Decision tables whose matches to cover.")
    parser.add_argument("--data-dir", default="data/open-data/data")
    parser.add_argument("--output-dir", default="outputs/creativity/context")
    args = parser.parse_args()
    started = time.time()
    decisions = pd.concat([pd.read_csv(p, usecols=["match_id", "event_uuid"]) for p in args.tables], ignore_index=True)
    frames = build_context(decisions["match_id"].unique(), Path(args.data_dir))
    frames["decisions"] = frames["decisions"][frames["decisions"]["event_uuid"].isin(set(decisions["event_uuid"]))]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(output / f"{name}.csv.gz" if name == "decisions" else output / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} rows")
    print(f"done in {time.time() - started:.0f}s; {len(decisions)} decisions, "
          f"{frames['decisions']['event_uuid'].nunique()} with context")


if __name__ == "__main__":
    main()
