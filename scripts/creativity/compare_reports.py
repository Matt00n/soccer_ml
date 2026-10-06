"""Headline metrics of several validation reports side by side.

    python scripts/creativity/compare_reports.py outputs/creativity/validation_c51 outputs/creativity/validation_c51_in_view
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the headline metrics of validation reports.")
    parser.add_argument("reports", nargs="+", help="Report directories (each with headline.csv).")
    parser.add_argument("--metrics", nargs="*", help="Metrics to show (default: all).")
    args = parser.parse_args()
    frames = [pd.read_csv(Path(report) / "headline.csv").assign(report=Path(report).name) for report in args.reports]
    data = pd.concat(frames, ignore_index=True).set_index(["measure", "report"])
    metrics = args.metrics or list(data.columns)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        for metric in metrics:
            if metric in data:
                print(f"\n== {metric}")
                print(data[metric].unstack("report").reindex(columns=[Path(r).name for r in args.reports]).to_string())


if __name__ == "__main__":
    main()
