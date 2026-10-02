"""Fit Expected Threat (xT) grids on StatsBomb open-data events.

Uses every event-only match except the validation matches of the 360
benchmark, so xT is evaluated out of sample like VAEP and our models.

    conda run -n soccer_ml python scripts/value/fit_expected_threat.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from soccer_ml.value.expected_threat import ExpectedThreat

_GRIDS: list[tuple[int, int]] = []


def _fit_match(path: str) -> list[ExpectedThreat] | None:
    try:
        events = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    models = [ExpectedThreat(m, n) for m, n in _GRIDS]
    for model in models:
        model.update(events)
    return models


def _init(grids: list[tuple[int, int]]) -> None:
    global _GRIDS
    _GRIDS = grids


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit xT grids on StatsBomb events.")
    parser.add_argument("--data-root", default="data/open-data/data")
    parser.add_argument("--val-summary", default="checkpoints/temporal_receiver/benchmark_summary.json")
    parser.add_argument("--grids", default="16x12,26x17", help="Comma-separated MxN grids (x bins x y bins).")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/value_baselines")
    args = parser.parse_args()

    grids = [tuple(int(v) for v in grid.split("x")) for grid in args.grids.split(",")]
    val_ids = {Path(pair[1]).stem for pair in json.loads(Path(args.val_summary).read_text())["split"]["val_pairs"]}
    paths = sorted(str(p) for p in Path(args.data_root, "events").glob("*.json") if p.stem not in val_ids)
    print(f"fitting xT {args.grids} on {len(paths)} matches (excluding {len(val_ids)} validation matches)")

    totals = [ExpectedThreat(m, n) for m, n in grids]
    started = time.time()
    with Pool(args.workers, initializer=_init, initargs=(grids,)) as pool:
        for done, models in enumerate(pool.imap_unordered(_fit_match, paths, chunksize=8), start=1):
            if models is not None:
                for total, model in zip(totals, models):
                    total.merge(model)
            if done % 500 == 0:
                print(f"  {done}/{len(paths)} matches ({time.time() - started:.0f}s)", flush=True)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for total in totals:
        values = total.solve()
        name = f"xt_{total.grid.grid_m}x{total.grid.grid_n}.json"
        total.save(output / name)
        print(f"  {name}: {total.num_matches} matches, {int(total.shots.sum())} shots, "
              f"{int(total.moves.sum())} moves | max xT {values.max():.3f}")


if __name__ == "__main__":
    main()
