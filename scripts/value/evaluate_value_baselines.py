"""Compare pass-value models with the un-xPass (KDD 2023) Table 3 protocol.

Task: for every completed open-play pass of the validation matches, predict
whether the passing team scores within the next 10 SPADL actions.  Labels and
VAEP predictions come from ``vaep_baseline.py``; xT predicts the threat of the
pass's end zone.  ``soft_logloss`` additionally scores the predictions against
the xG-sequence label, which is far less noisy than goals.

    conda run -n soccer_ml python scripts/value/evaluate_value_baselines.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from soccer_ml.training.metrics import probabilistic_binary_metrics
from soccer_ml.value.expected_threat import ExpectedThreat

# un-xPass Table 3 (EPL 2021/22 top-10 teams, binary goal labels) for reference.
UNXPASS_TABLE_3 = {
    "xT": (0.741, 0.012, 0.062),
    "VAEP": (0.757, 0.011, 0.056),
    "VAEP-xG": (0.766, 0.011, 0.056),
    "VAEP360": (0.776, 0.011, 0.055),
    "SoccerMap": (0.743, 0.011, 0.064),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--baseline-dir", default="outputs/value_baselines")
    parser.add_argument("--data-root", default="data/open-data/data")
    args = parser.parse_args()
    directory = Path(args.baseline_dir)

    rows = [
        row
        for row in csv.DictReader((directory / "vaep_val_predictions.csv").open())
        if row["type_name"] in ("pass", "cross") and row["result_name"] == "success"
    ]
    end_locations = {}
    for game_id in sorted({row["game_id"] for row in rows}):
        for event in json.loads(Path(args.data_root, "events", f"{game_id}.json").read_text()):
            if event["type"]["name"] == "Pass":
                end_locations[event["id"]] = event["pass"]["end_location"]
    rows = [row for row in rows if row["original_event_id"] in end_locations]

    labels = torch.tensor([float(row["scores"]) for row in rows])
    soft = torch.tensor([float(row["xg_scores"]) for row in rows])
    predictions = {
        "VAEP": [float(row["vaep_p_scores"]) for row in rows],
        "VAEP-xG": [float(row["vaep_xg_p_scores"]) for row in rows],
    }
    xt_models = {
        "xT (Karun 12x8)": directory / "karun_xt_12x8.json",
        "xT (ours 16x12)": directory / "xt_16x12.json",
        "xT (ours 26x17)": directory / "xt_26x17.json",
    }
    for name, path in xt_models.items():
        if not path.exists():
            continue
        model = ExpectedThreat.from_karun_grid(path) if "Karun" in name else ExpectedThreat.load(path)
        predictions[name] = [model.value(*end_locations[row["original_event_id"]][:2]) for row in rows]

    results = {
        name: probabilistic_binary_metrics(torch.tensor(values), labels, soft)
        for name, values in predictions.items()
    }
    print(f"Completed open-play passes in {len({r['game_id'] for r in rows})} validation matches: "
          f"n={len(rows)}, goal within 10 actions: {int(labels.sum())} ({labels.mean():.2%})\n")
    print(f"{'model':<18} {'AUC':>6} {'Brier':>8} {'log-loss':>9} {'xG log-loss':>12}")
    for name, metrics in sorted(results.items(), key=lambda item: -item[1]["auc"]):
        print(f"{name:<18} {metrics['auc']:6.3f} {metrics['brier']:8.4f} {metrics['logloss']:9.4f} "
              f"{metrics['soft_logloss']:12.4f}")
    print("\nun-xPass Table 3 (EPL 2021/22, different data; for orientation only):")
    for name, (auc, brier, logloss) in UNXPASS_TABLE_3.items():
        print(f"{name:<18} {auc:6.3f} {brier:8.3f} {logloss:9.3f}")
    (directory / "value_baseline_metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
