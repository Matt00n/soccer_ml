"""Does temperature scaling improve the policy's calibration, in particular in the rare tail?

Fits one temperature per head (type, zone, height) by grid search on the
policy run's validation matches, then compares T = 1 with the fitted
temperatures on held-out matches the temperatures never saw:

* cross-fitting runs: fit on the monitoring matches, test on the rest of the held-out fold;
* group hold-out runs (reference model): fit on the in-domain validation matches,
  test on the held-out matches (out of domain, e.g. club football);
* ordinary runs: fit on one half of the validation matches, test on the other.

Reports the joint NLL, observed / expected choices per log10-probability band,
and the uniformity of rarity (randomized PIT).

    python scripts/creativity/calibrate_policy.py --policy-checkpoint checkpoints/crossfit/fold0/next_action/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from soccer_ml.creativity.scoring import CALIBRATION_EDGES, NUM_CALIBRATION_BINS, calibration_bin_labels, set_rarity
from soccer_ml.creativity.validation import markdown
from soccer_ml.data.next_action_dataset import NextActionDataset, collate_next_action
from soccer_ml.data.splits import load_match_datasets
from soccer_ml.models.next_action import NextActionModel, action_log_likelihood
from soccer_ml.models.value import gather_action, observed_action_index, policy_action_log_probs
from soccer_ml.training.trainer import get_default_device

HEADS = ("type", "zone", "height")
GRID = np.round(np.arange(0.6, 1.6001, 0.025), 3)


def split_pairs(summary: dict) -> tuple[list, list, str]:
    split = summary["split"]
    val = [tuple(pair) for pair in split["val_pairs"]]
    if split.get("fold"):
        held_out = [tuple(pair) for pair in split["fold"]["held_out_pairs"]]
        return val, [pair for pair in held_out if pair not in set(val)], "monitoring matches -> rest of the fold"
    if split.get("holdout"):
        return val, [tuple(pair) for pair in split["holdout"]["held_out_pairs"]], "in-domain validation -> held-out groups"
    half = len(val) // 2
    return val[:half], val[half:], "validation halves"


def scaled(outputs: dict, temperatures: dict) -> dict:
    return {f"{head}_logits": outputs[f"{head}_logits"] / temperatures[head] for head in HEADS}


@torch.no_grad()
def fit_temperatures(model, loader, device) -> tuple[dict, pd.DataFrame]:
    """Per-head temperature minimizing the validation NLL (the joint NLL is a sum over heads)."""
    totals = np.zeros((len(GRID), len(HEADS)))
    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)
        for i, t in enumerate(GRID):
            nll = action_log_likelihood(scaled(outputs, dict.fromkeys(HEADS, float(t))), batch)
            totals[i] += [float(nll[f"{head}_nll"].sum()) for head in HEADS]
    best = {head: float(GRID[totals[:, k].argmin()]) for k, head in enumerate(HEADS)}
    curve = pd.DataFrame(totals, columns=[f"{head}_nll_sum" for head in HEADS]).assign(temperature=GRID)
    return best, curve


@torch.no_grad()
def evaluate(model, loader, device, settings: dict, seed: int = 0) -> pd.DataFrame:
    """NLL, band calibration and rarity uniformity of each temperature setting on the test matches."""
    generator = torch.Generator().manual_seed(seed)
    edges = torch.tensor(CALIBRATION_EDGES) * math.log(10.0)
    results = {name: {"nll": 0.0, "n": 0, "expected": np.zeros(NUM_CALIBRATION_BINS),
                      "observed": np.zeros(NUM_CALIBRATION_BINS), "pit": [], "rarity": []} for name in settings}
    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)
        uniform = torch.rand(batch.batch_size, generator=generator)
        for name, temperatures in settings.items():
            log_pi = policy_action_log_probs(scaled(outputs, temperatures)).cpu()
            pi = log_pi.exp()
            observed = observed_action_index(batch, model.num_zones).cpu()
            stats_ = set_rarity(pi, log_pi, None, observed)
            bins = torch.bucketize(log_pi, edges)
            r = results[name]
            r["nll"] += float(-gather_action(log_pi, observed).sum())
            r["n"] += batch.batch_size
            r["expected"] += torch.zeros(len(pi), NUM_CALIBRATION_BINS).scatter_add_(1, bins, pi).sum(0).numpy()
            r["observed"] += np.bincount(gather_action(bins, observed).numpy(), minlength=NUM_CALIBRATION_BINS)
            r["pit"].append((stats_["above"] + uniform * stats_["p_obs"]).numpy())
            r["rarity"].append(stats_["rarity"].numpy())
    rows = []
    for name, r in results.items():
        pit = np.concatenate(r["pit"])
        row = {"setting": name, **{f"T_{h}": settings[name][h] for h in HEADS}, "decisions": r["n"],
               "joint_nll": r["nll"] / r["n"], "mean_rarity": float(np.concatenate(r["rarity"]).mean()),
               "pit_ks": float(stats.kstest(pit, "uniform").statistic)}
        for label, expected, observed in zip(calibration_bin_labels(), r["expected"], r["observed"]):
            if expected > 1:
                row[f"ratio {label}"] = observed / expected
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temperature-scaling check of a next-action policy.")
    parser.add_argument("--policy-checkpoint", required=True)
    parser.add_argument("--max-test-matches", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", help="Default: outputs/creativity/calibration/<run name>.")
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else get_default_device()
    path = Path(args.policy_checkpoint)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = NextActionModel(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    summary = json.loads((path.parent / "summary.json").read_text())
    fit_pairs, test_pairs, how = split_pairs(summary)
    test_pairs = test_pairs[: args.max_test_matches]
    config = dict(checkpoint["data_config"])
    factory = lambda t, e: NextActionDataset(t, e, **config)  # noqa: E731
    fit_data, _ = load_match_datasets(fit_pairs, factory)
    test_data, _ = load_match_datasets(test_pairs, factory)
    loader = lambda data: DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_next_action)  # noqa: E731
    print(f"{path}: fit on {len(fit_pairs)} matches ({len(fit_data)} decisions), test on {len(test_pairs)} "
          f"({len(test_data)} decisions): {how}")
    best, curve = fit_temperatures(model, loader(fit_data), device)
    print(f"fitted temperatures: {best}")
    result = evaluate(model, loader(test_data), device, {"T = 1": dict.fromkeys(HEADS, 1.0), "fitted": best})
    output = Path(args.output_dir or f"outputs/creativity/calibration/{path.parent.parent.name}_{path.parent.name}")
    output.mkdir(parents=True, exist_ok=True)
    curve.to_csv(output / "fit_curve.csv", index=False)
    result.to_csv(output / "test.csv", index=False)
    text = (f"# Temperature check: `{path}`\n\nFit: {len(fit_pairs)} matches; test: {len(test_pairs)} matches ({how}).\n\n"
            f"Fitted temperatures: {best}\n\n" + markdown(result.set_index("setting").T.reset_index().rename(columns={"index": "metric"}), 4))
    (output / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
