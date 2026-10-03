"""Run every supported value-model configuration and compile one comparison table.

Configurations are the valid combinations of target x head x distribution x
training (20; quantiles only with the direct head and mc/sarsa targets).
Every run warm-starts from ``--policy-checkpoint``; ``joint`` runs then
fine-tune policy and value heads together.  Finished runs (with a
``summary.json``) are skipped, so the sweep can be resumed.

Screening at reduced scale, e.g. 100 training matches and 3 epochs:

    python scripts/value/run_value_sweep.py \\
        --policy-checkpoint checkpoints/next_action_v2/best_model.pt \\
        --output-dir checkpoints/value_sweep \\
        --train-dir data/open-data/data --val-dir data/open-data/data \\
        --train-file-count 60 --val-file-count 50 --sample-seed 42 --epochs 3

Every option the sweep does not know itself is passed to each
``train_value.py`` run (no ``--`` separator needed, which ``poetry run``
would intercept).  Only the table: ``--summarize``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TARGETS = ("mc", "sarsa", "expected_sarsa", "iql")
HEADS = ("direct", "decomposed")
DISTRIBUTIONS = ("expectation", "quantile")
TRAININGS = ("warm", "joint")


def configurations(args: argparse.Namespace) -> list[dict]:
    selected = []
    for target, head, distribution, training in itertools.product(TARGETS, HEADS, DISTRIBUTIONS, TRAININGS):
        if distribution == "quantile" and (head != "direct" or target not in ("mc", "sarsa")):
            continue
        if args.only and not all(any(token == value for value in (target, head, distribution, training))
                                 for token in args.only):
            continue
        selected.append({"target": target, "head": head, "distribution": distribution, "training": training})
        if args.include_frozen and training == "warm":
            selected.append({"target": target, "head": head, "distribution": distribution, "training": "warm",
                             "freeze_encoder": True})
    return selected


def run_name(config: dict) -> str:
    name = f"{config['target']}-{config['head']}-{config['distribution']}-{config['training']}"
    return name + ("-frozen" if config.get("freeze_encoder") else "")


TABLE_COLUMNS = [
    ("selection", "metrics", "selection"),
    ("score AUC", "metrics", "score_auc"),
    ("score xG ll", "metrics", "score_soft_logloss"),
    ("concede AUC", "metrics", "concede_auc"),
    ("protocol AUC", "metrics", "protocol_auc"),
    ("protocol xG ll", "metrics", "protocol_soft_logloss"),
    ("VAEP-xG AUC (same rows)", "comparison", "VAEP-xG.auc"),
    ("pass success AUC", "metrics", "success_auc_pass"),
    ("policy nll", "metrics", "policy_joint_nll"),
    ("return selection", "metrics", "return_selection"),
    ("C51 CRPS", "metrics", "distribution_crps"),
    ("C51 mean AUC", "metrics", "distribution_mean_score_auc"),
    ("best epoch", None, "best_epoch"),
    ("hours", None, "train_seconds"),
]


def summarize(output_dir: Path) -> None:
    rows = []
    for summary_path in sorted(output_dir.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        row = {"run": summary_path.parent.name}
        for label, section, key in TABLE_COLUMNS:
            source = summary.get(section, {}) if section else summary
            if "." in key:
                name, metric = key.split(".")
                value = source.get(name, {}).get(metric)
            else:
                value = source.get(key)
            if key == "train_seconds" and value is not None:
                value = value / 3600
            row[label] = value
        rows.append(row)
    if not rows:
        print("no finished runs")
        return
    rows.sort(key=lambda row: row["selection"] if row["selection"] is not None else float("inf"))
    with (output_dir / "sweep_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = ["run"] + [label for label, _, _ in TABLE_COLUMNS]
    lines = ["| " + " | ".join(labels) + " |", "|" + "---|" * len(labels)]
    for row in rows:
        cells = [row["run"]] + [
            "-" if row[label] is None else (f"{row[label]:.4f}" if isinstance(row[label], float) else str(row[label]))
            for label in labels[1:]
        ]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (output_dir / "sweep_summary.md").write_text(table + "\n")
    print(table)


def main() -> None:
    # allow_abbrev=False: a passthrough option must never be read as a prefix of a sweep option.
    parser = argparse.ArgumentParser(description="Sweep value-model configurations.", allow_abbrev=False)
    parser.add_argument("--policy-checkpoint")
    parser.add_argument("--output-dir", default="checkpoints/value_sweep")
    parser.add_argument("--only", nargs="*", help="Keep configurations containing all these values, e.g. mc warm.")
    parser.add_argument("--include-frozen", action="store_true", help="Also run warm configurations with a frozen encoder.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args, passthrough = parser.parse_known_args([arg for arg in sys.argv[1:] if arg != "--"])
    output_dir = Path(args.output_dir)
    if args.summarize:
        summarize(output_dir)
        return
    if not args.policy_checkpoint:
        parser.error("--policy-checkpoint is required to run the sweep.")

    selected = configurations(args)
    print(f"{len(selected)} configurations")
    for number, config in enumerate(selected, start=1):
        run_dir = output_dir / run_name(config)
        if (run_dir / "summary.json").exists():
            print(f"[{number}/{len(selected)}] {run_dir.name}: done, skipping")
            continue
        command = [
            sys.executable, "scripts/train_value.py", "--policy-checkpoint", args.policy_checkpoint,
            "--target", config["target"], "--head", config["head"], "--distribution", config["distribution"],
            "--training", config["training"], "--checkpoint-dir", str(run_dir), *passthrough,
        ]
        if config.get("freeze_encoder"):
            command.append("--freeze-encoder")
        print(f"[{number}/{len(selected)}] {run_dir.name}: {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with (run_dir / "train.log").open("w") as log:
            # Unbuffered, so train.log shows progress immediately rather than in 4 KB chunks.
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
        status = "ok" if result.returncode == 0 else f"FAILED (exit {result.returncode}, see {run_dir / 'train.log'})"
        print(f"    {status} after {(time.time() - started) / 60:.0f} min", flush=True)
    summarize(output_dir)


if __name__ == "__main__":
    main()
