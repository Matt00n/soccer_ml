"""Score every out-of-sample decision with a value model and its policy: the creativity table.

One row per decision with originality, value, option-set and calibration
statistics (``soccer_ml/creativity/scoring.py``), realized returns, and the
StatsBomb event details (player, timing, annotations).  The matches scored
are those the policy never trained on, read from the policy run's
``summary.json``:

* cross-fitting runs (``--num-folds``): the full held-out fold;
* group hold-out runs (``--holdout-*``): all held-out matches;
* ordinary runs: the validation matches.

The script refuses to score a match that either model trained on.

Examples (cross-fitted table = fold 0 + fold 1):

    python scripts/creativity/score_decisions.py --value-checkpoint checkpoints/crossfit/fold0/value_mc/best_model.pt \\
        --output outputs/creativity/crossfit_fold0.csv.gz
    python scripts/creativity/score_decisions.py --value-checkpoint checkpoints/value/expected_sarsa-decomposed-c51-v3/best_model.pt \\
        --output outputs/creativity/main_val_c51.csv.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from soccer_ml.creativity.events import event_fields, load_matches
from soccer_ml.creativity.scoring import decision_statistics
from soccer_ml.data.next_action_dataset import (
    ACTION_TYPES,
    DECISION_CONTEXTS,
    PASS_HEIGHTS,
    REWARDS,
    VALUE_LABELS,
    NextActionDataset,
    collate_next_action,
    lambda_returns,
    load_value_labels,
    node_layout_for,
)
from soccer_ml.data.splits import (
    load_match_datasets,
    match_id_from_filename,
    sample_disjoint_train_val_pairs,
    sample_fold_pairs,
    sample_holdout_pairs,
)
from soccer_ml.models.next_action import NextActionModel
from soccer_ml.models.value import ValueModel
from soccer_ml.training.trainer import get_default_device


def load_value_model(path: Path) -> tuple[ValueModel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ValueModel(NextActionModel(**checkpoint["policy_kwargs"]), **checkpoint["value_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval(), checkpoint


def load_policy(path: Path) -> tuple[NextActionModel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = NextActionModel(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval(), checkpoint


def evaluation_pairs(policy_summary: dict) -> tuple[list[tuple[str, str]], str]:
    """The policy's out-of-sample matches and how they were chosen."""
    split = policy_summary["split"]
    if split.get("fold"):
        return [tuple(pair) for pair in split["fold"]["held_out_pairs"]], f"fold {split['fold']['fold']}"
    if split.get("holdout"):
        return [tuple(pair) for pair in split["holdout"]["held_out_pairs"]], "group holdout"
    return [tuple(pair) for pair in split["val_pairs"]], "validation"


def value_training_pairs(value_summary: dict) -> list[tuple[str, str]]:
    """Recompute the value model's training matches from its run arguments."""
    args = value_summary["args"]
    if not args.get("train_dir"):
        raise ValueError("Single-match value runs have no out-of-sample matches.")
    if args.get("holdout_competitions") or args.get("holdout_teams"):
        return sample_holdout_pairs(
            args["train_dir"], args["holdout_competitions"], args["holdout_teams"],
            args["holdout_val_matches"], args["sample_seed"],
        )[0]
    if args.get("num_folds") is not None:
        return sample_fold_pairs(
            args["train_dir"], args["num_folds"], args["fold"], args["sample_seed"], args["monitor_val_matches"]
        )[0]
    return sample_disjoint_train_val_pairs(
        args["train_dir"], args["val_dir"], args["train_file_count"], args["val_file_count"], args["sample_seed"]
    )[0]


def _match_ids(pairs) -> set[str]:
    return {match_id_from_filename(Path(pair[0])) for pair in pairs}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score out-of-sample decisions for creativity analyses.")
    parser.add_argument("--value-checkpoint", required=True)
    parser.add_argument("--policy-checkpoint", help="Defaults to the policy the value model was warm-started from.")
    parser.add_argument("--output", required=True, help="Output table (.csv.gz).")
    parser.add_argument("--tag", help="Model tag stored in every row (default: the value run's directory name).")
    parser.add_argument("--max-matches", type=int, help="Smoke test: score only the first matches.")
    parser.add_argument("--tail-quantile", type=float, default=0.9, help="C51 positive-tail set: this quantile > V(s).")
    parser.add_argument("--value-labels", default="outputs/value_baselines/value_labels_360.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0, help="Seed of the randomized PIT.")
    parser.add_argument("--device")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    device = torch.device(args.device) if args.device else get_default_device()
    value_path = Path(args.value_checkpoint)
    value_model, value_checkpoint = load_value_model(value_path)
    policy_path = Path(args.policy_checkpoint or value_checkpoint["policy_checkpoint"])
    policy, policy_checkpoint = load_policy(policy_path)
    data_config = dict(policy_checkpoint["data_config"])
    if dict(value_checkpoint["data_config"]) != data_config:
        raise ValueError("The value model and the policy were trained on different data configurations.")

    policy_summary = json.loads((policy_path.parent / "summary.json").read_text())
    value_summary = json.loads((value_path.parent / "summary.json").read_text())
    pairs, split_kind = evaluation_pairs(policy_summary)
    evaluated = _match_ids(pairs)
    for name, training in (
        ("policy", policy_summary["split"]["train_pairs"]),
        ("value model", value_training_pairs(value_summary)),
    ):
        overlap = evaluated & _match_ids(training)
        if overlap:
            raise ValueError(f"{len(overlap)} evaluation matches were in the {name}'s training data.")
    if args.max_matches:
        pairs = pairs[: args.max_matches]
    gamma = float(value_summary["config"]["gamma"])
    tag = args.tag or value_path.parent.name
    print(f"scoring {len(pairs)} matches ({split_kind}) | value {value_path} | policy {policy_path} | "
          f"gamma {gamma} | device {device}")

    started = time.time()
    labels = load_value_labels(args.value_labels) if Path(args.value_labels).exists() else None
    dataset, skipped = load_match_datasets(pairs, lambda t, e: NextActionDataset(t, e, value_labels=labels, **data_config))
    parts = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    print(f"data: {len(dataset)} decisions in {len(parts)} matches ({time.time() - started:.0f}s), "
          f"{len(skipped)} skipped")

    policy.to(device)
    value_model.to(device)
    layout = node_layout_for(data_config.get("velocities", False))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_next_action)
    columns: dict[str, list[torch.Tensor]] = {}
    for batch in tqdm(loader, desc="scoring", disable=None):
        batch = batch.to(device)
        stats = decision_statistics(
            policy(batch), value_model(batch), batch, value_model, layout, args.tail_quantile, generator
        )
        for name, values in stats.items():
            columns.setdefault(name, []).append(values)
    table = pd.DataFrame({name: torch.cat(values).numpy() for name, values in columns.items()})

    # Per-decision facts from the dataset tensors (same order as the loader).
    tensors = {name: torch.cat([part._tensors[name] for part in parts]) for name in (
        "action_type", "decision_context", "target_zone", "pass_height", "success", "reward",
        "next_index", "next_flip", "done", "value_label", "graph_features",
    )}
    table["action_type"] = [ACTION_TYPES[i] for i in tensors["action_type"].tolist()]
    table["decision_context"] = [DECISION_CONTEXTS[i] for i in tensors["decision_context"].tolist()]
    table["target_zone"] = tensors["target_zone"].numpy()
    table["pass_height"] = [PASS_HEIGHTS[i] if i >= 0 else "" for i in tensors["pass_height"].tolist()]
    table["success"] = tensors["success"].float().where(tensors["success"] >= 0, torch.tensor(float("nan"))).numpy()
    names = parts[0].graph_feature_names
    table["open_play"] = tensors["graph_features"][:, names.index("open_play")].numpy()
    for i, name in enumerate(VALUE_LABELS):
        table[f"label_{name}"] = tensors["value_label"][:, i].numpy()
    reward = tensors["reward"]
    table["reward"] = (reward[:, REWARDS.index("our_xg")] - reward[:, REWARDS.index("their_xg")]).numpy()

    # Realized values: one-step TD advantage and the discounted Monte Carlo return.
    offsets = torch.tensor([0] + list(dataset.cumulative_sizes[:-1]) if isinstance(dataset, ConcatDataset) else [0])
    part_of = torch.cat([torch.full((len(part),), k) for k, part in enumerate(parts)])
    next_global = torch.where(tensors["next_index"] >= 0, tensors["next_index"] + offsets[part_of], -1)
    v = torch.tensor(table["v"].to_numpy())
    v_next = torch.where(next_global >= 0, v[next_global.clamp_min(0)], torch.zeros_like(v))
    v_next = torch.where(tensors["next_flip"], -v_next, v_next) * (~tensors["done"]).float()
    table["td_advantage"] = (torch.tensor(table["reward"].to_numpy()) + gamma * v_next - v).numpy()
    returns = lambda_returns(dataset, gamma, 1.0)
    table["mc_return"] = (returns[:, 0] - returns[:, 1]).numpy()
    table["mc_advantage"] = table["mc_return"] - table["v"]
    table["next_row"] = next_global.numpy()

    # StatsBomb details.
    matches_dir = Path(pairs[0][1]).parent.parent / "matches"
    matches = load_matches(str(matches_dir)) if matches_dir.is_dir() else {}
    details = []
    for part in parts:
        events = {event["id"]: event for event in json.loads(part.events_path.read_text())}
        match_id = part.events_path.stem
        info = matches.get(match_id, {})
        for meta in part._metadata:
            row = event_fields(events[meta["event_uuid"]])
            row.update(
                match_id=match_id, event_uuid=meta["event_uuid"], anchor_uuid=meta["anchor_uuid"],
                period=meta["period"], **{key: info.get(key) for key in (
                    "competition", "season", "match_date", "stage", "home_team", "away_team", "women"
                )},
            )
            row["opponent"] = info.get("away_team") if row["team"] == info.get("home_team") else info.get("home_team")
            details.append(row)
    table = pd.concat([pd.DataFrame(details), table], axis=1)
    table.insert(0, "model", tag)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, float_format="%.6g")
    provenance = {
        "value_checkpoint": str(value_path), "value_checkpoint_sha": _sha(value_path),
        "policy_checkpoint": str(policy_path), "policy_checkpoint_sha": _sha(policy_path),
        "split_kind": split_kind, "matches": len(parts), "skipped": skipped, "decisions": len(table),
        "gamma": gamma, "tail_quantile": args.tail_quantile, "seed": args.seed, "tag": tag,
        "value_config": value_summary["config"], "seconds": time.time() - started,
    }
    output.with_name(output.name.split(".")[0] + "_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"saved {len(table)} decisions to {output} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
