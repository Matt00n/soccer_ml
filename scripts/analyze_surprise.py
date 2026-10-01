"""Per-pass surprise analysis of a trained target-zone model on held-out matches.

For every validation pass of one or more benchmark runs this script records
the model's predicted zone distribution and joins it with the StatsBomb event
(player, position, pass flags, visibility).  It reports:

* model vs. closest-teammate agreement and accuracy where they disagree;
* results inside / outside the 360 visible area;
* an un-xPass-comparable subset (no dead-ball passes, foot passes, start and
  end inside the visible area) with their receiver-accuracy protocol;
* calibration of the top-1 probability (reliability table and ECE);
* event-level validation: surprise of through balls, switches, crosses, ...;
* position- and player-level surprise with 95% confidence intervals.

Per pass, ``surprise = -log p(true zone)`` and ``entropy = H(p)``.
``excess = surprise - entropy`` is how much more surprising the pass was than
the situation's spread of options.  ``rarity = P(Z rated more likely than
the chosen zone) + 0.5 * p(chosen zone)`` is a mid-PIT in [0, 1]: under a
calibrated model it averages 0.5 for *any* player, whatever their role, so it
is the situation-normalized score used for player rankings.

Only out-of-sample passes are meaningful.  Each ``--run-dir`` contributes the
validation matches of its own ``benchmark_summary.json``, scored with that
run's checkpoint.  Passing several runs whose validation folds partition the
data therefore yields cross-fitted surprise for every match.

Example:
    python scripts/analyze_surprise.py --run-dir checkpoints/temporal_receiver --model seq2spatial
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from soccer_ml.data.pitch_grid import PitchGrid, closest_teammate_zone_scores, receiver_candidates
from soccer_ml.data.schema import STATSBOMB_NODE_LAYOUT
from soccer_ml.data.temporal_receiver_dataset import TemporalReceiverDataset, collate_temporal_receiver_graphs
from soccer_ml.models.temporal_receiver import build_temporal_receiver_model
from soccer_ml.training.trainer import get_default_device

PASS_FLAGS = ("through_ball", "switch", "cross", "cut_back", "shot_assist", "goal_assist")
FOOT = ("Right Foot", "Left Foot")
# Restarts.  StatsBomb's "Recovery" and "Interception" pass types are open play.
DEAD_BALL = ("Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in")


def _point_in_polygon(polygon: list[float], x: float, y: float) -> bool:
    """Ray casting on a flat [x1, y1, x2, y2, ...] StatsBomb polygon."""
    points = list(zip(polygon[0::2], polygon[1::2]))
    inside = False
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


LEGACY_NODE_DIM = 9


def _to_legacy_layout(batch: Any) -> Any:
    """Rebuild the pre-October 9-channel input for old checkpoints.

    Old layout: x, y, v_x=0, v_y=0, height=1.80, weight=0.75, actor, teammate,
    keeper.  Only valid for models that read the passer from role features
    (every temporal model except ``spatial_gatv2``).
    """
    layout = STATSBOMB_NODE_LAYOUT
    nodes = batch.node_features
    constants = torch.tensor([0.0, 0.0, 1.80, 0.75], device=nodes.device).expand(*nodes.shape[:-1], 4)
    constants = constants * batch.player_mask.unsqueeze(-1).to(nodes.dtype)
    legacy = torch.cat(
        [nodes[..., [layout.x, layout.y]], constants, nodes[..., [layout.actor, layout.teammate, layout.keeper]]],
        dim=-1,
    )
    return dataclasses.replace(batch, node_features=legacy)


def _load_run(run_dir: Path, model_name: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    summary = json.loads((run_dir / "benchmark_summary.json").read_text())
    if summary.get("task", "receiver") != "target_zone":
        raise ValueError(f"{run_dir} is not a target_zone run.")
    checkpoint = torch.load(run_dir / model_name / "best_model.pt", map_location=device, weights_only=False)
    model = build_temporal_receiver_model(checkpoint["model_name"], **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint["model_kwargs"], summary


@torch.no_grad()
def score_passes(
    run_dir: Path, model_name: str, device: torch.device, max_matches: int | None, batch_size: int
) -> list[dict[str, Any]]:
    """Score every validation pass of one run and join it with its event."""
    model, model_kwargs, summary = _load_run(run_dir, model_name, device)
    legacy = model_kwargs["in_node_dim"] == LEGACY_NODE_DIM
    if legacy:
        if model_name == "spatial_gatv2":
            raise ValueError("Legacy 9-channel spatial_gatv2 checkpoints are not supported; retrain the model.")
        print(f"  [{run_dir.name}] legacy 9-channel checkpoint: rebuilding the old input layout")
    grid = PitchGrid(model_kwargs["grid_m"], model_kwargs["grid_n"])
    dataset_kwargs = {
        "context_length": model_kwargs["max_sequence_length"] - 1,
        "context_scope": summary["context_scope"],
        "grid_m": grid.grid_m,
        "grid_n": grid.grid_n,
        "include_target_pass_attributes": summary.get("include_target_pass_attributes", False),
    }
    pairs = summary["split"]["val_pairs"][:max_matches]
    rows: list[dict[str, Any]] = []
    for match_number, (three_sixty_path, events_path) in enumerate(pairs, start=1):
        print(f"  [{run_dir.name}] match {match_number}/{len(pairs)}: {Path(events_path).stem}", flush=True)
        dataset = TemporalReceiverDataset(three_sixty_path, events_path, **dataset_kwargs)
        events = {event["id"]: event for event in json.loads(Path(events_path).read_text())}
        frames = {frame["event_uuid"]: frame for frame in json.loads(Path(three_sixty_path).read_text())}
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_temporal_receiver_graphs)
        offset = 0
        for batch in loader:
            batch = batch.to(device)
            current = batch.current_graph_batch()
            model_input = _to_legacy_layout(batch) if legacy else batch
            log_probs = F.log_softmax(model(model_input)["target_zone_logits"], dim=-1)
            probs = log_probs.exp()
            target = batch.target_zone_label
            p_true = probs.gather(1, target[:, None]).squeeze(1)
            rarity = (probs * (probs > p_true[:, None])).sum(-1) + 0.5 * p_true
            entropy = -(probs * log_probs).sum(-1)
            rank = (probs > p_true[:, None]).sum(-1)
            prediction = log_probs.argmax(-1)
            teammate_prediction = closest_teammate_zone_scores(current, grid).argmax(-1)
            receivers = receiver_candidates(current, grid)
            model_receiver = receivers.nearest(grid.batch_zones_to_coords(prediction))
            teammate_receiver = receivers.nearest(grid.batch_zones_to_coords(teammate_prediction))
            for i in range(batch.batch_size):
                event_id = dataset.samples[offset + i].metadata["event_uuid"]
                event, details = events[event_id], events[event_id]["pass"]
                start, end = event["location"], details["end_location"]
                polygon = frames[event_id].get("visible_area", [])
                rows.append(
                    {
                        "run": run_dir.name,
                        "match_id": Path(events_path).stem,
                        "event_uuid": event_id,
                        "player_id": event["player"]["id"],
                        "player": event["player"]["name"],
                        "team": event["team"]["name"],
                        "position": event.get("position", {}).get("name", ""),
                        "pass_type": details.get("type", {}).get("name", "Open Play"),
                        "height": details.get("height", {}).get("name", ""),
                        "body_part": details.get("body_part", {}).get("name", ""),
                        **{flag: bool(details.get(flag)) for flag in PASS_FLAGS},
                        "length": float(details.get("length", 0.0)),
                        "start_x": start[0],
                        "start_y": start[1],
                        "end_x": end[0],
                        "end_y": end[1],
                        "start_visible": _point_in_polygon(polygon, *start[:2]) if polygon else False,
                        "end_visible": _point_in_polygon(polygon, *end[:2]) if polygon else False,
                        "target_zone": int(target[i]),
                        "model_zone": int(prediction[i]),
                        "teammate_zone": int(teammate_prediction[i]),
                        "p_top1": float(probs[i].max()),
                        "surprise": float(-log_probs[i, target[i]]),
                        "entropy": float(entropy[i]),
                        "rarity": float(rarity[i]),
                        "rank": int(rank[i]),
                        "has_receiver_candidate": bool(receivers.has_candidate[i]),
                        "model_receiver_correct": bool(model_receiver[i] == receivers.receiver[i]),
                        "teammate_receiver_correct": bool(teammate_receiver[i] == receivers.receiver[i]),
                    }
                )
            offset += batch.batch_size
    return rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read a passes.csv written by this script, restoring column types."""
    integer_columns = {"player_id", "target_zone", "model_zone", "teammate_zone", "rank"}
    text_columns = {"run", "match_id", "event_uuid", "player", "team", "position", "pass_type", "height", "body_part"}
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        for raw in csv.DictReader(file):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if key in text_columns:
                    row[key] = value
                elif key in integer_columns:
                    row[key] = int(value)
                elif value in ("True", "False"):
                    row[key] = value == "True"
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def _mean_ci(values: list[float]) -> tuple[float, float]:
    """Mean and normal-approximation 95% half-width."""
    n = len(values)
    mean = _mean(values)
    if n < 2:
        return mean, float("nan")
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    return mean, 1.96 * math.sqrt(variance / n)


def _accuracy(rows: list[dict[str, Any]], key: str) -> float:
    return _mean(float(row[key] == row["target_zone"]) for row in rows)


def _receiver_accuracy(rows: list[dict[str, Any]], key: str) -> float:
    eligible = [row for row in rows if row["has_receiver_candidate"]]
    return _mean(float(row[key]) for row in eligible)


def _surprise_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    ranks = sorted(row["rank"] for row in rows)
    return {
        "n": len(rows),
        "surprise": _mean(row["surprise"] for row in rows),
        "excess": _mean(row["surprise"] - row["entropy"] for row in rows),
        "rarity": _mean(row["rarity"] for row in rows),
        "median_rank": float(ranks[len(ranks) // 2]) if ranks else float("nan"),
    }


def build_report(rows: list[dict[str, Any]], min_player_passes: int, top_n: int) -> dict[str, Any]:
    report: dict[str, Any] = {"num_passes": len(rows), "num_matches": len({row["match_id"] for row in rows})}

    agree = [row for row in rows if row["model_zone"] == row["teammate_zone"]]
    disagree = [row for row in rows if row["model_zone"] != row["teammate_zone"]]
    report["overall"] = {
        "nll": _mean(row["surprise"] for row in rows),
        "model_top1": _accuracy(rows, "model_zone"),
        "closest_teammate_top1": _accuracy(rows, "teammate_zone"),
        "model_receiver_acc": _receiver_accuracy(rows, "model_receiver_correct"),
        "closest_teammate_receiver_acc": _receiver_accuracy(rows, "teammate_receiver_correct"),
        "top1_agreement_with_closest_teammate": len(agree) / max(len(rows), 1),
        "disagreement_model_top1": _accuracy(disagree, "model_zone"),
        "disagreement_closest_teammate_top1": _accuracy(disagree, "teammate_zone"),
    }

    report["visibility"] = {}
    for name, subset in (
        ("end_inside_visible_area", [row for row in rows if row["end_visible"]]),
        ("end_outside_visible_area", [row for row in rows if not row["end_visible"]]),
    ):
        report["visibility"][name] = {
            "n": len(subset),
            "share": len(subset) / max(len(rows), 1),
            "nll": _mean(row["surprise"] for row in subset),
            "model_top1": _accuracy(subset, "model_zone"),
            "closest_teammate_top1": _accuracy(subset, "teammate_zone"),
        }

    comparable = [
        row
        for row in rows
        if row["pass_type"] not in DEAD_BALL and row["body_part"] in FOOT and row["start_visible"] and row["end_visible"]
    ]
    report["unxpass_comparable_subset"] = {
        "definition": "no dead-ball passes, foot, start and end inside the 360 visible area (completed passes only)",
        "n": len(comparable),
        "share": len(comparable) / max(len(rows), 1),
        "nll": _mean(row["surprise"] for row in comparable),
        "model_receiver_acc": _receiver_accuracy(comparable, "model_receiver_correct"),
        "closest_teammate_receiver_acc": _receiver_accuracy(comparable, "teammate_receiver_correct"),
        "model_top1": _accuracy(comparable, "model_zone"),
    }

    calibration = []
    weighted_gap = 0.0
    for low, high in ((0.0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.01)):
        subset = [row for row in rows if low <= row["p_top1"] < high]
        if not subset:
            continue
        confidence, accuracy = _mean(row["p_top1"] for row in subset), _accuracy(subset, "model_zone")
        weighted_gap += len(subset) * abs(confidence - accuracy)
        calibration.append({"bin": [low, min(high, 1.0)], "n": len(subset), "confidence": confidence, "accuracy": accuracy})
    report["calibration"] = {"bins": calibration, "ece": weighted_gap / max(len(rows), 1)}

    report["event_flags"] = {"all": _surprise_summary(rows)}
    for flag in PASS_FLAGS:
        report["event_flags"][flag] = _surprise_summary([row for row in rows if row[flag]])
    for height in ("Ground Pass", "Low Pass", "High Pass"):
        report["event_flags"][f"height:{height}"] = _surprise_summary([row for row in rows if row["height"] == height])
    for pass_type in sorted({row["pass_type"] for row in rows}):
        report["event_flags"][f"type:{pass_type}"] = _surprise_summary(
            [row for row in rows if row["pass_type"] == pass_type]
        )

    open_play = [row for row in rows if row["pass_type"] not in DEAD_BALL]
    by_position: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in open_play:
        by_position[row["position"]].append(row)
    report["positions"] = {
        position: _surprise_summary(subset)
        for position, subset in sorted(by_position.items(), key=lambda item: -_mean(r["rarity"] for r in item[1]))
    }

    by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in open_play:
        by_player[row["player_id"]].append(row)
    players = []
    for subset in by_player.values():
        if len(subset) < min_player_passes:
            continue
        rarity, rarity_ci = _mean_ci([row["rarity"] for row in subset])
        players.append(
            {
                "player": subset[0]["player"],
                "team": Counter(row["team"] for row in subset).most_common(1)[0][0],
                "position": Counter(row["position"] for row in subset).most_common(1)[0][0],
                "n": len(subset),
                "rarity": rarity,
                "rarity_ci95": rarity_ci,
                "surprise": _mean(row["surprise"] for row in subset),
                "excess": _mean(row["surprise"] - row["entropy"] for row in subset),
                "through_ball_rate": _mean(float(row["through_ball"]) for row in subset),
                "mean_length": _mean(row["length"] for row in subset),
            }
        )
    players.sort(key=lambda player: -player["rarity"])
    report["players"] = {
        "min_passes": min_player_passes,
        "num_players": len(players),
        "top": players[:top_n],
        "bottom": players[-top_n:][::-1],
        "all": players,
    }
    return report


def print_report(report: dict[str, Any]) -> None:
    overall = report["overall"]
    print("=" * 96)
    print(f"Surprise analysis: {report['num_passes']} passes in {report['num_matches']} held-out matches")
    print("=" * 96)
    print(
        f"NLL {overall['nll']:.3f} | top-1 model {overall['model_top1']:.2%} vs closest teammate "
        f"{overall['closest_teammate_top1']:.2%} | receiver-acc model {overall['model_receiver_acc']:.2%} "
        f"vs closest teammate {overall['closest_teammate_receiver_acc']:.2%}"
    )
    print(
        f"Top-1 agreement with closest teammate {overall['top1_agreement_with_closest_teammate']:.1%}; "
        f"where they disagree: model {overall['disagreement_model_top1']:.2%}, "
        f"closest teammate {overall['disagreement_closest_teammate_top1']:.2%}"
    )
    for name, values in report["visibility"].items():
        print(
            f"  {name:<26} n={values['n']:6d} ({values['share']:.0%}) nll={values['nll']:.2f} "
            f"top-1 model {values['model_top1']:.2%} / closest {values['closest_teammate_top1']:.2%}"
        )
    comparable = report["unxpass_comparable_subset"]
    print(
        f"\nun-xPass-comparable subset ({comparable['definition']}): n={comparable['n']} ({comparable['share']:.0%})\n"
        f"  receiver-acc model {comparable['model_receiver_acc']:.2%} vs closest teammate "
        f"{comparable['closest_teammate_receiver_acc']:.2%} | zone top-1 {comparable['model_top1']:.2%} "
        f"| nll {comparable['nll']:.3f}"
    )
    print(f"\nCalibration of the top-1 probability (ECE {report['calibration']['ece']:.4f}):")
    for entry in report["calibration"]["bins"]:
        low, high = entry["bin"]
        print(f"  p in [{low:.2f}, {high:.2f}) n={entry['n']:6d} confidence={entry['confidence']:.3f} accuracy={entry['accuracy']:.3f}")

    def line(name: str, values: dict[str, float]) -> str:
        return (
            f"  {name:<28} n={values['n']:6d} surprise={values['surprise']:.2f} excess={values['excess']:+.2f} "
            f"rarity={values['rarity']:.3f} median_rank={values['median_rank']:.0f}"
        )

    print("\nEvent-level validation (rarity ~0.5 = typical under a calibrated model):")
    for name, values in report["event_flags"].items():
        if values["n"]:
            print(line(name, values))
    print("\nOpen-play passes by position (sorted by rarity):")
    for name, values in report["positions"].items():
        if values["n"] >= 300:
            print(line(name, values))
    players = report["players"]
    print(f"\nPlayers with >= {players['min_passes']} open-play passes: {players['num_players']}")
    for label in ("top", "bottom"):
        print(f"  -- {label} by rarity")
        for player in players[label]:
            print(
                f"     {player['player'][:30]:<30} {player['position'][:24]:<24} n={player['n']:4d} "
                f"rarity={player['rarity']:.3f}±{player['rarity_ci95']:.3f} surprise={player['surprise']:.2f} "
                f"excess={player['excess']:+.2f} len={player['mean_length']:.0f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-pass surprise analysis of a target-zone model.")
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        help="Benchmark output directory (repeatable; default checkpoints/temporal_receiver).",
    )
    parser.add_argument("--model", default="seq2spatial")
    parser.add_argument("--from-csv", type=Path, help="Rebuild the report from an existing passes.csv (no scoring).")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <first run dir>/analysis_<model>.")
    parser.add_argument("--max-matches", type=int, help="Score only the first N validation matches per run.")
    parser.add_argument("--min-player-passes", type=int, default=150)
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = args.run_dir or [Path("checkpoints/temporal_receiver")]
    if args.from_csv:
        rows = load_rows(args.from_csv)
    else:
        device = get_default_device()
        rows = []
        for run_dir in run_dirs:
            rows.extend(score_passes(run_dir, args.model, device, args.max_matches, args.batch_size))

    report = build_report(rows, args.min_player_passes, args.top_n)
    print_report(report)

    output_dir = args.output_dir or (args.from_csv.parent if args.from_csv else run_dirs[0] / f"analysis_{args.model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.from_csv:
        with (output_dir / "passes.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    report["run_dirs"] = sorted({row["run"] for row in rows})
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved {output_dir / 'passes.csv'} and {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
