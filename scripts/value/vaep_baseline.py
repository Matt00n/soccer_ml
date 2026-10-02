"""VAEP and VAEP-xG value baselines (Decroos et al., KDD 2019) via socceraction.

Runs in the separate ``soccer_vaep`` env (socceraction needs Python < 3.13):

    ~/miniforge3/envs/soccer_vaep/bin/python scripts/value/vaep_baseline.py

Both variants use socceraction's standard VAEP features of the last three
SPADL actions and two gradient-boosted classifiers per variant:

* VAEP:    P(team scores / concedes a goal within the next 10 actions)
* VAEP-xG: the same with xG-sequence labels, 1 - prod(1 - xG) over the team's
  (or opponent's) shots in the window, as in un-xPass.  Soft labels are less
  noisy than goals.

Training uses event-only matches (no 360 data needed), never the validation
matches of the 360 benchmark.  Outputs in ``--output-dir``:

* ``vaep_val_predictions.csv``: every SPADL action of the validation matches
  with labels, P(score), P(concede) and VAEP values of both variants;
* ``value_labels_360.csv``: labels for every SPADL action of all 360 matches,
  for training our own value model with the same targets;
* ``vaep_metrics.json``: the un-xPass Table 3 protocol (completed open-play
  passes, binary goal labels) for both variants.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

warnings.filterwarnings("ignore")

import socceraction.spadl as spadl  # noqa: E402
import socceraction.vaep.features as fs  # noqa: E402
import socceraction.vaep.labels as lab  # noqa: E402
from socceraction.data.statsbomb import StatsBombLoader  # noqa: E402
from socceraction.vaep.formula import value as vaep_value  # noqa: E402

NB_PREV_ACTIONS = 3
NR_ACTIONS = 10
FEATURE_FNS = [
    fs.actiontype_onehot,
    fs.result_onehot,
    fs.actiontype_result_onehot,
    fs.bodypart_onehot,
    fs.time,
    fs.startlocation,
    fs.endlocation,
    fs.startpolar,
    fs.endpolar,
    fs.movement,
    fs.team,
    fs.time_delta,
    fs.space_delta,
    fs.goalscore,
]
SHOT_TYPES = {"shot", "shot_freekick", "shot_penalty"}
META_COLUMNS = [
    "game_id", "original_event_id", "action_id", "period_id", "time_seconds", "team_id",
    "player_id", "type_name", "result_name", "bodypart_name",
]
_ROOT: str = ""


def _match_index(root: str) -> dict[int, dict]:
    index = {}
    for path in glob.glob(f"{root}/matches/*/*.json"):
        for match in json.loads(Path(path).read_text()):
            index[int(match["match_id"])] = {
                "home_team_id": match["home_team"]["home_team_id"],
                "competition": match["competition"]["competition_name"],
                "season": match["season"]["season_name"],
            }
    return index


def _xg_sequence(actions: pd.DataFrame, xg: np.ndarray, same_team: bool) -> np.ndarray:
    """1 - prod(1 - xG) over shots of the (opposing) team in the next NR_ACTIONS actions."""
    team = actions["team_id"].to_numpy()
    log_miss = np.zeros(len(actions))
    for offset in range(NR_ACTIONS):
        shifted_xg = np.zeros(len(actions))
        shifted_team = np.full(len(actions), -1, dtype=team.dtype)
        shifted_xg[: len(actions) - offset] = xg[offset:]
        shifted_team[: len(actions) - offset] = team[offset:]
        mask = (shifted_team == team) if same_team else ((shifted_team != team) & (shifted_team != -1))
        log_miss += np.where(mask, np.log1p(-np.clip(shifted_xg, 0.0, 0.999)), 0.0)
    return 1.0 - np.exp(log_miss)


def process_match(job: tuple[int, int, bool]) -> tuple[pd.DataFrame, np.ndarray | None, list[str] | None] | None:
    """SPADL actions, labels and (optionally) float32 VAEP features of one match."""
    game_id, home_team_id, with_features = job
    try:
        events = StatsBombLoader(getter="local", root=_ROOT).events(game_id)
        actions = spadl.add_names(spadl.statsbomb.convert_to_actions(events, home_team_id=home_team_id))
    except Exception as error:  # broken upstream files must not stop the run
        print(f"  skipping {game_id}: {type(error).__name__}: {str(error)[:80]}", flush=True)
        return None
    actions = actions.reset_index(drop=True)
    raw = json.loads(Path(f"{_ROOT}/events/{game_id}.json").read_text())
    xg_by_event = {e["id"]: e["shot"].get("statsbomb_xg", 0.0) for e in raw if e["type"]["name"] == "Shot"}
    xg = np.array([
        xg_by_event.get(event_id, 0.0) if type_name in SHOT_TYPES else 0.0
        for event_id, type_name in zip(actions["original_event_id"], actions["type_name"])
    ])
    meta = actions[META_COLUMNS].copy()
    meta["scores"] = lab.scores(actions, nr_actions=NR_ACTIONS)["scores"].to_numpy().astype(np.int8)
    meta["concedes"] = lab.concedes(actions, nr_actions=NR_ACTIONS)["concedes"].to_numpy().astype(np.int8)
    meta["xg_scores"] = _xg_sequence(actions, xg, same_team=True).astype(np.float32)
    meta["xg_concedes"] = _xg_sequence(actions, xg, same_team=False).astype(np.float32)
    if not with_features:
        return meta, None, None
    gamestates = fs.play_left_to_right(fs.gamestates(actions, NB_PREV_ACTIONS), home_team_id)
    features = pd.concat([fn(gamestates) for fn in FEATURE_FNS], axis=1)
    # float16 halves memory; exact for one-hots and far below xgboost's 256 histogram bins otherwise.
    return meta, features.to_numpy(np.float16), list(features.columns)


def _init(root: str) -> None:
    global _ROOT
    _ROOT = root


def collect(game_ids, match_index, root: str, workers: int, with_features: bool, label: str):
    jobs = [(g, match_index[g]["home_team_id"], with_features) for g in game_ids if g in match_index]
    metas, features, names = [], [], None
    started = time.time()
    with Pool(workers, initializer=_init, initargs=(root,)) as pool:
        for done, result in enumerate(pool.imap(process_match, jobs, chunksize=4), start=1):
            if result is not None:
                meta, x, feature_names = result
                metas.append(meta)
                if x is not None:
                    if names is not None and feature_names != names:
                        raise ValueError(f"Feature columns differ in match {meta['game_id'].iloc[0]}.")
                    features.append(x)
                    names = feature_names
            if done % 100 == 0 or done == len(jobs):
                print(f"  [{label}] {done}/{len(jobs)} matches ({time.time() - started:.0f}s)", flush=True)
    return metas, features, names


class _Chunks(xgb.DataIter):
    """Feed per-match float16 feature chunks to xgboost without stacking them."""

    def __init__(self, chunks: list[np.ndarray]):
        self.chunks, self.position = chunks, 0
        super().__init__()

    def next(self, input_data) -> bool:
        if self.position == len(self.chunks):
            return False
        input_data(data=self.chunks[self.position].astype(np.float32))
        self.position += 1
        return True

    def reset(self) -> None:
        self.position = 0


def quantized_split(chunks: list[np.ndarray], seed: int) -> tuple[xgb.QuantileDMatrix, xgb.QuantileDMatrix, np.ndarray]:
    """Train/early-stopping matrices; 10% of the training matches are held out."""
    rng = np.random.default_rng(seed)
    is_valid = np.zeros(len(chunks), dtype=bool)
    is_valid[rng.choice(len(chunks), size=max(1, len(chunks) // 10), replace=False)] = True
    train = xgb.QuantileDMatrix(_Chunks([c for c, v in zip(chunks, is_valid) if not v]), max_bin=256)
    valid = xgb.QuantileDMatrix(_Chunks([c for c, v in zip(chunks, is_valid) if v]), ref=train)
    return train, valid, is_valid


def train_classifier(train, valid, threads: int, seed: int) -> xgb.Booster:
    """Logistic boosted trees with early stopping on the held-out matches."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "nthread": threads,
        "seed": seed,
    }
    booster = xgb.train(
        params, train, num_boost_round=1500, evals=[(valid, "held_out")], early_stopping_rounds=50, verbose_eval=False
    )
    print(f"    best iteration {booster.best_iteration}, held-out logloss {booster.best_score:.5f}", flush=True)
    return booster


def unxpass_protocol(frame: pd.DataFrame, column: str) -> dict[str, float]:
    """un-xPass Table 3: completed open-play passes, binary goal-within-10-actions label."""
    rows = frame[frame["type_name"].isin(["pass", "cross"]) & (frame["result_name"] == "success")]
    y, p = rows["scores"].to_numpy(), np.clip(rows[column].to_numpy(), 1e-6, 1 - 1e-6)
    return {"n": int(len(rows)), "positives": int(y.sum()), "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", default="data/open-data/data")
    parser.add_argument("--val-summary", default="checkpoints/temporal_receiver/benchmark_summary.json",
                        help="Benchmark summary whose split.val_pairs are the held-out matches.")
    parser.add_argument("--train-matches", type=int, default=600, help="Event-only training matches (memory bound).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/value_baselines")
    parser.add_argument("--max-val-matches", type=int, help="Quick runs only.")
    parser.add_argument("--skip-360-labels", action="store_true", help="Quick runs only.")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    match_index = _match_index(args.data_root)
    event_ids = {int(Path(p).stem) for p in glob.glob(f"{args.data_root}/events/*.json")}
    ids_360 = sorted(int(Path(p).stem) for p in glob.glob(f"{args.data_root}/three-sixty/*.json"))
    val_ids = sorted(int(Path(pair[1]).stem) for pair in json.loads(Path(args.val_summary).read_text())["split"]["val_pairs"])
    pool = sorted((event_ids & set(match_index)) - set(val_ids))
    val_ids = val_ids[: args.max_val_matches]
    train_ids = sorted(random.Random(args.seed).sample(pool, min(args.train_matches, len(pool))))
    print(f"{len(event_ids)} event matches | {len(val_ids)} validation | training on {len(train_ids)} of {len(pool)}")

    train_metas, train_chunks, feature_names = collect(train_ids, match_index, args.data_root, args.workers, True, "train")
    rows = sum(len(chunk) for chunk in train_chunks)
    print(f"training actions: {rows:,} x {len(feature_names)} features "
          f"({sum(chunk.nbytes for chunk in train_chunks) / 1e9:.1f} GB as float16)", flush=True)
    train, valid, is_valid = quantized_split(train_chunks, args.seed)
    del train_chunks
    models = {}
    for variant, targets in (("vaep", ("scores", "concedes")), ("vaep_xg", ("xg_scores", "xg_concedes"))):
        for target in targets:
            print(f"  fitting {variant}:{target}", flush=True)
            train.set_label(np.concatenate([m[target].to_numpy(np.float32) for m, v in zip(train_metas, is_valid) if not v]))
            valid.set_label(np.concatenate([m[target].to_numpy(np.float32) for m, v in zip(train_metas, is_valid) if v]))
            models[(variant, target)] = train_classifier(train, valid, args.threads, args.seed)
    del train, valid

    val_metas, val_chunks, _ = collect(val_ids, match_index, args.data_root, args.workers, True, "validation")
    val_meta = pd.concat(val_metas, ignore_index=True)
    dval = xgb.DMatrix(np.concatenate(val_chunks).astype(np.float32))
    for variant, (score_target, concede_target) in (("vaep", ("scores", "concedes")),
                                                    ("vaep_xg", ("xg_scores", "xg_concedes"))):
        for target, name in ((score_target, "p_scores"), (concede_target, "p_concedes")):
            booster = models[(variant, target)]
            val_meta[f"{variant}_{name}"] = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
        values = []
        for _, game in val_meta.groupby("game_id", sort=False):
            values.append(vaep_value(game, game[f"{variant}_p_scores"], game[f"{variant}_p_concedes"]))
        rated = pd.concat(values)
        val_meta[f"{variant}_offensive_value"] = rated["offensive_value"].to_numpy()
        val_meta[f"{variant}_defensive_value"] = rated["defensive_value"].to_numpy()
        val_meta[f"{variant}_value"] = rated["vaep_value"].to_numpy()
    val_meta.to_csv(output / "vaep_val_predictions.csv", index=False)

    metrics = {variant: unxpass_protocol(val_meta, f"{variant}_p_scores") for variant in ("vaep", "vaep_xg")}
    metrics["config"] = {"train_matches": len(train_ids), "val_matches": len(val_ids), "features": len(feature_names),
                         "nr_actions": NR_ACTIONS, "nb_prev_actions": NB_PREV_ACTIONS}
    (output / "vaep_metrics.json").write_text(json.dumps(metrics, indent=2))
    for variant in ("vaep", "vaep_xg"):
        m = metrics[variant]
        print(f"{variant:>8}: AUC {m['auc']:.3f} | Brier {m['brier']:.4f} | log-loss {m['logloss']:.4f} "
              f"(n={m['n']}, positives={m['positives']})")

    if args.skip_360_labels:
        return
    print("exporting labels for all 360 matches ...", flush=True)
    labels_360, _, _ = collect(ids_360, match_index, args.data_root, args.workers, False, "360 labels")
    pd.concat(labels_360, ignore_index=True).to_csv(output / "value_labels_360.csv", index=False)
    print(f"saved to {output}")


if __name__ == "__main__":
    main()
