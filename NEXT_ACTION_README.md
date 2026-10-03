# Next-Action Prediction

`scripts/train_next_action.py` trains the model that predicts the next on-ball
action from a StatsBomb 360 freeze frame and the possession's event history.
This is part A of the creativity work: it gives the likelihood of the action
that was actually chosen, and from it the surprise.

## Task

One sample is one on-ball decision whose event has a 360 freeze frame. That
covers every pass (completed and failed), every carry of at least one grid
cell (4.6 yd), and every shot, dribble and clearance. The actor's team can be
in possession or not (most clearances are by the defending team). Shorter
carries ("receive, touch") are not decisions; they stay in the history.

The likelihood factorizes as

```
log P(action) = log P(type) + log P(zone | type) + log P(height | pass, zone)
```

| factor | classes | applies to |
|---|---|---|
| type | pass, carry, shot, dribble, clearance | all |
| zone | 26 × 17 grid (pass and carry heads) | passes (observed end location, also for failed passes), carries |
| height | ground / low / high | passes |

Surprise therefore splits into an unexpected action, an unexpected
destination, and an unexpected technique for that destination.

### Decision moments

StatsBomb starts every carry at the exact timestamp the player gained the
ball, and never follows a carry with another carry by the same player. If
every action were sampled at its own timestamp, the type head could read the
answer off these conventions. The first run did exactly that: 95% type
accuracy, 98.7% of it after the player's own short carry, where a carry is
impossible by annotation. Decisions are therefore aligned:

| context | anchored at | type set | share (toy match) |
|---|---|---|---|
| `spell_start` | the event in which the player gained the ball (receipt, recovery, interception, …). Covers long carries, one-touch actions, and actions after a short touch. | all 5 | ~67% |
| `after_carry` | release time, after the player's own long carry | carry masked out (`type_mask`) | ~25% |
| `other` | the action itself (first-time clearances, actions after an own dribble, …) | all 5 | ~8% |

Frame, history, timing and situation features come from the anchor. The
trainer reports type accuracy and NLL for each context; `spell_start` is
where carrying vs. releasing is genuinely decided.

## Inputs

- **Freeze frame** (5 node channels): x, y, is_actor, is_teammate, is_keeper.
  It is given in the actor's team frame. Edges carry the opposing-team flag.
- **Graph features** (15) describe the situation only:
  - second half, extra time, minute / 90, under pressure;
  - open play, or a one-hot restart type (corner, free kick, goal kick, kick
    off, throw-in, penalty), which is fixed by the preceding stoppage;
  - women's competition, looked up in `matches/`;
  - fraction of visible zones, visible teammates / 11, visible opponents / 11.
- **Visible area**: the camera polygon rasterized onto the 26 × 17 grid. Zones
  inside and outside it use separate linear maps in the zone head.
- **History**: the last 16 events of the possession from the full event
  stream; these need no freeze frame. Each event carries:
  - type, outcome, body part and pass sub-type;
  - time gap, duration, start and end location, length and direction,
    pressure, and same-team-as-actor.

  Opponent events are rotated into the actor's frame. The decision slot holds
  only the time since the previous event.

Leakage rules:
- The target's type, height and body part are never inputs.
- The opponent's reactions that StatsBomb indexes *before* the action they
  react to (e.g. "Dribbled Past", duels at the same timestamp) are dropped
  from the history.

## Optional inputs (v3)

All are off by default, so v2 checkpoints and their data configuration keep
working unchanged. They are stored in the checkpoint's `data_config` and
`model_kwargs`, so value models built on a v3 encoder pick them up
automatically.

| flag | adds | notes |
|---|---|---|
| `--actor-role` | one-hot of the acting player's role (goalkeeper, centre-back, full-back, defensive / central / attacking midfield, wide, forward), from StatsBomb's position | Side-agnostic, so invariant under the wing reflection. **Surprise becomes role-relative** ("unusual for a centre-back"); train without it for population-relative surprise. |
| `--game-state` | goal difference before the decision (clipped to ±3, / 3) and home/away | Home/away is looked up in `matches/` (0.5 if unavailable). |
| `--velocities` | v_x, v_y and a "velocity known" flag per player | Estimated by matching each team's players to the previous freeze frame (0.2–3 s earlier) with a gated Hungarian assignment; matches implying more than 11 yd/s are rejected. About 80% of visible players get a velocity. Node features then use `STATSBOMB_VELOCITY_NODE_LAYOUT`, and reflections negate the velocity channels. |
| `--edge-distance` | pairwise player distance (yards / 40) as a second edge feature | Computed in the model; distances don't change under reflections. |

## Model

`NextActionModel` uses the `seq2spatial` design, which beat the spatial-only
GATv2 on the pass task: val NLL 3.519 vs 3.608.

1. A masked GRU encodes the history plus a query token.
2. Its state is appended to the graph features of the wing-flip GATv2 (4
   layers) and to the readout [pooled, actor, context].
3. Linear heads produce the type, zone and height logits.

Zone and height grids are decoded per view, reflected back and averaged, so a
mirrored situation gets a mirrored prediction (tested). Type logits are
averaged over the views.

## Metrics and baselines

- **Headline:** joint NLL. It is the training loss and the
  checkpoint-selection criterion.
- **Factors:** the NLL of each factor, type accuracy, macro and per-class
  recall, and pass/carry zone top-k, within-one-zone, distance and receiver
  accuracy (un-xPass protocol). Height accuracy is reported for 3 classes and
  for ground vs lofted.
- **Baselines**, fitted on train:
  - `marginal`: plain frequencies;
  - `zone_prior`: type given the actor's zone, a zone-transition table
    smoothed towards the pass/carry displacement prior, and height given the
    pass displacement.

## Run

```bash
conda run -n soccer_ml python scripts/train_next_action.py \
  --train-dir data/open-data/data --val-dir data/open-data/data \
  --train-file-count 375 --val-file-count 50 --sample-seed 42 --epochs 12
```

This uses all matches, with the same 50 validation matches as the pass
benchmark (broken matches are skipped and listed in `summary.json`). It takes
about 8 minutes to load and about 43 minutes per epoch on MPS. The outputs in
`checkpoints/next_action/` are `best_model.pt` (self-describing: model
kwargs and data config), `history.json` and `summary.json`.

Ablation flags: `--no-history`, `--no-visibility`, `--symmetry {y_flip,d2,none}`,
`--use-mlp`, `--context-length`, `--carry-min-length`, and `--device cpu` to
use the second lane.
