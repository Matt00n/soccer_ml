# Temporal Pass Target Prediction

The original single-freeze-frame GATv2 receiver benchmark is untouched in
`scripts/train_receiver.py`. The temporal benchmark is
`scripts/train_temporal_receiver.py`. It trains every selected model and
evaluates the naive baselines on the same split.

## Task

By default (`--task target_zone`) models predict **where a completed pass
ends**. The 120 × 80 StatsBomb pitch is split into a **26 × 17 grid**
(442 zones of about 4.6 × 4.7 yd, which is about 4 m cells on a 105 × 68 m
pitch). The label is the zone of `pass.end_location`. Zone index
`k = bin_x * 17 + bin_y`, so logits reshape row-major to `(26, 17)`. The grid
can be changed with `--grid-m/--grid-n`. This target avoids the noisy
"closest teammate to the end location" receiver label, which is often wrong
and sometimes not even visible in the 360 frame. `--task receiver` reproduces
the earlier receiver benchmark.

Reported metrics, for models and baselines alike:

- **NLL** of the true zone, the headline metric. It is a proper scoring rule,
  and per pass it is exactly the "surprise" used for creativity analysis.
  Every baseline except the hard closest-teammate ranking has one; its
  probabilistic counterpart is `teammate_kernel`.
- top-1 / top-3 / top-5 zone accuracy
- within-1-zone accuracy: the top-1 zone is the true zone or one of its 8 neighbours
- mean and median distance from the top-1 zone center to the true end
  location, in StatsBomb yards and in meters after rescaling to 105 × 68 m
- receiver accuracy, following the un-xPass protocol (Robberechts et al., KDD
  2023). The predicted receiver is the visible teammate closest to the top-1
  zone; the true receiver is the visible teammate closest to the actual end
  location.

The best checkpoint is selected by validation NLL.

## Inputs and leakage rules

- Node features (5, `STATSBOMB_NODE_LAYOUT`): `x, y, is_actor, is_teammate,
  is_keeper`. 360 frames have no velocities or player attributes, so no
  constant placeholder channels are used. (Checkpoints from before this change
  expect 9 channels; `scripts/analyze_surprise.py` can still score
  `seq2spatial` and the history models from those.)
- Graph features (7): `is_pass, is_shot, period, minute/90, under_pressure,
  pass_height, pass_body_part`. The target pass's `pass_height` and
  `pass_body_part` describe the decision being predicted, and a "High Pass"
  all but reveals a long pass. They are **zeroed on the target frame by
  default**; `--include-target-pass-attributes` keeps them, which turns the
  task into "given how the pass is played, where does it go?".
- History frames and events are in the **target passer's frame**.
  StatsBomb stores every event in the frame of the team performing it, so
  opponent events (pressures, duels, clearances, …; about 13% of 360 frames)
  are rotated by 180° and their `teammate` flags are inverted relative to the
  passer's team.
- Per-event history features (`EVENT_NUMERIC_DIM = 11`): time gap, duration,
  start→end length and direction (for passes, carries and shots), pressure,
  start and end location, and whether the event was performed by the
  passer's team. Plus categorical type, outcome and body part. The target
  event's own labels are never used.

## Reflection symmetry (`--symmetry`)

The GATv2 backbone computes one representation per view of a reflection
group, and outputs are frame-averaged over those views. The receiver logits
are invariant. The **zone logits are equivariant**: each view is decoded in its
own frame, its logit grid is reflected back, and then the grids are averaged.
If views were averaged *before* decoding, a left-wing situation and its
mirrored right-wing twin would get the same prediction.

- `y_flip` (default): only the wing reflection. StatsBomb always has the team
  in possession attacking left→right, so reflecting the pitch length reverses
  the direction of attack and is not a symmetry of pass destinations. It is
  also 4× cheaper than D2 (2×2 instead of 4×4 GATv2 calls per layer).
- `d2`: all four TacticAI reflections (for ablations).
- `none`: identity view only.

Event features (locations, directions) are reflected into each view as well,
so every model is exactly equivariant. `tests/test_temporal_receiver.py`
checks this.

## Models

| name | spatial encoder | temporal model |
|---|---|---|
| `spatial_gatv2` | GATv2 on the target frame only (`SoccerGATv2Model`) | none |
| `graph_transformer` | GATv2 on every history snapshot + target | causal Transformer over graph latents |
| `graph_gru` | same | masked stacked GRU over graph latents |
| `transport_memory` | same | per-player memory aligned by Sinkhorn transport |
| `seq2spatial` | GATv2 on the target frame only | GRU (or `--sequence-encoder transformer`) over the **event** sequence |

`seq2spatial` (`EventSeqToSpatialGATv2`) encodes the possession's event
history. The final slot holds a learned query token plus the time since the
last event, and the resulting state `h_temporal` is used in two places:

- It is appended to the target frame's graph features, where it conditions
  every GATv2 attention logit.
- It is concatenated into the zone head's input. This second path is needed
  because GATv2 graph features only enter attention logits, not messages.

The graph backbone runs once, on the target frame. There are no redundant
per-snapshot backbone passes.

The history models only run the backbone on non-padded timesteps. Heads are a
single linear layer unless `--use-mlp` is passed. Fourier pitch features are
off unless `--num-fourier-bands > 0`.

## Baselines

All baselines output a full ranking over zones:

- `teammate_kernel`: a probabilistic closest-teammate baseline. It is a
  mixture of Gaussians on visible teammates, weighted by their distance to the
  passer, with (τ, σ, ε) fitted on train.
- `passer_zone_prior`: a zone-transition table P(target zone | passer zone),
  fitted on the training split and smoothed towards `pass_vector_prior`.
- `pass_vector_prior`: the empirical distribution of pass displacements,
  shifted to the passer's zone.
- `most_frequent_zone`: the marginal target-zone frequency.
- `closest_teammate_zone`: zones ranked by the distance of the teammate they
  contain to the passer.
- `random_zone`.

The priors are fitted on the training split, so their training-split scores
are in-sample.

## Benchmarking

Prefer match-level splits for reported numbers:

```bash
conda run -n soccer_ml python scripts/train_temporal_receiver.py \
  --train-dir data/open-data/train --val-dir data/open-data/validation \
  --train-file-count 20 --val-file-count 5 --sample-seed 42 \
  --models spatial_gatv2 seq2spatial graph_gru
```

Quick single-match smoke run (random pass split):

```bash
conda run -n soccer_ml python scripts/train_temporal_receiver.py \
  --three-sixty data_toy_receiver/three_sixty_3764440.json \
  --events data_toy_receiver/events_3764440.json --epochs 2
```

Useful flags: `--symmetry {y_flip,d2,none}`, `--grid-m/--grid-n`,
`--zone-label-sigma` (Gaussian label smoothing in yards, 0 = hard CE),
`--use-mlp`, `--sequence-encoder {gru,transformer}`,
`--include-target-pass-attributes`, `--task receiver`.

The run writes `benchmark_summary.json`, per-model metric histories, and
self-describing `best_model.pt` checkpoints under
`checkpoints/temporal_receiver/`. The summary includes hashes of the target
event IDs, so the exact split can be audited. In directory mode, validation
matches are sampled first with `sample_seed + 1`. Their IDs are removed before
training matches are sampled with `sample_seed`, even when both arguments
point at the same folder.

Evaluate a checkpoint. The task and grid are read from the checkpoint, and
the `--prior-*` files fit the prior baselines; without them the priors are fit
in-sample:

```bash
conda run -n soccer_ml python scripts/evaluate_temporal_receiver.py \
  --three-sixty data_toy_receiver/three_sixty_3764661.json \
  --events data_toy_receiver/events_3764661.json \
  --prior-three-sixty data_toy_receiver/three_sixty_3764440.json \
  --prior-events data_toy_receiver/events_3764440.json \
  --checkpoint checkpoints/temporal_receiver/seq2spatial/best_model.pt
```

## Surprise analysis

`scripts/analyze_surprise.py` scores every validation pass of a run with its
checkpoint and joins it with the StatsBomb event. It reports:

- agreement with the closest-teammate baseline;
- results inside and outside the visible area;
- an un-xPass-comparable subset;
- calibration;
- surprise by pass flag (through ball, switch, cross, …) and by position;
- player rankings with 95% CIs, based on **rarity**. Rarity is the probability
  mass on zones the model rated more likely than the chosen one, plus half the
  chosen zone's probability. Under a calibrated model it averages 0.5 for any
  role, so it normalizes for the situation.

It writes `passes.csv` and `report.json`. You can pass several `--run-dir`
values whose validation folds partition the matches; scoring each fold with
its own run's checkpoint gives out-of-fold (cross-fitted) surprise for every
match. `--from-csv` rebuilds the report without rescoring.

```bash
conda run -n soccer_ml python scripts/analyze_surprise.py \
  --run-dir checkpoints/temporal_receiver --model seq2spatial
```

`--context-length` must match the checkpoint. The parser only links events
that have a StatsBomb 360 freeze frame; no regular tracking frames are
interpolated. Checkpoints from before the target-zone rewrite (receiver-only,
D2 + Fourier) cannot be loaded by the current model classes.
