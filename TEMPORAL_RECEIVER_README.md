# Temporal Receiver Prediction

The original single-freeze-frame GATv2 benchmark is untouched in
`scripts/train_receiver.py`.  The parallel temporal benchmark is
`scripts/train_temporal_receiver.py` and trains two models against the same
receiver targets and the same three naive baselines.

## Implemented models

- `spatial_gatv2`: An adapter around the existing, unmodified
  `SoccerGATv2Model`. It consumes only the final target freeze frame and is the
  learned spatial baseline in every temporal benchmark run.
- `graph_transformer`: A D2 GATv2 encoder and attention/actor graph readout at
  every event snapshot, followed by a causal Transformer over graph latents.
  It receives irregular time gaps and embeddings for *prior* event type,
  outcome, body part, pass geometry, pressure, duration, and location.
- `graph_gru`: The same Fourier-enhanced D2 GATv2 graph encoder and event
  latents as `graph_transformer`, with a mask-aware stacked GRU over events
  instead of self-attention.
- `transport_memory`: The same spatial encoder, plus a per-visible-node GRU
  memory. Consecutive node sets are joined with differentiable Sinkhorn
  transport using learned embedding affinity, pitch displacement, and team
  side. This avoids treating a freeze-frame node index as a player ID.

Both encoders add continuous 2D Fourier pitch features. They are calculated in
each of the four existing D2 views before view averaging, retaining the spatial
benefit without bypassing the repository's reflection-invariant readout.

Each sequence is left-padded to `context_length + 1`; its final graph is the
target pass. Event labels and the recipient proxy are populated only for prior
graphs. In particular, the target pass outcome and target-recipient proxy are
zeroed, so they cannot leak the receiver label.

## Benchmarking

Single-match mode uses the same deterministic pass-level split for the spatial
GATv2 baseline, both temporal models, and all naive baselines:

```bash
conda run -n soccer_ml python scripts/train_temporal_receiver.py \
  --three-sixty data_toy_receiver/three_sixty_3764440.json \
  --events data_toy_receiver/events_3764440.json \
  --context-length 8 --context-scope possession \
  --models graph_transformer transport_memory
```

For a stronger generalization benchmark, supply separate match directories—the
same layouts accepted by `train_receiver.py` are supported:

```bash
conda run -n soccer_ml python scripts/train_temporal_receiver.py \
  --train-dir data/open-data/train --val-dir data/open-data/validation \
  --train-file-count 20 --val-file-count 5 --sample-seed 42
```

The run writes `benchmark_summary.json`, per-model metric histories, and
self-describing `best_model.pt` checkpoints under
`checkpoints/temporal_receiver/`. The summary includes hashes of target event
IDs, so the exact split can be audited. The random-pass split is useful for
rapid ablations, but match-level train/validation directories are preferable
for reporting generalization. In directory mode, validation matches are sampled
first with `sample_seed + 1`; their IDs are removed before training matches are
sampled with `sample_seed`, even when both arguments point at the same folder.

Evaluate a checkpoint with the identical target-graph baselines:

```bash
conda run -n soccer_ml python scripts/evaluate_temporal_receiver.py \
  --three-sixty data_toy_receiver/three_sixty_3764440.json \
  --events data_toy_receiver/events_3764440.json \
  --context-length 8 --context-scope possession \
  --checkpoint checkpoints/temporal_receiver/graph_transformer/best_model.pt
```

`context-length` must match the checkpoint.  The parser only links events that
have a StatsBomb 360 freeze frame; no regular tracking frames are interpolated.
