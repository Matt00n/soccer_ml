# Value Models

`scripts/train_value.py` trains action-value models Q(s, a) on the same
decision-anchored states and the same flat action space as the next-action
policy (`NEXT_ACTION_README.md`). That makes advantages
A(s,a) = Q(s,a) − Σ π(a′|s)·Q(s,a′) exact tensor operations.

The flat action space has 1,771 actions: pass (3 heights × 442 zones), carry
(442 zones), shot, dribble and clearance. Values have two components from the
actor's perspective: *score* and *concede* (VAEP style). The components swap
when the other team makes the next decision.

## Configuration axes

| flag | options | meaning |
|---|---|---|
| `--target` | `mc` | Monte Carlo: P(score / concede within the next 10 SPADL actions), with xG-sequence (`--mc-label xg`, default) or goal labels |
| | `sarsa` | TD: y = r + γ·Q_target(s′, a′_observed) |
| | `expected_sarsa` | TD: y = r + γ·Σ π(a′\|s′)·Q_target(s′, a′), with the next-action policy π |
| | `iql` | Implicit Q-learning: y = r + γ·V_target(s′); V is fitted by expectile regression (`--expectile`, default 0.7) to Q_target(s, a) |
| `--head` | `direct` | Q(s, a) for every action |
| | `decomposed` | P(success) · V(success) + (1 − P(success)) · V(failure) (un-xPass style); success means a pass completed, a shot scored, a dribble won, or for carries/clearances the own team acting next |
| `--distribution` | `expectation` | sigmoid probabilities, trained with binary cross-entropy against soft targets |
| | `quantile` | `--num-quantiles` quantiles of the return (QR-DQN quantile-Huber loss); distributional RL |
| `--training` | `warm` | value heads on a copy of a trained next-action encoder (`--policy-checkpoint`); the encoder is fine-tuned at `--encoder-lr-scale` (0.1), or kept fixed with `--freeze-encoder` |
| | `joint` | policy and value heads trained together; from scratch, or initialized from `--policy-checkpoint` |

Common TD and IQL details:

- r is the xG (`--reward goal`: goals) of shots between the action and the
  next decision, including shots without a 360 frame.
- γ = 0.9 per decision (`--gamma`); the bootstrap term is zero at the end of a
  period.
- Target networks use Polyak averaging (`--target-tau` 0.005).

**Not supported, by design** (they raise an error):

- *Quantiles with the decomposed head.* Quantiles of a success/failure mixture
  are not a function of the two branches' quantiles.
- *Quantiles with expected SARSA.* That would need the π-mixture of the
  next-state distributions.
- *Quantiles with IQL.* IQL fits expectiles of an expected Q.

That leaves 20 configurations. For the "upside of risky actions" idea:

- the decomposed head gives the upside V(success) and the risk 1 − P(success);
- the quantile head gives upper quantiles of the return directly.

## Metrics

Computed on the validation decisions. `selection` (used for checkpoint
selection) is the sum of the two soft log-losses.

- `score_*` / `concede_*`: the ex-ante Q(s, a) of the observed action against
  goal labels (AUC, log-loss) and xG-sequence labels (`soft_logloss`).
- `protocol_*`: un-xPass Table 3, i.e. completed open-play passes; the
  decomposed head uses V(success). At the end, the best checkpoint is
  compared with **VAEP, VAEP-xG and xT on the identical passes**.
- `success_auc_<type>` (decomposed head) and `quantile_coverage_<i>`
  (quantile head; it should match the fractions 0.1, 0.3, …).
- `policy_joint_nll` (joint training).

`val_predictions.csv` stores, per validation decision, the value predictions
at the observed action, keyed by event ID.

## Running

A single configuration:

```bash
conda run -n soccer_ml python scripts/train_value.py \
  --policy-checkpoint checkpoints/next_action_v2/best_model.pt \
  --train-dir data/open-data/data --val-dir data/open-data/data \
  --train-file-count 375 --val-file-count 50 --sample-seed 42 \
  --target mc --head decomposed --epochs 4 --checkpoint-dir checkpoints/value/mc-decomposed
```

All 20 configurations (resumable; writes `sweep_summary.md` and `.csv`):

```bash
conda run -n soccer_ml python scripts/value/run_value_sweep.py \
  --policy-checkpoint checkpoints/next_action_v2/best_model.pt \
  --output-dir checkpoints/value_sweep -- \
  --train-dir data/open-data/data --val-dir data/open-data/data \
  --train-file-count 60 --val-file-count 50 --sample-seed 42 --epochs 3
```

Use `--only` to filter (e.g. `--only warm`), `--include-frozen` to add the
frozen-encoder variants, and `--summarize` to rebuild the table. Every sweep
run warm-starts from the checkpoint, so `joint` means fine-tuning policy and
value together. A from-scratch joint run is `train_value.py --training joint`
without `--policy-checkpoint`.
