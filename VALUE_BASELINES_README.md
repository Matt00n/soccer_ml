# Value Baselines

These are established pass-value models, evaluated with the protocol of
un-xPass (Robberechts et al., KDD 2023, Table 3). For every completed
open-play pass in the validation matches, each model predicts whether the
passing team scores within the next 10 actions. Metrics are AUC, Brier and
log-loss, plus log-loss against the less noisy xG-sequence label. The
validation matches are the same 48 used by the 360 benchmarks.

| model | what it values | context | trained on |
|---|---|---|---|
| xT, Karun Singh's published 12×8 grid | the zone the ball is in | ball location | his data |
| xT, ours (16×12, 26×17) | same; Markov chain over zones, StatsBomb xG for shots | ball location | all ~4,190 non-validation event matches |
| VAEP | P(score) and P(concede) within the next 10 actions | the last 3 actions (SPADL features) | 600 random non-validation matches |
| VAEP-xG | same, with xG-sequence labels 1 − Π(1 − xG) | same | same |

StatsBomb's own On-Ball Value (OBV) is not in the open data (0 of 4,235 event
files), so it can't be benchmarked here.

## Run

```bash
# 1. xT (main env, about 1 minute)
conda run -n soccer_ml python scripts/value/fit_expected_threat.py
curl -sSL -o outputs/value_baselines/karun_xt_12x8.json https://karun.in/blog/data/open_xt_12x8_v1.json

# 2. VAEP / VAEP-xG (separate env, see below; ~15-30 min, ~3 GB RAM)
~/miniforge3/envs/soccer_vaep/bin/python scripts/value/vaep_baseline.py

# 3. Comparison table (main env)
conda run -n soccer_ml python scripts/value/evaluate_value_baselines.py
```

Outputs go to `outputs/value_baselines/` (gitignored):

- `xt_*.json`: the fitted xT grids;
- `vaep_val_predictions.csv`: every SPADL action of the validation matches,
  with labels and both VAEP variants;
- `value_labels_360.csv`: goal and xG-sequence labels for every action of all
  360 matches, ready as targets for our own value model;
- `vaep_metrics.json`, `value_baseline_metrics.json`.

## The `soccer_vaep` environment

`socceraction` 1.5.3 needs Python < 3.13, but the main env runs 3.13, so it
lives in its own env:

```bash
conda create -n soccer_vaep python=3.12 -y
conda install -n soccer_vaep -c conda-forge llvm-openmp -y   # xgboost needs OpenMP on macOS
~/miniforge3/envs/soccer_vaep/bin/python -m pip install socceraction xgboost "multimethod<2"
```

`multimethod<2` is needed because socceraction pins pandera 0.17, which
breaks with multimethod 2. Call the env's own `python -m pip`: on this
machine a plain `pip` inside `conda run` resolves to the system Python 3.9.
