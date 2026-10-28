#!/usr/bin/env bash
# Rescore all out-of-sample tables with the opportunity and pressure statistics (step 2, C1).
set -u
PY=/Users/matthiaslehmann/miniforge3/envs/soccer_ml/bin/python
OUT=outputs/creativity/step2/scored
stamp() { echo "[$(date '+%a %H:%M')] $*"; }
for spec in "checkpoints/crossfit/fold0/value_mc_c51/best_model.pt crossfit_c51_fold0" \
            "checkpoints/crossfit/fold1/value_mc_c51/best_model.pt crossfit_c51_fold1" \
            "checkpoints/reference/value_mc/best_model.pt reference_holdout"; do
  set -- $spec
  stamp "scoring $2"
  "$PY" scripts/creativity/score_decisions.py --value-checkpoint "$1" --output "$OUT/$2.csv.gz" > "$OUT/$2.log" 2>&1 || stamp "FAILED $2"
done
stamp "done"
