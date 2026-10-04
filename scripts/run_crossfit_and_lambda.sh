#!/usr/bin/env bash
# Unattended runs (about 12-13 h):
#   1. 2-fold cross-fitting: a v3 next-action model and an MC value model per fold, so
#      every match gets out-of-sample predictions for the creativity analysis.
#   2. Stabilized TD value model on the main split: λ-return targets (λ = 0.8),
#      refreshed once per epoch instead of bootstrapping every batch.
#
# GPU lane: fold 0 policy -> fold 0 value -> λ = 0.8 value (main split, v3 encoder)
# CPU lane: fold 1 policy -> fold 1 value
#
# Usage (repo root, soccer_ml env activated):
#   nohup scripts/run_crossfit_and_lambda.sh > checkpoints/overnight.log 2>&1 &
set -u
PYTHON=${PYTHON:-python}
DATA=data/open-data/data
OUT=checkpoints/crossfit
SEED=42
POLICY_FLAGS="--game-state --velocities --edge-distance --epochs 16"
mkdir -p "$OUT" checkpoints/value

stamp() { echo "[$(date '+%a %H:%M')] $*"; }

policy() {  # fold, extra flags...
  local fold=$1; shift
  stamp "fold $fold: next-action model"
  "$PYTHON" scripts/train_next_action.py --train-dir "$DATA" --num-folds 2 --fold "$fold" --sample-seed $SEED \
    $POLICY_FLAGS "$@" --checkpoint-dir "$OUT/fold$fold/next_action" > "$OUT/fold${fold}_next_action.log" 2>&1
}

value() {  # fold, extra flags...
  local fold=$1; shift
  stamp "fold $fold: MC value model"
  "$PYTHON" scripts/train_value.py --policy-checkpoint "$OUT/fold$fold/next_action/best_model.pt" \
    --train-dir "$DATA" --num-folds 2 --fold "$fold" --sample-seed $SEED \
    --target mc --head decomposed --epochs 4 "$@" \
    --checkpoint-dir "$OUT/fold$fold/value_mc" > "$OUT/fold${fold}_value_mc.log" 2>&1
}

gpu_lane() {
  policy 0 && value 0 || stamp "fold 0 FAILED, see $OUT/fold0_*.log"
  stamp "main split: λ = 0.8 value model"
  "$PYTHON" scripts/train_value.py --policy-checkpoint checkpoints/next_action_v3/best_model.pt \
    --train-dir "$DATA" --val-dir "$DATA" --train-file-count 375 --val-file-count 50 --sample-seed $SEED \
    --target lambda_return --lam 0.8 --head decomposed --epochs 4 \
    --checkpoint-dir checkpoints/value/lambda0.8-decomposed-v3 > checkpoints/value_lambda0.8_v3.log 2>&1 \
    || stamp "λ run FAILED, see checkpoints/value_lambda0.8_v3.log"
  stamp "GPU lane done"
}

cpu_lane() {
  export OMP_NUM_THREADS=5
  policy 1 --device cpu && value 1 --device cpu || stamp "fold 1 FAILED, see $OUT/fold1_*.log"
  stamp "CPU lane done"
}

stamp "python: $(command -v "$PYTHON") ($("$PYTHON" -c 'import sys, torch; print(sys.version.split()[0], "torch", torch.__version__)'))"
gpu_lane &
cpu_lane &
wait
stamp "all runs finished"
