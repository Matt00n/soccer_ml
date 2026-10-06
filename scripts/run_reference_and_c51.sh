#!/usr/bin/env bash
# Unattended runs for the preliminary creativity analyses (about 12-14 h):
#   1. Reference model: next-action + MC value model trained without the single-team club seasons
#      (Barcelona, PSG, Leverkusen, Inter Miami) and Argentina's World Cup matches, then the 140 held-out
#      matches are scored against this common yardstick (team and change analyses, Messi across contexts).
#   2. Cross-fitted MC + C51 value models (warm-started from the fold policies), so the C51 positive-tail
#      measure can be evaluated on every match instead of 49.
#
# GPU lane: reference policy (10 epochs) -> reference MC value (3 epochs) -> score held-out matches -> report
# CPU lane: fold 0 MC + C51 value (3 epochs) -> score fold 0 -> fold 1 MC + C51 value -> score fold 1
#
# Usage (repo root, soccer_ml env activated):
#   nohup scripts/run_reference_and_c51.sh > checkpoints/reference_c51.log 2>&1 &
set -u
PYTHON=${PYTHON:-python}
DATA=data/open-data/data
SEED=42
HOLDOUT=(--holdout-competitions "La Liga" "Ligue 1" "1. Bundesliga" "Major League Soccer" --holdout-teams Argentina)
REF=checkpoints/reference
CF=checkpoints/crossfit
OUT=outputs/creativity
mkdir -p "$REF" "$OUT"

stamp() { echo "[$(date '+%a %H:%M')] $*"; }

gpu_lane() {
  stamp "reference: next-action model"
  "$PYTHON" scripts/train_next_action.py --train-dir "$DATA" "${HOLDOUT[@]}" --sample-seed $SEED \
    --game-state --velocities --edge-distance --epochs 10 \
    --checkpoint-dir "$REF/next_action" > "$REF/next_action.log" 2>&1 \
  && stamp "reference: MC value model" \
  && "$PYTHON" scripts/train_value.py --policy-checkpoint "$REF/next_action/best_model.pt" \
    --train-dir "$DATA" "${HOLDOUT[@]}" --sample-seed $SEED --target mc --head decomposed --epochs 3 \
    --checkpoint-dir "$REF/value_mc" > "$REF/value_mc.log" 2>&1 \
  && stamp "reference: scoring the held-out matches" \
  && "$PYTHON" scripts/creativity/score_decisions.py --value-checkpoint "$REF/value_mc/best_model.pt" \
    --output "$OUT/reference_holdout.csv.gz" > "$OUT/reference_holdout.log" 2>&1 \
  && stamp "reference: validation report" \
  && "$PYTHON" scripts/creativity/validate_measures.py --tables "$OUT/reference_holdout.csv.gz" \
    --compare "$OUT/crossfit_fold0.csv.gz" "$OUT/crossfit_fold1.csv.gz" \
    --known-players validation/known_creative_players.csv \
    --output-dir "$OUT/validation_reference" > "$OUT/validation_reference.log" 2>&1 \
  || stamp "reference lane FAILED, see $REF/*.log and $OUT/reference_holdout.log"
  stamp "GPU lane done"
}

cpu_lane() {
  export OMP_NUM_THREADS=5
  for fold in 0 1; do
    stamp "fold $fold: MC + C51 value model"
    "$PYTHON" scripts/train_value.py --policy-checkpoint "$CF/fold$fold/next_action/best_model.pt" \
      --train-dir "$DATA" --num-folds 2 --fold "$fold" --sample-seed $SEED \
      --target mc --head decomposed --distribution categorical --epochs 3 --device cpu \
      --checkpoint-dir "$CF/fold$fold/value_mc_c51" > "$CF/fold${fold}_value_mc_c51.log" 2>&1 \
    && stamp "fold $fold: scoring with C51" \
    && "$PYTHON" scripts/creativity/score_decisions.py --device cpu \
      --value-checkpoint "$CF/fold$fold/value_mc_c51/best_model.pt" \
      --output "$OUT/crossfit_c51_fold$fold.csv.gz" > "$OUT/crossfit_c51_fold$fold.log" 2>&1 \
    || stamp "fold $fold C51 FAILED, see $CF/fold${fold}_value_mc_c51.log"
  done
  stamp "CPU lane done"
}

stamp "python: $(command -v "$PYTHON") ($("$PYTHON" -c 'import sys, torch; print(sys.version.split()[0], "torch", torch.__version__)'))"
gpu_lane &
cpu_lane &
wait
stamp "all runs finished"
