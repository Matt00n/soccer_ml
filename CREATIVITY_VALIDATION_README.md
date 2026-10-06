# Creativity measures and their validation

Creativity = originality × effectiveness (Runco & Jaeger's standard definition; multiplicative as in
Simonton 2012). Originality comes from the next-action policy π, effectiveness from the action-value
model Q. There is no ground truth for creativity, so validation is a *validity argument*: several
independent kinds of evidence, each fixed in advance and each able to fail.

## Pipeline

```
score_decisions.py   out-of-sample decisions -> per-decision table (originality, value, option sets, calibration, event details)
validate_measures.py tables -> automated validity report (calibration, artefacts, controls, reliability, players, external criteria)
make_review_kit.py   tables -> blinded rating page + separate key
score_review.py      ratings + key -> expert-agreement report
```

All scripts live in `scripts/creativity/`; the logic is in `soccer_ml/creativity/`
(`scoring.py`, `measures.py`, `validation.py`, `review.py`).

**Only out-of-sample decisions are scored.** `score_decisions.py` reads the matches from the policy run's
`summary.json`:
- cross-fitting runs: the full held-out fold;
- group hold-out runs (reference model): all held-out matches;
- ordinary runs: the validation matches.

It refuses to score a match that either the policy or the value model trained on.

```bash
PY=python  # the soccer_ml env
$PY scripts/creativity/score_decisions.py --value-checkpoint checkpoints/crossfit/fold0/value_mc/best_model.pt \
    --output outputs/creativity/crossfit_fold0.csv.gz          # likewise fold 1
$PY scripts/creativity/validate_measures.py \
    --tables outputs/creativity/crossfit_fold0.csv.gz outputs/creativity/crossfit_fold1.csv.gz \
    --compare outputs/creativity/main_val_mc.csv.gz \
    --known-players validation/known_creative_players.csv --external-ratings validation/external_ratings.csv \
    --famous-moments validation/famous_moments.csv --output-dir outputs/creativity/validation
$PY scripts/creativity/make_review_kit.py \
    --tables outputs/creativity/crossfit_fold0.csv.gz outputs/creativity/crossfit_fold1.csv.gz \
    --output-dir outputs/creativity/review_kit1
$PY scripts/creativity/score_review.py --kit-dir outputs/creativity/review_kit1 --ratings ~/Downloads/ratings_*.csv
```

`outputs/creativity/run_scoring.sh` runs the first three steps for the cross-fitted folds and the
main-split models.

### Scope and camera view

- **`--scope intended`** (default) drops passes whose end location is not the intended target:
  - incomplete passes, whose end is where the ball was intercepted or went out;
  - deflected passes.

  Their rarity is inflated (mean 0.66 vs. 0.51 for completed passes), and they made miscommunications
  and deflections look creative. Carries, shots, dribbles and clearances stay. `--scope all` keeps
  every decision.
- **`--view in` / `--view out`** restricts the checks to actions ending inside or outside the 360
  camera's visible area.
- **Comparing runs:** `scripts/creativity/compare_reports.py` puts the `headline.csv` of several
  reports side by side.

### Further analyses

```bash
# Temperature scaling: fit per-head temperatures on validation matches, test on held-out matches.
$PY scripts/creativity/calibrate_policy.py --policy-checkpoint checkpoints/reference/next_action/best_model.pt
# Teams vs. opponents, players across contexts (vs. teammates), change over a season, on the reference yardstick.
$PY scripts/creativity/team_analyses.py --tables outputs/creativity/reference_holdout.csv.gz \
    --contrast outputs/creativity/crossfit_c51_fold0.csv.gz outputs/creativity/crossfit_c51_fold1.csv.gz
```

### Second-order analyses

```bash
# Match context (score state, players on the pitch, goals, substitutions, cards, possessions, results).
$PY scripts/creativity/build_context.py --tables outputs/creativity/crossfit_c51_fold0.csv.gz outputs/creativity/crossfit_c51_fold1.csv.gz
# Every analysis -> outputs/creativity/second_order/: hypotheses.csv with BH q-values, index.md, manifest.json, CSVs per section.
$PY scripts/creativity/run_second_order.py \
    --tables outputs/creativity/crossfit_c51_fold0.csv.gz outputs/creativity/crossfit_c51_fold1.csv.gz \
    --reference outputs/creativity/reference_holdout.csv.gz
```

The sections are:
1. surprise premium;
2. sequences (possessions, chain position, creative flow);
3. team level and concentration vs. success;
4. reactions to goals, comebacks and game management;
5. substitutions, on/off, red cards, tactical shifts and star absence;
6. time (minute profile, fresh legs, knockouts, editions);
7. opponents (creativity allowed, opponent strength, pressure, pitch maps, novelty);
8. player profiles, consistency, enablers and duos.

The primary measure is `good_surprise`, z-scored over intended-scope decisions. The models condition on goal
difference and minute, so game-state effects are relative to what is typical in that state.

Cross-team comparisons must use the reference model: a cross-fitted model learned each single-team
club's style from its other matches. Under cross-fitting, club players sit at the 35th percentile of
their position versus the 53rd for tournament players. Within-team comparisons (player minus teammates
in the same matches) are robust to this.

## Per-decision statistics

They are computed over the flat action space (pass height × zone, carry zone, shot, dribble,
clearance); see `soccer_ml/creativity/scoring.py`.

| Statistic | Meaning |
|---|---|
| `surprise` | −log π(a), with type / zone / height factors |
| `rarity` | π(actions rated more likely than a) + π(a)/2, a mid-PIT in [0, 1]. It does not depend on how many options the situation offers, and it is uniform under a calibrated policy |
| `q`, `v`, `advantage` | Q(s,a), V(s) = E_π Q(s,·), A = Q − V (net = score − concede) |
| `td_advantage`, `mc_advantage` | Realized one-step advantage r + γV(s') − V(s), and the realized return minus V(s) |
| `p_success`, `v_success`, `v_failure`, `upside` | Decomposed head: success probability, branch values, E[(return − V)+] |
| `c51_*` | C51 head: mean, 10%/90% quantiles, P(return > V), upside, upper-tail advantage |
| `<set>_mass`, `in_<set>`, `<set>_rarity`, `<set>_surprise` | Rarity of a *within* an option set (π renormalized on the set) |

The option sets:
- `good`: positive advantage;
- `tail`: the success branch beats V(s);
- `c51_tail`: a heavier right tail than the state's return distribution;
- `pass`: passes only.

## Candidate measures

All measures are "higher = more creative"; see `soccer_ml/creativity/measures.py`.

| Family | Measure | Definition |
|---|---|---|
| Baseline | `rarity` | Originality only |
| Baseline | `advantage` | Value only |
| Baseline | `difficulty` | 1 − P(success) of completed actions |
| Product | `surprise_x_adv` | −log π(a) × max(A, 0) |
| Product | `rarity_x_adv` | rarity × max(A, 0) |
| Product | `rarity_x_adv_completed` | The same for completed actions only (ex post, un-xPass-like) |
| Product | `rarity_x_td` | rarity × max(realized one-step advantage, 0). Robust to errors in Q for rare actions, but includes execution luck |
| Product | `rarity_x_upside` | rarity × E[(return − V)+]: a positive tail counts even when the mean advantage is negative |
| Set | `good_rarity`, `good_surprise` | Rarity among positive-advantage actions (0 if a is not among them) |
| Set | `tail_rarity` | Rarity among actions whose success branch beats V(s) |
| Set | `c51_tail_rarity` | Rarity among actions with a heavier right tail than the state (C51 models only) |
| Set | `unxpass` | Passes: rarity among passes if A > 0 (un-xPass style) |

## Automated checks (`validate_measures.py`)

| Check | Question | Fails if |
|---|---|---|
| Policy calibration by probability band, by gender | Is surprise calibrated down to the rare tail? | Observed/expected far from 1 in rare bands |
| Rarity uniformity by competition, gender, context, action type | Is originality comparable across contexts? | Mean ≠ 0.5, large KS distance in some groups |
| Value calibration by rarity decile | Is Q inflated for rare actions? | Bias grows with rarity |
| Winner's curse | Do top decisions have real value, or overestimated value? | Predicted ≫ realized advantage |
| Artefact rates | Is the top driven by out-of-view targets, restarts, miscommunications, deflections, failed actions? | Large lifts vs. all decisions |
| StatsBomb annotations | Positive controls (through balls, dummies, nutmegs, flair shots) and negative controls (miscommunications, deflections, goal kicks, throw-ins) | Positive AUC ≤ 0.5 or negative AUC ≥ 0.5 |
| Line-breaking | Opponents bypassed (model-free) | No positive association |
| Discriminant | Just position, volume, risk or team? | Strong correlations; large team/position η² |
| Split-half, cross-competition | Is player creativity a stable trait? | Low Spearman–Brown, no cross-season stability |
| Model agreement | Same decisions scored by an independently trained model | Low action- or player-level agreement |
| Player rankings | With 90% match-bootstrap rank intervals | Implausible or unstable top players |
| Known groups, external ratings, famous moments | From the templates in `validation/` | See the hypotheses below |

## Pre-registration draft (directional hypotheses)

Fill in `validation/*.csv` from external sources **before** looking at any ranking, and commit them; the
report records each file's hash and timestamp. Measure variants are chosen on a development set of
expert ratings, and confirmatory tests use separate clips.

- **H1 calibration:** observed/expected within about [0.8, 1.25] in all bands with π ≥ 10⁻⁴, and mean rarity
  0.5 ± 0.02 in every competition and gender. If not: recalibrate the policy (temperature) before any
  creativity analysis.
- **H2 controls:** creativity measures give AUC > 0.5 for through balls, dummies, nutmegs and flair
  shots, and AUC < 0.5 for miscommunications, deflections, goal kicks and throw-ins.
- **H3 components (expert ratings):** flagged clips are rated more creative than value-matched
  low-originality controls and than originality-matched low-value controls.
- **H4 beyond components:** the best creativity measure correlates more with expert creativity ratings
  than `rarity`, `advantage` and `difficulty` alone.
- **H5 known groups:** known creative players lie above the median within their position group
  (AUC > 0.5).
- **H6 convergent / discriminant:** correlation with vision > with dribbling or ball control; about 0
  with strength or jumping (within position).
- **H7 reliability:** split-half Spearman–Brown ≥ 0.5 for players with ≥ 200 decisions.

## Quick expert verification (first pass, about 2 hours)

1. Open `outputs/creativity/review_kit1/review.html` in a browser. Do **not** open `key_DO_NOT_OPEN.csv`.
2. For every clip, rate the 2D scene first: creative, original and effective (1–5), and flag a data
   issue if the scene looks wrong. Rate the decision, not the execution or the outcome.
3. Optionally click *Reveal* (player, match, time, video search) and add a rating after watching the
   video. The page records whether you rated before revealing.
4. Download the ratings CSV and run `score_review.py`.

About 62 clips:

| Clips | Selection |
|---|---|
| 3 per measure | Random draws from its top 1% |
| 10 | Value-matched low-originality controls |
| 8 | Originality-matched low-value controls |
| 8 | Random decisions |

Set pieces are excluded by default (`--include-restarts` adds them). With one rater and about 60 clips
this is a first verification. Read effect sizes and intervals: the rigorous study needs several
blinded raters, inter-rater reliability, and separate development and confirmation clips.

## Caveats

- **Team style is absorbed.** Under cross-fitting, a club's other matches are in training (the club
  data are single-team releases), so surprise is measured against "how this team plays". Player and
  team comparisons across contexts (Messi, Leverkusen) need the reference model trained without
  these teams (`--holdout-*` in the training scripts).
- **Q-based measures are ex ante.** A failed pass can be creative under them; `rarity_x_adv_completed`
  is the ex-post variant.
- **The 360 camera view is limited.** Targets outside it can look original only because the receiver
  is not visible; check the `end_out_of_view` lift and the rater flags.
