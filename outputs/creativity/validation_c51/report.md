# Creativity measure validation

Tables: `outputs/creativity/crossfit_c51_fold0.csv.gz` (sha256 6dea815abdb2, modified Tue Oct  6 00:17:20 2026), `outputs/creativity/crossfit_c51_fold1.csv.gz` (sha256 ea0c62543802, modified Tue Oct  6 02:54:53 2026)

536223 decisions in 418 matches (469733 in scope: scope 'intended', view 'all'); player checks use players with at least 200 in-scope decisions; 'top' = top 1% per measure.

## Measures

| measure | family | definition | in_scope | available |
|---|---|---|---|---|
| rarity | baseline | originality only: rarity (mid-PIT) of the chosen action | 469733 | True |
| advantage | baseline | value only: Q(s,a) - V(s) | 469733 | True |
| difficulty | baseline | 1 - P(success) of completed actions, 0 if failed | 469564 | True |
| surprise_x_adv | product | -log π(a) x max(A, 0) | 469733 | True |
| rarity_x_adv | product | rarity x max(A, 0) | 469733 | True |
| rarity_x_adv_completed | product | rarity x max(A, 0), completed actions only | 469564 | True |
| rarity_x_td | product | rarity x max(r + γV(s') - V(s), 0) (realized) | 469733 | True |
| rarity_x_upside | product | rarity x E[(return - V)+] | 469733 | True |
| good_rarity | set | rarity among positive-advantage actions | 469733 | True |
| good_surprise | set | -log π(a | positive-advantage actions) | 469733 | True |
| tail_rarity | set | rarity among actions whose success branch beats V(s) | 469733 | True |
| c51_tail_rarity | set | rarity among actions with a heavier right tail than the state (C51) | 469733 | True |
| unxpass | set | passes: rarity among passes if A > 0 | 306312 | True |


## Headline metrics

One row per measure, collected from the checks below (compare runs with scripts/creativity/compare_reports.py).

| measure | known_auc | known_auc_low | known_vs_advantage | known_vs_rarity | auc_through_ball | auc_flair_shot | auc_goal_kick | auc_throw_in | line_breaking_rho | split_half | cross_competition_rho | model_agreement_players | eta2_position | eta2_team | top_out_of_view | realized_over_predicted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rarity | 0.588 | 0.499 | 0.033 |  | 0.829 | 0.629 | 0.417 | 0.438 | 0.257 | 0.575 | 0.359 | 1.000 | 0.177 | 0.212 | 0.530 |  |
| advantage | 0.554 | 0.470 |  | -0.033 | 0.804 | 0.662 | 0.559 | 0.515 | 0.208 | 0.544 | 0.400 | 0.972 | 0.282 | 0.101 | 0.239 | 0.858 |
| difficulty | 0.478 | 0.389 | -0.076 | -0.110 | 0.936 | 0.529 | 0.444 | 0.695 | 0.525 | 0.892 | 0.636 | 0.999 | 0.462 | 0.322 | 0.355 |  |
| surprise_x_adv | 0.583 | 0.497 | 0.029 | -0.005 | 0.837 | 0.605 | 0.481 | 0.497 | 0.291 | 0.882 | 0.758 | 0.991 | 0.702 | 0.082 | 0.393 | 0.815 |
| rarity_x_adv | 0.582 | 0.492 | 0.028 | -0.006 | 0.837 | 0.687 | 0.483 | 0.496 | 0.291 | 0.889 | 0.760 | 0.992 | 0.708 | 0.082 | 0.362 | 0.829 |
| rarity_x_adv_completed | 0.589 | 0.499 | 0.034 | 0.001 | 0.837 | 0.528 | 0.483 | 0.496 | 0.291 | 0.864 | 0.734 | 0.990 | 0.682 | 0.080 | 0.366 | 0.892 |
| rarity_x_td | 0.599 | 0.512 | 0.044 | 0.011 | 0.913 | 0.470 | 0.564 | 0.441 | 0.345 | 0.854 | 0.706 | 0.998 | 0.649 | 0.097 | 0.291 | 6.593 |
| rarity_x_upside | 0.564 | 0.473 | 0.010 | -0.024 | 0.924 | 0.691 | 0.518 | 0.513 | 0.353 | 0.904 | 0.776 | 0.994 | 0.720 | 0.081 | 0.319 | 1.034 |
| good_rarity | 0.645 | 0.562 | 0.091 | 0.058 | 0.824 | 0.623 | 0.485 | 0.494 | 0.277 | 0.638 | 0.392 | 0.931 | 0.286 | 0.146 | 0.467 | 1.415 |
| good_surprise | 0.650 | 0.566 | 0.097 | 0.063 | 0.817 | 0.495 | 0.490 | 0.495 | 0.279 | 0.581 | 0.371 | 0.939 | 0.091 | 0.206 | 0.533 | 1.451 |
| tail_rarity | 0.598 | 0.512 | 0.045 | 0.011 | 0.894 | 0.616 | 0.516 | 0.497 | 0.340 | 0.719 | 0.364 | 0.961 | 0.253 | 0.225 | 0.460 | 2.219 |
| c51_tail_rarity | 0.634 | 0.539 | 0.079 | 0.046 | 0.803 | 0.591 | 0.452 | 0.477 | 0.288 | 0.760 | 0.620 |  | 0.496 | 0.082 | 0.440 | 1.897 |
| unxpass | 0.645 | 0.557 | 0.090 | 0.057 | 0.812 |  | 0.490 | 0.497 | 0.276 | 0.556 | 0.238 | 0.933 | 0.047 | 0.243 | 0.511 | 1.852 |


## Policy calibration by probability band

Observed / expected choices per log10 π band. 1 = calibrated; the rare bands drive surprise-based measures.

| group | log10_p_band | expected | observed | ratio | ratio_low | ratio_high |
|---|---|---|---|---|---|---|
| all | (-inf, -6] | 31.779 | 27 | 0.850 | 0.560 | 1.236 |
| all | (-6, -5] | 1450.979 | 1038 | 0.715 | 0.673 | 0.760 |
| all | (-5, -4] | 10858.919 | 10796 | 0.994 | 0.976 | 1.013 |
| all | (-4, -3] | 44987.726 | 50396 | 1.120 | 1.110 | 1.130 |
| all | (-3, -2.5] | 57136.045 | 60890 | 1.066 | 1.057 | 1.074 |
| all | (-2.5, -2] | 106061.248 | 106174 | 1.001 | 0.995 | 1.007 |
| all | (-2, -1.5] | 154107.618 | 147879 | 0.960 | 0.955 | 0.964 |
| all | (-1.5, -1] | 111862.617 | 110343 | 0.986 | 0.981 | 0.992 |
| all | (-1, -0.5] | 24037.482 | 23191 | 0.965 | 0.952 | 0.977 |
| all | (-0.5, 0] | 25688.587 | 25489 | 0.992 | 0.980 | 1.004 |


## Policy calibration by gender

| group | log10_p_band | expected | observed | ratio | ratio_low | ratio_high |
|---|---|---|---|---|---|---|
| men | (-inf, -6] | 23.906 | 19 | 0.795 | 0.479 | 1.241 |
| men | (-6, -5] | 1066.866 | 805 | 0.755 | 0.703 | 0.809 |
| men | (-5, -4] | 7780.753 | 7907 | 1.016 | 0.994 | 1.039 |
| men | (-4, -3] | 31834.645 | 35627 | 1.119 | 1.108 | 1.131 |
| men | (-3, -2.5] | 40181.144 | 42748 | 1.064 | 1.054 | 1.074 |
| men | (-2.5, -2] | 75771.697 | 75688 | 0.999 | 0.992 | 1.006 |
| men | (-2, -1.5] | 113074.670 | 108646 | 0.961 | 0.955 | 0.967 |
| men | (-1.5, -1] | 84674.723 | 83647 | 0.988 | 0.981 | 0.995 |
| men | (-1, -0.5] | 18024.076 | 17492 | 0.970 | 0.956 | 0.985 |
| men | (-0.5, 0] | 17074.519 | 16928 | 0.991 | 0.977 | 1.006 |
| women | (-inf, -6] | 7.873 | 8 | 1.016 | 0.439 | 2.002 |
| women | (-6, -5] | 384.112 | 233 | 0.607 | 0.531 | 0.690 |
| women | (-5, -4] | 3078.165 | 2889 | 0.939 | 0.905 | 0.973 |
| women | (-4, -3] | 13153.081 | 14769 | 1.123 | 1.105 | 1.141 |
| women | (-3, -2.5] | 16954.901 | 18142 | 1.070 | 1.055 | 1.086 |
| women | (-2.5, -2] | 30289.551 | 30486 | 1.006 | 0.995 | 1.018 |
| women | (-2, -1.5] | 41032.948 | 39233 | 0.956 | 0.947 | 0.966 |
| women | (-1.5, -1] | 27187.894 | 26696 | 0.982 | 0.970 | 0.994 |
| women | (-1, -0.5] | 6013.406 | 5699 | 0.948 | 0.923 | 0.973 |
| women | (-0.5, 0] | 8614.068 | 8561 | 0.994 | 0.973 | 1.015 |


## Rarity uniformity (PIT) by context

Under calibration mean rarity = 0.5 and KS distance ~ 0 in every group (measurement invariance).

| group | n | mean_rarity | ks_distance | share_top_decile | share_bottom_decile | max_decile_deviation | by |
|---|---|---|---|---|---|---|---|
| all | 536223 | 0.511 | 0.015 | 0.107 | 0.091 | 0.009 | None |
| men | 389507 | 0.510 | 0.015 | 0.107 | 0.092 | 0.008 | gender |
| women | 146716 | 0.512 | 0.017 | 0.106 | 0.090 | 0.010 | gender |
| 1. Bundesliga 2023/2024 | 42637 | 0.505 | 0.009 | 0.101 | 0.092 | 0.008 | competition_season |
| FIFA World Cup 2022 | 82307 | 0.518 | 0.027 | 0.113 | 0.089 | 0.013 | competition_season |
| La Liga 2020/2021 | 52355 | 0.507 | 0.010 | 0.106 | 0.093 | 0.007 | competition_season |
| Ligue 1 2021/2022 | 34061 | 0.507 | 0.012 | 0.107 | 0.094 | 0.007 | competition_season |
| Ligue 1 2022/2023 | 42343 | 0.501 | 0.006 | 0.097 | 0.095 | 0.005 | competition_season |
| Major League Soccer 2023 | 3678 | 0.524 | 0.043 | 0.116 | 0.088 | 0.016 | competition_season |
| UEFA Euro 2020 | 66329 | 0.517 | 0.027 | 0.117 | 0.090 | 0.017 | competition_season |
| UEFA Euro 2024 | 65796 | 0.506 | 0.012 | 0.103 | 0.095 | 0.005 | competition_season |
| UEFA Women's Euro 2022 | 36341 | 0.508 | 0.016 | 0.100 | 0.090 | 0.010 | competition_season |
| UEFA Women's Euro 2025 | 34964 | 0.505 | 0.009 | 0.101 | 0.093 | 0.007 | competition_season |
| Women's World Cup 2023 | 75411 | 0.518 | 0.026 | 0.112 | 0.088 | 0.012 | competition_season |
| after_carry | 114431 | 0.499 | 0.008 | 0.104 | 0.097 | 0.004 | decision_context |
| other | 73748 | 0.514 | 0.020 | 0.104 | 0.086 | 0.014 | decision_context |
| spell_start | 348044 | 0.514 | 0.021 | 0.109 | 0.091 | 0.009 | decision_context |
| carry | 126803 | 0.474 | 0.050 | 0.065 | 0.095 | 0.035 | action_type |
| clearance | 15061 | 0.441 | 0.090 | 0.063 | 0.133 | 0.037 | action_type |
| dribble | 11194 | 0.313 | 0.265 | 0.018 | 0.282 | 0.182 | action_type |
| pass | 372802 | 0.538 | 0.054 | 0.128 | 0.079 | 0.028 | action_type |
| shot | 10363 | 0.312 | 0.278 | 0.012 | 0.227 | 0.127 | action_type |


## Value calibration by rarity decile

Is Q inflated for rare actions? bias = mean(Q - realized return).

| rarity | n | pred_q_score | label_xg_scores | score_ratio | pred_q_net | realized_return_net | bias | bias_ci |
|---|---|---|---|---|---|---|---|---|
| (-0.001, 0.1] | 46733 | 0.009 | 0.009 | 0.979 | 0.008 | 0.010 | -0.002 | 0.000 |
| (0.1, 0.2] | 49830 | 0.010 | 0.011 | 0.937 | 0.009 | 0.011 | -0.002 | 0.000 |
| (0.2, 0.3] | 52343 | 0.011 | 0.012 | 0.931 | 0.010 | 0.012 | -0.002 | 0.000 |
| (0.3, 0.4] | 54003 | 0.013 | 0.014 | 0.943 | 0.012 | 0.014 | -0.002 | 0.000 |
| (0.4, 0.5] | 67558 | 0.019 | 0.019 | 0.969 | 0.014 | 0.016 | -0.001 | 0.000 |
| (0.5, 0.6] | 50301 | 0.010 | 0.010 | 1.047 | 0.009 | 0.009 | -0.001 | 0.000 |
| (0.6, 0.7] | 51256 | 0.011 | 0.011 | 1.012 | 0.009 | 0.010 | -0.001 | 0.000 |
| (0.7, 0.8] | 53256 | 0.012 | 0.012 | 0.998 | 0.010 | 0.011 | -0.002 | 0.000 |
| (0.8, 0.9] | 54653 | 0.013 | 0.012 | 1.042 | 0.010 | 0.011 | -0.001 | 0.000 |
| (0.9, 1.0] | 56290 | 0.014 | 0.013 | 1.108 | 0.010 | 0.011 | -0.001 | 0.000 |


## Predicted vs. realized advantage of each measure's top decisions

Top 1% per measure.

| measure | n | pred_advantage | realized_td_advantage | realized_mc_advantage | realized_mc_ci | pred_q | realized_return | pred_q_score | label_xg_scores |
|---|---|---|---|---|---|---|---|---|---|
| rarity | 4698 | -0.002 | 0.002 | 0.005 | 0.002 | 0.008 | 0.015 | 0.015 | 0.017 |
| advantage | 4698 | 0.061 | 0.041 | 0.053 | 0.004 | 0.123 | 0.115 | 0.127 | 0.124 |
| difficulty | 4696 | -0.000 | 0.027 | 0.028 | 0.003 | 0.023 | 0.051 | 0.027 | 0.057 |
| surprise_x_adv | 4698 | 0.056 | 0.036 | 0.045 | 0.004 | 0.083 | 0.073 | 0.087 | 0.081 |
| rarity_x_adv | 4698 | 0.056 | 0.037 | 0.046 | 0.004 | 0.092 | 0.082 | 0.095 | 0.091 |
| rarity_x_adv_completed | 4696 | 0.054 | 0.039 | 0.048 | 0.004 | 0.087 | 0.082 | 0.091 | 0.091 |
| rarity_x_td | 4698 | 0.023 | 0.141 | 0.153 | 0.005 | 0.096 | 0.225 | 0.099 | 0.238 |
| rarity_x_upside | 4698 | 0.053 | 0.042 | 0.055 | 0.004 | 0.105 | 0.107 | 0.109 | 0.116 |
| good_rarity | 4698 | 0.013 | 0.014 | 0.018 | 0.003 | 0.025 | 0.030 | 0.029 | 0.034 |
| good_surprise | 4698 | 0.009 | 0.010 | 0.013 | 0.002 | 0.014 | 0.018 | 0.019 | 0.020 |
| tail_rarity | 4698 | 0.007 | 0.012 | 0.016 | 0.002 | 0.019 | 0.028 | 0.025 | 0.032 |
| c51_tail_rarity | 4698 | 0.012 | 0.020 | 0.022 | 0.003 | 0.023 | 0.034 | 0.027 | 0.038 |
| unxpass | 3064 | 0.008 | 0.012 | 0.015 | 0.002 | 0.013 | 0.020 | 0.017 | 0.022 |


## Artefact rates among top decisions

Shares among each measure's top decisions vs. all decisions. High lifts flag measures driven by data artefacts.

| measure | n | end_out_of_view | restart | miscommunication | deflected | failed | clearance | goalkeeper |
|---|---|---|---|---|---|---|---|---|
| all decisions | 536223 | 0.171 | 0.034 | 0.000 | 0.001 | 0.181 | 0.028 | 0.038 |
| rarity | 4698 | 0.530 | 0.015 | 0.000 | 0.000 | 0.012 | 0.000 | 0.027 |
| advantage | 4698 | 0.239 | 0.013 | 0.000 | 0.000 | 0.117 | 0.000 | 0.002 |
| difficulty | 4696 | 0.355 | 0.056 | 0.000 | 0.008 | 0.000 | 0.081 | 0.080 |
| surprise_x_adv | 4698 | 0.393 | 0.011 | 0.000 | 0.000 | 0.048 | 0.000 | 0.004 |
| rarity_x_adv | 4698 | 0.362 | 0.014 | 0.000 | 0.000 | 0.053 | 0.000 | 0.003 |
| rarity_x_adv_completed | 4696 | 0.366 | 0.016 | 0.000 | 0.000 | 0.000 | 0.000 | 0.003 |
| rarity_x_td | 4698 | 0.291 | 0.029 | 0.000 | 0.002 | 0.145 | 0.000 | 0.003 |
| rarity_x_upside | 4698 | 0.319 | 0.014 | 0.000 | 0.000 | 0.066 | 0.000 | 0.002 |
| good_rarity | 4698 | 0.467 | 0.019 | 0.000 | 0.000 | 0.024 | 0.000 | 0.020 |
| good_surprise | 4698 | 0.533 | 0.016 | 0.000 | 0.000 | 0.021 | 0.000 | 0.034 |
| tail_rarity | 4698 | 0.460 | 0.020 | 0.000 | 0.000 | 0.021 | 0.000 | 0.022 |
| c51_tail_rarity | 4698 | 0.440 | 0.025 | 0.000 | 0.000 | 0.024 | 0.000 | 0.016 |
| unxpass | 3064 | 0.511 | 0.023 | 0.000 | 0.000 | 0.000 | 0.000 | 0.028 |


## StatsBomb annotations as controls (AUC within action type)

Positive (+) controls should exceed 0.5, negative (-) fall below; '?' exploratory. through_ball: through balls (StatsBomb technique); no_touch: dummies: body part 'No Touch'; nutmeg: nutmegs; flair_shot: backheel, lob or overhead-kick shots; cut_back: cut-backs; switch: switches of play; first_time: first-time passes; miscommunication: miscommunication passes; deflected: deflected passes (end location not intended); goal_kick: goal kicks; throw_in: throw-ins

| control | expected | scope | n_annotated | advantage | c51_tail_rarity | difficulty | good_rarity | good_surprise | rarity | rarity_x_adv | rarity_x_adv_completed | rarity_x_td | rarity_x_upside | surprise_x_adv | tail_rarity | unxpass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cut_back | ? | pass | 362 | 0.530 | 0.596 | 0.932 | 0.603 | 0.579 | 0.568 | 0.641 | 0.641 | 0.746 | 0.816 | 0.645 | 0.750 | 0.581 |
| flair_shot | + | shot | 176 | 0.662 | 0.591 | 0.529 | 0.623 | 0.495 | 0.629 | 0.687 | 0.528 | 0.470 | 0.691 | 0.605 | 0.616 |  |
| goal_kick | - | pass | 1059 | 0.559 | 0.452 | 0.444 | 0.485 | 0.490 | 0.417 | 0.483 | 0.483 | 0.564 | 0.518 | 0.481 | 0.516 | 0.490 |
| no_touch | + | pass | 315 | 0.462 | 0.505 | 0.661 | 0.511 | 0.513 | 0.510 | 0.524 | 0.524 | 0.536 | 0.555 | 0.527 | 0.538 | 0.508 |
| nutmeg | + | dribble | 823 | 0.485 | 0.487 | 0.486 | 0.488 | 0.487 | 0.481 | 0.485 | 0.477 | 0.482 | 0.485 | 0.482 | 0.493 |  |
| switch | ? | pass | 5677 | 0.577 | 0.614 | 0.728 | 0.639 | 0.638 | 0.865 | 0.617 | 0.617 | 0.666 | 0.657 | 0.615 | 0.678 | 0.640 |
| through_ball | + | pass | 702 | 0.804 | 0.803 | 0.936 | 0.824 | 0.817 | 0.829 | 0.837 | 0.837 | 0.913 | 0.924 | 0.837 | 0.894 | 0.812 |
| throw_in | - | pass | 7018 | 0.515 | 0.477 | 0.695 | 0.494 | 0.495 | 0.438 | 0.496 | 0.496 | 0.441 | 0.513 | 0.497 | 0.497 | 0.497 |


## Line-breaking (opponents bypassed) vs. measures

| measure | spearman_forward_passes | mean_bypassed_top | mean_bypassed_all_forward |
|---|---|---|---|
| rarity | 0.257 | 4.297 | 2.765 |
| advantage | 0.208 | 4.420 | 2.765 |
| difficulty | 0.525 | 4.734 | 2.765 |
| surprise_x_adv | 0.291 | 4.833 | 2.765 |
| rarity_x_adv | 0.291 | 4.739 | 2.765 |
| rarity_x_adv_completed | 0.291 | 4.769 | 2.765 |
| rarity_x_td | 0.345 | 4.883 | 2.765 |
| rarity_x_upside | 0.353 | 4.506 | 2.765 |
| good_rarity | 0.277 | 4.721 | 2.765 |
| good_surprise | 0.279 | 4.706 | 2.765 |
| tail_rarity | 0.340 | 4.536 | 2.765 |
| c51_tail_rarity | 0.288 | 4.937 | 2.765 |
| unxpass | 0.276 | 4.931 | 2.765 |


## Discriminant checks (is it just position, volume, risk, team?)

| measure | action_vs_start_x | action_vs_p_success | action_vs_advantage | player_vs_decisions | player_vs_mean_advantage | player_vs_final_third | player_vs_mean_p_success | eta2_team | eta2_group |
|---|---|---|---|---|---|---|---|---|---|
| rarity | -0.007 | -0.175 | 0.056 | -0.052 | 0.289 | 0.197 | -0.158 | 0.212 | 0.177 |
| advantage | -0.110 | -0.112 | 1.000 | -0.039 | 0.893 | 0.259 | -0.370 | 0.101 | 0.282 |
| difficulty | 0.172 | -0.678 | 0.025 | -0.246 | 0.382 | 0.646 | -0.944 | 0.322 | 0.462 |
| surprise_x_adv | 0.082 | -0.208 | 0.861 | -0.036 | 0.575 | 0.851 | -0.608 | 0.082 | 0.702 |
| rarity_x_adv | 0.093 | -0.219 | 0.860 | -0.036 | 0.569 | 0.856 | -0.608 | 0.082 | 0.708 |
| rarity_x_adv_completed | 0.044 | -0.131 | 0.799 | -0.021 | 0.575 | 0.845 | -0.589 | 0.080 | 0.682 |
| rarity_x_td | 0.001 | -0.236 | 0.314 | -0.029 | 0.490 | 0.811 | -0.669 | 0.097 | 0.649 |
| rarity_x_upside | 0.058 | -0.403 | 0.804 | -0.056 | 0.563 | 0.860 | -0.662 | 0.081 | 0.720 |
| good_rarity | 0.035 | -0.211 | 0.819 | -0.016 | 0.682 | 0.379 | -0.423 | 0.146 | 0.286 |
| good_surprise | 0.009 | -0.189 | 0.817 | 0.006 | 0.578 | 0.144 | -0.241 | 0.206 | 0.091 |
| tail_rarity | -0.018 | -0.369 | 0.730 | -0.127 | 0.655 | 0.362 | -0.639 | 0.225 | 0.253 |
| c51_tail_rarity | 0.123 | -0.191 | 0.269 | 0.039 | 0.544 | 0.614 | -0.390 | 0.082 | 0.496 |
| unxpass | -0.015 | -0.225 | 0.803 | 0.028 | 0.423 | -0.182 | -0.073 | 0.243 | 0.047 |


## Split-half reliability (alternating matches per player)

| measure | players | split_half_spearman | spearman_brown |
|---|---|---|---|
| rarity | 507 | 0.404 | 0.575 |
| advantage | 507 | 0.373 | 0.544 |
| difficulty | 507 | 0.804 | 0.892 |
| surprise_x_adv | 507 | 0.789 | 0.882 |
| rarity_x_adv | 507 | 0.800 | 0.889 |
| rarity_x_adv_completed | 507 | 0.761 | 0.864 |
| rarity_x_td | 507 | 0.745 | 0.854 |
| rarity_x_upside | 507 | 0.824 | 0.904 |
| good_rarity | 507 | 0.469 | 0.638 |
| good_surprise | 507 | 0.409 | 0.581 |
| tail_rarity | 507 | 0.562 | 0.719 |
| c51_tail_rarity | 507 | 0.612 | 0.760 |
| unxpass | 507 | 0.385 | 0.556 |


## Stability across competition-seasons

| first | second | players | measure | spearman |
|---|---|---|---|---|
| 1. Bundesliga 2023/2024 | UEFA Euro 2024 | 13 | rarity | 0.159 |
| 1. Bundesliga 2023/2024 | UEFA Euro 2024 | 13 | advantage | 0.379 |
| 1. Bundesliga 2023/2024 | UEFA Euro 2024 | 13 | difficulty | 0.659 |
| 1. Bundesliga 2023/2024 | UEFA Euro 2024 | 13 | surprise_x_adv | 0.808 |
| 1. Bundesliga 2023/2024 | UEFA Euro 2024 | 13 | rarity_x_adv | 0.857 |
| 1. Bundesliga 2023/2024 | UEFA Euro 2024 | 13 | rarity_x_adv_completed | 0.808 |
| 1. Bundesliga 2023/2024 | UEFA Euro 2024 | 13 | rarity_x_td | 0.863 |
| 1. Bundesliga 2023/2024 | UEFA Euro 2024 | 13 | rarity_x_upside | 0.841 |
| 1. Bundesliga 2023/2024 | UEFA Euro 2024 | 13 | good_rarity | 0.104 |
| 1. Bundesliga 2023/2024 | UEFA Euro 2024 | 13 | good_surprise | 0.412 |
| 1. Bundesliga 2023/2024 | UEFA Euro 2024 | 13 | tail_rarity | -0.027 |
| 1. Bundesliga 2023/2024 | UEFA Euro 2024 | 13 | c51_tail_rarity | 0.681 |
| 1. Bundesliga 2023/2024 | UEFA Euro 2024 | 13 | unxpass | -0.066 |
| FIFA World Cup 2022 | La Liga 2020/2021 | 14 | rarity | 0.481 |
| FIFA World Cup 2022 | La Liga 2020/2021 | 14 | advantage | -0.165 |
| FIFA World Cup 2022 | La Liga 2020/2021 | 14 | difficulty | 0.538 |
| FIFA World Cup 2022 | La Liga 2020/2021 | 14 | surprise_x_adv | 0.626 |
| FIFA World Cup 2022 | La Liga 2020/2021 | 14 | rarity_x_adv | 0.675 |
| FIFA World Cup 2022 | La Liga 2020/2021 | 14 | rarity_x_adv_completed | 0.648 |
| FIFA World Cup 2022 | La Liga 2020/2021 | 14 | rarity_x_td | 0.763 |
| FIFA World Cup 2022 | La Liga 2020/2021 | 14 | rarity_x_upside | 0.732 |
| FIFA World Cup 2022 | La Liga 2020/2021 | 14 | good_rarity | 0.270 |
| FIFA World Cup 2022 | La Liga 2020/2021 | 14 | good_surprise | 0.143 |
| FIFA World Cup 2022 | La Liga 2020/2021 | 14 | tail_rarity | 0.292 |
| FIFA World Cup 2022 | La Liga 2020/2021 | 14 | c51_tail_rarity | 0.802 |
| FIFA World Cup 2022 | La Liga 2020/2021 | 14 | unxpass | -0.007 |
| FIFA World Cup 2022 | Ligue 1 2021/2022 | 10 | rarity | 0.515 |
| FIFA World Cup 2022 | Ligue 1 2021/2022 | 10 | advantage | 0.879 |
| FIFA World Cup 2022 | Ligue 1 2021/2022 | 10 | difficulty | 0.879 |
| FIFA World Cup 2022 | Ligue 1 2021/2022 | 10 | surprise_x_adv | 0.842 |
| FIFA World Cup 2022 | Ligue 1 2021/2022 | 10 | rarity_x_adv | 0.867 |
| FIFA World Cup 2022 | Ligue 1 2021/2022 | 10 | rarity_x_adv_completed | 0.867 |
| FIFA World Cup 2022 | Ligue 1 2021/2022 | 10 | rarity_x_td | 0.806 |
| FIFA World Cup 2022 | Ligue 1 2021/2022 | 10 | rarity_x_upside | 0.855 |
| FIFA World Cup 2022 | Ligue 1 2021/2022 | 10 | good_rarity | 0.939 |
| FIFA World Cup 2022 | Ligue 1 2021/2022 | 10 | good_surprise | 0.867 |
| FIFA World Cup 2022 | Ligue 1 2021/2022 | 10 | tail_rarity | 0.794 |
| FIFA World Cup 2022 | Ligue 1 2021/2022 | 10 | c51_tail_rarity | 0.939 |
| FIFA World Cup 2022 | Ligue 1 2021/2022 | 10 | unxpass | 0.648 |
| FIFA World Cup 2022 | UEFA Euro 2020 | 76 | rarity | 0.586 |
| FIFA World Cup 2022 | UEFA Euro 2020 | 76 | advantage | 0.290 |
| FIFA World Cup 2022 | UEFA Euro 2020 | 76 | difficulty | 0.746 |
| FIFA World Cup 2022 | UEFA Euro 2020 | 76 | surprise_x_adv | 0.757 |
| FIFA World Cup 2022 | UEFA Euro 2020 | 76 | rarity_x_adv | 0.740 |
| FIFA World Cup 2022 | UEFA Euro 2020 | 76 | rarity_x_adv_completed | 0.729 |
| FIFA World Cup 2022 | UEFA Euro 2020 | 76 | rarity_x_td | 0.756 |
| FIFA World Cup 2022 | UEFA Euro 2020 | 76 | rarity_x_upside | 0.774 |
| FIFA World Cup 2022 | UEFA Euro 2020 | 76 | good_rarity | 0.418 |
| FIFA World Cup 2022 | UEFA Euro 2020 | 76 | good_surprise | 0.372 |
| FIFA World Cup 2022 | UEFA Euro 2020 | 76 | tail_rarity | 0.615 |
| FIFA World Cup 2022 | UEFA Euro 2020 | 76 | c51_tail_rarity | 0.731 |
| FIFA World Cup 2022 | UEFA Euro 2020 | 76 | unxpass | 0.403 |
| FIFA World Cup 2022 | UEFA Euro 2024 | 76 | rarity | 0.334 |
| FIFA World Cup 2022 | UEFA Euro 2024 | 76 | advantage | 0.543 |
| FIFA World Cup 2022 | UEFA Euro 2024 | 76 | difficulty | 0.667 |
| FIFA World Cup 2022 | UEFA Euro 2024 | 76 | surprise_x_adv | 0.836 |
| FIFA World Cup 2022 | UEFA Euro 2024 | 76 | rarity_x_adv | 0.833 |
| FIFA World Cup 2022 | UEFA Euro 2024 | 76 | rarity_x_adv_completed | 0.808 |
| FIFA World Cup 2022 | UEFA Euro 2024 | 76 | rarity_x_td | 0.698 |
| FIFA World Cup 2022 | UEFA Euro 2024 | 76 | rarity_x_upside | 0.843 |
| FIFA World Cup 2022 | UEFA Euro 2024 | 76 | good_rarity | 0.412 |
| FIFA World Cup 2022 | UEFA Euro 2024 | 76 | good_surprise | 0.373 |
| FIFA World Cup 2022 | UEFA Euro 2024 | 76 | tail_rarity | 0.367 |
| FIFA World Cup 2022 | UEFA Euro 2024 | 76 | c51_tail_rarity | 0.664 |
| FIFA World Cup 2022 | UEFA Euro 2024 | 76 | unxpass | 0.271 |
| La Liga 2020/2021 | UEFA Euro 2020 | 12 | rarity | 0.175 |
| La Liga 2020/2021 | UEFA Euro 2020 | 12 | advantage | 0.462 |
| La Liga 2020/2021 | UEFA Euro 2020 | 12 | difficulty | 0.497 |
| La Liga 2020/2021 | UEFA Euro 2020 | 12 | surprise_x_adv | 0.608 |
| La Liga 2020/2021 | UEFA Euro 2020 | 12 | rarity_x_adv | 0.580 |
| La Liga 2020/2021 | UEFA Euro 2020 | 12 | rarity_x_adv_completed | 0.580 |
| La Liga 2020/2021 | UEFA Euro 2020 | 12 | rarity_x_td | 0.699 |
| La Liga 2020/2021 | UEFA Euro 2020 | 12 | rarity_x_upside | 0.643 |
| La Liga 2020/2021 | UEFA Euro 2020 | 12 | good_rarity | 0.070 |
| La Liga 2020/2021 | UEFA Euro 2020 | 12 | good_surprise | 0.077 |
| La Liga 2020/2021 | UEFA Euro 2020 | 12 | tail_rarity | 0.098 |
| La Liga 2020/2021 | UEFA Euro 2020 | 12 | c51_tail_rarity | 0.510 |
| La Liga 2020/2021 | UEFA Euro 2020 | 12 | unxpass | 0.203 |
| La Liga 2020/2021 | UEFA Euro 2024 | 10 | rarity | -0.139 |
| La Liga 2020/2021 | UEFA Euro 2024 | 10 | advantage | 0.697 |
| La Liga 2020/2021 | UEFA Euro 2024 | 10 | difficulty | 0.236 |
| La Liga 2020/2021 | UEFA Euro 2024 | 10 | surprise_x_adv | 0.867 |
| La Liga 2020/2021 | UEFA Euro 2024 | 10 | rarity_x_adv | 0.855 |
| La Liga 2020/2021 | UEFA Euro 2024 | 10 | rarity_x_adv_completed | 0.818 |
| La Liga 2020/2021 | UEFA Euro 2024 | 10 | rarity_x_td | 0.503 |
| La Liga 2020/2021 | UEFA Euro 2024 | 10 | rarity_x_upside | 0.830 |
| La Liga 2020/2021 | UEFA Euro 2024 | 10 | good_rarity | 0.527 |
| La Liga 2020/2021 | UEFA Euro 2024 | 10 | good_surprise | 0.491 |
| La Liga 2020/2021 | UEFA Euro 2024 | 10 | tail_rarity | 0.152 |
| La Liga 2020/2021 | UEFA Euro 2024 | 10 | c51_tail_rarity | 0.358 |
| La Liga 2020/2021 | UEFA Euro 2024 | 10 | unxpass | 0.006 |
| Ligue 1 2021/2022 | Ligue 1 2022/2023 | 18 | rarity | 0.822 |
| Ligue 1 2021/2022 | Ligue 1 2022/2023 | 18 | advantage | 0.666 |
| Ligue 1 2021/2022 | Ligue 1 2022/2023 | 18 | difficulty | 0.759 |
| Ligue 1 2021/2022 | Ligue 1 2022/2023 | 18 | surprise_x_adv | 0.911 |
| Ligue 1 2021/2022 | Ligue 1 2022/2023 | 18 | rarity_x_adv | 0.884 |
| Ligue 1 2021/2022 | Ligue 1 2022/2023 | 18 | rarity_x_adv_completed | 0.895 |
| Ligue 1 2021/2022 | Ligue 1 2022/2023 | 18 | rarity_x_td | 0.928 |
| Ligue 1 2021/2022 | Ligue 1 2022/2023 | 18 | rarity_x_upside | 0.917 |
| Ligue 1 2021/2022 | Ligue 1 2022/2023 | 18 | good_rarity | 0.610 |
| Ligue 1 2021/2022 | Ligue 1 2022/2023 | 18 | good_surprise | 0.604 |
| Ligue 1 2021/2022 | Ligue 1 2022/2023 | 18 | tail_rarity | 0.589 |
| Ligue 1 2021/2022 | Ligue 1 2022/2023 | 18 | c51_tail_rarity | 0.759 |
| Ligue 1 2021/2022 | Ligue 1 2022/2023 | 18 | unxpass | 0.688 |
| UEFA Euro 2020 | UEFA Euro 2024 | 81 | rarity | 0.252 |
| UEFA Euro 2020 | UEFA Euro 2024 | 81 | advantage | 0.400 |
| UEFA Euro 2020 | UEFA Euro 2024 | 81 | difficulty | 0.632 |
| UEFA Euro 2020 | UEFA Euro 2024 | 81 | surprise_x_adv | 0.836 |
| UEFA Euro 2020 | UEFA Euro 2024 | 81 | rarity_x_adv | 0.827 |
| UEFA Euro 2020 | UEFA Euro 2024 | 81 | rarity_x_adv_completed | 0.810 |
| UEFA Euro 2020 | UEFA Euro 2024 | 81 | rarity_x_td | 0.588 |
| UEFA Euro 2020 | UEFA Euro 2024 | 81 | rarity_x_upside | 0.805 |
| UEFA Euro 2020 | UEFA Euro 2024 | 81 | good_rarity | 0.480 |
| UEFA Euro 2020 | UEFA Euro 2024 | 81 | good_surprise | 0.413 |
| UEFA Euro 2020 | UEFA Euro 2024 | 81 | tail_rarity | 0.447 |
| UEFA Euro 2020 | UEFA Euro 2024 | 81 | c51_tail_rarity | 0.696 |
| UEFA Euro 2020 | UEFA Euro 2024 | 81 | unxpass | 0.139 |
| UEFA Women's Euro 2022 | UEFA Women's Euro 2025 | 59 | rarity | 0.415 |
| UEFA Women's Euro 2022 | UEFA Women's Euro 2025 | 59 | advantage | 0.286 |
| UEFA Women's Euro 2022 | UEFA Women's Euro 2025 | 59 | difficulty | 0.639 |
| UEFA Women's Euro 2022 | UEFA Women's Euro 2025 | 59 | surprise_x_adv | 0.647 |
| UEFA Women's Euro 2022 | UEFA Women's Euro 2025 | 59 | rarity_x_adv | 0.619 |
| UEFA Women's Euro 2022 | UEFA Women's Euro 2025 | 59 | rarity_x_adv_completed | 0.576 |
| UEFA Women's Euro 2022 | UEFA Women's Euro 2025 | 59 | rarity_x_td | 0.730 |
| UEFA Women's Euro 2022 | UEFA Women's Euro 2025 | 59 | rarity_x_upside | 0.650 |
| UEFA Women's Euro 2022 | UEFA Women's Euro 2025 | 59 | good_rarity | 0.248 |
| UEFA Women's Euro 2022 | UEFA Women's Euro 2025 | 59 | good_surprise | 0.109 |
| UEFA Women's Euro 2022 | UEFA Women's Euro 2025 | 59 | tail_rarity | 0.361 |
| UEFA Women's Euro 2022 | UEFA Women's Euro 2025 | 59 | c51_tail_rarity | 0.438 |
| UEFA Women's Euro 2022 | UEFA Women's Euro 2025 | 59 | unxpass | 0.044 |
| UEFA Women's Euro 2022 | Women's World Cup 2023 | 76 | rarity | 0.395 |
| UEFA Women's Euro 2022 | Women's World Cup 2023 | 76 | advantage | 0.148 |
| UEFA Women's Euro 2022 | Women's World Cup 2023 | 76 | difficulty | 0.673 |
| UEFA Women's Euro 2022 | Women's World Cup 2023 | 76 | surprise_x_adv | 0.691 |
| UEFA Women's Euro 2022 | Women's World Cup 2023 | 76 | rarity_x_adv | 0.697 |
| UEFA Women's Euro 2022 | Women's World Cup 2023 | 76 | rarity_x_adv_completed | 0.645 |
| UEFA Women's Euro 2022 | Women's World Cup 2023 | 76 | rarity_x_td | 0.609 |
| UEFA Women's Euro 2022 | Women's World Cup 2023 | 76 | rarity_x_upside | 0.736 |
| UEFA Women's Euro 2022 | Women's World Cup 2023 | 76 | good_rarity | 0.424 |
| UEFA Women's Euro 2022 | Women's World Cup 2023 | 76 | good_surprise | 0.339 |
| UEFA Women's Euro 2022 | Women's World Cup 2023 | 76 | tail_rarity | 0.491 |
| UEFA Women's Euro 2022 | Women's World Cup 2023 | 76 | c51_tail_rarity | 0.482 |
| UEFA Women's Euro 2022 | Women's World Cup 2023 | 76 | unxpass | 0.377 |
| UEFA Women's Euro 2025 | Women's World Cup 2023 | 60 | rarity | 0.313 |
| UEFA Women's Euro 2025 | Women's World Cup 2023 | 60 | advantage | 0.214 |
| UEFA Women's Euro 2025 | Women's World Cup 2023 | 60 | difficulty | 0.704 |
| UEFA Women's Euro 2025 | Women's World Cup 2023 | 60 | surprise_x_adv | 0.660 |
| UEFA Women's Euro 2025 | Women's World Cup 2023 | 60 | rarity_x_adv | 0.688 |
| UEFA Women's Euro 2025 | Women's World Cup 2023 | 60 | rarity_x_adv_completed | 0.623 |
| UEFA Women's Euro 2025 | Women's World Cup 2023 | 60 | rarity_x_td | 0.534 |
| UEFA Women's Euro 2025 | Women's World Cup 2023 | 60 | rarity_x_upside | 0.685 |
| UEFA Women's Euro 2025 | Women's World Cup 2023 | 60 | good_rarity | 0.206 |
| UEFA Women's Euro 2025 | Women's World Cup 2023 | 60 | good_surprise | 0.254 |
| UEFA Women's Euro 2025 | Women's World Cup 2023 | 60 | tail_rarity | 0.187 |
| UEFA Women's Euro 2025 | Women's World Cup 2023 | 60 | c51_tail_rarity | 0.373 |
| UEFA Women's Euro 2025 | Women's World Cup 2023 | 60 | unxpass | 0.155 |


## Agreement with an independently trained model

| measure | decisions | action_spearman | top_jaccard | players | player_spearman |
|---|---|---|---|---|---|
| rarity | 536223 | 1.000 | 1.000 | 1255 | 1.000 |
| advantage | 536223 | 0.949 | 0.667 | 1255 | 0.972 |
| difficulty | 536223 | 0.995 | 0.784 | 1255 | 0.999 |
| surprise_x_adv | 536223 | 0.897 | 0.686 | 1255 | 0.991 |
| rarity_x_adv | 536223 | 0.897 | 0.694 | 1255 | 0.992 |
| rarity_x_adv_completed | 536223 | 0.897 | 0.699 | 1255 | 0.990 |
| rarity_x_td | 536223 | 0.972 | 0.876 | 1255 | 0.998 |
| rarity_x_upside | 536223 | 0.920 | 0.701 | 1255 | 0.994 |
| good_rarity | 536223 | 0.855 | 0.790 | 1255 | 0.931 |
| good_surprise | 536223 | 0.856 | 0.832 | 1255 | 0.939 |
| tail_rarity | 536223 | 0.879 | 0.817 | 1255 | 0.961 |
| unxpass | 536223 | 0.860 | 0.861 | 1255 | 0.933 |


## Top players per measure (90% match-bootstrap rank intervals)

| measure | rank | player | team | position | decisions | score | rank_p05 | rank_p95 |
|---|---|---|---|---|---|---|---|---|
| rarity | 1 | Ruslan Malinovskiy | Ukraine | Left Wing | 220 | 0.580 | 1.000 | 11.100 |
| rarity | 2 | Toni Kroos | Germany | Left Defensive Midfield | 1118 | 0.559 | 2.000 | 29.050 |
| rarity | 3 | Rafael Alexandre Conceição Leão | Portugal | Left Wing | 204 | 0.555 | 1.000 | 162.850 |
| rarity | 4 | Giorgi Kochorashvili | Georgia | Left Defensive Midfield | 226 | 0.554 | 2.950 | 66.800 |
| rarity | 5 | Denis Vavro | Slovakia | Right Center Back | 262 | 0.546 | 6.000 | 28.150 |
| rarity | 6 | Luka Modrić | Croatia | Right Center Midfield | 1281 | 0.542 | 9.000 | 40.000 |
| rarity | 7 | Johan Gastien | Clermont Foot | Left Defensive Midfield | 249 | 0.541 | 6.000 | 84.700 |
| rarity | 8 | Tomáš Kalas | Czech Republic | Left Center Back | 206 | 0.541 | 7.900 | 61.400 |
| rarity | 9 | Aaron Mooy | Australia | Left Defensive Midfield | 229 | 0.540 | 2.850 | 167.150 |
| rarity | 10 | Ivan Perišić | Croatia | Left Wing | 447 | 0.540 | 7.000 | 98.100 |
| rarity | 11 | Sergej Milinković-Savić | Serbia | Left Attacking Midfield | 208 | 0.540 | 5.000 | 120.100 |
| rarity | 12 | Federico Santiago Valverde Dipetta | Uruguay | Right Defensive Midfield | 275 | 0.539 | 3.000 | 237.700 |
| rarity | 13 | Stanislav Lobotka | Slovakia | Center Defensive Midfield | 326 | 0.538 | 7.900 | 156.100 |
| rarity | 14 | Lorenzo Insigne | Italy | Left Wing | 380 | 0.538 | 4.000 | 130.300 |
| rarity | 15 | Sofiane Boufal | Morocco | Left Wing | 228 | 0.538 | 2.000 | 172.050 |
| rarity | 16 | Eden Hazard | Belgium | Left Attacking Midfield | 281 | 0.538 | 9.000 | 136.050 |
| rarity | 17 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | Center Attacking Midfield | 279 | 0.537 | 2.000 | 126.050 |
| rarity | 18 | Leandro Trossard | Belgium | Right Wing | 207 | 0.537 | 3.000 | 187.000 |
| rarity | 19 | Fabian Lukas Schär | Switzerland | Right Center Back | 509 | 0.536 | 7.000 | 103.150 |
| rarity | 20 | Thomas Teye Partey | Ghana | Right Defensive Midfield | 208 | 0.534 | 3.000 | 296.800 |
| rarity | 21 | Oleksandr Karavaev | Ukraine | Right Back | 287 | 0.534 | 7.000 | 141.500 |
| rarity | 22 | Philipp Lienhart | Austria | Left Center Back | 243 | 0.533 | 2.950 | 217.100 |
| rarity | 23 | Montassar Omar Talbi | Tunisia | Left Center Back | 218 | 0.533 | 10.800 | 120.600 |
| rarity | 24 | Marcos Javier Acuña | Argentina | Left Back | 226 | 0.533 | 2.000 | 208.000 |
| rarity | 25 | Kerolin Nicoli Israel Ferraz | Brazil Women's | Right Defensive Midfield | 208 | 0.530 | 20.450 | 160.000 |
| advantage | 1 | Victor Okoh Boniface | Bayer Leverkusen | Center Forward | 560 | 0.004 | 1.000 | 14.000 |
| advantage | 2 | Pernille Mosegaard Harder | Denmark Women's | Center Attacking Midfield | 320 | 0.004 | 1.000 | 32.050 |
| advantage | 3 | Cody Mathès Gakpo | Netherlands | Left Wing | 323 | 0.003 | 1.000 | 47.200 |
| advantage | 4 | Delphine Cascarino | France Women's | Right Wing | 347 | 0.003 | 2.000 | 40.200 |
| advantage | 5 | Christoph Baumgartner | Austria | Center Attacking Midfield | 254 | 0.003 | 1.000 | 78.050 |
| advantage | 6 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | Center Attacking Midfield | 279 | 0.003 | 3.950 | 56.000 |
| advantage | 7 | Athenea del Castillo Belvide | Spain Women's | Right Wing | 260 | 0.003 | 2.950 | 40.150 |
| advantage | 8 | Lamine Yamal Nasraoui Ebana | Spain | Right Wing | 296 | 0.003 | 2.000 | 75.500 |
| advantage | 9 | Kylian Mbappé Lottin | Paris Saint-Germain | Center Forward | 3542 | 0.003 | 6.000 | 24.000 |
| advantage | 10 | Caitlin Jade Foord | Australia Women's | Left Midfield | 212 | 0.003 | 1.000 | 231.000 |
| advantage | 11 | Mary Boio Fowler | Australia Women's | Left Center Forward | 204 | 0.003 | 5.000 | 78.700 |
| advantage | 12 | Ruslan Malinovskiy | Ukraine | Left Wing | 220 | 0.003 | 3.000 | 117.150 |
| advantage | 13 | Jamal Musiala | Germany | Left Wing | 558 | 0.003 | 2.000 | 104.100 |
| advantage | 14 | Sofiane Boufal | Morocco | Left Wing | 228 | 0.003 | 5.000 | 56.200 |
| advantage | 15 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | Right Center Forward | 8537 | 0.002 | 11.950 | 30.050 |
| advantage | 16 | Rafael Alexandre Conceição Leão | Portugal | Left Wing | 204 | 0.002 | 8.000 | 127.400 |
| advantage | 17 | Jeremy Doku | Belgium | Left Wing | 332 | 0.002 | 9.000 | 46.050 |
| advantage | 18 | Mikkel Damsgaard | Denmark | Left Wing | 243 | 0.002 | 7.000 | 79.500 |
| advantage | 19 | Johanna Rytting-Kaneryd | Sweden Women's | Right Wing | 470 | 0.002 | 4.000 | 280.950 |
| advantage | 20 | Emma Stina Blackstenius | Sweden Women's | Center Forward | 308 | 0.002 | 2.000 | 451.100 |
| advantage | 21 | Mason Mount | England | Center Attacking Midfield | 294 | 0.002 | 9.950 | 86.250 |
| advantage | 22 | Yannick Ferreira Carrasco | Belgium | Left Wing Back | 343 | 0.002 | 6.000 | 251.300 |
| advantage | 23 | Amine Adli | Bayer Leverkusen | Left Attacking Midfield | 447 | 0.002 | 5.000 | 147.950 |
| advantage | 24 | Nathan Tella | Bayer Leverkusen | Right Wing Back | 317 | 0.002 | 6.000 | 149.200 |
| advantage | 25 | Nicholas Williams Arthuer | Spain | Left Wing | 362 | 0.002 | 6.000 | 377.800 |
| difficulty | 1 | Manuela Vanegas Cataño | Colombia Women's | Left Back | 213 | 0.185 | 1.000 | 23.000 |
| difficulty | 2 | Clara Mateo | France Women's | Left Center Midfield | 278 | 0.176 | 1.000 | 32.000 |
| difficulty | 3 | Caitlin Jade Foord | Australia Women's | Left Midfield | 212 | 0.173 | 2.000 | 37.050 |
| difficulty | 4 | Kosovare Asllani | Sweden Women's | Center Attacking Midfield | 487 | 0.173 | 2.000 | 21.000 |
| difficulty | 5 | Lisa Boattin | Italy Women's | Left Back | 500 | 0.171 | 2.000 | 40.050 |
| difficulty | 6 | Jonna Ann-Charlotte Andersson | Sweden Women's | Left Back | 443 | 0.170 | 1.000 | 37.100 |
| difficulty | 7 | Cody Mathès Gakpo | Netherlands | Left Wing | 323 | 0.169 | 2.000 | 46.100 |
| difficulty | 8 | Pernille Mosegaard Harder | Denmark Women's | Center Attacking Midfield | 320 | 0.162 | 5.000 | 46.000 |
| difficulty | 9 | Ana-Maria Crnogorčević | Switzerland Women's | Center Forward | 295 | 0.161 | 7.950 | 36.050 |
| difficulty | 10 | Leicy Maria Santos Herrera | Colombia Women's | Center Attacking Midfield | 237 | 0.159 | 6.000 | 44.050 |
| difficulty | 11 | Tessa Wullaert | Belgium Women's | Center Forward | 265 | 0.159 | 3.000 | 60.000 |
| difficulty | 12 | Janni Thomsen | Denmark Women's | Right Wing Back | 294 | 0.159 | 2.000 | 90.300 |
| difficulty | 13 | Robert Lewandowski | Poland | Left Center Forward | 241 | 0.157 | 2.000 | 93.100 |
| difficulty | 14 | Jéssica da Silva | Portugal Women's | Right Center Forward | 226 | 0.157 | 10.950 | 42.000 |
| difficulty | 15 | Patrik Schick | Bayer Leverkusen | Center Forward | 447 | 0.157 | 7.000 | 47.050 |
| difficulty | 16 | Lindsey Michelle Heaps | United States Women's | Left Center Midfield | 236 | 0.156 | 2.000 | 104.700 |
| difficulty | 17 | Jun Endo | Japan Women's | Left Wing Back | 216 | 0.156 | 2.000 | 140.100 |
| difficulty | 18 | Johanna Rytting-Kaneryd | Sweden Women's | Right Wing | 470 | 0.156 | 9.000 | 53.100 |
| difficulty | 19 | Caroline Graham Hansen | Norway Women's | Right Midfield | 410 | 0.155 | 6.900 | 52.050 |
| difficulty | 20 | Bethany Mead | England Women's | Right Wing | 251 | 0.153 | 8.950 | 74.050 |
| difficulty | 21 | Emma Stina Blackstenius | Sweden Women's | Center Forward | 308 | 0.153 | 11.000 | 57.000 |
| difficulty | 22 | Alessia Russo | England Women's | Center Forward | 390 | 0.153 | 8.950 | 53.000 |
| difficulty | 23 | Delphine Cascarino | France Women's | Right Wing | 347 | 0.152 | 13.900 | 53.100 |
| difficulty | 24 | Chloe Kelly | England Women's | Right Wing | 316 | 0.151 | 10.950 | 69.100 |
| difficulty | 25 | Vladimír Coufal | Czech Republic | Right Back | 406 | 0.150 | 6.000 | 79.000 |
| surprise_x_adv | 1 | Victor Okoh Boniface | Bayer Leverkusen | Center Forward | 560 | 0.034 | 1.000 | 15.050 |
| surprise_x_adv | 2 | Athenea del Castillo Belvide | Spain Women's | Right Wing | 260 | 0.034 | 1.000 | 38.300 |
| surprise_x_adv | 3 | Pernille Mosegaard Harder | Denmark Women's | Center Attacking Midfield | 320 | 0.034 | 1.000 | 37.000 |
| surprise_x_adv | 4 | Cody Mathès Gakpo | Netherlands | Left Wing | 323 | 0.032 | 1.000 | 30.200 |
| surprise_x_adv | 5 | Christoph Baumgartner | Austria | Center Attacking Midfield | 254 | 0.030 | 1.000 | 66.050 |
| surprise_x_adv | 6 | Ruslan Malinovskiy | Ukraine | Left Wing | 220 | 0.030 | 3.000 | 45.000 |
| surprise_x_adv | 7 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | Center Attacking Midfield | 279 | 0.030 | 4.000 | 42.050 |
| surprise_x_adv | 8 | Jamal Musiala | Germany | Left Wing | 558 | 0.029 | 2.000 | 43.050 |
| surprise_x_adv | 9 | Johanna Rytting-Kaneryd | Sweden Women's | Right Wing | 470 | 0.029 | 2.000 | 62.000 |
| surprise_x_adv | 10 | Bethany Mead | England Women's | Right Wing | 251 | 0.028 | 3.950 | 43.150 |
| surprise_x_adv | 11 | Kylian Mbappé Lottin | Paris Saint-Germain | Center Forward | 3542 | 0.028 | 8.950 | 26.000 |
| surprise_x_adv | 12 | Emma Stina Blackstenius | Sweden Women's | Center Forward | 308 | 0.027 | 4.000 | 77.150 |
| surprise_x_adv | 13 | Caitlin Jade Foord | Australia Women's | Left Midfield | 212 | 0.027 | 4.000 | 70.100 |
| surprise_x_adv | 14 | Jeremy Doku | Belgium | Left Wing | 332 | 0.027 | 3.000 | 50.050 |
| surprise_x_adv | 15 | Delphine Cascarino | France Women's | Right Wing | 347 | 0.027 | 6.800 | 44.100 |
| surprise_x_adv | 16 | Nicholas Williams Arthuer | Spain | Left Wing | 362 | 0.027 | 4.000 | 82.550 |
| surprise_x_adv | 17 | Clara Mateo | France Women's | Left Center Midfield | 278 | 0.027 | 6.000 | 57.100 |
| surprise_x_adv | 18 | Rafael Alexandre Conceição Leão | Portugal | Left Wing | 204 | 0.026 | 7.000 | 64.000 |
| surprise_x_adv | 19 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | Right Center Forward | 8537 | 0.026 | 13.000 | 29.050 |
| surprise_x_adv | 20 | Vinícius José Paixão de Oliveira Júnior | Brazil | Left Wing | 213 | 0.026 | 5.000 | 71.200 |
| surprise_x_adv | 21 | Ousmane Dembélé | Barcelona | Right Wing | 1761 | 0.025 | 13.000 | 44.000 |
| surprise_x_adv | 22 | Amine Adli | Bayer Leverkusen | Left Attacking Midfield | 447 | 0.025 | 3.000 | 86.150 |
| surprise_x_adv | 23 | Lamine Yamal Nasraoui Ebana | Spain | Right Wing | 296 | 0.025 | 7.000 | 78.450 |
| surprise_x_adv | 24 | Klara Bühl | Germany Women's | Left Wing | 557 | 0.025 | 7.950 | 77.300 |
| surprise_x_adv | 25 | Nathan Tella | Bayer Leverkusen | Right Wing Back | 317 | 0.024 | 4.000 | 97.050 |
| rarity_x_adv | 1 | Victor Okoh Boniface | Bayer Leverkusen | Center Forward | 560 | 0.005 | 1.000 | 8.000 |
| rarity_x_adv | 2 | Pernille Mosegaard Harder | Denmark Women's | Center Attacking Midfield | 320 | 0.004 | 1.000 | 31.050 |
| rarity_x_adv | 3 | Athenea del Castillo Belvide | Spain Women's | Right Wing | 260 | 0.004 | 1.000 | 50.050 |
| rarity_x_adv | 4 | Cody Mathès Gakpo | Netherlands | Left Wing | 323 | 0.004 | 2.000 | 35.000 |
| rarity_x_adv | 5 | Bethany Mead | England Women's | Right Wing | 251 | 0.004 | 2.000 | 35.050 |
| rarity_x_adv | 6 | Christoph Baumgartner | Austria | Center Attacking Midfield | 254 | 0.004 | 1.950 | 60.200 |
| rarity_x_adv | 7 | Caitlin Jade Foord | Australia Women's | Left Midfield | 212 | 0.004 | 2.000 | 63.100 |
| rarity_x_adv | 8 | Emma Stina Blackstenius | Sweden Women's | Center Forward | 308 | 0.004 | 3.000 | 56.100 |
| rarity_x_adv | 9 | Kylian Mbappé Lottin | Paris Saint-Germain | Center Forward | 3542 | 0.004 | 8.000 | 21.000 |
| rarity_x_adv | 10 | Ruslan Malinovskiy | Ukraine | Left Wing | 220 | 0.004 | 5.000 | 57.100 |
| rarity_x_adv | 11 | Jeremy Doku | Belgium | Left Wing | 332 | 0.004 | 3.000 | 44.150 |
| rarity_x_adv | 12 | Nicholas Williams Arthuer | Spain | Left Wing | 362 | 0.004 | 3.000 | 79.050 |
| rarity_x_adv | 13 | Jamal Musiala | Germany | Left Wing | 558 | 0.004 | 3.950 | 52.100 |
| rarity_x_adv | 14 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | Center Attacking Midfield | 279 | 0.004 | 5.000 | 76.000 |
| rarity_x_adv | 15 | Johanna Rytting-Kaneryd | Sweden Women's | Right Wing | 470 | 0.003 | 4.000 | 75.100 |
| rarity_x_adv | 16 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | Right Center Forward | 8537 | 0.003 | 15.000 | 33.000 |
| rarity_x_adv | 17 | Delphine Cascarino | France Women's | Right Wing | 347 | 0.003 | 10.000 | 50.000 |
| rarity_x_adv | 18 | Amine Adli | Bayer Leverkusen | Left Attacking Midfield | 447 | 0.003 | 4.000 | 80.250 |
| rarity_x_adv | 19 | Lamine Yamal Nasraoui Ebana | Spain | Right Wing | 296 | 0.003 | 8.000 | 77.050 |
| rarity_x_adv | 20 | Ousmane Dembélé | Barcelona | Right Wing | 1761 | 0.003 | 12.950 | 44.000 |
| rarity_x_adv | 21 | Vinícius José Paixão de Oliveira Júnior | Brazil | Left Wing | 213 | 0.003 | 5.000 | 87.750 |
| rarity_x_adv | 22 | Klara Bühl | Germany Women's | Left Wing | 557 | 0.003 | 8.000 | 72.000 |
| rarity_x_adv | 23 | Rafael Alexandre Conceição Leão | Portugal | Left Wing | 204 | 0.003 | 10.000 | 66.150 |
| rarity_x_adv | 24 | Mikel Oyarzabal Ugarte | Spain | Right Wing | 245 | 0.003 | 2.000 | 141.250 |
| rarity_x_adv | 25 | Clara Mateo | France Women's | Left Center Midfield | 278 | 0.003 | 10.950 | 66.100 |
| rarity_x_adv_completed | 1 | Victor Okoh Boniface | Bayer Leverkusen | Center Forward | 560 | 0.004 | 1.000 | 16.050 |
| rarity_x_adv_completed | 2 | Pernille Mosegaard Harder | Denmark Women's | Center Attacking Midfield | 320 | 0.004 | 1.000 | 26.050 |
| rarity_x_adv_completed | 3 | Bethany Mead | England Women's | Right Wing | 251 | 0.004 | 1.000 | 37.000 |
| rarity_x_adv_completed | 4 | Cody Mathès Gakpo | Netherlands | Left Wing | 323 | 0.004 | 2.000 | 36.000 |
| rarity_x_adv_completed | 5 | Caitlin Jade Foord | Australia Women's | Left Midfield | 212 | 0.003 | 2.000 | 71.000 |
| rarity_x_adv_completed | 6 | Christoph Baumgartner | Austria | Center Attacking Midfield | 254 | 0.003 | 1.000 | 88.000 |
| rarity_x_adv_completed | 7 | Jeremy Doku | Belgium | Left Wing | 332 | 0.003 | 2.000 | 41.050 |
| rarity_x_adv_completed | 8 | Jamal Musiala | Germany | Left Wing | 558 | 0.003 | 2.950 | 45.000 |
| rarity_x_adv_completed | 9 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | Center Attacking Midfield | 279 | 0.003 | 3.950 | 64.000 |
| rarity_x_adv_completed | 10 | Nicholas Williams Arthuer | Spain | Left Wing | 362 | 0.003 | 2.000 | 92.500 |
| rarity_x_adv_completed | 11 | Athenea del Castillo Belvide | Spain Women's | Right Wing | 260 | 0.003 | 1.000 | 59.100 |
| rarity_x_adv_completed | 12 | Ruslan Malinovskiy | Ukraine | Left Wing | 220 | 0.003 | 5.000 | 52.000 |
| rarity_x_adv_completed | 13 | Johanna Rytting-Kaneryd | Sweden Women's | Right Wing | 470 | 0.003 | 2.950 | 79.100 |
| rarity_x_adv_completed | 14 | Kylian Mbappé Lottin | Paris Saint-Germain | Center Forward | 3542 | 0.003 | 9.000 | 26.000 |
| rarity_x_adv_completed | 15 | Nathan Tella | Bayer Leverkusen | Right Wing Back | 317 | 0.003 | 3.000 | 83.250 |
| rarity_x_adv_completed | 16 | Emma Stina Blackstenius | Sweden Women's | Center Forward | 308 | 0.003 | 4.000 | 99.200 |
| rarity_x_adv_completed | 17 | Delphine Cascarino | France Women's | Right Wing | 347 | 0.003 | 10.000 | 38.100 |
| rarity_x_adv_completed | 18 | Ousmane Dembélé | Barcelona | Right Wing | 1761 | 0.003 | 10.000 | 40.000 |
| rarity_x_adv_completed | 19 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | Right Center Forward | 8537 | 0.003 | 15.000 | 30.000 |
| rarity_x_adv_completed | 20 | Mikel Oyarzabal Ugarte | Spain | Right Wing | 245 | 0.003 | 1.000 | 156.100 |
| rarity_x_adv_completed | 21 | Lamine Yamal Nasraoui Ebana | Spain | Right Wing | 296 | 0.003 | 8.000 | 86.100 |
| rarity_x_adv_completed | 22 | Clara Mateo | France Women's | Left Center Midfield | 278 | 0.003 | 10.000 | 62.200 |
| rarity_x_adv_completed | 23 | Karim Benzema | France | Center Forward | 261 | 0.003 | 1.000 | 176.100 |
| rarity_x_adv_completed | 24 | Amine Adli | Bayer Leverkusen | Left Attacking Midfield | 447 | 0.003 | 4.000 | 83.250 |
| rarity_x_adv_completed | 25 | Klara Bühl | Germany Women's | Left Wing | 557 | 0.003 | 8.000 | 81.150 |
| rarity_x_td | 1 | Robert Lewandowski | Poland | Left Center Forward | 241 | 0.009 | 1.000 | 85.200 |
| rarity_x_td | 2 | Emma Stina Blackstenius | Sweden Women's | Center Forward | 308 | 0.008 | 1.000 | 28.250 |
| rarity_x_td | 3 | Kai Havertz | Germany | Center Forward | 351 | 0.007 | 1.000 | 48.400 |
| rarity_x_td | 4 | Victor Okoh Boniface | Bayer Leverkusen | Center Forward | 560 | 0.007 | 3.000 | 16.000 |
| rarity_x_td | 5 | Mary Boio Fowler | Australia Women's | Left Center Forward | 204 | 0.007 | 2.000 | 61.100 |
| rarity_x_td | 6 | Bethany Mead | England Women's | Right Wing | 251 | 0.007 | 2.000 | 56.050 |
| rarity_x_td | 7 | Nathan Tella | Bayer Leverkusen | Right Wing Back | 317 | 0.007 | 2.000 | 49.250 |
| rarity_x_td | 8 | Andrija Živković | Serbia | Right Wing Back | 229 | 0.007 | 1.000 | 186.050 |
| rarity_x_td | 9 | Ruslan Malinovskiy | Ukraine | Left Wing | 220 | 0.007 | 2.000 | 84.300 |
| rarity_x_td | 10 | Kylian Mbappé Lottin | Paris Saint-Germain | Center Forward | 3542 | 0.006 | 6.000 | 25.000 |
| rarity_x_td | 11 | Tessa Wullaert | Belgium Women's | Center Forward | 265 | 0.006 | 1.950 | 99.150 |
| rarity_x_td | 12 | Cristiano Ronaldo dos Santos Aveiro | Portugal | Center Forward | 391 | 0.006 | 3.000 | 55.150 |
| rarity_x_td | 13 | Chloe Kelly | England Women's | Right Wing | 316 | 0.006 | 5.950 | 47.150 |
| rarity_x_td | 14 | Alessia Russo | England Women's | Center Forward | 390 | 0.006 | 3.000 | 67.050 |
| rarity_x_td | 15 | Kadidiatou Diani | France Women's | Right Wing | 460 | 0.006 | 3.950 | 83.000 |
| rarity_x_td | 16 | Harry Kane | England | Center Forward | 480 | 0.006 | 7.000 | 53.000 |
| rarity_x_td | 17 | Jeremy Doku | Belgium | Left Wing | 332 | 0.006 | 2.000 | 98.100 |
| rarity_x_td | 18 | Álvaro Borja Morata Martín | Spain | Center Forward | 356 | 0.006 | 7.950 | 68.050 |
| rarity_x_td | 19 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | Center Attacking Midfield | 279 | 0.005 | 7.000 | 133.000 |
| rarity_x_td | 20 | Breel-Donald Embolo | Switzerland | Center Forward | 318 | 0.005 | 4.000 | 88.050 |
| rarity_x_td | 21 | Lamine Yamal Nasraoui Ebana | Spain | Right Wing | 296 | 0.005 | 11.950 | 77.150 |
| rarity_x_td | 22 | Memphis Depay | Netherlands | Left Center Forward | 488 | 0.005 | 11.950 | 59.050 |
| rarity_x_td | 23 | Adam Hložek | Bayer Leverkusen | Right Attacking Midfield | 271 | 0.005 | 8.950 | 80.050 |
| rarity_x_td | 24 | Clara Mateo | France Women's | Left Center Midfield | 278 | 0.005 | 12.000 | 72.000 |
| rarity_x_td | 25 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | Right Center Forward | 8537 | 0.005 | 22.000 | 41.000 |
| rarity_x_upside | 1 | Victor Okoh Boniface | Bayer Leverkusen | Center Forward | 560 | 0.006 | 1.000 | 7.000 |
| rarity_x_upside | 2 | Athenea del Castillo Belvide | Spain Women's | Right Wing | 260 | 0.005 | 1.000 | 45.050 |
| rarity_x_upside | 3 | Emma Stina Blackstenius | Sweden Women's | Center Forward | 308 | 0.005 | 1.000 | 29.050 |
| rarity_x_upside | 4 | Pernille Mosegaard Harder | Denmark Women's | Center Attacking Midfield | 320 | 0.005 | 2.000 | 32.000 |
| rarity_x_upside | 5 | Bethany Mead | England Women's | Right Wing | 251 | 0.005 | 2.000 | 35.000 |
| rarity_x_upside | 6 | Cody Mathès Gakpo | Netherlands | Left Wing | 323 | 0.005 | 2.000 | 35.050 |
| rarity_x_upside | 7 | Christoph Baumgartner | Austria | Center Attacking Midfield | 254 | 0.005 | 2.000 | 51.050 |
| rarity_x_upside | 8 | Kylian Mbappé Lottin | Paris Saint-Germain | Center Forward | 3542 | 0.005 | 8.000 | 20.000 |
| rarity_x_upside | 9 | Caitlin Jade Foord | Australia Women's | Left Midfield | 212 | 0.005 | 2.950 | 66.100 |
| rarity_x_upside | 10 | Jeremy Doku | Belgium | Left Wing | 332 | 0.004 | 4.000 | 48.000 |
| rarity_x_upside | 11 | Nicholas Williams Arthuer | Spain | Left Wing | 362 | 0.004 | 4.000 | 82.250 |
| rarity_x_upside | 12 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | Center Attacking Midfield | 279 | 0.004 | 5.000 | 84.050 |
| rarity_x_upside | 13 | Delphine Cascarino | France Women's | Right Wing | 347 | 0.004 | 8.900 | 38.000 |
| rarity_x_upside | 14 | Clara Mateo | France Women's | Left Center Midfield | 278 | 0.004 | 7.950 | 51.000 |
| rarity_x_upside | 15 | Ruslan Malinovskiy | Ukraine | Left Wing | 220 | 0.004 | 6.000 | 70.000 |
| rarity_x_upside | 16 | Jamal Musiala | Germany | Left Wing | 558 | 0.004 | 5.000 | 59.100 |
| rarity_x_upside | 17 | Esther Gonzalez Rodríguez | Spain Women's | Center Forward | 272 | 0.004 | 4.000 | 78.050 |
| rarity_x_upside | 18 | Vinícius José Paixão de Oliveira Júnior | Brazil | Left Wing | 213 | 0.004 | 2.000 | 98.150 |
| rarity_x_upside | 19 | Johanna Rytting-Kaneryd | Sweden Women's | Right Wing | 470 | 0.004 | 5.000 | 79.050 |
| rarity_x_upside | 20 | Klara Bühl | Germany Women's | Left Wing | 557 | 0.004 | 8.000 | 62.100 |
| rarity_x_upside | 21 | Lamine Yamal Nasraoui Ebana | Spain | Right Wing | 296 | 0.004 | 10.000 | 88.200 |
| rarity_x_upside | 22 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | Right Center Forward | 8537 | 0.004 | 17.000 | 35.050 |
| rarity_x_upside | 23 | Amine Adli | Bayer Leverkusen | Left Attacking Midfield | 447 | 0.004 | 6.000 | 78.050 |
| rarity_x_upside | 24 | Mikel Oyarzabal Ugarte | Spain | Right Wing | 245 | 0.004 | 2.000 | 121.050 |
| rarity_x_upside | 25 | Chloe Kelly | England Women's | Right Wing | 316 | 0.004 | 11.000 | 65.050 |
| good_rarity | 1 | Ruslan Malinovskiy | Ukraine | Left Wing | 220 | 0.290 | 1.000 | 16.000 |
| good_rarity | 2 | Rafael Alexandre Conceição Leão | Portugal | Left Wing | 204 | 0.274 | 2.000 | 57.100 |
| good_rarity | 3 | Jeremy Doku | Belgium | Left Wing | 332 | 0.269 | 1.000 | 27.050 |
| good_rarity | 4 | Kerolin Nicoli Israel Ferraz | Brazil Women's | Right Defensive Midfield | 208 | 0.268 | 2.000 | 25.550 |
| good_rarity | 5 | Mary Boio Fowler | Australia Women's | Left Center Forward | 204 | 0.265 | 1.000 | 87.200 |
| good_rarity | 6 | Pernille Mosegaard Harder | Denmark Women's | Center Attacking Midfield | 320 | 0.260 | 3.000 | 37.050 |
| good_rarity | 7 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | Right Center Forward | 8537 | 0.258 | 6.000 | 17.000 |
| good_rarity | 8 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | Center Attacking Midfield | 279 | 0.253 | 3.000 | 63.000 |
| good_rarity | 9 | Lamine Yamal Nasraoui Ebana | Spain | Right Wing | 296 | 0.251 | 2.950 | 111.800 |
| good_rarity | 10 | Clara Mateo | France Women's | Left Center Midfield | 278 | 0.250 | 4.950 | 70.600 |
| good_rarity | 11 | Victor Okoh Boniface | Bayer Leverkusen | Center Forward | 560 | 0.247 | 6.000 | 47.000 |
| good_rarity | 12 | Christoph Baumgartner | Austria | Center Attacking Midfield | 254 | 0.247 | 2.950 | 87.600 |
| good_rarity | 13 | Ana Borges | Portugal Women's | Right Back | 326 | 0.247 | 3.000 | 100.050 |
| good_rarity | 14 | Neymar da Silva Santos Junior | Paris Saint-Germain | Left Wing | 2650 | 0.244 | 13.000 | 37.000 |
| good_rarity | 15 | Caitlin Jade Foord | Australia Women's | Left Midfield | 212 | 0.242 | 4.000 | 76.100 |
| good_rarity | 16 | Piotr Zieliński | Poland | Right Center Midfield | 508 | 0.242 | 13.000 | 41.050 |
| good_rarity | 17 | Sofiane Boufal | Morocco | Left Wing | 228 | 0.240 | 11.000 | 64.100 |
| good_rarity | 18 | Jule Brand | Germany Women's | Right Wing | 414 | 0.239 | 8.950 | 76.200 |
| good_rarity | 19 | Chloe Kelly | England Women's | Right Wing | 316 | 0.238 | 6.950 | 109.250 |
| good_rarity | 20 | Harry Kane | England | Center Forward | 480 | 0.237 | 6.950 | 105.500 |
| good_rarity | 21 | Federico Chiesa | Italy | Right Wing | 282 | 0.237 | 12.000 | 78.050 |
| good_rarity | 22 | Lindsey Michelle Heaps | United States Women's | Left Center Midfield | 236 | 0.236 | 1.000 | 133.100 |
| good_rarity | 23 | Delphine Cascarino | France Women's | Right Wing | 347 | 0.236 | 10.900 | 114.000 |
| good_rarity | 24 | Xavi Simons | Netherlands | Right Wing | 354 | 0.236 | 14.000 | 78.250 |
| good_rarity | 25 | Ousmane Dembélé | Barcelona | Right Wing | 1761 | 0.236 | 21.900 | 54.050 |
| good_surprise | 1 | Ruslan Malinovskiy | Ukraine | Left Wing | 220 | 2.046 | 1.000 | 26.350 |
| good_surprise | 2 | Kerolin Nicoli Israel Ferraz | Brazil Women's | Right Defensive Midfield | 208 | 1.941 | 2.000 | 9.000 |
| good_surprise | 3 | Ana Borges | Portugal Women's | Right Back | 326 | 1.865 | 1.950 | 57.150 |
| good_surprise | 4 | Katie Bowen | New Zealand Women's | Left Center Back | 255 | 1.818 | 4.000 | 25.450 |
| good_surprise | 5 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | Center Attacking Midfield | 279 | 1.817 | 2.000 | 27.000 |
| good_surprise | 6 | Rafael Alexandre Conceição Leão | Portugal | Left Wing | 204 | 1.804 | 3.000 | 116.550 |
| good_surprise | 7 | Luka Modrić | Croatia | Right Center Midfield | 1281 | 1.764 | 5.000 | 41.000 |
| good_surprise | 8 | Giorgi Kochorashvili | Georgia | Left Defensive Midfield | 226 | 1.744 | 2.000 | 70.300 |
| good_surprise | 9 | Mary Boio Fowler | Australia Women's | Left Center Forward | 204 | 1.740 | 3.900 | 96.100 |
| good_surprise | 10 | Piotr Zieliński | Poland | Right Center Midfield | 508 | 1.724 | 10.000 | 40.050 |
| good_surprise | 11 | Jeremy Doku | Belgium | Left Wing | 332 | 1.722 | 3.000 | 82.100 |
| good_surprise | 12 | Thomas Teye Partey | Ghana | Right Defensive Midfield | 208 | 1.713 | 14.000 | 48.400 |
| good_surprise | 13 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | Right Center Forward | 8537 | 1.694 | 16.000 | 38.050 |
| good_surprise | 14 | Clara Mateo | France Women's | Left Center Midfield | 278 | 1.693 | 6.000 | 147.250 |
| good_surprise | 15 | Toni Kroos | Germany | Left Defensive Midfield | 1118 | 1.688 | 8.000 | 83.050 |
| good_surprise | 16 | Pernille Mosegaard Harder | Denmark Women's | Center Attacking Midfield | 320 | 1.686 | 6.000 | 136.050 |
| good_surprise | 17 | Aaron Mooy | Australia | Left Defensive Midfield | 229 | 1.676 | 1.000 | 154.600 |
| good_surprise | 18 | Lindsey Michelle Heaps | United States Women's | Left Center Midfield | 236 | 1.672 | 1.000 | 183.000 |
| good_surprise | 19 | Neymar da Silva Santos Junior | Paris Saint-Germain | Left Wing | 2650 | 1.662 | 22.950 | 49.000 |
| good_surprise | 20 | Rafaelle Leone Carvalho Souza | Brazil Women's | Left Center Back | 278 | 1.660 | 2.000 | 119.550 |
| good_surprise | 21 | Hakim Ziyech | Morocco | Right Wing | 336 | 1.650 | 9.950 | 94.050 |
| good_surprise | 22 | Sherida Spitse | Netherlands Women's | Right Center Back | 671 | 1.648 | 4.000 | 163.500 |
| good_surprise | 23 | Jule Brand | Germany Women's | Right Wing | 414 | 1.644 | 8.000 | 137.000 |
| good_surprise | 24 | Sofiane Boufal | Morocco | Left Wing | 228 | 1.641 | 14.000 | 102.200 |
| good_surprise | 25 | Christoph Baumgartner | Austria | Center Attacking Midfield | 254 | 1.640 | 5.000 | 151.100 |
| tail_rarity | 1 | Ruslan Malinovskiy | Ukraine | Left Wing | 220 | 0.332 | 1.000 | 21.050 |
| tail_rarity | 2 | Rafael Alexandre Conceição Leão | Portugal | Left Wing | 204 | 0.330 | 1.000 | 58.650 |
| tail_rarity | 3 | Clara Mateo | France Women's | Left Center Midfield | 278 | 0.308 | 2.000 | 22.000 |
| tail_rarity | 4 | Mary Boio Fowler | Australia Women's | Left Center Forward | 204 | 0.307 | 1.900 | 72.600 |
| tail_rarity | 5 | Jeremy Doku | Belgium | Left Wing | 332 | 0.302 | 2.000 | 40.150 |
| tail_rarity | 6 | Jule Brand | Germany Women's | Right Wing | 414 | 0.300 | 2.000 | 48.000 |
| tail_rarity | 7 | Kerolin Nicoli Israel Ferraz | Brazil Women's | Right Defensive Midfield | 208 | 0.298 | 4.000 | 58.000 |
| tail_rarity | 8 | Pernille Mosegaard Harder | Denmark Women's | Center Attacking Midfield | 320 | 0.297 | 4.000 | 43.150 |
| tail_rarity | 9 | Victor Okoh Boniface | Bayer Leverkusen | Center Forward | 560 | 0.296 | 6.000 | 38.100 |
| tail_rarity | 10 | Romain Saïss | Morocco | Left Center Back | 204 | 0.295 | 2.900 | 262.700 |
| tail_rarity | 11 | Ana Borges | Portugal Women's | Right Back | 326 | 0.291 | 3.000 | 52.100 |
| tail_rarity | 12 | Christoph Baumgartner | Austria | Center Attacking Midfield | 254 | 0.289 | 4.000 | 75.000 |
| tail_rarity | 13 | Robert Lewandowski | Poland | Left Center Forward | 241 | 0.287 | 4.950 | 66.050 |
| tail_rarity | 14 | Katie Bowen | New Zealand Women's | Left Center Back | 255 | 0.287 | 9.000 | 46.450 |
| tail_rarity | 15 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | Center Attacking Midfield | 279 | 0.285 | 3.000 | 104.150 |
| tail_rarity | 16 | Caitlin Jade Foord | Australia Women's | Left Midfield | 212 | 0.284 | 5.950 | 92.100 |
| tail_rarity | 17 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | Right Center Forward | 8537 | 0.283 | 17.000 | 37.000 |
| tail_rarity | 18 | Delphine Cascarino | France Women's | Right Wing | 347 | 0.277 | 9.000 | 109.200 |
| tail_rarity | 19 | Vladimír Coufal | Czech Republic | Right Back | 406 | 0.277 | 8.000 | 92.000 |
| tail_rarity | 20 | Cody Mathès Gakpo | Netherlands | Left Wing | 323 | 0.276 | 8.000 | 120.450 |
| tail_rarity | 21 | Harry Kane | England | Center Forward | 480 | 0.276 | 11.000 | 93.200 |
| tail_rarity | 22 | Lamine Yamal Nasraoui Ebana | Spain | Right Wing | 296 | 0.275 | 9.950 | 178.250 |
| tail_rarity | 23 | Breel-Donald Embolo | Switzerland | Center Forward | 318 | 0.275 | 12.900 | 90.050 |
| tail_rarity | 24 | Giorgi Kochorashvili | Georgia | Left Defensive Midfield | 226 | 0.274 | 5.000 | 86.050 |
| tail_rarity | 25 | Piotr Zieliński | Poland | Right Center Midfield | 508 | 0.273 | 21.000 | 53.000 |
| c51_tail_rarity | 1 | Ruslan Malinovskiy | Ukraine | Left Wing | 220 | 0.165 | 1.000 | 15.050 |
| c51_tail_rarity | 2 | Rafael Alexandre Conceição Leão | Portugal | Left Wing | 204 | 0.159 | 1.000 | 22.100 |
| c51_tail_rarity | 3 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | Right Center Forward | 8537 | 0.157 | 2.000 | 8.000 |
| c51_tail_rarity | 4 | Jeremy Doku | Belgium | Left Wing | 332 | 0.151 | 2.000 | 21.000 |
| c51_tail_rarity | 5 | Leandro Trossard | Belgium | Right Wing | 207 | 0.145 | 3.000 | 31.050 |
| c51_tail_rarity | 6 | Lamine Yamal Nasraoui Ebana | Spain | Right Wing | 296 | 0.144 | 5.000 | 20.000 |
| c51_tail_rarity | 7 | Neymar da Silva Santos Junior | Paris Saint-Germain | Left Wing | 2650 | 0.143 | 5.000 | 22.000 |
| c51_tail_rarity | 8 | Jamal Musiala | Germany | Left Wing | 558 | 0.138 | 3.000 | 63.000 |
| c51_tail_rarity | 9 | Kevin De Bruyne | Belgium | Center Attacking Midfield | 578 | 0.136 | 4.000 | 42.000 |
| c51_tail_rarity | 10 | Vinícius José Paixão de Oliveira Júnior | Brazil | Left Wing | 213 | 0.135 | 7.000 | 36.000 |
| c51_tail_rarity | 11 | Cody Mathès Gakpo | Netherlands | Left Wing | 323 | 0.135 | 3.000 | 95.050 |
| c51_tail_rarity | 12 | Ramona Bachmann | Switzerland Women's | Center Forward | 330 | 0.133 | 5.000 | 71.300 |
| c51_tail_rarity | 13 | Ángel Fabián Di María Hernández | Paris Saint-Germain | Right Wing | 1337 | 0.132 | 5.950 | 40.000 |
| c51_tail_rarity | 14 | Pernille Mosegaard Harder | Denmark Women's | Center Attacking Midfield | 320 | 0.132 | 3.000 | 97.250 |
| c51_tail_rarity | 15 | Ousmane Dembélé | Barcelona | Right Wing | 1761 | 0.131 | 10.000 | 39.000 |
| c51_tail_rarity | 16 | Kylian Mbappé Lottin | Paris Saint-Germain | Center Forward | 3542 | 0.130 | 13.950 | 39.000 |
| c51_tail_rarity | 17 | Jule Brand | Germany Women's | Right Wing | 414 | 0.128 | 5.000 | 72.400 |
| c51_tail_rarity | 18 | Andrija Živković | Serbia | Right Wing Back | 229 | 0.128 | 6.000 | 124.300 |
| c51_tail_rarity | 19 | Florian Wirtz | Bayer Leverkusen | Left Attacking Midfield | 2235 | 0.127 | 15.000 | 46.050 |
| c51_tail_rarity | 20 | Delphine Cascarino | France Women's | Right Wing | 347 | 0.126 | 7.900 | 109.200 |
| c51_tail_rarity | 21 | Leroy Sané | Germany | Right Wing | 399 | 0.126 | 4.950 | 122.100 |
| c51_tail_rarity | 22 | Aleksandr Golovin | Russia | Left Attacking Midfield | 225 | 0.124 | 12.950 | 109.100 |
| c51_tail_rarity | 23 | Jéssica da Silva | Portugal Women's | Right Center Forward | 226 | 0.124 | 2.950 | 174.350 |
| c51_tail_rarity | 24 | Victor Okoh Boniface | Bayer Leverkusen | Center Forward | 560 | 0.124 | 11.000 | 80.100 |
| c51_tail_rarity | 25 | Clara Mateo | France Women's | Left Center Midfield | 278 | 0.123 | 3.000 | 131.550 |
| unxpass | 1 | Ruslan Malinovskiy | Ukraine | Left Wing | 220 | 0.273 | 2.000 | 54.000 |
| unxpass | 2 | Lindsey Michelle Heaps | United States Women's | Left Center Midfield | 236 | 0.271 | 1.000 | 88.200 |
| unxpass | 3 | Ana Borges | Portugal Women's | Right Back | 326 | 0.268 | 2.000 | 60.050 |
| unxpass | 4 | Katie Bowen | New Zealand Women's | Left Center Back | 255 | 0.267 | 2.000 | 152.000 |
| unxpass | 5 | Mary Boio Fowler | Australia Women's | Left Center Forward | 204 | 0.267 | 1.000 | 166.400 |
| unxpass | 6 | Giorgi Kochorashvili | Georgia | Left Defensive Midfield | 226 | 0.262 | 3.000 | 60.050 |
| unxpass | 7 | Toni Kroos | Germany | Left Defensive Midfield | 1118 | 0.258 | 4.950 | 53.150 |
| unxpass | 8 | Aaron Mooy | Australia | Left Defensive Midfield | 229 | 0.257 | 2.000 | 62.600 |
| unxpass | 9 | Hakim Ziyech | Morocco | Right Wing | 336 | 0.255 | 4.000 | 51.000 |
| unxpass | 10 | Clara Mateo | France Women's | Left Center Midfield | 278 | 0.252 | 4.000 | 167.000 |
| unxpass | 11 | Leonardo Bonucci | Italy | Right Center Back | 562 | 0.251 | 5.000 | 64.000 |
| unxpass | 12 | Rafaelle Leone Carvalho Souza | Brazil Women's | Left Center Back | 278 | 0.251 | 2.000 | 57.550 |
| unxpass | 13 | Ondrej Celustka | Czech Republic | Right Center Back | 210 | 0.251 | 1.900 | 110.500 |
| unxpass | 14 | Romain Saïss | Morocco | Left Center Back | 204 | 0.251 | 2.000 | 406.800 |
| unxpass | 15 | Woo-Young Jung | South Korea | Right Defensive Midfield | 215 | 0.251 | 1.850 | 69.300 |
| unxpass | 16 | Vanessa Brigitte Gilles | Canada Women's | Left Center Back | 264 | 0.250 | 2.000 | 103.000 |
| unxpass | 17 | Luka Modrić | Croatia | Right Center Midfield | 1281 | 0.248 | 9.950 | 64.000 |
| unxpass | 18 | Natalia Kuikka | WNT Finland | Right Center Back | 364 | 0.246 | 8.000 | 117.500 |
| unxpass | 19 | Daley Blind | Netherlands | Left Center Back | 627 | 0.244 | 16.000 | 49.000 |
| unxpass | 20 | Sherida Spitse | Netherlands Women's | Right Center Back | 671 | 0.242 | 6.950 | 112.200 |
| unxpass | 21 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | Right Center Forward | 8537 | 0.240 | 20.950 | 51.050 |
| unxpass | 22 | Neymar da Silva Santos Junior | Paris Saint-Germain | Left Wing | 2650 | 0.240 | 22.000 | 57.100 |
| unxpass | 23 | Thomas Teye Partey | Ghana | Right Defensive Midfield | 208 | 0.239 | 19.000 | 68.700 |
| unxpass | 24 | Fabian Lukas Schär | Switzerland | Right Center Back | 509 | 0.239 | 4.950 | 150.050 |
| unxpass | 25 | Kadeisha Buchanan | Canada Women's | Right Center Back | 210 | 0.238 | 5.000 | 132.000 |


## Top and bottom players by gender and position (headline measures)

Within-group percentiles; players with at least 200 in-scope decisions, position and team = most frequent.

| measure | gender | group | players | side | rank | player | team | decisions | score | percentile |
|---|---|---|---|---|---|---|---|---|---|---|
| good_surprise | men | AM | 20 | top | 1 | Christoph Baumgartner | Austria | 254 | 1.640 | 1.000 |
| good_surprise | men | AM | 20 | top | 2 | Bruno Miguel Borges Fernandes | Portugal | 645 | 1.604 | 0.950 |
| good_surprise | men | AM | 20 | top | 3 | Kevin De Bruyne | Belgium | 578 | 1.593 | 0.900 |
| good_surprise | men | AM | 20 | top | 4 | Florian Wirtz | Bayer Leverkusen | 2235 | 1.539 | 0.850 |
| good_surprise | men | AM | 20 | top | 5 | Xherdan Shaqiri | Switzerland | 343 | 1.515 | 0.800 |
| good_surprise | men | AM | 20 | bottom | 1 | Adam Hložek | Bayer Leverkusen | 271 | 1.114 | 0.050 |
| good_surprise | men | AM | 20 | bottom | 2 | Thomas Müller | Germany | 281 | 1.175 | 0.100 |
| good_surprise | men | AM | 20 | bottom | 3 | Amine Adli | Bayer Leverkusen | 447 | 1.199 | 0.150 |
| good_surprise | men | CB | 118 | top | 1 | Romain Saïss | Morocco | 204 | 1.633 | 1.000 |
| good_surprise | men | CB | 118 | top | 2 | Abdelkarim Hassan Al Haj Fadlalla | Qatar | 207 | 1.613 | 0.992 |
| good_surprise | men | CB | 118 | top | 3 | Leonardo Bonucci | Italy | 562 | 1.589 | 0.983 |
| good_surprise | men | CB | 118 | top | 4 | Fabian Lukas Schär | Switzerland | 509 | 1.584 | 0.975 |
| good_surprise | men | CB | 118 | top | 5 | David Olatukunbo Alaba | Austria | 341 | 1.534 | 0.966 |
| good_surprise | men | CB | 118 | bottom | 1 | Kamil Glik | Poland | 301 | 0.739 | 0.008 |
| good_surprise | men | CB | 118 | bottom | 2 | Merih Demiral | Turkey | 271 | 0.799 | 0.017 |
| good_surprise | men | CB | 118 | bottom | 3 | Ronald Federico Araújo da Silva | Barcelona | 938 | 0.879 | 0.025 |
| good_surprise | men | CM | 29 | top | 1 | Luka Modrić | Croatia | 1281 | 1.764 | 1.000 |
| good_surprise | men | CM | 29 | top | 2 | Piotr Zieliński | Poland | 508 | 1.724 | 0.966 |
| good_surprise | men | CM | 29 | top | 3 | Mateo Kovačić | Croatia | 1033 | 1.536 | 0.931 |
| good_surprise | men | CM | 29 | top | 4 | Aaron Ramsey | Wales | 449 | 1.536 | 0.897 |
| good_surprise | men | CM | 29 | top | 5 | Mykola Shaparenko | Ukraine | 430 | 1.460 | 0.862 |
| good_surprise | men | CM | 29 | bottom | 1 | Carlos Soler Barragán | Paris Saint-Germain | 696 | 0.964 | 0.034 |
| good_surprise | men | CM | 29 | bottom | 2 | Moriba Kourouma Kourouma | Barcelona | 430 | 0.977 | 0.069 |
| good_surprise | men | CM | 29 | bottom | 3 | Georginio Wijnaldum | Paris Saint-Germain | 617 | 1.105 | 0.103 |
| good_surprise | men | DM | 66 | top | 1 | Giorgi Kochorashvili | Georgia | 226 | 1.744 | 1.000 |
| good_surprise | men | DM | 66 | top | 2 | Thomas Teye Partey | Ghana | 208 | 1.713 | 0.985 |
| good_surprise | men | DM | 66 | top | 3 | Toni Kroos | Germany | 1118 | 1.688 | 0.970 |
| good_surprise | men | DM | 66 | top | 4 | Aaron Mooy | Australia | 229 | 1.676 | 0.955 |
| good_surprise | men | DM | 66 | top | 5 | Federico Santiago Valverde Dipetta | Uruguay | 275 | 1.633 | 0.939 |
| good_surprise | men | DM | 66 | bottom | 1 | Marten de Roon | Netherlands | 259 | 0.906 | 0.015 |
| good_surprise | men | DM | 66 | bottom | 2 | Morten Hjulmand | Denmark | 222 | 0.944 | 0.030 |
| good_surprise | men | DM | 66 | bottom | 3 | Leander Dendoncker | Belgium | 222 | 0.960 | 0.045 |
| good_surprise | men | FB | 72 | top | 1 | Antonee Robinson | United States | 217 | 1.627 | 1.000 |
| good_surprise | men | FB | 72 | top | 2 | Vladimír Coufal | Czech Republic | 406 | 1.594 | 0.986 |
| good_surprise | men | FB | 72 | top | 3 | Marcos Javier Acuña | Argentina | 226 | 1.542 | 0.972 |
| good_surprise | men | FB | 72 | top | 4 | Thomas Meunier | Belgium | 332 | 1.490 | 0.958 |
| good_surprise | men | FB | 72 | top | 5 | Pervis Josué Estupiñán Tenorio | Ecuador | 260 | 1.482 | 0.944 |
| good_surprise | men | FB | 72 | bottom | 1 | Vitalii Mykolenko | Ukraine | 206 | 0.831 | 0.014 |
| good_surprise | men | FB | 72 | bottom | 2 | Emerson Palmieri dos Santos | Italy | 211 | 0.873 | 0.028 |
| good_surprise | men | FB | 72 | bottom | 3 | Nélson Cabral Semedo | Portugal | 211 | 0.929 | 0.042 |
| good_surprise | men | FW | 16 | top | 1 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | 8537 | 1.694 | 1.000 |
| good_surprise | men | FW | 16 | top | 2 | Harry Kane | England | 480 | 1.553 | 0.938 |
| good_surprise | men | FW | 16 | top | 3 | Arda Güler | Turkey | 240 | 1.426 | 0.875 |
| good_surprise | men | FW | 16 | top | 4 | Kylian Mbappé Lottin | Paris Saint-Germain | 3542 | 1.399 | 0.812 |
| good_surprise | men | FW | 16 | top | 5 | Breel-Donald Embolo | Switzerland | 318 | 1.397 | 0.750 |
| good_surprise | men | FW | 16 | bottom | 1 | Cristiano Ronaldo dos Santos Aveiro | Portugal | 391 | 1.017 | 0.062 |
| good_surprise | men | FW | 16 | bottom | 2 | Álvaro Borja Morata Martín | Spain | 356 | 1.023 | 0.125 |
| good_surprise | men | FW | 16 | bottom | 3 | Patrik Schick | Bayer Leverkusen | 447 | 1.025 | 0.188 |
| good_surprise | men | GK | 11 | top | 1 | Kasper Schmeichel | Denmark | 247 | 1.549 | 1.000 |
| good_surprise | men | GK | 11 | top | 2 | Marc-André ter Stegen | Barcelona | 845 | 1.505 | 0.909 |
| good_surprise | men | GK | 11 | top | 3 | Manuel Neuer | Germany | 371 | 1.410 | 0.818 |
| good_surprise | men | GK | 11 | top | 4 | Yassine Bounou | Morocco | 205 | 1.367 | 0.727 |
| good_surprise | men | GK | 11 | top | 5 | Thibaut Courtois | Belgium | 225 | 1.310 | 0.636 |
| good_surprise | men | GK | 11 | bottom | 1 | Gianluigi Donnarumma | Paris Saint-Germain | 867 | 1.063 | 0.091 |
| good_surprise | men | GK | 11 | bottom | 2 | Unai Simón Mendibil | Spain | 522 | 1.118 | 0.182 |
| good_surprise | men | GK | 11 | bottom | 3 | Lukáš Hrádecký | Bayer Leverkusen | 641 | 1.133 | 0.273 |
| good_surprise | men | W | 37 | top | 1 | Ruslan Malinovskiy | Ukraine | 220 | 2.046 | 1.000 |
| good_surprise | men | W | 37 | top | 2 | Rafael Alexandre Conceição Leão | Portugal | 204 | 1.804 | 0.973 |
| good_surprise | men | W | 37 | top | 3 | Jeremy Doku | Belgium | 332 | 1.722 | 0.946 |
| good_surprise | men | W | 37 | top | 4 | Neymar da Silva Santos Junior | Paris Saint-Germain | 2650 | 1.662 | 0.919 |
| good_surprise | men | W | 37 | top | 5 | Hakim Ziyech | Morocco | 336 | 1.650 | 0.892 |
| good_surprise | men | W | 37 | bottom | 1 | Ferrán Torres García | Spain | 306 | 1.053 | 0.027 |
| good_surprise | men | W | 37 | bottom | 2 | Pablo Sarabia García | Spain | 251 | 1.066 | 0.054 |
| good_surprise | men | W | 37 | bottom | 3 | Martin Braithwaite Christensen | Barcelona | 566 | 1.115 | 0.081 |
| good_surprise | women | AM | 6 | top | 1 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | 279 | 1.817 | 1.000 |
| good_surprise | women | AM | 6 | top | 2 | Pernille Mosegaard Harder | Denmark Women's | 320 | 1.686 | 0.833 |
| good_surprise | women | AM | 6 | top | 3 | Lauren James | England Women's | 442 | 1.586 | 0.667 |
| good_surprise | women | AM | 6 | top | 4 | Kosovare Asllani | Sweden Women's | 487 | 1.344 | 0.500 |
| good_surprise | women | AM | 6 | top | 5 | Leicy Maria Santos Herrera | Colombia Women's | 237 | 1.311 | 0.333 |
| good_surprise | women | AM | 6 | bottom | 1 | Ella Toone | England Women's | 498 | 1.204 | 0.167 |
| good_surprise | women | AM | 6 | bottom | 2 | Leicy Maria Santos Herrera | Colombia Women's | 237 | 1.311 | 0.333 |
| good_surprise | women | AM | 6 | bottom | 3 | Kosovare Asllani | Sweden Women's | 487 | 1.344 | 0.500 |
| good_surprise | women | CB | 52 | top | 1 | Katie Bowen | New Zealand Women's | 255 | 1.818 | 1.000 |
| good_surprise | women | CB | 52 | top | 2 | Rafaelle Leone Carvalho Souza | Brazil Women's | 278 | 1.660 | 0.981 |
| good_surprise | women | CB | 52 | top | 3 | Sherida Spitse | Netherlands Women's | 671 | 1.648 | 0.962 |
| good_surprise | women | CB | 52 | top | 4 | Naomi Haile Girma | United States Women's | 302 | 1.616 | 0.942 |
| good_surprise | women | CB | 52 | top | 5 | Kadeisha Buchanan | Canada Women's | 210 | 1.527 | 0.923 |
| good_surprise | women | CB | 52 | bottom | 1 | Amanda Ilestedt | Sweden Women's | 734 | 1.008 | 0.019 |
| good_surprise | women | CB | 52 | bottom | 2 | Laia Aleixandri López | Spain Women's | 516 | 1.035 | 0.038 |
| good_surprise | women | CB | 52 | bottom | 3 | Noelle Maritz | Switzerland Women's | 614 | 1.044 | 0.058 |
| good_surprise | women | CM | 16 | top | 1 | Clara Mateo | France Women's | 278 | 1.693 | 1.000 |
| good_surprise | women | CM | 16 | top | 2 | Lindsey Michelle Heaps | United States Women's | 236 | 1.672 | 0.938 |
| good_surprise | women | CM | 16 | top | 3 | Andreia Alexandra Norton | Portugal Women's | 314 | 1.523 | 0.875 |
| good_surprise | women | CM | 16 | top | 4 | Jennifer Hermoso Fuentes | Spain Women's | 493 | 1.444 | 0.812 |
| good_surprise | women | CM | 16 | top | 5 | Aitana Bonmati Conca | Spain Women's | 1293 | 1.420 | 0.750 |
| good_surprise | women | CM | 16 | bottom | 1 | Coumba Sow | Switzerland Women's | 232 | 0.808 | 0.062 |
| good_surprise | women | CM | 16 | bottom | 2 | Lina Magull | Germany Women's | 240 | 0.979 | 0.125 |
| good_surprise | women | CM | 16 | bottom | 3 | Jill Roord | Netherlands Women's | 510 | 1.000 | 0.188 |
| good_surprise | women | DM | 25 | top | 1 | Kerolin Nicoli Israel Ferraz | Brazil Women's | 208 | 1.941 | 1.000 |
| good_surprise | women | DM | 25 | top | 2 | Katrina Gorry | Australia Women's | 385 | 1.559 | 0.960 |
| good_surprise | women | DM | 25 | top | 3 | Lena Oberdorf | Germany Women's | 313 | 1.516 | 0.920 |
| good_surprise | women | DM | 25 | top | 4 | Vilde Bøe Risa | Norway Women's | 359 | 1.485 | 0.880 |
| good_surprise | women | DM | 25 | top | 5 | Ingrid Syrstad Engen | Norway Women's | 382 | 1.463 | 0.840 |
| good_surprise | women | DM | 25 | bottom | 1 | Charlotte Bilbault | France Women's | 214 | 1.135 | 0.040 |
| good_surprise | women | DM | 25 | bottom | 2 | Sjoeke Nüsken | Germany Women's | 217 | 1.139 | 0.080 |
| good_surprise | women | DM | 25 | bottom | 3 | Georgia Stanway | England Women's | 838 | 1.151 | 0.120 |
| good_surprise | women | FB | 37 | top | 1 | Ana Borges | Portugal Women's | 326 | 1.865 | 1.000 |
| good_surprise | women | FB | 37 | top | 2 | Eve Perisset | France Women's | 459 | 1.528 | 0.973 |
| good_surprise | women | FB | 37 | top | 3 | Joana Filipa Gaspar Silva Marchão | Portugal Women's | 251 | 1.517 | 0.946 |
| good_surprise | women | FB | 37 | top | 4 | Catarina Amado | Portugal Women's | 257 | 1.514 | 0.919 |
| good_surprise | women | FB | 37 | top | 5 | Selma Bacha | France Women's | 606 | 1.512 | 0.892 |
| good_surprise | women | FB | 37 | bottom | 1 | Leila Ouahabi El Ouahabi | Spain Women's | 285 | 0.822 | 0.027 |
| good_surprise | women | FB | 37 | bottom | 2 | Oihane Hernández Zurbano | Spain Women's | 270 | 1.021 | 0.054 |
| good_surprise | women | FB | 37 | bottom | 3 | Lucia Di Guglielmo | Italy Women's | 362 | 1.036 | 0.081 |
| good_surprise | women | FW | 10 | top | 1 | Mary Boio Fowler | Australia Women's | 204 | 1.740 | 1.000 |
| good_surprise | women | FW | 10 | top | 2 | Tessa Wullaert | Belgium Women's | 265 | 1.570 | 0.900 |
| good_surprise | women | FW | 10 | top | 3 | Diana Micaela Abreu de Sousa e Silva | Portugal Women's | 217 | 1.404 | 0.800 |
| good_surprise | women | FW | 10 | top | 4 | Lieke Martens | Netherlands Women's | 402 | 1.401 | 0.700 |
| good_surprise | women | FW | 10 | top | 5 | Ramona Bachmann | Switzerland Women's | 330 | 1.322 | 0.600 |
| good_surprise | women | FW | 10 | bottom | 1 | Esther Gonzalez Rodríguez | Spain Women's | 272 | 0.745 | 0.100 |
| good_surprise | women | FW | 10 | bottom | 2 | Emma Stina Blackstenius | Sweden Women's | 308 | 0.850 | 0.200 |
| good_surprise | women | FW | 10 | bottom | 3 | Alessia Russo | England Women's | 390 | 0.998 | 0.300 |
| good_surprise | women | GK | 7 | top | 1 | Ann-Katrin Berger | Germany Women's | 216 | 1.532 | 1.000 |
| good_surprise | women | GK | 7 | top | 2 | Daphne van Domselaar | Netherlands Women's | 259 | 1.419 | 0.857 |
| good_surprise | women | GK | 7 | top | 3 | Merle Frohms | Germany Women's | 237 | 1.181 | 0.714 |
| good_surprise | women | GK | 7 | top | 4 | Mary Alexandra Earps | England Women's | 209 | 1.159 | 0.571 |
| good_surprise | women | GK | 7 | top | 5 | Pauline Peyraud Magnin | France Women's | 271 | 1.026 | 0.429 |
| good_surprise | women | GK | 7 | bottom | 1 | Gaëlle Thalmann | Switzerland Women's | 207 | 0.871 | 0.143 |
| good_surprise | women | GK | 7 | bottom | 2 | Laura Giuliani | Italy Women's | 201 | 0.946 | 0.286 |
| good_surprise | women | GK | 7 | bottom | 3 | Pauline Peyraud Magnin | France Women's | 271 | 1.026 | 0.429 |
| good_surprise | women | W | 17 | top | 1 | Jule Brand | Germany Women's | 414 | 1.644 | 1.000 |
| good_surprise | women | W | 17 | top | 2 | Linda Lizeth Caicedo Alegría | Colombia Women's | 226 | 1.578 | 0.941 |
| good_surprise | women | W | 17 | top | 3 | Johanna Rytting-Kaneryd | Sweden Women's | 470 | 1.519 | 0.882 |
| good_surprise | women | W | 17 | top | 4 | Delphine Cascarino | France Women's | 347 | 1.509 | 0.824 |
| good_surprise | women | W | 17 | top | 5 | María Francesca Caldentey Oliver | Spain Women's | 1179 | 1.497 | 0.765 |
| good_surprise | women | W | 17 | bottom | 1 | Salma Paralluelo Ayingono | Spain Women's | 335 | 1.081 | 0.059 |
| good_surprise | women | W | 17 | bottom | 2 | Claudia Pina Medina | Spain Women's | 253 | 1.157 | 0.118 |
| good_surprise | women | W | 17 | bottom | 3 | Bethany Mead | England Women's | 251 | 1.213 | 0.176 |
| good_rarity | men | AM | 20 | top | 1 | Christoph Baumgartner | Austria | 254 | 0.247 | 1.000 |
| good_rarity | men | AM | 20 | top | 2 | Kevin De Bruyne | Belgium | 578 | 0.226 | 0.950 |
| good_rarity | men | AM | 20 | top | 3 | Florian Wirtz | Bayer Leverkusen | 2235 | 0.225 | 0.900 |
| good_rarity | men | AM | 20 | top | 4 | Xherdan Shaqiri | Switzerland | 343 | 0.222 | 0.850 |
| good_rarity | men | AM | 20 | top | 5 | Bruno Miguel Borges Fernandes | Portugal | 645 | 0.216 | 0.800 |
| good_rarity | men | AM | 20 | bottom | 1 | Thomas Müller | Germany | 281 | 0.182 | 0.050 |
| good_rarity | men | AM | 20 | bottom | 2 | Amine Adli | Bayer Leverkusen | 447 | 0.182 | 0.100 |
| good_rarity | men | AM | 20 | bottom | 3 | Christian Dannemann Eriksen | Denmark | 473 | 0.183 | 0.150 |
| good_rarity | men | CB | 118 | top | 1 | Romain Saïss | Morocco | 204 | 0.235 | 1.000 |
| good_rarity | men | CB | 118 | top | 2 | Victor Nilsson Lindelöf | Sweden | 266 | 0.216 | 0.992 |
| good_rarity | men | CB | 118 | top | 3 | Fabian Lukas Schär | Switzerland | 509 | 0.212 | 0.983 |
| good_rarity | men | CB | 118 | top | 4 | Steven de Sousa Vitoria | Canada | 207 | 0.209 | 0.975 |
| good_rarity | men | CB | 118 | top | 5 | Abdelkarim Hassan Al Haj Fadlalla | Qatar | 207 | 0.206 | 0.966 |
| good_rarity | men | CB | 118 | bottom | 1 | Grant Campbell Hanley | Scotland | 272 | 0.123 | 0.008 |
| good_rarity | men | CB | 118 | bottom | 2 | Ronald Federico Araújo da Silva | Barcelona | 938 | 0.123 | 0.017 |
| good_rarity | men | CB | 118 | bottom | 3 | Kamil Glik | Poland | 301 | 0.125 | 0.025 |
| good_rarity | men | CM | 29 | top | 1 | Piotr Zieliński | Poland | 508 | 0.242 | 1.000 |
| good_rarity | men | CM | 29 | top | 2 | Luka Modrić | Croatia | 1281 | 0.230 | 0.966 |
| good_rarity | men | CM | 29 | top | 3 | Mateo Kovačić | Croatia | 1033 | 0.209 | 0.931 |
| good_rarity | men | CM | 29 | top | 4 | Tomáš Souček | Czech Republic | 355 | 0.204 | 0.897 |
| good_rarity | men | CM | 29 | top | 5 | Aaron Ramsey | Wales | 449 | 0.194 | 0.862 |
| good_rarity | men | CM | 29 | bottom | 1 | Carlos Soler Barragán | Paris Saint-Germain | 696 | 0.133 | 0.034 |
| good_rarity | men | CM | 29 | bottom | 2 | Moriba Kourouma Kourouma | Barcelona | 430 | 0.139 | 0.069 |
| good_rarity | men | CM | 29 | bottom | 3 | Jordan Brian Henderson | England | 291 | 0.145 | 0.103 |
| good_rarity | men | DM | 66 | top | 1 | Giorgi Kochorashvili | Georgia | 226 | 0.235 | 1.000 |
| good_rarity | men | DM | 66 | top | 2 | Toni Kroos | Germany | 1118 | 0.233 | 0.985 |
| good_rarity | men | DM | 66 | top | 3 | Aaron Mooy | Australia | 229 | 0.220 | 0.970 |
| good_rarity | men | DM | 66 | top | 4 | Sofyan Amrabat | Morocco | 354 | 0.215 | 0.955 |
| good_rarity | men | DM | 66 | top | 5 | Federico Santiago Valverde Dipetta | Uruguay | 275 | 0.214 | 0.939 |
| good_rarity | men | DM | 66 | bottom | 1 | Morten Hjulmand | Denmark | 222 | 0.122 | 0.015 |
| good_rarity | men | DM | 66 | bottom | 2 | Leander Dendoncker | Belgium | 222 | 0.125 | 0.030 |
| good_rarity | men | DM | 66 | bottom | 3 | Marten de Roon | Netherlands | 259 | 0.126 | 0.045 |
| good_rarity | men | FB | 72 | top | 1 | Alphonso Davies | Canada | 223 | 0.219 | 1.000 |
| good_rarity | men | FB | 72 | top | 2 | Antonee Robinson | United States | 217 | 0.218 | 0.986 |
| good_rarity | men | FB | 72 | top | 3 | Vladimír Coufal | Czech Republic | 406 | 0.212 | 0.972 |
| good_rarity | men | FB | 72 | top | 4 | Andrija Živković | Serbia | 229 | 0.208 | 0.958 |
| good_rarity | men | FB | 72 | top | 5 | Marcos Javier Acuña | Argentina | 226 | 0.207 | 0.944 |
| good_rarity | men | FB | 72 | bottom | 1 | Vitalii Mykolenko | Ukraine | 206 | 0.109 | 0.014 |
| good_rarity | men | FB | 72 | bottom | 2 | Emerson Palmieri dos Santos | Italy | 211 | 0.116 | 0.028 |
| good_rarity | men | FB | 72 | bottom | 3 | Borna Sosa | Croatia | 265 | 0.127 | 0.042 |
| good_rarity | men | FW | 16 | top | 1 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | 8537 | 0.258 | 1.000 |
| good_rarity | men | FW | 16 | top | 2 | Victor Okoh Boniface | Bayer Leverkusen | 560 | 0.247 | 0.938 |
| good_rarity | men | FW | 16 | top | 3 | Harry Kane | England | 480 | 0.237 | 0.875 |
| good_rarity | men | FW | 16 | top | 4 | Kylian Mbappé Lottin | Paris Saint-Germain | 3542 | 0.234 | 0.812 |
| good_rarity | men | FW | 16 | top | 5 | Arda Güler | Turkey | 240 | 0.215 | 0.750 |
| good_rarity | men | FW | 16 | bottom | 1 | Roland Sallai | Hungary | 203 | 0.174 | 0.062 |
| good_rarity | men | FW | 16 | bottom | 2 | Álvaro Borja Morata Martín | Spain | 356 | 0.178 | 0.125 |
| good_rarity | men | FW | 16 | bottom | 3 | Hugo Ekitike | Paris Saint-Germain | 369 | 0.190 | 0.188 |
| good_rarity | men | GK | 11 | top | 1 | Kasper Schmeichel | Denmark | 247 | 0.193 | 1.000 |
| good_rarity | men | GK | 11 | top | 2 | Marc-André ter Stegen | Barcelona | 845 | 0.190 | 0.909 |
| good_rarity | men | GK | 11 | top | 3 | Manuel Neuer | Germany | 371 | 0.164 | 0.818 |
| good_rarity | men | GK | 11 | top | 4 | Yassine Bounou | Morocco | 205 | 0.158 | 0.727 |
| good_rarity | men | GK | 11 | top | 5 | Thibaut Courtois | Belgium | 225 | 0.158 | 0.636 |
| good_rarity | men | GK | 11 | bottom | 1 | Gianluigi Donnarumma | Paris Saint-Germain | 867 | 0.125 | 0.091 |
| good_rarity | men | GK | 11 | bottom | 2 | Lukáš Hrádecký | Bayer Leverkusen | 641 | 0.138 | 0.182 |
| good_rarity | men | GK | 11 | bottom | 3 | Dominik Livaković | Croatia | 259 | 0.141 | 0.273 |
| good_rarity | men | W | 37 | top | 1 | Ruslan Malinovskiy | Ukraine | 220 | 0.290 | 1.000 |
| good_rarity | men | W | 37 | top | 2 | Rafael Alexandre Conceição Leão | Portugal | 204 | 0.274 | 0.973 |
| good_rarity | men | W | 37 | top | 3 | Jeremy Doku | Belgium | 332 | 0.269 | 0.946 |
| good_rarity | men | W | 37 | top | 4 | Lamine Yamal Nasraoui Ebana | Spain | 296 | 0.251 | 0.919 |
| good_rarity | men | W | 37 | top | 5 | Neymar da Silva Santos Junior | Paris Saint-Germain | 2650 | 0.244 | 0.892 |
| good_rarity | men | W | 37 | bottom | 1 | Pablo Sarabia García | Spain | 251 | 0.162 | 0.027 |
| good_rarity | men | W | 37 | bottom | 2 | Andrej Kramarić | Croatia | 442 | 0.164 | 0.054 |
| good_rarity | men | W | 37 | bottom | 3 | Francisco António Machado Mota de Castro Trincão | Barcelona | 506 | 0.167 | 0.081 |
| good_rarity | women | AM | 6 | top | 1 | Pernille Mosegaard Harder | Denmark Women's | 320 | 0.260 | 1.000 |
| good_rarity | women | AM | 6 | top | 2 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | 279 | 0.253 | 0.833 |
| good_rarity | women | AM | 6 | top | 3 | Lauren James | England Women's | 442 | 0.224 | 0.667 |
| good_rarity | women | AM | 6 | top | 4 | Kosovare Asllani | Sweden Women's | 487 | 0.204 | 0.500 |
| good_rarity | women | AM | 6 | top | 5 | Leicy Maria Santos Herrera | Colombia Women's | 237 | 0.182 | 0.333 |
| good_rarity | women | AM | 6 | bottom | 1 | Ella Toone | England Women's | 498 | 0.176 | 0.167 |
| good_rarity | women | AM | 6 | bottom | 2 | Leicy Maria Santos Herrera | Colombia Women's | 237 | 0.182 | 0.333 |
| good_rarity | women | AM | 6 | bottom | 3 | Kosovare Asllani | Sweden Women's | 487 | 0.204 | 0.500 |
| good_rarity | women | CB | 52 | top | 1 | Katie Bowen | New Zealand Women's | 255 | 0.235 | 1.000 |
| good_rarity | women | CB | 52 | top | 2 | Sherida Spitse | Netherlands Women's | 671 | 0.219 | 0.981 |
| good_rarity | women | CB | 52 | top | 3 | Rafaelle Leone Carvalho Souza | Brazil Women's | 278 | 0.218 | 0.962 |
| good_rarity | women | CB | 52 | top | 4 | Kadeisha Buchanan | Canada Women's | 210 | 0.216 | 0.942 |
| good_rarity | women | CB | 52 | top | 5 | Julie Beth Ertz | United States Women's | 243 | 0.209 | 0.923 |
| good_rarity | women | CB | 52 | bottom | 1 | Noelle Maritz | Switzerland Women's | 614 | 0.137 | 0.019 |
| good_rarity | women | CB | 52 | bottom | 2 | Laia Aleixandri López | Spain Women's | 516 | 0.138 | 0.038 |
| good_rarity | women | CB | 52 | bottom | 3 | María Méndez Fernández | Spain Women's | 295 | 0.138 | 0.058 |
| good_rarity | women | CM | 16 | top | 1 | Clara Mateo | France Women's | 278 | 0.250 | 1.000 |
| good_rarity | women | CM | 16 | top | 2 | Lindsey Michelle Heaps | United States Women's | 236 | 0.236 | 0.938 |
| good_rarity | women | CM | 16 | top | 3 | Jennifer Hermoso Fuentes | Spain Women's | 493 | 0.216 | 0.875 |
| good_rarity | women | CM | 16 | top | 4 | Aitana Bonmati Conca | Spain Women's | 1293 | 0.204 | 0.812 |
| good_rarity | women | CM | 16 | top | 5 | Andreia Alexandra Norton | Portugal Women's | 314 | 0.201 | 0.750 |
| good_rarity | women | CM | 16 | bottom | 1 | Coumba Sow | Switzerland Women's | 232 | 0.117 | 0.062 |
| good_rarity | women | CM | 16 | bottom | 2 | Lina Magull | Germany Women's | 240 | 0.141 | 0.125 |
| good_rarity | women | CM | 16 | bottom | 3 | Jill Roord | Netherlands Women's | 510 | 0.143 | 0.188 |
| good_rarity | women | DM | 25 | top | 1 | Kerolin Nicoli Israel Ferraz | Brazil Women's | 208 | 0.268 | 1.000 |
| good_rarity | women | DM | 25 | top | 2 | Katrina Gorry | Australia Women's | 385 | 0.202 | 0.960 |
| good_rarity | women | DM | 25 | top | 3 | Ingrid Filippa Angeldal | Sweden Women's | 709 | 0.196 | 0.920 |
| good_rarity | women | DM | 25 | top | 4 | Sara Däbritz | Germany Women's | 433 | 0.194 | 0.880 |
| good_rarity | women | DM | 25 | top | 5 | Frida Maanum | Norway Women's | 361 | 0.194 | 0.840 |
| good_rarity | women | DM | 25 | bottom | 1 | Charlotte Bilbault | France Women's | 214 | 0.147 | 0.040 |
| good_rarity | women | DM | 25 | bottom | 2 | Karen Holmgaard | Denmark Women's | 281 | 0.161 | 0.080 |
| good_rarity | women | DM | 25 | bottom | 3 | Georgia Stanway | England Women's | 838 | 0.162 | 0.120 |
| good_rarity | women | FB | 37 | top | 1 | Ana Borges | Portugal Women's | 326 | 0.247 | 1.000 |
| good_rarity | women | FB | 37 | top | 2 | Janni Thomsen | Denmark Women's | 294 | 0.212 | 0.973 |
| good_rarity | women | FB | 37 | top | 3 | Selma Bacha | France Women's | 606 | 0.203 | 0.946 |
| good_rarity | women | FB | 37 | top | 4 | Victoria Pelova | Netherlands Women's | 499 | 0.194 | 0.919 |
| good_rarity | women | FB | 37 | top | 5 | Eve Perisset | France Women's | 459 | 0.193 | 0.892 |
| good_rarity | women | FB | 37 | bottom | 1 | Leila Ouahabi El Ouahabi | Spain Women's | 285 | 0.110 | 0.027 |
| good_rarity | women | FB | 37 | bottom | 2 | Oihane Hernández Zurbano | Spain Women's | 270 | 0.135 | 0.054 |
| good_rarity | women | FB | 37 | bottom | 3 | Olga  Carmona García | Spain Women's | 799 | 0.142 | 0.081 |
| good_rarity | women | FW | 10 | top | 1 | Mary Boio Fowler | Australia Women's | 204 | 0.265 | 1.000 |
| good_rarity | women | FW | 10 | top | 2 | Tessa Wullaert | Belgium Women's | 265 | 0.220 | 0.900 |
| good_rarity | women | FW | 10 | top | 3 | Jéssica da Silva | Portugal Women's | 226 | 0.209 | 0.800 |
| good_rarity | women | FW | 10 | top | 4 | Lieke Martens | Netherlands Women's | 402 | 0.208 | 0.700 |
| good_rarity | women | FW | 10 | top | 5 | Diana Micaela Abreu de Sousa e Silva | Portugal Women's | 217 | 0.197 | 0.600 |
| good_rarity | women | FW | 10 | bottom | 1 | Ana-Maria Crnogorčević | Switzerland Women's | 295 | 0.175 | 0.100 |
| good_rarity | women | FW | 10 | bottom | 2 | Alessia Russo | England Women's | 390 | 0.184 | 0.200 |
| good_rarity | women | FW | 10 | bottom | 3 | Emma Stina Blackstenius | Sweden Women's | 308 | 0.187 | 0.300 |
| good_rarity | women | GK | 7 | top | 1 | Daphne van Domselaar | Netherlands Women's | 259 | 0.176 | 1.000 |
| good_rarity | women | GK | 7 | top | 2 | Ann-Katrin Berger | Germany Women's | 216 | 0.175 | 0.857 |
| good_rarity | women | GK | 7 | top | 3 | Merle Frohms | Germany Women's | 237 | 0.140 | 0.714 |
| good_rarity | women | GK | 7 | top | 4 | Mary Alexandra Earps | England Women's | 209 | 0.136 | 0.571 |
| good_rarity | women | GK | 7 | top | 5 | Pauline Peyraud Magnin | France Women's | 271 | 0.136 | 0.429 |
| good_rarity | women | GK | 7 | bottom | 1 | Gaëlle Thalmann | Switzerland Women's | 207 | 0.108 | 0.143 |
| good_rarity | women | GK | 7 | bottom | 2 | Laura Giuliani | Italy Women's | 201 | 0.116 | 0.286 |
| good_rarity | women | GK | 7 | bottom | 3 | Pauline Peyraud Magnin | France Women's | 271 | 0.136 | 0.429 |
| good_rarity | women | W | 17 | top | 1 | Caitlin Jade Foord | Australia Women's | 212 | 0.242 | 1.000 |
| good_rarity | women | W | 17 | top | 2 | Jule Brand | Germany Women's | 414 | 0.239 | 0.941 |
| good_rarity | women | W | 17 | top | 3 | Chloe Kelly | England Women's | 316 | 0.238 | 0.882 |
| good_rarity | women | W | 17 | top | 4 | Delphine Cascarino | France Women's | 347 | 0.236 | 0.824 |
| good_rarity | women | W | 17 | top | 5 | Klara Bühl | Germany Women's | 557 | 0.232 | 0.765 |
| good_rarity | women | W | 17 | bottom | 1 | Claudia Pina Medina | Spain Women's | 253 | 0.182 | 0.059 |
| good_rarity | women | W | 17 | bottom | 2 | Salma Paralluelo Ayingono | Spain Women's | 335 | 0.183 | 0.118 |
| good_rarity | women | W | 17 | bottom | 3 | Svenja Huth | Germany Women's | 421 | 0.189 | 0.176 |
| c51_tail_rarity | men | AM | 20 | top | 1 | Kevin De Bruyne | Belgium | 578 | 0.136 | 1.000 |
| c51_tail_rarity | men | AM | 20 | top | 2 | Florian Wirtz | Bayer Leverkusen | 2235 | 0.127 | 0.950 |
| c51_tail_rarity | men | AM | 20 | top | 3 | Aleksandr Golovin | Russia | 225 | 0.124 | 0.900 |
| c51_tail_rarity | men | AM | 20 | top | 4 | Christoph Baumgartner | Austria | 254 | 0.121 | 0.850 |
| c51_tail_rarity | men | AM | 20 | top | 5 | Eden Hazard | Belgium | 281 | 0.120 | 0.800 |
| c51_tail_rarity | men | AM | 20 | bottom | 1 | Adam Hložek | Bayer Leverkusen | 271 | 0.085 | 0.050 |
| c51_tail_rarity | men | AM | 20 | bottom | 2 | Philippe Coutinho Correia | Barcelona | 658 | 0.087 | 0.100 |
| c51_tail_rarity | men | AM | 20 | bottom | 3 | Antoine Griezmann | Barcelona | 2163 | 0.097 | 0.150 |
| c51_tail_rarity | men | CB | 118 | top | 1 | Daley Blind | Netherlands | 627 | 0.106 | 1.000 |
| c51_tail_rarity | men | CB | 118 | top | 2 | Fabian Lukas Schär | Switzerland | 509 | 0.099 | 0.992 |
| c51_tail_rarity | men | CB | 118 | top | 3 | Boualem Khoukhi | Qatar | 222 | 0.093 | 0.983 |
| c51_tail_rarity | men | CB | 118 | top | 4 | Leonardo Bonucci | Italy | 562 | 0.093 | 0.975 |
| c51_tail_rarity | men | CB | 118 | top | 5 | Tomáš Holeš | Czech Republic | 241 | 0.092 | 0.966 |
| c51_tail_rarity | men | CB | 118 | bottom | 1 | Joe Rodon | Wales | 374 | 0.029 | 0.008 |
| c51_tail_rarity | men | CB | 118 | bottom | 2 | Merih Demiral | Turkey | 271 | 0.030 | 0.017 |
| c51_tail_rarity | men | CB | 118 | bottom | 3 | Jason Denayer | Belgium | 214 | 0.032 | 0.025 |
| c51_tail_rarity | men | CM | 29 | top | 1 | Piotr Zieliński | Poland | 508 | 0.120 | 1.000 |
| c51_tail_rarity | men | CM | 29 | top | 2 | Mateo Kovačić | Croatia | 1033 | 0.118 | 0.966 |
| c51_tail_rarity | men | CM | 29 | top | 3 | Luka Modrić | Croatia | 1281 | 0.115 | 0.931 |
| c51_tail_rarity | men | CM | 29 | top | 4 | Aaron Ramsey | Wales | 449 | 0.102 | 0.897 |
| c51_tail_rarity | men | CM | 29 | top | 5 | Ondrej Duda | Slovakia | 287 | 0.095 | 0.862 |
| c51_tail_rarity | men | CM | 29 | bottom | 1 | Moriba Kourouma Kourouma | Barcelona | 430 | 0.050 | 0.034 |
| c51_tail_rarity | men | CM | 29 | bottom | 2 | Georginio Wijnaldum | Paris Saint-Germain | 617 | 0.060 | 0.069 |
| c51_tail_rarity | men | CM | 29 | bottom | 3 | Carlos Soler Barragán | Paris Saint-Germain | 696 | 0.065 | 0.103 |
| c51_tail_rarity | men | DM | 66 | top | 1 | Toni Kroos | Germany | 1118 | 0.123 | 1.000 |
| c51_tail_rarity | men | DM | 66 | top | 2 | Thomas Teye Partey | Ghana | 208 | 0.111 | 0.985 |
| c51_tail_rarity | men | DM | 66 | top | 3 | Pedro Chirivella Burgos | Nantes | 223 | 0.107 | 0.970 |
| c51_tail_rarity | men | DM | 66 | top | 4 | Grzegorz Krychowiak | Poland | 215 | 0.104 | 0.955 |
| c51_tail_rarity | men | DM | 66 | top | 5 | Tijjani Reijnders | Netherlands | 364 | 0.102 | 0.939 |
| c51_tail_rarity | men | DM | 66 | bottom | 1 | Serhii Sydorchuk | Ukraine | 222 | 0.032 | 0.015 |
| c51_tail_rarity | men | DM | 66 | bottom | 2 | Leander Dendoncker | Belgium | 222 | 0.039 | 0.030 |
| c51_tail_rarity | men | DM | 66 | bottom | 3 | Ander Herrera Agüera | Paris Saint-Germain | 340 | 0.047 | 0.045 |
| c51_tail_rarity | men | FB | 72 | top | 1 | Andrija Živković | Serbia | 229 | 0.128 | 1.000 |
| c51_tail_rarity | men | FB | 72 | top | 2 | Alphonso Davies | Canada | 223 | 0.111 | 0.986 |
| c51_tail_rarity | men | FB | 72 | top | 3 | Jeremie Frimpong | Bayer Leverkusen | 1103 | 0.102 | 0.972 |
| c51_tail_rarity | men | FB | 72 | top | 4 | Nathan Tella | Bayer Leverkusen | 317 | 0.101 | 0.958 |
| c51_tail_rarity | men | FB | 72 | top | 5 | Vladimír Coufal | Czech Republic | 406 | 0.098 | 0.944 |
| c51_tail_rarity | men | FB | 72 | bottom | 1 | Thilo Kehrer | Paris Saint-Germain | 644 | 0.028 | 0.014 |
| c51_tail_rarity | men | FB | 72 | bottom | 2 | Danilo Luiz da Silva | Brazil | 252 | 0.033 | 0.028 |
| c51_tail_rarity | men | FB | 72 | bottom | 3 | Sergi Roberto Carnicer | Barcelona | 980 | 0.044 | 0.042 |
| c51_tail_rarity | men | FW | 16 | top | 1 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | 8537 | 0.157 | 1.000 |
| c51_tail_rarity | men | FW | 16 | top | 2 | Kylian Mbappé Lottin | Paris Saint-Germain | 3542 | 0.130 | 0.938 |
| c51_tail_rarity | men | FW | 16 | top | 3 | Victor Okoh Boniface | Bayer Leverkusen | 560 | 0.124 | 0.875 |
| c51_tail_rarity | men | FW | 16 | top | 4 | Breel-Donald Embolo | Switzerland | 318 | 0.120 | 0.812 |
| c51_tail_rarity | men | FW | 16 | top | 5 | Memphis Depay | Netherlands | 488 | 0.116 | 0.750 |
| c51_tail_rarity | men | FW | 16 | bottom | 1 | Arda Güler | Turkey | 240 | 0.071 | 0.062 |
| c51_tail_rarity | men | FW | 16 | bottom | 2 | Roland Sallai | Hungary | 203 | 0.086 | 0.125 |
| c51_tail_rarity | men | FW | 16 | bottom | 3 | Cristiano Ronaldo dos Santos Aveiro | Portugal | 391 | 0.088 | 0.188 |
| c51_tail_rarity | men | GK | 11 | top | 1 | Kasper Schmeichel | Denmark | 247 | 0.052 | 1.000 |
| c51_tail_rarity | men | GK | 11 | top | 2 | Marc-André ter Stegen | Barcelona | 845 | 0.041 | 0.909 |
| c51_tail_rarity | men | GK | 11 | top | 3 | Dominik Livaković | Croatia | 259 | 0.041 | 0.818 |
| c51_tail_rarity | men | GK | 11 | top | 4 | Manuel Neuer | Germany | 371 | 0.037 | 0.727 |
| c51_tail_rarity | men | GK | 11 | top | 5 | Jordan Pickford | England | 549 | 0.036 | 0.636 |
| c51_tail_rarity | men | GK | 11 | bottom | 1 | Yann Sommer | Switzerland | 334 | 0.014 | 0.091 |
| c51_tail_rarity | men | GK | 11 | bottom | 2 | Yassine Bounou | Morocco | 205 | 0.030 | 0.182 |
| c51_tail_rarity | men | GK | 11 | bottom | 3 | Gianluigi Donnarumma | Paris Saint-Germain | 867 | 0.030 | 0.273 |
| c51_tail_rarity | men | W | 37 | top | 1 | Ruslan Malinovskiy | Ukraine | 220 | 0.165 | 1.000 |
| c51_tail_rarity | men | W | 37 | top | 2 | Rafael Alexandre Conceição Leão | Portugal | 204 | 0.159 | 0.973 |
| c51_tail_rarity | men | W | 37 | top | 3 | Jeremy Doku | Belgium | 332 | 0.151 | 0.946 |
| c51_tail_rarity | men | W | 37 | top | 4 | Leandro Trossard | Belgium | 207 | 0.145 | 0.919 |
| c51_tail_rarity | men | W | 37 | top | 5 | Lamine Yamal Nasraoui Ebana | Spain | 296 | 0.144 | 0.892 |
| c51_tail_rarity | men | W | 37 | bottom | 1 | Gerard Moreno Balaguero | Spain | 213 | 0.067 | 0.027 |
| c51_tail_rarity | men | W | 37 | bottom | 2 | Francisco António Machado Mota de Castro Trincão | Barcelona | 506 | 0.068 | 0.054 |
| c51_tail_rarity | men | W | 37 | bottom | 3 | Martin Braithwaite Christensen | Barcelona | 566 | 0.068 | 0.081 |
| c51_tail_rarity | women | AM | 6 | top | 1 | Pernille Mosegaard Harder | Denmark Women's | 320 | 0.132 | 1.000 |
| c51_tail_rarity | women | AM | 6 | top | 2 | Lauren James | England Women's | 442 | 0.108 | 0.833 |
| c51_tail_rarity | women | AM | 6 | top | 3 | Leicy Maria Santos Herrera | Colombia Women's | 237 | 0.095 | 0.667 |
| c51_tail_rarity | women | AM | 6 | top | 4 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | 279 | 0.094 | 0.500 |
| c51_tail_rarity | women | AM | 6 | top | 5 | Ella Toone | England Women's | 498 | 0.093 | 0.333 |
| c51_tail_rarity | women | AM | 6 | bottom | 1 | Kosovare Asllani | Sweden Women's | 487 | 0.079 | 0.167 |
| c51_tail_rarity | women | AM | 6 | bottom | 2 | Ella Toone | England Women's | 498 | 0.093 | 0.333 |
| c51_tail_rarity | women | AM | 6 | bottom | 3 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | 279 | 0.094 | 0.500 |
| c51_tail_rarity | women | CB | 52 | top | 1 | Sherida Spitse | Netherlands Women's | 671 | 0.105 | 1.000 |
| c51_tail_rarity | women | CB | 52 | top | 2 | Leah Williamson | England Women's | 929 | 0.100 | 0.981 |
| c51_tail_rarity | women | CB | 52 | top | 3 | Rafaelle Leone Carvalho Souza | Brazil Women's | 278 | 0.098 | 0.962 |
| c51_tail_rarity | women | CB | 52 | top | 4 | Katie Bowen | New Zealand Women's | 255 | 0.092 | 0.942 |
| c51_tail_rarity | women | CB | 52 | top | 5 | Julie Beth Ertz | United States Women's | 243 | 0.083 | 0.923 |
| c51_tail_rarity | women | CB | 52 | bottom | 1 | Sara Doorsoun-Khajeh | Germany Women's | 249 | 0.027 | 0.019 |
| c51_tail_rarity | women | CB | 52 | bottom | 2 | Eva Nyström | WNT Finland | 282 | 0.036 | 0.038 |
| c51_tail_rarity | women | CB | 52 | bottom | 3 | Laia Codina Panedas | Spain Women's | 240 | 0.038 | 0.058 |
| c51_tail_rarity | women | CM | 16 | top | 1 | Clara Mateo | France Women's | 278 | 0.123 | 1.000 |
| c51_tail_rarity | women | CM | 16 | top | 2 | Jennifer Hermoso Fuentes | Spain Women's | 493 | 0.116 | 0.938 |
| c51_tail_rarity | women | CM | 16 | top | 3 | Lindsey Michelle Heaps | United States Women's | 236 | 0.104 | 0.875 |
| c51_tail_rarity | women | CM | 16 | top | 4 | Arianna Caruso | Italy Women's | 323 | 0.098 | 0.812 |
| c51_tail_rarity | women | CM | 16 | top | 5 | Aitana Bonmati Conca | Spain Women's | 1293 | 0.097 | 0.750 |
| c51_tail_rarity | women | CM | 16 | bottom | 1 | Lina Magull | Germany Women's | 240 | 0.060 | 0.062 |
| c51_tail_rarity | women | CM | 16 | bottom | 2 | Géraldine Reuteler | Switzerland Women's | 523 | 0.061 | 0.125 |
| c51_tail_rarity | women | CM | 16 | bottom | 3 | Coumba Sow | Switzerland Women's | 232 | 0.064 | 0.188 |
| c51_tail_rarity | women | DM | 25 | top | 1 | Eveliina Summanen | WNT Finland | 276 | 0.109 | 1.000 |
| c51_tail_rarity | women | DM | 25 | top | 2 | Keira Walsh | England Women's | 985 | 0.105 | 0.960 |
| c51_tail_rarity | women | DM | 25 | top | 3 | Kerolin Nicoli Israel Ferraz | Brazil Women's | 208 | 0.103 | 0.920 |
| c51_tail_rarity | women | DM | 25 | top | 4 | Vilde Bøe Risa | Norway Women's | 359 | 0.101 | 0.880 |
| c51_tail_rarity | women | DM | 25 | top | 5 | Ingrid Syrstad Engen | Norway Women's | 382 | 0.101 | 0.840 |
| c51_tail_rarity | women | DM | 25 | bottom | 1 | Sjoeke Nüsken | Germany Women's | 217 | 0.051 | 0.040 |
| c51_tail_rarity | women | DM | 25 | bottom | 2 | Karen Holmgaard | Denmark Women's | 281 | 0.052 | 0.080 |
| c51_tail_rarity | women | DM | 25 | bottom | 3 | Charlotte Bilbault | France Women's | 214 | 0.053 | 0.120 |
| c51_tail_rarity | women | FB | 37 | top | 1 | Janni Thomsen | Denmark Women's | 294 | 0.110 | 1.000 |
| c51_tail_rarity | women | FB | 37 | top | 2 | Selma Bacha | France Women's | 606 | 0.097 | 0.973 |
| c51_tail_rarity | women | FB | 37 | top | 3 | Gun Nathalie Björn | Sweden Women's | 805 | 0.090 | 0.946 |
| c51_tail_rarity | women | FB | 37 | top | 4 | Esmee Brugts | Netherlands Women's | 370 | 0.090 | 0.919 |
| c51_tail_rarity | women | FB | 37 | top | 5 | Victoria Pelova | Netherlands Women's | 499 | 0.084 | 0.892 |
| c51_tail_rarity | women | FB | 37 | bottom | 1 | Thea Bjelde | Norway Women's | 235 | 0.039 | 0.027 |
| c51_tail_rarity | women | FB | 37 | bottom | 2 | Oihane Hernández Zurbano | Spain Women's | 270 | 0.041 | 0.054 |
| c51_tail_rarity | women | FB | 37 | bottom | 3 | Eseosa Aigbogun | Switzerland Women's | 280 | 0.044 | 0.081 |
| c51_tail_rarity | women | FW | 10 | top | 1 | Ramona Bachmann | Switzerland Women's | 330 | 0.133 | 1.000 |
| c51_tail_rarity | women | FW | 10 | top | 2 | Jéssica da Silva | Portugal Women's | 226 | 0.124 | 0.900 |
| c51_tail_rarity | women | FW | 10 | top | 3 | Lieke Martens | Netherlands Women's | 402 | 0.104 | 0.800 |
| c51_tail_rarity | women | FW | 10 | top | 4 | Tessa Wullaert | Belgium Women's | 265 | 0.099 | 0.700 |
| c51_tail_rarity | women | FW | 10 | top | 5 | Mary Boio Fowler | Australia Women's | 204 | 0.094 | 0.600 |
| c51_tail_rarity | women | FW | 10 | bottom | 1 | Alessia Russo | England Women's | 390 | 0.058 | 0.100 |
| c51_tail_rarity | women | FW | 10 | bottom | 2 | Diana Micaela Abreu de Sousa e Silva | Portugal Women's | 217 | 0.062 | 0.200 |
| c51_tail_rarity | women | FW | 10 | bottom | 3 | Ana-Maria Crnogorčević | Switzerland Women's | 295 | 0.067 | 0.300 |
| c51_tail_rarity | women | GK | 7 | top | 1 | Ann-Katrin Berger | Germany Women's | 216 | 0.057 | 1.000 |
| c51_tail_rarity | women | GK | 7 | top | 2 | Gaëlle Thalmann | Switzerland Women's | 207 | 0.043 | 0.857 |
| c51_tail_rarity | women | GK | 7 | top | 3 | Merle Frohms | Germany Women's | 237 | 0.041 | 0.714 |
| c51_tail_rarity | women | GK | 7 | top | 4 | Daphne van Domselaar | Netherlands Women's | 259 | 0.039 | 0.571 |
| c51_tail_rarity | women | GK | 7 | top | 5 | Laura Giuliani | Italy Women's | 201 | 0.030 | 0.429 |
| c51_tail_rarity | women | GK | 7 | bottom | 1 | Mary Alexandra Earps | England Women's | 209 | 0.019 | 0.143 |
| c51_tail_rarity | women | GK | 7 | bottom | 2 | Pauline Peyraud Magnin | France Women's | 271 | 0.029 | 0.286 |
| c51_tail_rarity | women | GK | 7 | bottom | 3 | Laura Giuliani | Italy Women's | 201 | 0.030 | 0.429 |
| c51_tail_rarity | women | W | 17 | top | 1 | Jule Brand | Germany Women's | 414 | 0.128 | 1.000 |
| c51_tail_rarity | women | W | 17 | top | 2 | Delphine Cascarino | France Women's | 347 | 0.126 | 0.941 |
| c51_tail_rarity | women | W | 17 | top | 3 | Klara Bühl | Germany Women's | 557 | 0.121 | 0.882 |
| c51_tail_rarity | women | W | 17 | top | 4 | Johanna Rytting-Kaneryd | Sweden Women's | 470 | 0.117 | 0.824 |
| c51_tail_rarity | women | W | 17 | top | 5 | Caitlin Jade Foord | Australia Women's | 212 | 0.116 | 0.765 |
| c51_tail_rarity | women | W | 17 | bottom | 1 | Linda Lizeth Caicedo Alegría | Colombia Women's | 226 | 0.075 | 0.059 |
| c51_tail_rarity | women | W | 17 | bottom | 2 | Bethany Mead | England Women's | 251 | 0.082 | 0.118 |
| c51_tail_rarity | women | W | 17 | bottom | 3 | Fridolina Rolfö | Sweden Women's | 504 | 0.083 | 0.176 |
| unxpass | men | AM | 20 | top | 1 | Kevin De Bruyne | Belgium | 578 | 0.231 | 1.000 |
| unxpass | men | AM | 20 | top | 2 | Bruno Miguel Borges Fernandes | Portugal | 645 | 0.226 | 0.950 |
| unxpass | men | AM | 20 | top | 3 | Dušan Tadić | Serbia | 232 | 0.224 | 0.900 |
| unxpass | men | AM | 20 | top | 4 | Christoph Baumgartner | Austria | 254 | 0.210 | 0.850 |
| unxpass | men | AM | 20 | top | 5 | Marcel Sabitzer | Austria | 437 | 0.206 | 0.800 |
| unxpass | men | AM | 20 | bottom | 1 | Adam Hložek | Bayer Leverkusen | 271 | 0.124 | 0.050 |
| unxpass | men | AM | 20 | bottom | 2 | Thomas Müller | Germany | 281 | 0.149 | 0.100 |
| unxpass | men | AM | 20 | bottom | 3 | Philippe Coutinho Correia | Barcelona | 658 | 0.161 | 0.150 |
| unxpass | men | CB | 118 | top | 1 | Leonardo Bonucci | Italy | 562 | 0.251 | 1.000 |
| unxpass | men | CB | 118 | top | 2 | Ondrej Celustka | Czech Republic | 210 | 0.251 | 0.992 |
| unxpass | men | CB | 118 | top | 3 | Romain Saïss | Morocco | 204 | 0.251 | 0.983 |
| unxpass | men | CB | 118 | top | 4 | Daley Blind | Netherlands | 627 | 0.244 | 0.975 |
| unxpass | men | CB | 118 | top | 5 | Fabian Lukas Schär | Switzerland | 509 | 0.239 | 0.966 |
| unxpass | men | CB | 118 | bottom | 1 | Ronald Federico Araújo da Silva | Barcelona | 938 | 0.124 | 0.008 |
| unxpass | men | CB | 118 | bottom | 2 | Kamil Glik | Poland | 301 | 0.124 | 0.017 |
| unxpass | men | CB | 118 | bottom | 3 | Ľubomír Šatka | Slovakia | 218 | 0.124 | 0.025 |
| unxpass | men | CM | 29 | top | 1 | Luka Modrić | Croatia | 1281 | 0.248 | 1.000 |
| unxpass | men | CM | 29 | top | 2 | Piotr Zieliński | Poland | 508 | 0.219 | 0.966 |
| unxpass | men | CM | 29 | top | 3 | Aaron Ramsey | Wales | 449 | 0.203 | 0.931 |
| unxpass | men | CM | 29 | top | 4 | Tomáš Souček | Czech Republic | 355 | 0.202 | 0.897 |
| unxpass | men | CM | 29 | top | 5 | Jorge Resurrección Merodio | Spain | 669 | 0.191 | 0.862 |
| unxpass | men | CM | 29 | bottom | 1 | Moriba Kourouma Kourouma | Barcelona | 430 | 0.101 | 0.034 |
| unxpass | men | CM | 29 | bottom | 2 | Carlos Soler Barragán | Paris Saint-Germain | 696 | 0.129 | 0.069 |
| unxpass | men | CM | 29 | bottom | 3 | Georginio Wijnaldum | Paris Saint-Germain | 617 | 0.132 | 0.103 |
| unxpass | men | DM | 66 | top | 1 | Giorgi Kochorashvili | Georgia | 226 | 0.262 | 1.000 |
| unxpass | men | DM | 66 | top | 2 | Toni Kroos | Germany | 1118 | 0.258 | 0.985 |
| unxpass | men | DM | 66 | top | 3 | Aaron Mooy | Australia | 229 | 0.257 | 0.970 |
| unxpass | men | DM | 66 | top | 4 | Woo-Young Jung | South Korea | 215 | 0.251 | 0.955 |
| unxpass | men | DM | 66 | top | 5 | Thomas Teye Partey | Ghana | 208 | 0.239 | 0.939 |
| unxpass | men | DM | 66 | bottom | 1 | Kobbie Mainoo | England | 246 | 0.109 | 0.015 |
| unxpass | men | DM | 66 | bottom | 2 | Marten de Roon | Netherlands | 259 | 0.119 | 0.030 |
| unxpass | men | DM | 66 | bottom | 3 | Morten Hjulmand | Denmark | 222 | 0.120 | 0.045 |
| unxpass | men | FB | 72 | top | 1 | Marcos Javier Acuña | Argentina | 226 | 0.226 | 1.000 |
| unxpass | men | FB | 72 | top | 2 | Eduardo Camavinga | France | 216 | 0.225 | 0.986 |
| unxpass | men | FB | 72 | top | 3 | Vladimír Coufal | Czech Republic | 406 | 0.220 | 0.972 |
| unxpass | men | FB | 72 | top | 4 | Thomas Meunier | Belgium | 332 | 0.218 | 0.958 |
| unxpass | men | FB | 72 | top | 5 | Pervis Josué Estupiñán Tenorio | Ecuador | 260 | 0.218 | 0.944 |
| unxpass | men | FB | 72 | bottom | 1 | Emerson Palmieri dos Santos | Italy | 211 | 0.103 | 0.014 |
| unxpass | men | FB | 72 | bottom | 2 | Warren Zaire Emery | Paris Saint-Germain | 546 | 0.104 | 0.028 |
| unxpass | men | FB | 72 | bottom | 3 | Vitalii Mykolenko | Ukraine | 206 | 0.112 | 0.042 |
| unxpass | men | FW | 16 | top | 1 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | 8537 | 0.240 | 1.000 |
| unxpass | men | FW | 16 | top | 2 | Harry Kane | England | 480 | 0.223 | 0.938 |
| unxpass | men | FW | 16 | top | 3 | Kylian Mbappé Lottin | Paris Saint-Germain | 3542 | 0.190 | 0.875 |
| unxpass | men | FW | 16 | top | 4 | Karim Benzema | France | 261 | 0.180 | 0.812 |
| unxpass | men | FW | 16 | top | 5 | Memphis Depay | Netherlands | 488 | 0.175 | 0.750 |
| unxpass | men | FW | 16 | bottom | 1 | Cristiano Ronaldo dos Santos Aveiro | Portugal | 391 | 0.131 | 0.062 |
| unxpass | men | FW | 16 | bottom | 2 | Patrik Schick | Bayer Leverkusen | 447 | 0.132 | 0.125 |
| unxpass | men | FW | 16 | bottom | 3 | Álvaro Borja Morata Martín | Spain | 356 | 0.140 | 0.188 |
| unxpass | men | GK | 11 | top | 1 | Kasper Schmeichel | Denmark | 247 | 0.202 | 1.000 |
| unxpass | men | GK | 11 | top | 2 | Marc-André ter Stegen | Barcelona | 845 | 0.200 | 0.909 |
| unxpass | men | GK | 11 | top | 3 | Manuel Neuer | Germany | 371 | 0.185 | 0.818 |
| unxpass | men | GK | 11 | top | 4 | Jordan Pickford | England | 549 | 0.185 | 0.727 |
| unxpass | men | GK | 11 | top | 5 | Yassine Bounou | Morocco | 205 | 0.180 | 0.636 |
| unxpass | men | GK | 11 | bottom | 1 | Gianluigi Donnarumma | Paris Saint-Germain | 867 | 0.127 | 0.091 |
| unxpass | men | GK | 11 | bottom | 2 | Thibaut Courtois | Belgium | 225 | 0.148 | 0.182 |
| unxpass | men | GK | 11 | bottom | 3 | Unai Simón Mendibil | Spain | 522 | 0.159 | 0.273 |
| unxpass | men | W | 37 | top | 1 | Ruslan Malinovskiy | Ukraine | 220 | 0.273 | 1.000 |
| unxpass | men | W | 37 | top | 2 | Hakim Ziyech | Morocco | 336 | 0.255 | 0.973 |
| unxpass | men | W | 37 | top | 3 | Neymar da Silva Santos Junior | Paris Saint-Germain | 2650 | 0.240 | 0.946 |
| unxpass | men | W | 37 | top | 4 | Leandro Trossard | Belgium | 207 | 0.234 | 0.919 |
| unxpass | men | W | 37 | top | 5 | Lamine Yamal Nasraoui Ebana | Spain | 296 | 0.224 | 0.892 |
| unxpass | men | W | 37 | bottom | 1 | Andrej Kramarić | Croatia | 442 | 0.130 | 0.027 |
| unxpass | men | W | 37 | bottom | 2 | Francisco António Machado Mota de Castro Trincão | Barcelona | 506 | 0.132 | 0.054 |
| unxpass | men | W | 37 | bottom | 3 | Martin Braithwaite Christensen | Barcelona | 566 | 0.135 | 0.081 |
| unxpass | women | AM | 6 | top | 1 | Pernille Mosegaard Harder | Denmark Women's | 320 | 0.215 | 1.000 |
| unxpass | women | AM | 6 | top | 2 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | 279 | 0.214 | 0.833 |
| unxpass | women | AM | 6 | top | 3 | Leicy Maria Santos Herrera | Colombia Women's | 237 | 0.191 | 0.667 |
| unxpass | women | AM | 6 | top | 4 | Lauren James | England Women's | 442 | 0.183 | 0.500 |
| unxpass | women | AM | 6 | top | 5 | Kosovare Asllani | Sweden Women's | 487 | 0.167 | 0.333 |
| unxpass | women | AM | 6 | bottom | 1 | Ella Toone | England Women's | 498 | 0.154 | 0.167 |
| unxpass | women | AM | 6 | bottom | 2 | Kosovare Asllani | Sweden Women's | 487 | 0.167 | 0.333 |
| unxpass | women | AM | 6 | bottom | 3 | Lauren James | England Women's | 442 | 0.183 | 0.500 |
| unxpass | women | CB | 52 | top | 1 | Katie Bowen | New Zealand Women's | 255 | 0.267 | 1.000 |
| unxpass | women | CB | 52 | top | 2 | Rafaelle Leone Carvalho Souza | Brazil Women's | 278 | 0.251 | 0.981 |
| unxpass | women | CB | 52 | top | 3 | Vanessa Brigitte Gilles | Canada Women's | 264 | 0.250 | 0.962 |
| unxpass | women | CB | 52 | top | 4 | Natalia Kuikka | WNT Finland | 364 | 0.246 | 0.942 |
| unxpass | women | CB | 52 | top | 5 | Sherida Spitse | Netherlands Women's | 671 | 0.242 | 0.923 |
| unxpass | women | CB | 52 | bottom | 1 | Mathilde Hauge Harviken | Norway Women's | 382 | 0.135 | 0.019 |
| unxpass | women | CB | 52 | bottom | 2 | Noelle Maritz | Switzerland Women's | 614 | 0.136 | 0.038 |
| unxpass | women | CB | 52 | bottom | 3 | Laia Aleixandri López | Spain Women's | 516 | 0.147 | 0.058 |
| unxpass | women | CM | 16 | top | 1 | Lindsey Michelle Heaps | United States Women's | 236 | 0.271 | 1.000 |
| unxpass | women | CM | 16 | top | 2 | Clara Mateo | France Women's | 278 | 0.252 | 0.938 |
| unxpass | women | CM | 16 | top | 3 | Jennifer Hermoso Fuentes | Spain Women's | 493 | 0.207 | 0.875 |
| unxpass | women | CM | 16 | top | 4 | Andreia Alexandra Norton | Portugal Women's | 314 | 0.202 | 0.812 |
| unxpass | women | CM | 16 | top | 5 | Tatiana Vanessa Ferreira Pinto | Portugal Women's | 352 | 0.198 | 0.750 |
| unxpass | women | CM | 16 | bottom | 1 | Coumba Sow | Switzerland Women's | 232 | 0.106 | 0.062 |
| unxpass | women | CM | 16 | bottom | 2 | Jill Roord | Netherlands Women's | 510 | 0.120 | 0.125 |
| unxpass | women | CM | 16 | bottom | 3 | Kathrine Møller Kühl | Denmark Women's | 242 | 0.136 | 0.188 |
| unxpass | women | DM | 25 | top | 1 | Kerolin Nicoli Israel Ferraz | Brazil Women's | 208 | 0.220 | 1.000 |
| unxpass | women | DM | 25 | top | 2 | Katrina Gorry | Australia Women's | 385 | 0.209 | 0.960 |
| unxpass | women | DM | 25 | top | 3 | Keira Walsh | England Women's | 985 | 0.205 | 0.920 |
| unxpass | women | DM | 25 | top | 4 | Ingrid Syrstad Engen | Norway Women's | 382 | 0.203 | 0.880 |
| unxpass | women | DM | 25 | top | 5 | Vilde Bøe Risa | Norway Women's | 359 | 0.202 | 0.840 |
| unxpass | women | DM | 25 | bottom | 1 | Charlotte Bilbault | France Women's | 214 | 0.146 | 0.040 |
| unxpass | women | DM | 25 | bottom | 2 | Sjoeke Nüsken | Germany Women's | 217 | 0.147 | 0.080 |
| unxpass | women | DM | 25 | bottom | 3 | Karen Holmgaard | Denmark Women's | 281 | 0.151 | 0.120 |
| unxpass | women | FB | 37 | top | 1 | Ana Borges | Portugal Women's | 326 | 0.268 | 1.000 |
| unxpass | women | FB | 37 | top | 2 | Eve Perisset | France Women's | 459 | 0.228 | 0.973 |
| unxpass | women | FB | 37 | top | 3 | Joana Filipa Gaspar Silva Marchão | Portugal Women's | 251 | 0.216 | 0.946 |
| unxpass | women | FB | 37 | top | 4 | Risa Shimizu | Japan Women's | 210 | 0.210 | 0.919 |
| unxpass | women | FB | 37 | top | 5 | Elisa De Almeida | France Women's | 392 | 0.209 | 0.892 |
| unxpass | women | FB | 37 | bottom | 1 | Leila Ouahabi El Ouahabi | Spain Women's | 285 | 0.120 | 0.027 |
| unxpass | women | FB | 37 | bottom | 2 | Oihane Hernández Zurbano | Spain Women's | 270 | 0.123 | 0.054 |
| unxpass | women | FB | 37 | bottom | 3 | Lucia Di Guglielmo | Italy Women's | 362 | 0.125 | 0.081 |
| unxpass | women | FW | 10 | top | 1 | Mary Boio Fowler | Australia Women's | 204 | 0.267 | 1.000 |
| unxpass | women | FW | 10 | top | 2 | Tessa Wullaert | Belgium Women's | 265 | 0.229 | 0.900 |
| unxpass | women | FW | 10 | top | 3 | Diana Micaela Abreu de Sousa e Silva | Portugal Women's | 217 | 0.195 | 0.800 |
| unxpass | women | FW | 10 | top | 4 | Lieke Martens | Netherlands Women's | 402 | 0.169 | 0.700 |
| unxpass | women | FW | 10 | top | 5 | Ana-Maria Crnogorčević | Switzerland Women's | 295 | 0.143 | 0.600 |
| unxpass | women | FW | 10 | bottom | 1 | Emma Stina Blackstenius | Sweden Women's | 308 | 0.104 | 0.100 |
| unxpass | women | FW | 10 | bottom | 2 | Esther Gonzalez Rodríguez | Spain Women's | 272 | 0.108 | 0.200 |
| unxpass | women | FW | 10 | bottom | 3 | Alessia Russo | England Women's | 390 | 0.123 | 0.300 |
| unxpass | women | GK | 7 | top | 1 | Ann-Katrin Berger | Germany Women's | 216 | 0.233 | 1.000 |
| unxpass | women | GK | 7 | top | 2 | Daphne van Domselaar | Netherlands Women's | 259 | 0.186 | 0.857 |
| unxpass | women | GK | 7 | top | 3 | Merle Frohms | Germany Women's | 237 | 0.175 | 0.714 |
| unxpass | women | GK | 7 | top | 4 | Pauline Peyraud Magnin | France Women's | 271 | 0.170 | 0.571 |
| unxpass | women | GK | 7 | top | 5 | Mary Alexandra Earps | England Women's | 209 | 0.150 | 0.429 |
| unxpass | women | GK | 7 | bottom | 1 | Gaëlle Thalmann | Switzerland Women's | 207 | 0.116 | 0.143 |
| unxpass | women | GK | 7 | bottom | 2 | Laura Giuliani | Italy Women's | 201 | 0.137 | 0.286 |
| unxpass | women | GK | 7 | bottom | 3 | Mary Alexandra Earps | England Women's | 209 | 0.150 | 0.429 |
| unxpass | women | W | 17 | top | 1 | Caitlin Jade Foord | Australia Women's | 212 | 0.222 | 1.000 |
| unxpass | women | W | 17 | top | 2 | María Francesca Caldentey Oliver | Spain Women's | 1179 | 0.197 | 0.941 |
| unxpass | women | W | 17 | top | 3 | Kadidiatou Diani | France Women's | 460 | 0.192 | 0.882 |
| unxpass | women | W | 17 | top | 4 | Chloe Kelly | England Women's | 316 | 0.192 | 0.824 |
| unxpass | women | W | 17 | top | 5 | Linda Lizeth Caicedo Alegría | Colombia Women's | 226 | 0.186 | 0.765 |
| unxpass | women | W | 17 | bottom | 1 | Klara Bühl | Germany Women's | 557 | 0.139 | 0.059 |
| unxpass | women | W | 17 | bottom | 2 | Lauren Hemp | England Women's | 581 | 0.143 | 0.118 |
| unxpass | women | W | 17 | bottom | 3 | Athenea del Castillo Belvide | Spain Women's | 260 | 0.144 | 0.176 |
| rarity_x_adv | men | AM | 20 | top | 1 | Christoph Baumgartner | Austria | 254 | 0.004 | 1.000 |
| rarity_x_adv | men | AM | 20 | top | 2 | Amine Adli | Bayer Leverkusen | 447 | 0.003 | 0.950 |
| rarity_x_adv | men | AM | 20 | top | 3 | Xherdan Shaqiri | Switzerland | 343 | 0.003 | 0.900 |
| rarity_x_adv | men | AM | 20 | top | 4 | Kevin De Bruyne | Belgium | 578 | 0.003 | 0.850 |
| rarity_x_adv | men | AM | 20 | top | 5 | Florian Wirtz | Bayer Leverkusen | 2235 | 0.003 | 0.800 |
| rarity_x_adv | men | AM | 20 | bottom | 1 | Sergej Milinković-Savić | Serbia | 208 | 0.002 | 0.050 |
| rarity_x_adv | men | AM | 20 | bottom | 2 | Christian Dannemann Eriksen | Denmark | 473 | 0.002 | 0.100 |
| rarity_x_adv | men | AM | 20 | bottom | 3 | Thomas Müller | Germany | 281 | 0.002 | 0.150 |
| rarity_x_adv | men | CB | 118 | top | 1 | Scott McTominay | Scotland | 231 | 0.001 | 1.000 |
| rarity_x_adv | men | CB | 118 | top | 2 | Tomáš Holeš | Czech Republic | 241 | 0.001 | 0.992 |
| rarity_x_adv | men | CB | 118 | top | 3 | Romain Saïss | Morocco | 204 | 0.001 | 0.983 |
| rarity_x_adv | men | CB | 118 | top | 4 | Abdelkarim Hassan Al Haj Fadlalla | Qatar | 207 | 0.001 | 0.975 |
| rarity_x_adv | men | CB | 118 | top | 5 | Kamal Miller | Canada | 417 | 0.001 | 0.966 |
| rarity_x_adv | men | CB | 118 | bottom | 1 | Kamil Glik | Poland | 301 | 0.000 | 0.008 |
| rarity_x_adv | men | CB | 118 | bottom | 2 | Grant Campbell Hanley | Scotland | 272 | 0.000 | 0.017 |
| rarity_x_adv | men | CB | 118 | bottom | 3 | Jason Denayer | Belgium | 214 | 0.000 | 0.025 |
| rarity_x_adv | men | CM | 29 | top | 1 | Yunus Dimoara Musah | United States | 236 | 0.002 | 1.000 |
| rarity_x_adv | men | CM | 29 | top | 2 | Piotr Zieliński | Poland | 508 | 0.002 | 0.966 |
| rarity_x_adv | men | CM | 29 | top | 3 | Aaron Ramsey | Wales | 449 | 0.002 | 0.931 |
| rarity_x_adv | men | CM | 29 | top | 4 | N'Golo Kanté | France | 617 | 0.002 | 0.897 |
| rarity_x_adv | men | CM | 29 | top | 5 | Luka Modrić | Croatia | 1281 | 0.002 | 0.862 |
| rarity_x_adv | men | CM | 29 | bottom | 1 | Carlos Soler Barragán | Paris Saint-Germain | 696 | 0.001 | 0.034 |
| rarity_x_adv | men | CM | 29 | bottom | 2 | András Schäfer | Hungary | 257 | 0.001 | 0.069 |
| rarity_x_adv | men | CM | 29 | bottom | 3 | Idrissa Gana Gueye | Paris Saint-Germain | 1387 | 0.001 | 0.103 |
| rarity_x_adv | men | DM | 66 | top | 1 | Konrad Laimer | Austria | 354 | 0.002 | 1.000 |
| rarity_x_adv | men | DM | 66 | top | 2 | Federico Santiago Valverde Dipetta | Uruguay | 275 | 0.002 | 0.985 |
| rarity_x_adv | men | DM | 66 | top | 3 | Tijjani Reijnders | Netherlands | 364 | 0.002 | 0.970 |
| rarity_x_adv | men | DM | 66 | top | 4 | Leon Goretzka | Germany | 272 | 0.002 | 0.955 |
| rarity_x_adv | men | DM | 66 | top | 5 | Paul Pogba | France | 340 | 0.002 | 0.939 |
| rarity_x_adv | men | DM | 66 | bottom | 1 | Leander Dendoncker | Belgium | 222 | 0.000 | 0.015 |
| rarity_x_adv | men | DM | 66 | bottom | 2 | Marten de Roon | Netherlands | 259 | 0.001 | 0.030 |
| rarity_x_adv | men | DM | 66 | bottom | 3 | Morten Hjulmand | Denmark | 222 | 0.001 | 0.045 |
| rarity_x_adv | men | FB | 72 | top | 1 | Nathan Tella | Bayer Leverkusen | 317 | 0.003 | 1.000 |
| rarity_x_adv | men | FB | 72 | top | 2 | Yannick Ferreira Carrasco | Belgium | 343 | 0.003 | 0.986 |
| rarity_x_adv | men | FB | 72 | top | 3 | Jeremie Frimpong | Bayer Leverkusen | 1103 | 0.003 | 0.972 |
| rarity_x_adv | men | FB | 72 | top | 4 | Andrija Živković | Serbia | 229 | 0.003 | 0.958 |
| rarity_x_adv | men | FB | 72 | top | 5 | Oleksandr Karavaev | Ukraine | 287 | 0.003 | 0.944 |
| rarity_x_adv | men | FB | 72 | bottom | 1 | Thilo Kehrer | Paris Saint-Germain | 644 | 0.000 | 0.014 |
| rarity_x_adv | men | FB | 72 | bottom | 2 | Vitalii Mykolenko | Ukraine | 206 | 0.001 | 0.028 |
| rarity_x_adv | men | FB | 72 | bottom | 3 | Éder Gabriel Militão | Brazil | 303 | 0.001 | 0.042 |
| rarity_x_adv | men | FW | 16 | top | 1 | Victor Okoh Boniface | Bayer Leverkusen | 560 | 0.005 | 1.000 |
| rarity_x_adv | men | FW | 16 | top | 2 | Kylian Mbappé Lottin | Paris Saint-Germain | 3542 | 0.004 | 0.938 |
| rarity_x_adv | men | FW | 16 | top | 3 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | 8537 | 0.003 | 0.875 |
| rarity_x_adv | men | FW | 16 | top | 4 | Karim Benzema | France | 261 | 0.003 | 0.812 |
| rarity_x_adv | men | FW | 16 | top | 5 | Breel-Donald Embolo | Switzerland | 318 | 0.003 | 0.750 |
| rarity_x_adv | men | FW | 16 | bottom | 1 | Arda Güler | Turkey | 240 | 0.002 | 0.062 |
| rarity_x_adv | men | FW | 16 | bottom | 2 | Roland Sallai | Hungary | 203 | 0.002 | 0.125 |
| rarity_x_adv | men | FW | 16 | bottom | 3 | Hugo Ekitike | Paris Saint-Germain | 369 | 0.002 | 0.188 |
| rarity_x_adv | men | GK | 11 | top | 1 | Kasper Schmeichel | Denmark | 247 | 0.001 | 1.000 |
| rarity_x_adv | men | GK | 11 | top | 2 | Dominik Livaković | Croatia | 259 | 0.001 | 0.909 |
| rarity_x_adv | men | GK | 11 | top | 3 | Thibaut Courtois | Belgium | 225 | 0.001 | 0.818 |
| rarity_x_adv | men | GK | 11 | top | 4 | Marc-André ter Stegen | Barcelona | 845 | 0.001 | 0.727 |
| rarity_x_adv | men | GK | 11 | top | 5 | Yassine Bounou | Morocco | 205 | 0.000 | 0.636 |
| rarity_x_adv | men | GK | 11 | bottom | 1 | Yann Sommer | Switzerland | 334 | 0.000 | 0.091 |
| rarity_x_adv | men | GK | 11 | bottom | 2 | Lukáš Hrádecký | Bayer Leverkusen | 641 | 0.000 | 0.182 |
| rarity_x_adv | men | GK | 11 | bottom | 3 | Gianluigi Donnarumma | Paris Saint-Germain | 867 | 0.000 | 0.273 |
| rarity_x_adv | men | W | 37 | top | 1 | Cody Mathès Gakpo | Netherlands | 323 | 0.004 | 1.000 |
| rarity_x_adv | men | W | 37 | top | 2 | Ruslan Malinovskiy | Ukraine | 220 | 0.004 | 0.973 |
| rarity_x_adv | men | W | 37 | top | 3 | Jeremy Doku | Belgium | 332 | 0.004 | 0.946 |
| rarity_x_adv | men | W | 37 | top | 4 | Nicholas Williams Arthuer | Spain | 362 | 0.004 | 0.919 |
| rarity_x_adv | men | W | 37 | top | 5 | Jamal Musiala | Germany | 558 | 0.004 | 0.892 |
| rarity_x_adv | men | W | 37 | bottom | 1 | Timothy Weah | United States | 213 | 0.001 | 0.027 |
| rarity_x_adv | men | W | 37 | bottom | 2 | Lovro Majer | Croatia | 348 | 0.001 | 0.054 |
| rarity_x_adv | men | W | 37 | bottom | 3 | Hakim Ziyech | Morocco | 336 | 0.001 | 0.081 |
| rarity_x_adv | women | AM | 6 | top | 1 | Pernille Mosegaard Harder | Denmark Women's | 320 | 0.004 | 1.000 |
| rarity_x_adv | women | AM | 6 | top | 2 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | 279 | 0.004 | 0.833 |
| rarity_x_adv | women | AM | 6 | top | 3 | Lauren James | England Women's | 442 | 0.002 | 0.667 |
| rarity_x_adv | women | AM | 6 | top | 4 | Kosovare Asllani | Sweden Women's | 487 | 0.002 | 0.500 |
| rarity_x_adv | women | AM | 6 | top | 5 | Leicy Maria Santos Herrera | Colombia Women's | 237 | 0.002 | 0.333 |
| rarity_x_adv | women | AM | 6 | bottom | 1 | Ella Toone | England Women's | 498 | 0.002 | 0.167 |
| rarity_x_adv | women | AM | 6 | bottom | 2 | Leicy Maria Santos Herrera | Colombia Women's | 237 | 0.002 | 0.333 |
| rarity_x_adv | women | AM | 6 | bottom | 3 | Kosovare Asllani | Sweden Women's | 487 | 0.002 | 0.500 |
| rarity_x_adv | women | CB | 52 | top | 1 | Rafaelle Leone Carvalho Souza | Brazil Women's | 278 | 0.001 | 1.000 |
| rarity_x_adv | women | CB | 52 | top | 2 | Katie Bowen | New Zealand Women's | 255 | 0.001 | 0.981 |
| rarity_x_adv | women | CB | 52 | top | 3 | Sherida Spitse | Netherlands Women's | 671 | 0.001 | 0.962 |
| rarity_x_adv | women | CB | 52 | top | 4 | Diana Catarina Ribeiro Gomes | Portugal Women's | 381 | 0.001 | 0.942 |
| rarity_x_adv | women | CB | 52 | top | 5 | Janice Cayman | Belgium Women's | 204 | 0.001 | 0.923 |
| rarity_x_adv | women | CB | 52 | bottom | 1 | Sara Doorsoun-Khajeh | Germany Women's | 249 | 0.000 | 0.019 |
| rarity_x_adv | women | CB | 52 | bottom | 2 | Eva Nyström | WNT Finland | 282 | 0.000 | 0.038 |
| rarity_x_adv | women | CB | 52 | bottom | 3 | Luana Bühler | Switzerland Women's | 250 | 0.000 | 0.058 |
| rarity_x_adv | women | CM | 16 | top | 1 | Clara Mateo | France Women's | 278 | 0.003 | 1.000 |
| rarity_x_adv | women | CM | 16 | top | 2 | Arianna Caruso | Italy Women's | 323 | 0.003 | 0.938 |
| rarity_x_adv | women | CM | 16 | top | 3 | Lindsey Michelle Heaps | United States Women's | 236 | 0.002 | 0.875 |
| rarity_x_adv | women | CM | 16 | top | 4 | Alexia Putellas Segura | Spain Women's | 635 | 0.002 | 0.812 |
| rarity_x_adv | women | CM | 16 | top | 5 | Aitana Bonmati Conca | Spain Women's | 1293 | 0.002 | 0.750 |
| rarity_x_adv | women | CM | 16 | bottom | 1 | Jill Roord | Netherlands Women's | 510 | 0.001 | 0.062 |
| rarity_x_adv | women | CM | 16 | bottom | 2 | Tatiana Vanessa Ferreira Pinto | Portugal Women's | 352 | 0.001 | 0.125 |
| rarity_x_adv | women | CM | 16 | bottom | 3 | Kathrine Møller Kühl | Denmark Women's | 242 | 0.001 | 0.188 |
| rarity_x_adv | women | DM | 25 | top | 1 | Kerolin Nicoli Israel Ferraz | Brazil Women's | 208 | 0.003 | 1.000 |
| rarity_x_adv | women | DM | 25 | top | 2 | Frida Maanum | Norway Women's | 361 | 0.002 | 0.960 |
| rarity_x_adv | women | DM | 25 | top | 3 | Sara Däbritz | Germany Women's | 433 | 0.002 | 0.920 |
| rarity_x_adv | women | DM | 25 | top | 4 | Ingrid Filippa Angeldal | Sweden Women's | 709 | 0.002 | 0.880 |
| rarity_x_adv | women | DM | 25 | top | 5 | Yui Hasegawa | Japan Women's | 308 | 0.002 | 0.840 |
| rarity_x_adv | women | DM | 25 | bottom | 1 | Lia Wälti | Switzerland Women's | 615 | 0.001 | 0.040 |
| rarity_x_adv | women | DM | 25 | bottom | 2 | Charlotte Bilbault | France Women's | 214 | 0.001 | 0.080 |
| rarity_x_adv | women | DM | 25 | bottom | 3 | Elin Ingrid Johanna Rubensson | Sweden Women's | 331 | 0.001 | 0.120 |
| rarity_x_adv | women | FB | 37 | top | 1 | Janni Thomsen | Denmark Women's | 294 | 0.002 | 1.000 |
| rarity_x_adv | women | FB | 37 | top | 2 | Victoria Pelova | Netherlands Women's | 499 | 0.002 | 0.973 |
| rarity_x_adv | women | FB | 37 | top | 3 | Risa Shimizu | Japan Women's | 210 | 0.002 | 0.946 |
| rarity_x_adv | women | FB | 37 | top | 4 | Nadine Riesen | Switzerland Women's | 254 | 0.002 | 0.919 |
| rarity_x_adv | women | FB | 37 | top | 5 | Selma Bacha | France Women's | 606 | 0.002 | 0.892 |
| rarity_x_adv | women | FB | 37 | bottom | 1 | Thea Bjelde | Norway Women's | 235 | 0.001 | 0.027 |
| rarity_x_adv | women | FB | 37 | bottom | 2 | Felicitas Rauch | Germany Women's | 332 | 0.001 | 0.054 |
| rarity_x_adv | women | FB | 37 | bottom | 3 | Oihane Hernández Zurbano | Spain Women's | 270 | 0.001 | 0.081 |
| rarity_x_adv | women | FW | 10 | top | 1 | Emma Stina Blackstenius | Sweden Women's | 308 | 0.004 | 1.000 |
| rarity_x_adv | women | FW | 10 | top | 2 | Esther Gonzalez Rodríguez | Spain Women's | 272 | 0.003 | 0.900 |
| rarity_x_adv | women | FW | 10 | top | 3 | Jéssica da Silva | Portugal Women's | 226 | 0.003 | 0.800 |
| rarity_x_adv | women | FW | 10 | top | 4 | Tessa Wullaert | Belgium Women's | 265 | 0.003 | 0.700 |
| rarity_x_adv | women | FW | 10 | top | 5 | Mary Boio Fowler | Australia Women's | 204 | 0.003 | 0.600 |
| rarity_x_adv | women | FW | 10 | bottom | 1 | Diana Micaela Abreu de Sousa e Silva | Portugal Women's | 217 | 0.002 | 0.100 |
| rarity_x_adv | women | FW | 10 | bottom | 2 | Ana-Maria Crnogorčević | Switzerland Women's | 295 | 0.002 | 0.200 |
| rarity_x_adv | women | FW | 10 | bottom | 3 | Ramona Bachmann | Switzerland Women's | 330 | 0.002 | 0.300 |
| rarity_x_adv | women | GK | 7 | top | 1 | Daphne van Domselaar | Netherlands Women's | 259 | 0.001 | 1.000 |
| rarity_x_adv | women | GK | 7 | top | 2 | Ann-Katrin Berger | Germany Women's | 216 | 0.001 | 0.857 |
| rarity_x_adv | women | GK | 7 | top | 3 | Mary Alexandra Earps | England Women's | 209 | 0.000 | 0.714 |
| rarity_x_adv | women | GK | 7 | top | 4 | Merle Frohms | Germany Women's | 237 | 0.000 | 0.571 |
| rarity_x_adv | women | GK | 7 | top | 5 | Gaëlle Thalmann | Switzerland Women's | 207 | 0.000 | 0.429 |
| rarity_x_adv | women | GK | 7 | bottom | 1 | Laura Giuliani | Italy Women's | 201 | 0.000 | 0.143 |
| rarity_x_adv | women | GK | 7 | bottom | 2 | Pauline Peyraud Magnin | France Women's | 271 | 0.000 | 0.286 |
| rarity_x_adv | women | GK | 7 | bottom | 3 | Gaëlle Thalmann | Switzerland Women's | 207 | 0.000 | 0.429 |
| rarity_x_adv | women | W | 17 | top | 1 | Athenea del Castillo Belvide | Spain Women's | 260 | 0.004 | 1.000 |
| rarity_x_adv | women | W | 17 | top | 2 | Bethany Mead | England Women's | 251 | 0.004 | 0.941 |
| rarity_x_adv | women | W | 17 | top | 3 | Caitlin Jade Foord | Australia Women's | 212 | 0.004 | 0.882 |
| rarity_x_adv | women | W | 17 | top | 4 | Johanna Rytting-Kaneryd | Sweden Women's | 470 | 0.003 | 0.824 |
| rarity_x_adv | women | W | 17 | top | 5 | Delphine Cascarino | France Women's | 347 | 0.003 | 0.765 |
| rarity_x_adv | women | W | 17 | bottom | 1 | Linda Lizeth Caicedo Alegría | Colombia Women's | 226 | 0.002 | 0.059 |
| rarity_x_adv | women | W | 17 | bottom | 2 | Svenja Huth | Germany Women's | 421 | 0.002 | 0.118 |
| rarity_x_adv | women | W | 17 | bottom | 3 | Claudia Pina Medina | Spain Women's | 253 | 0.002 | 0.176 |
| rarity | men | AM | 20 | top | 1 | Sergej Milinković-Savić | Serbia | 208 | 0.540 | 1.000 |
| rarity | men | AM | 20 | top | 2 | Eden Hazard | Belgium | 281 | 0.538 | 0.950 |
| rarity | men | AM | 20 | top | 3 | Bruno Miguel Borges Fernandes | Portugal | 645 | 0.529 | 0.900 |
| rarity | men | AM | 20 | top | 4 | Dušan Tadić | Serbia | 232 | 0.528 | 0.850 |
| rarity | men | AM | 20 | top | 5 | Jude Bellingham | England | 803 | 0.521 | 0.800 |
| rarity | men | AM | 20 | bottom | 1 | Philippe Coutinho Correia | Barcelona | 658 | 0.467 | 0.050 |
| rarity | men | AM | 20 | bottom | 2 | Amine Adli | Bayer Leverkusen | 447 | 0.482 | 0.100 |
| rarity | men | AM | 20 | bottom | 3 | Mason Mount | England | 294 | 0.485 | 0.150 |
| rarity | men | CB | 118 | top | 1 | Denis Vavro | Slovakia | 262 | 0.546 | 1.000 |
| rarity | men | CB | 118 | top | 2 | Tomáš Kalas | Czech Republic | 206 | 0.541 | 0.992 |
| rarity | men | CB | 118 | top | 3 | Fabian Lukas Schär | Switzerland | 509 | 0.536 | 0.983 |
| rarity | men | CB | 118 | top | 4 | Philipp Lienhart | Austria | 243 | 0.533 | 0.975 |
| rarity | men | CB | 118 | top | 5 | Montassar Omar Talbi | Tunisia | 218 | 0.533 | 0.966 |
| rarity | men | CB | 118 | bottom | 1 | Abdülkerim Bardakcı | Turkey | 225 | 0.441 | 0.008 |
| rarity | men | CB | 118 | bottom | 2 | Piero Martín Hincapié Reyna | Bayer Leverkusen | 1656 | 0.443 | 0.017 |
| rarity | men | CB | 118 | bottom | 3 | Óscar Mingueza García | Barcelona | 1718 | 0.446 | 0.025 |
| rarity | men | CM | 29 | top | 1 | Luka Modrić | Croatia | 1281 | 0.542 | 1.000 |
| rarity | men | CM | 29 | top | 2 | Jorge Resurrección Merodio | Spain | 669 | 0.530 | 0.966 |
| rarity | men | CM | 29 | top | 3 | Tomáš Souček | Czech Republic | 355 | 0.529 | 0.931 |
| rarity | men | CM | 29 | top | 4 | Aaron Ramsey | Wales | 449 | 0.516 | 0.897 |
| rarity | men | CM | 29 | top | 5 | Mateo Kovačić | Croatia | 1033 | 0.514 | 0.862 |
| rarity | men | CM | 29 | bottom | 1 | Moriba Kourouma Kourouma | Barcelona | 430 | 0.448 | 0.034 |
| rarity | men | CM | 29 | bottom | 2 | Carlos Soler Barragán | Paris Saint-Germain | 696 | 0.466 | 0.069 |
| rarity | men | CM | 29 | bottom | 3 | Bryan Cristante | Italy | 216 | 0.469 | 0.103 |
| rarity | men | DM | 66 | top | 1 | Toni Kroos | Germany | 1118 | 0.559 | 1.000 |
| rarity | men | DM | 66 | top | 2 | Giorgi Kochorashvili | Georgia | 226 | 0.554 | 0.985 |
| rarity | men | DM | 66 | top | 3 | Johan Gastien | Clermont Foot | 249 | 0.541 | 0.970 |
| rarity | men | DM | 66 | top | 4 | Aaron Mooy | Australia | 229 | 0.540 | 0.955 |
| rarity | men | DM | 66 | top | 5 | Federico Santiago Valverde Dipetta | Uruguay | 275 | 0.539 | 0.939 |
| rarity | men | DM | 66 | bottom | 1 | Leander Dendoncker | Belgium | 222 | 0.423 | 0.015 |
| rarity | men | DM | 66 | bottom | 2 | Morten Hjulmand | Denmark | 222 | 0.447 | 0.030 |
| rarity | men | DM | 66 | bottom | 3 | Axel Witsel | Belgium | 461 | 0.466 | 0.045 |
| rarity | men | FB | 72 | top | 1 | Oleksandr Karavaev | Ukraine | 287 | 0.534 | 1.000 |
| rarity | men | FB | 72 | top | 2 | Marcos Javier Acuña | Argentina | 226 | 0.533 | 0.986 |
| rarity | men | FB | 72 | top | 3 | Andrija Živković | Serbia | 229 | 0.530 | 0.972 |
| rarity | men | FB | 72 | top | 4 | Ferdi Erenay Kadıoğlu | Turkey | 286 | 0.522 | 0.958 |
| rarity | men | FB | 72 | top | 5 | Connor Roberts | Wales | 205 | 0.518 | 0.944 |
| rarity | men | FB | 72 | bottom | 1 | Emerson Palmieri dos Santos | Italy | 211 | 0.431 | 0.014 |
| rarity | men | FB | 72 | bottom | 2 | Warren Zaire Emery | Paris Saint-Germain | 546 | 0.439 | 0.028 |
| rarity | men | FB | 72 | bottom | 3 | Sergino Dest | Barcelona | 1646 | 0.441 | 0.042 |
| rarity | men | FW | 16 | top | 1 | Harry Kane | England | 480 | 0.525 | 1.000 |
| rarity | men | FW | 16 | top | 2 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | 8537 | 0.523 | 0.938 |
| rarity | men | FW | 16 | top | 3 | Arda Güler | Turkey | 240 | 0.521 | 0.875 |
| rarity | men | FW | 16 | top | 4 | Serge Gnabry | Germany | 269 | 0.514 | 0.812 |
| rarity | men | FW | 16 | top | 5 | Patrik Schick | Bayer Leverkusen | 447 | 0.502 | 0.750 |
| rarity | men | FW | 16 | bottom | 1 | Memphis Depay | Netherlands | 488 | 0.473 | 0.062 |
| rarity | men | FW | 16 | bottom | 2 | Kai Havertz | Germany | 351 | 0.484 | 0.125 |
| rarity | men | FW | 16 | bottom | 3 | Kylian Mbappé Lottin | Paris Saint-Germain | 3542 | 0.485 | 0.188 |
| rarity | men | GK | 11 | top | 1 | Marc-André ter Stegen | Barcelona | 845 | 0.513 | 1.000 |
| rarity | men | GK | 11 | top | 2 | Kasper Schmeichel | Denmark | 247 | 0.504 | 0.909 |
| rarity | men | GK | 11 | top | 3 | Manuel Neuer | Germany | 371 | 0.483 | 0.818 |
| rarity | men | GK | 11 | top | 4 | Jordan Pickford | England | 549 | 0.477 | 0.727 |
| rarity | men | GK | 11 | top | 5 | Unai Simón Mendibil | Spain | 522 | 0.468 | 0.636 |
| rarity | men | GK | 11 | bottom | 1 | Yassine Bounou | Morocco | 205 | 0.455 | 0.091 |
| rarity | men | GK | 11 | bottom | 2 | Lukáš Hrádecký | Bayer Leverkusen | 641 | 0.460 | 0.182 |
| rarity | men | GK | 11 | bottom | 3 | Yann Sommer | Switzerland | 334 | 0.461 | 0.273 |
| rarity | men | W | 37 | top | 1 | Ruslan Malinovskiy | Ukraine | 220 | 0.580 | 1.000 |
| rarity | men | W | 37 | top | 2 | Rafael Alexandre Conceição Leão | Portugal | 204 | 0.555 | 0.973 |
| rarity | men | W | 37 | top | 3 | Ivan Perišić | Croatia | 447 | 0.540 | 0.946 |
| rarity | men | W | 37 | top | 4 | Lorenzo Insigne | Italy | 380 | 0.538 | 0.919 |
| rarity | men | W | 37 | top | 5 | Sofiane Boufal | Morocco | 228 | 0.538 | 0.892 |
| rarity | men | W | 37 | bottom | 1 | Timothy Weah | United States | 213 | 0.452 | 0.027 |
| rarity | men | W | 37 | bottom | 2 | Anssumane Fati | Barcelona | 330 | 0.453 | 0.054 |
| rarity | men | W | 37 | bottom | 3 | Daniel Olmo Carvajal | Spain | 805 | 0.456 | 0.081 |
| rarity | women | AM | 6 | top | 1 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | 279 | 0.537 | 1.000 |
| rarity | women | AM | 6 | top | 2 | Lauren James | England Women's | 442 | 0.515 | 0.833 |
| rarity | women | AM | 6 | top | 3 | Pernille Mosegaard Harder | Denmark Women's | 320 | 0.505 | 0.667 |
| rarity | women | AM | 6 | top | 4 | Leicy Maria Santos Herrera | Colombia Women's | 237 | 0.499 | 0.500 |
| rarity | women | AM | 6 | top | 5 | Ella Toone | England Women's | 498 | 0.496 | 0.333 |
| rarity | women | AM | 6 | bottom | 1 | Kosovare Asllani | Sweden Women's | 487 | 0.461 | 0.167 |
| rarity | women | AM | 6 | bottom | 2 | Ella Toone | England Women's | 498 | 0.496 | 0.333 |
| rarity | women | AM | 6 | bottom | 3 | Leicy Maria Santos Herrera | Colombia Women's | 237 | 0.499 | 0.500 |
| rarity | women | CB | 52 | top | 1 | Katie Bowen | New Zealand Women's | 255 | 0.530 | 1.000 |
| rarity | women | CB | 52 | top | 2 | Leah Williamson | England Women's | 929 | 0.525 | 0.981 |
| rarity | women | CB | 52 | top | 3 | Sherida Spitse | Netherlands Women's | 671 | 0.516 | 0.962 |
| rarity | women | CB | 52 | top | 4 | Kadeisha Buchanan | Canada Women's | 210 | 0.516 | 0.942 |
| rarity | women | CB | 52 | top | 5 | Marina Hegering | Germany Women's | 472 | 0.513 | 0.923 |
| rarity | women | CB | 52 | bottom | 1 | Sara Doorsoun-Khajeh | Germany Women's | 249 | 0.433 | 0.019 |
| rarity | women | CB | 52 | bottom | 2 | Laia Aleixandri López | Spain Women's | 516 | 0.438 | 0.038 |
| rarity | women | CB | 52 | bottom | 3 | Noelle Maritz | Switzerland Women's | 614 | 0.438 | 0.058 |
| rarity | women | CM | 16 | top | 1 | Jill Roord | Netherlands Women's | 510 | 0.518 | 1.000 |
| rarity | women | CM | 16 | top | 2 | Aitana Bonmati Conca | Spain Women's | 1293 | 0.512 | 0.938 |
| rarity | women | CM | 16 | top | 3 | Tatiana Vanessa Ferreira Pinto | Portugal Women's | 352 | 0.511 | 0.875 |
| rarity | women | CM | 16 | top | 4 | Danielle van de Donk | Netherlands Women's | 403 | 0.510 | 0.812 |
| rarity | women | CM | 16 | top | 5 | Kathrine Møller Kühl | Denmark Women's | 242 | 0.505 | 0.750 |
| rarity | women | CM | 16 | bottom | 1 | Lina Magull | Germany Women's | 240 | 0.448 | 0.062 |
| rarity | women | CM | 16 | bottom | 2 | Arianna Caruso | Italy Women's | 323 | 0.458 | 0.125 |
| rarity | women | CM | 16 | bottom | 3 | Coumba Sow | Switzerland Women's | 232 | 0.458 | 0.188 |
| rarity | women | DM | 25 | top | 1 | Kerolin Nicoli Israel Ferraz | Brazil Women's | 208 | 0.530 | 1.000 |
| rarity | women | DM | 25 | top | 2 | Elin Ingrid Johanna Rubensson | Sweden Women's | 331 | 0.524 | 0.960 |
| rarity | women | DM | 25 | top | 3 | Fuka Nagano | Japan Women's | 219 | 0.518 | 0.920 |
| rarity | women | DM | 25 | top | 4 | Lena Oberdorf | Germany Women's | 313 | 0.516 | 0.880 |
| rarity | women | DM | 25 | top | 5 | Keira Walsh | England Women's | 985 | 0.516 | 0.840 |
| rarity | women | DM | 25 | bottom | 1 | Sjoeke Nüsken | Germany Women's | 217 | 0.452 | 0.040 |
| rarity | women | DM | 25 | bottom | 2 | Ingrid Filippa Angeldal | Sweden Women's | 709 | 0.467 | 0.080 |
| rarity | women | DM | 25 | bottom | 3 | Karen Holmgaard | Denmark Women's | 281 | 0.473 | 0.120 |
| rarity | women | FB | 37 | top | 1 | Ana Borges | Portugal Women's | 326 | 0.524 | 1.000 |
| rarity | women | FB | 37 | top | 2 | Esmee Brugts | Netherlands Women's | 370 | 0.522 | 0.973 |
| rarity | women | FB | 37 | top | 3 | Janni Thomsen | Denmark Women's | 294 | 0.515 | 0.946 |
| rarity | women | FB | 37 | top | 4 | Joana Filipa Gaspar Silva Marchão | Portugal Women's | 251 | 0.513 | 0.919 |
| rarity | women | FB | 37 | top | 5 | Risa Shimizu | Japan Women's | 210 | 0.508 | 0.892 |
| rarity | women | FB | 37 | bottom | 1 | Olga  Carmona García | Spain Women's | 799 | 0.424 | 0.027 |
| rarity | women | FB | 37 | bottom | 2 | Thea Bjelde | Norway Women's | 235 | 0.437 | 0.054 |
| rarity | women | FB | 37 | bottom | 3 | Eseosa Aigbogun | Switzerland Women's | 280 | 0.441 | 0.081 |
| rarity | women | FW | 10 | top | 1 | Tessa Wullaert | Belgium Women's | 265 | 0.526 | 1.000 |
| rarity | women | FW | 10 | top | 2 | Diana Micaela Abreu de Sousa e Silva | Portugal Women's | 217 | 0.524 | 0.900 |
| rarity | women | FW | 10 | top | 3 | Mary Boio Fowler | Australia Women's | 204 | 0.522 | 0.800 |
| rarity | women | FW | 10 | top | 4 | Alessia Russo | England Women's | 390 | 0.516 | 0.700 |
| rarity | women | FW | 10 | top | 5 | Ramona Bachmann | Switzerland Women's | 330 | 0.511 | 0.600 |
| rarity | women | FW | 10 | bottom | 1 | Jéssica da Silva | Portugal Women's | 226 | 0.477 | 0.100 |
| rarity | women | FW | 10 | bottom | 2 | Ana-Maria Crnogorčević | Switzerland Women's | 295 | 0.482 | 0.200 |
| rarity | women | FW | 10 | bottom | 3 | Esther Gonzalez Rodríguez | Spain Women's | 272 | 0.490 | 0.300 |
| rarity | women | GK | 7 | top | 1 | Daphne van Domselaar | Netherlands Women's | 259 | 0.481 | 1.000 |
| rarity | women | GK | 7 | top | 2 | Ann-Katrin Berger | Germany Women's | 216 | 0.478 | 0.857 |
| rarity | women | GK | 7 | top | 3 | Mary Alexandra Earps | England Women's | 209 | 0.457 | 0.714 |
| rarity | women | GK | 7 | top | 4 | Pauline Peyraud Magnin | France Women's | 271 | 0.440 | 0.571 |
| rarity | women | GK | 7 | top | 5 | Merle Frohms | Germany Women's | 237 | 0.437 | 0.429 |
| rarity | women | GK | 7 | bottom | 1 | Laura Giuliani | Italy Women's | 201 | 0.415 | 0.143 |
| rarity | women | GK | 7 | bottom | 2 | Gaëlle Thalmann | Switzerland Women's | 207 | 0.428 | 0.286 |
| rarity | women | GK | 7 | bottom | 3 | Merle Frohms | Germany Women's | 237 | 0.437 | 0.429 |
| rarity | women | W | 17 | top | 1 | Jule Brand | Germany Women's | 414 | 0.527 | 1.000 |
| rarity | women | W | 17 | top | 2 | María Francesca Caldentey Oliver | Spain Women's | 1179 | 0.517 | 0.941 |
| rarity | women | W | 17 | top | 3 | Linda Lizeth Caicedo Alegría | Colombia Women's | 226 | 0.513 | 0.882 |
| rarity | women | W | 17 | top | 4 | Fridolina Rolfö | Sweden Women's | 504 | 0.508 | 0.824 |
| rarity | women | W | 17 | top | 5 | Svenja Huth | Germany Women's | 421 | 0.501 | 0.765 |
| rarity | women | W | 17 | bottom | 1 | Claudia Pina Medina | Spain Women's | 253 | 0.451 | 0.059 |
| rarity | women | W | 17 | bottom | 2 | Athenea del Castillo Belvide | Spain Women's | 260 | 0.474 | 0.118 |
| rarity | women | W | 17 | bottom | 3 | Delphine Cascarino | France Women's | 347 | 0.476 | 0.176 |
| advantage | men | AM | 20 | top | 1 | Christoph Baumgartner | Austria | 254 | 0.003 | 1.000 |
| advantage | men | AM | 20 | top | 2 | Mason Mount | England | 294 | 0.002 | 0.950 |
| advantage | men | AM | 20 | top | 3 | Amine Adli | Bayer Leverkusen | 447 | 0.002 | 0.900 |
| advantage | men | AM | 20 | top | 4 | Kevin De Bruyne | Belgium | 578 | 0.002 | 0.850 |
| advantage | men | AM | 20 | top | 5 | Xherdan Shaqiri | Switzerland | 343 | 0.002 | 0.800 |
| advantage | men | AM | 20 | bottom | 1 | Thomas Müller | Germany | 281 | -0.002 | 0.050 |
| advantage | men | AM | 20 | bottom | 2 | Adam Hložek | Bayer Leverkusen | 271 | -0.001 | 0.100 |
| advantage | men | AM | 20 | bottom | 3 | Ruben Vargas | Switzerland | 345 | -0.001 | 0.150 |
| advantage | men | CB | 118 | top | 1 | Romain Saïss | Morocco | 204 | 0.001 | 1.000 |
| advantage | men | CB | 118 | top | 2 | Kamal Miller | Canada | 417 | 0.000 | 0.992 |
| advantage | men | CB | 118 | top | 3 | Joachim Andersen | Denmark | 630 | 0.000 | 0.983 |
| advantage | men | CB | 118 | top | 4 | Tomáš Kalas | Czech Republic | 206 | 0.000 | 0.975 |
| advantage | men | CB | 118 | top | 5 | Leonardo Bonucci | Italy | 562 | 0.000 | 0.966 |
| advantage | men | CB | 118 | bottom | 1 | Nordi Mukiele Mulere | Paris Saint-Germain | 495 | -0.001 | 0.008 |
| advantage | men | CB | 118 | bottom | 2 | Grant Campbell Hanley | Scotland | 272 | -0.001 | 0.017 |
| advantage | men | CB | 118 | bottom | 3 | Jason Denayer | Belgium | 214 | -0.001 | 0.025 |
| advantage | men | CM | 29 | top | 1 | Piotr Zieliński | Poland | 508 | 0.001 | 1.000 |
| advantage | men | CM | 29 | top | 2 | Aaron Ramsey | Wales | 449 | 0.001 | 0.966 |
| advantage | men | CM | 29 | top | 3 | N'Golo Kanté | France | 617 | 0.001 | 0.931 |
| advantage | men | CM | 29 | top | 4 | Luka Modrić | Croatia | 1281 | 0.001 | 0.897 |
| advantage | men | CM | 29 | top | 5 | Yunus Dimoara Musah | United States | 236 | 0.000 | 0.862 |
| advantage | men | CM | 29 | bottom | 1 | Alexis Mac Allister | Argentina | 320 | -0.002 | 0.034 |
| advantage | men | CM | 29 | bottom | 2 | Carlos Soler Barragán | Paris Saint-Germain | 696 | -0.002 | 0.069 |
| advantage | men | CM | 29 | bottom | 3 | Oleksandr Zinchenko | Ukraine | 478 | -0.001 | 0.103 |
| advantage | men | DM | 66 | top | 1 | Giorgi Kochorashvili | Georgia | 226 | 0.001 | 1.000 |
| advantage | men | DM | 66 | top | 2 | Federico Santiago Valverde Dipetta | Uruguay | 275 | 0.001 | 0.985 |
| advantage | men | DM | 66 | top | 3 | Kristjan Asllani | Albania | 216 | 0.001 | 0.970 |
| advantage | men | DM | 66 | top | 4 | Tijjani Reijnders | Netherlands | 364 | 0.001 | 0.955 |
| advantage | men | DM | 66 | top | 5 | Paul Pogba | France | 340 | 0.001 | 0.939 |
| advantage | men | DM | 66 | bottom | 1 | Morten Hjulmand | Denmark | 222 | -0.002 | 0.015 |
| advantage | men | DM | 66 | bottom | 2 | Ander Herrera Agüera | Paris Saint-Germain | 340 | -0.001 | 0.030 |
| advantage | men | DM | 66 | bottom | 3 | Aïssa Bilal Laïdouni | Tunisia | 207 | -0.001 | 0.045 |
| advantage | men | FB | 72 | top | 1 | Yannick Ferreira Carrasco | Belgium | 343 | 0.002 | 1.000 |
| advantage | men | FB | 72 | top | 2 | Nathan Tella | Bayer Leverkusen | 317 | 0.002 | 0.986 |
| advantage | men | FB | 72 | top | 3 | Andrija Živković | Serbia | 229 | 0.001 | 0.972 |
| advantage | men | FB | 72 | top | 4 | Alphonso Davies | Canada | 223 | 0.001 | 0.958 |
| advantage | men | FB | 72 | top | 5 | Joakim Mæhle | Denmark | 699 | 0.001 | 0.944 |
| advantage | men | FB | 72 | bottom | 1 | Nicolás Alejandro Tagliafico | Argentina | 211 | -0.002 | 0.014 |
| advantage | men | FB | 72 | bottom | 2 | Daniel Wass | Denmark | 260 | -0.002 | 0.028 |
| advantage | men | FB | 72 | bottom | 3 | Sergi Roberto Carnicer | Barcelona | 980 | -0.002 | 0.042 |
| advantage | men | FW | 16 | top | 1 | Victor Okoh Boniface | Bayer Leverkusen | 560 | 0.004 | 1.000 |
| advantage | men | FW | 16 | top | 2 | Kylian Mbappé Lottin | Paris Saint-Germain | 3542 | 0.003 | 0.938 |
| advantage | men | FW | 16 | top | 3 | Lionel Andrés Messi Cuccittini | Paris Saint-Germain | 8537 | 0.002 | 0.875 |
| advantage | men | FW | 16 | top | 4 | Kai Havertz | Germany | 351 | 0.001 | 0.812 |
| advantage | men | FW | 16 | top | 5 | Memphis Depay | Netherlands | 488 | 0.001 | 0.750 |
| advantage | men | FW | 16 | bottom | 1 | Hugo Ekitike | Paris Saint-Germain | 369 | -0.002 | 0.062 |
| advantage | men | FW | 16 | bottom | 2 | Serge Gnabry | Germany | 269 | -0.002 | 0.125 |
| advantage | men | FW | 16 | bottom | 3 | Robert Lewandowski | Poland | 241 | -0.001 | 0.188 |
| advantage | men | GK | 11 | top | 1 | Manuel Neuer | Germany | 371 | -0.000 | 1.000 |
| advantage | men | GK | 11 | top | 2 | Kasper Schmeichel | Denmark | 247 | -0.000 | 0.909 |
| advantage | men | GK | 11 | top | 3 | Thibaut Courtois | Belgium | 225 | -0.000 | 0.818 |
| advantage | men | GK | 11 | top | 4 | Dominik Livaković | Croatia | 259 | -0.000 | 0.727 |
| advantage | men | GK | 11 | top | 5 | Marc-André ter Stegen | Barcelona | 845 | -0.000 | 0.636 |
| advantage | men | GK | 11 | bottom | 1 | Yann Sommer | Switzerland | 334 | -0.001 | 0.091 |
| advantage | men | GK | 11 | bottom | 2 | Lukáš Hrádecký | Bayer Leverkusen | 641 | -0.000 | 0.182 |
| advantage | men | GK | 11 | bottom | 3 | Jordan Pickford | England | 549 | -0.000 | 0.273 |
| advantage | men | W | 37 | top | 1 | Cody Mathès Gakpo | Netherlands | 323 | 0.003 | 1.000 |
| advantage | men | W | 37 | top | 2 | Lamine Yamal Nasraoui Ebana | Spain | 296 | 0.003 | 0.973 |
| advantage | men | W | 37 | top | 3 | Ruslan Malinovskiy | Ukraine | 220 | 0.003 | 0.946 |
| advantage | men | W | 37 | top | 4 | Jamal Musiala | Germany | 558 | 0.003 | 0.919 |
| advantage | men | W | 37 | top | 5 | Sofiane Boufal | Morocco | 228 | 0.003 | 0.892 |
| advantage | men | W | 37 | bottom | 1 | Andrej Kramarić | Croatia | 442 | -0.001 | 0.027 |
| advantage | men | W | 37 | bottom | 2 | Lovro Majer | Croatia | 348 | -0.001 | 0.054 |
| advantage | men | W | 37 | bottom | 3 | Timothy Weah | United States | 213 | -0.001 | 0.081 |
| advantage | women | AM | 6 | top | 1 | Pernille Mosegaard Harder | Denmark Women's | 320 | 0.004 | 1.000 |
| advantage | women | AM | 6 | top | 2 | Francisca Ramos Ribeiro Nazareth Sousa | Portugal Women's | 279 | 0.003 | 0.833 |
| advantage | women | AM | 6 | top | 3 | Lauren James | England Women's | 442 | 0.002 | 0.667 |
| advantage | women | AM | 6 | top | 4 | Kosovare Asllani | Sweden Women's | 487 | 0.001 | 0.500 |
| advantage | women | AM | 6 | top | 5 | Leicy Maria Santos Herrera | Colombia Women's | 237 | 0.000 | 0.333 |
| advantage | women | AM | 6 | bottom | 1 | Ella Toone | England Women's | 498 | -0.000 | 0.167 |
| advantage | women | AM | 6 | bottom | 2 | Leicy Maria Santos Herrera | Colombia Women's | 237 | 0.000 | 0.333 |
| advantage | women | AM | 6 | bottom | 3 | Kosovare Asllani | Sweden Women's | 487 | 0.001 | 0.500 |
| advantage | women | CB | 52 | top | 1 | Katie Bowen | New Zealand Women's | 255 | 0.001 | 1.000 |
| advantage | women | CB | 52 | top | 2 | Rafaelle Leone Carvalho Souza | Brazil Women's | 278 | 0.001 | 0.981 |
| advantage | women | CB | 52 | top | 3 | Glódís Perla Viggósdóttir | Iceland Women's | 232 | 0.001 | 0.962 |
| advantage | women | CB | 52 | top | 4 | Diana Catarina Ribeiro Gomes | Portugal Women's | 381 | 0.000 | 0.942 |
| advantage | women | CB | 52 | top | 5 | Kadeisha Buchanan | Canada Women's | 210 | 0.000 | 0.923 |
| advantage | women | CB | 52 | bottom | 1 | María Méndez Fernández | Spain Women's | 295 | -0.001 | 0.019 |
| advantage | women | CB | 52 | bottom | 2 | Rikke Læntver Sevecke | Denmark Women's | 315 | -0.001 | 0.038 |
| advantage | women | CB | 52 | bottom | 3 | Laia Aleixandri López | Spain Women's | 516 | -0.001 | 0.058 |
| advantage | women | CM | 16 | top | 1 | Clara Mateo | France Women's | 278 | 0.001 | 1.000 |
| advantage | women | CM | 16 | top | 2 | Lindsey Michelle Heaps | United States Women's | 236 | 0.001 | 0.938 |
| advantage | women | CM | 16 | top | 3 | Arianna Caruso | Italy Women's | 323 | 0.001 | 0.875 |
| advantage | women | CM | 16 | top | 4 | Alexia Putellas Segura | Spain Women's | 635 | 0.000 | 0.812 |
| advantage | women | CM | 16 | top | 5 | Jennifer Hermoso Fuentes | Spain Women's | 493 | 0.000 | 0.750 |
| advantage | women | CM | 16 | bottom | 1 | Kathrine Møller Kühl | Denmark Women's | 242 | -0.001 | 0.062 |
| advantage | women | CM | 16 | bottom | 2 | Jill Roord | Netherlands Women's | 510 | -0.001 | 0.125 |
| advantage | women | CM | 16 | bottom | 3 | Lina Magull | Germany Women's | 240 | -0.001 | 0.188 |
| advantage | women | DM | 25 | top | 1 | Ingrid Filippa Angeldal | Sweden Women's | 709 | 0.001 | 1.000 |
| advantage | women | DM | 25 | top | 2 | Karen Holmgaard | Denmark Women's | 281 | 0.001 | 0.960 |
| advantage | women | DM | 25 | top | 3 | Frida Maanum | Norway Women's | 361 | 0.001 | 0.920 |
| advantage | women | DM | 25 | top | 4 | Sara Däbritz | Germany Women's | 433 | 0.000 | 0.880 |
| advantage | women | DM | 25 | top | 5 | Eveliina Summanen | WNT Finland | 276 | 0.000 | 0.840 |
| advantage | women | DM | 25 | bottom | 1 | Sjoeke Nüsken | Germany Women's | 217 | -0.001 | 0.040 |
| advantage | women | DM | 25 | bottom | 2 | Georgia Stanway | England Women's | 838 | -0.001 | 0.080 |
| advantage | women | DM | 25 | bottom | 3 | Sandie Toletti | France Women's | 501 | -0.001 | 0.120 |
| advantage | women | FB | 37 | top | 1 | Janni Thomsen | Denmark Women's | 294 | 0.001 | 1.000 |
| advantage | women | FB | 37 | top | 2 | Ana Borges | Portugal Women's | 326 | 0.001 | 0.973 |
| advantage | women | FB | 37 | top | 3 | Ellie Madison Carpenter | Australia Women's | 318 | 0.000 | 0.946 |
| advantage | women | FB | 37 | top | 4 | Victoria Pelova | Netherlands Women's | 499 | 0.000 | 0.919 |
| advantage | women | FB | 37 | top | 5 | Sakina Karchaoui | France Women's | 858 | 0.000 | 0.892 |
| advantage | women | FB | 37 | bottom | 1 | Leila Ouahabi El Ouahabi | Spain Women's | 285 | -0.002 | 0.027 |
| advantage | women | FB | 37 | bottom | 2 | Oihane Hernández Zurbano | Spain Women's | 270 | -0.002 | 0.054 |
| advantage | women | FB | 37 | bottom | 3 | Rachel Daly | England Women's | 609 | -0.001 | 0.081 |
| advantage | women | FW | 10 | top | 1 | Mary Boio Fowler | Australia Women's | 204 | 0.003 | 1.000 |
| advantage | women | FW | 10 | top | 2 | Emma Stina Blackstenius | Sweden Women's | 308 | 0.002 | 0.900 |
| advantage | women | FW | 10 | top | 3 | Tessa Wullaert | Belgium Women's | 265 | 0.001 | 0.800 |
| advantage | women | FW | 10 | top | 4 | Ramona Bachmann | Switzerland Women's | 330 | 0.001 | 0.700 |
| advantage | women | FW | 10 | top | 5 | Jéssica da Silva | Portugal Women's | 226 | 0.001 | 0.600 |
| advantage | women | FW | 10 | bottom | 1 | Alessia Russo | England Women's | 390 | -0.002 | 0.100 |
| advantage | women | FW | 10 | bottom | 2 | Esther Gonzalez Rodríguez | Spain Women's | 272 | -0.002 | 0.200 |
| advantage | women | FW | 10 | bottom | 3 | Diana Micaela Abreu de Sousa e Silva | Portugal Women's | 217 | -0.001 | 0.300 |
| advantage | women | GK | 7 | top | 1 | Ann-Katrin Berger | Germany Women's | 216 | -0.000 | 1.000 |
| advantage | women | GK | 7 | top | 2 | Daphne van Domselaar | Netherlands Women's | 259 | -0.000 | 0.857 |
| advantage | women | GK | 7 | top | 3 | Merle Frohms | Germany Women's | 237 | -0.000 | 0.714 |
| advantage | women | GK | 7 | top | 4 | Mary Alexandra Earps | England Women's | 209 | -0.000 | 0.571 |
| advantage | women | GK | 7 | top | 5 | Gaëlle Thalmann | Switzerland Women's | 207 | -0.001 | 0.429 |
| advantage | women | GK | 7 | bottom | 1 | Laura Giuliani | Italy Women's | 201 | -0.001 | 0.143 |
| advantage | women | GK | 7 | bottom | 2 | Pauline Peyraud Magnin | France Women's | 271 | -0.001 | 0.286 |
| advantage | women | GK | 7 | bottom | 3 | Gaëlle Thalmann | Switzerland Women's | 207 | -0.001 | 0.429 |
| advantage | women | W | 17 | top | 1 | Delphine Cascarino | France Women's | 347 | 0.003 | 1.000 |
| advantage | women | W | 17 | top | 2 | Athenea del Castillo Belvide | Spain Women's | 260 | 0.003 | 0.941 |
| advantage | women | W | 17 | top | 3 | Caitlin Jade Foord | Australia Women's | 212 | 0.003 | 0.882 |
| advantage | women | W | 17 | top | 4 | Johanna Rytting-Kaneryd | Sweden Women's | 470 | 0.002 | 0.824 |
| advantage | women | W | 17 | top | 5 | Bethany Mead | England Women's | 251 | 0.002 | 0.765 |
| advantage | women | W | 17 | bottom | 1 | Salma Paralluelo Ayingono | Spain Women's | 335 | -0.001 | 0.059 |
| advantage | women | W | 17 | bottom | 2 | Linda Lizeth Caicedo Alegría | Colombia Women's | 226 | 0.000 | 0.118 |
| advantage | women | W | 17 | bottom | 3 | Caroline Graham Hansen | Norway Women's | 410 | 0.000 | 0.176 |


## Known creative players: name resolution (`validation/known_creative_players.csv` (sha256 50e31708de4f, modified Mon Oct  5 19:32:15 2026))

| player_name | player_id | status |
|---|---|---|
| Lionel Messi | 5503.000 | ok |
| Neymar | 4320.000 | ok |
| Kevin De Bruyne | 3089.000 | ok |
| Luka Modrić | 5463.000 | ok |
| Pedri | 30486.000 | ok |
| Florian Wirtz | 40724.000 | ok |
| Jamal Musiala | 39565.000 | ok |
| Lamine Yamal | 316046.000 | ok |
| Toni Kroos | 5574.000 | ok |
| Bruno Fernandes | 5204.000 | ok |
| Bernardo Silva | 3193.000 | ok (player_id given) |
| Antoine Griezmann | 5487.000 | ok |
| Ángel Di María | 2995.000 | ok |
| Hakim Ziyech | 5237.000 | ok |
| Christian Eriksen | 3043.000 | ok |
| Hakan Çalhanoğlu | 7039.000 | ok |
| Arda Güler | 116587.000 | ok |
| Khvicha Kvaratskhelia |  | not found (or fewer than --min-decisions decisions) |
| Dani Olmo | 16532.000 | ok (player_id given) |
| Phil Foden | 4354.000 | ok |
| Jack Grealish |  | not found (or fewer than --min-decisions decisions) |
| Lorenzo Insigne | 7037.000 | ok |
| Marco Verratti | 3166.000 | ok |
| Ilkay Gündoğan | 10287.000 | ok |
| Xavi Simons | 39167.000 | ok |
| Piotr Zieliński | 5660.000 | ok |
| Aleksandr Golovin | 5177.000 | ok |
| João Cancelo | 7005.000 | ok |
| Trent Alexander-Arnold |  | not found (or fewer than --min-decisions decisions) |
| Alejandro Grimaldo | 10336.000 | ok |
| Aitana Bonmatí | 15284.000 | ok |
| Alexia Putellas | 10143.000 | ok |
| Jenni Hermoso | 10151.000 | ok (player_id given) |
| Mariona Caldentey | 10161.000 | ok |
| Caroline Graham Hansen | 10386.000 | ok |
| Guro Reiten | 10514.000 | ok |
| Pernille Harder | 10108.000 | ok |
| Kosovare Asllani | 10220.000 | ok |
| Lina Magull | 10259.000 | ok |
| Fran Kirby | 4641.000 | ok (player_id given) |
| Lauren James | 31531.000 | ok |
| Keira Walsh | 4658.000 | ok |
| Rose Lavelle |  | not found (or fewer than --min-decisions decisions) |
| Linda Caicedo | 386339.000 | ok |
| Yui Hasegawa | 25632.000 | ok |
| Daniëlle van de Donk | 10658.000 | ok |
| Debinha |  | not found (or fewer than --min-decisions decisions) |
| Kika Nazareth | 62913.000 | ok |


## Known creative players vs. others (within position groups)

| measure | known_players | auc_within_position | auc_se | auc_low | auc_high | mean_known_percentile | known_in_top_decile | diff_vs_advantage | diff_vs_advantage_low | diff_vs_advantage_high | diff_vs_rarity | diff_vs_rarity_low | diff_vs_rarity_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rarity | 42 | 0.588 | 0.048 | 0.499 | 0.680 | 0.589 | 0.095 | 0.033 | -0.071 | 0.141 |  |  |  |
| advantage | 42 | 0.554 | 0.047 | 0.470 | 0.640 | 0.557 | 0.095 |  |  |  | -0.033 | -0.141 | 0.071 |
| difficulty | 42 | 0.478 | 0.046 | 0.389 | 0.571 | 0.487 | 0.119 | -0.076 | -0.189 | 0.040 | -0.110 | -0.252 | 0.026 |
| surprise_x_adv | 42 | 0.583 | 0.048 | 0.497 | 0.667 | 0.584 | 0.143 | 0.029 | -0.025 | 0.086 | -0.005 | -0.119 | 0.104 |
| rarity_x_adv | 42 | 0.582 | 0.048 | 0.492 | 0.670 | 0.583 | 0.119 | 0.028 | -0.032 | 0.090 | -0.006 | -0.123 | 0.106 |
| rarity_x_adv_completed | 42 | 0.589 | 0.048 | 0.499 | 0.676 | 0.589 | 0.143 | 0.034 | -0.020 | 0.091 | 0.001 | -0.111 | 0.110 |
| rarity_x_td | 42 | 0.599 | 0.048 | 0.512 | 0.684 | 0.599 | 0.167 | 0.044 | -0.041 | 0.127 | 0.011 | -0.119 | 0.133 |
| rarity_x_upside | 42 | 0.564 | 0.048 | 0.473 | 0.653 | 0.567 | 0.119 | 0.010 | -0.053 | 0.075 | -0.024 | -0.148 | 0.094 |
| good_rarity | 42 | 0.645 | 0.048 | 0.562 | 0.726 | 0.642 | 0.214 | 0.091 | 0.026 | 0.158 | 0.058 | -0.017 | 0.129 |
| good_surprise | 42 | 0.650 | 0.047 | 0.566 | 0.731 | 0.646 | 0.190 | 0.097 | 0.020 | 0.179 | 0.063 | -0.002 | 0.126 |
| tail_rarity | 42 | 0.598 | 0.048 | 0.512 | 0.681 | 0.598 | 0.119 | 0.045 | -0.021 | 0.115 | 0.011 | -0.073 | 0.095 |
| c51_tail_rarity | 42 | 0.634 | 0.048 | 0.539 | 0.727 | 0.631 | 0.262 | 0.079 | -0.000 | 0.157 | 0.046 | -0.060 | 0.141 |
| unxpass | 42 | 0.645 | 0.048 | 0.557 | 0.728 | 0.641 | 0.238 | 0.090 | -0.001 | 0.182 | 0.057 | -0.013 | 0.126 |

