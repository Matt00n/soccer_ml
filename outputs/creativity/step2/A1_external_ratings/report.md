# External validation with FIFA ratings

768 outfield player-contexts with >= 150 in-scope decisions; 667 matched (499 contemporaneous). Correlations are Spearman of within (competition-season x position group) percentiles; 95% player-clustered bootstrap intervals in correlations.csv.

## Coverage

| competition_season | ambiguous | matched | not found |
|---|---|---|---|
| 1. Bundesliga 2023/2024 | 0 | 13 | 7 |
| FIFA World Cup 2022 | 5 | 150 | 1 |
| La Liga 2020/2021 | 1 | 22 | 3 |
| Ligue 1 2021/2022 | 0 | 21 | 0 |
| Ligue 1 2022/2023 | 0 | 19 | 3 |
| Major League Soccer 2023 | 0 | 0 | 5 |
| UEFA Euro 2020 | 1 | 137 | 2 |
| UEFA Euro 2024 | 5 | 121 | 3 |
| UEFA Women's Euro 2022 | 2 | 54 | 10 |
| UEFA Women's Euro 2025 | 2 | 34 | 16 |
| Women's World Cup 2023 | 2 | 96 | 33 |


## Correlations (contemporaneous ratings)

| role | attribute | advantage | c51_tail_rarity | good_rarity | good_surprise | rarity | rarity_x_adv | unxpass |
|---|---|---|---|---|---|---|---|---|
| convergent | attacking_short_passing | 0.058 | 0.132 | 0.090 | 0.117 | 0.036 | 0.038 | 0.131 |
| convergent | mentality_composure | 0.010 | 0.123 | 0.044 | 0.071 | 0.089 | 0.018 | 0.093 |
| convergent | mentality_vision | 0.015 | 0.187 | 0.123 | 0.171 | 0.135 | 0.109 | 0.145 |
| convergent | skill_ball_control | 0.054 | 0.185 | 0.137 | 0.140 | 0.098 | 0.090 | 0.103 |
| convergent | skill_curve | 0.032 | 0.186 | 0.135 | 0.133 | 0.137 | 0.111 | 0.164 |
| convergent | skill_dribbling | 0.053 | 0.174 | 0.159 | 0.162 | 0.121 | 0.117 | 0.088 |
| convergent | skill_long_passing | 0.055 | 0.206 | 0.149 | 0.185 | 0.106 | 0.094 | 0.211 |
| discriminant | defending_standing_tackle | -0.019 | -0.071 | -0.044 | -0.022 | -0.082 | -0.106 | 0.021 |
| discriminant | movement_sprint_speed | -0.003 | -0.017 | 0.047 | 0.036 | -0.018 | 0.042 | -0.070 |
| discriminant | power_jumping | 0.021 | -0.020 | -0.043 | -0.052 | 0.006 | -0.100 | -0.030 |
| discriminant | power_stamina | -0.002 | 0.022 | 0.042 | 0.039 | 0.052 | 0.021 | 0.051 |
| discriminant | power_strength | 0.074 | 0.054 | 0.068 | 0.077 | 0.011 | 0.030 | 0.124 |
| quality | overall | 0.058 | 0.122 | 0.117 | 0.114 | 0.071 | 0.012 | 0.145 |


## Partial correlations, controlling for overall rating

| role | attribute | advantage | c51_tail_rarity | good_rarity | good_surprise | rarity | rarity_x_adv | unxpass |
|---|---|---|---|---|---|---|---|---|
| convergent | attacking_short_passing | 0.020 | 0.059 | -0.003 | 0.045 | -0.031 | 0.046 | 0.028 |
| convergent | mentality_composure | -0.045 | 0.052 | -0.056 | -0.014 | 0.056 | 0.013 | -0.015 |
| convergent | mentality_vision | -0.026 | 0.143 | 0.065 | 0.129 | 0.116 | 0.130 | 0.072 |
| convergent | skill_ball_control | 0.017 | 0.141 | 0.077 | 0.084 | 0.068 | 0.120 | -0.004 |
| convergent | skill_curve | 0.005 | 0.146 | 0.091 | 0.091 | 0.118 | 0.120 | 0.109 |
| convergent | skill_dribbling | 0.025 | 0.129 | 0.114 | 0.120 | 0.099 | 0.132 | 0.010 |
| convergent | skill_long_passing | 0.019 | 0.170 | 0.095 | 0.149 | 0.080 | 0.122 | 0.155 |
| discriminant | defending_standing_tackle | -0.068 | -0.185 | -0.147 | -0.117 | -0.158 | -0.143 | -0.086 |
| discriminant | movement_sprint_speed | -0.018 | -0.051 | 0.018 | 0.007 | -0.038 | 0.040 | -0.113 |
| discriminant | power_jumping | 0.005 | -0.056 | -0.079 | -0.087 | -0.014 | -0.108 | -0.073 |
| discriminant | power_stamina | -0.034 | -0.042 | -0.016 | -0.017 | 0.021 | 0.018 | -0.021 |
| discriminant | power_strength | 0.062 | 0.024 | 0.041 | 0.050 | -0.007 | 0.028 | 0.092 |


## Vision: paired differences between measures

| measure | baseline | difference | low | high |
|---|---|---|---|---|
| good_surprise | advantage | 0.154 | 0.069 | 0.237 |
| good_surprise | rarity | 0.034 | -0.059 | 0.132 |
| good_surprise | rarity_x_adv | 0.059 | -0.029 | 0.146 |
| good_rarity | advantage | 0.106 | 0.024 | 0.189 |
| good_rarity | rarity | -0.015 | -0.105 | 0.073 |
| good_rarity | rarity_x_adv | 0.010 | -0.070 | 0.095 |
| c51_tail_rarity | advantage | 0.170 | 0.075 | 0.262 |
| c51_tail_rarity | rarity | 0.050 | -0.042 | 0.140 |
| c51_tail_rarity | rarity_x_adv | 0.075 | -0.015 | 0.164 |
| unxpass | advantage | 0.129 | 0.034 | 0.214 |
| unxpass | rarity | 0.009 | -0.087 | 0.099 |
| unxpass | rarity_x_adv | 0.034 | -0.062 | 0.135 |
| rarity | advantage | 0.120 | 0.008 | 0.240 |
| rarity | rarity_x_adv | 0.025 | -0.083 | 0.132 |
| advantage | rarity | -0.120 | -0.240 | -0.008 |
| advantage | rarity_x_adv | -0.095 | -0.169 | -0.020 |
| rarity_x_adv | advantage | 0.095 | 0.020 | 0.169 |
| rarity_x_adv | rarity | -0.025 | -0.132 | 0.083 |


## Known groups from FIFA labels (AUC of within-group percentiles)

| label | labelled | advantage | c51_tail_rarity | good_rarity | good_surprise | rarity | rarity_x_adv | unxpass |
|---|---|---|---|---|---|---|---|---|
| flair | 89 | 0.560 | 0.563 | 0.582 | 0.580 | 0.542 | 0.560 | 0.548 |
| long_passer | 70 | 0.545 | 0.575 | 0.564 | 0.566 | 0.545 | 0.529 | 0.612 |
| playmaker | 78 | 0.592 | 0.664 | 0.601 | 0.616 | 0.560 | 0.597 | 0.622 |
| technical_dribbler | 92 | 0.583 | 0.603 | 0.589 | 0.572 | 0.533 | 0.588 | 0.542 |

