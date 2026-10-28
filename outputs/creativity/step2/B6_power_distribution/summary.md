# Power of team-level nulls; distribution of creativity within possessions

## team_level_power (13 rows)

| outcome | regressor | estimate | se | mde_80_power | n |
|---|---|---|---|---|---|
| xg_diff | level_mean_z | -0.0140 | 0.0569 | 0.1592 | 761 |
| xg_diff | gini_z | -0.0329 | 0.0551 | 0.1543 | 761 |
| xg_diff | opponent_strength | -0.5011 | 0.0491 | 0.1375 | 761 |
| goal_diff | level_mean_z | -0.0804 | 0.0980 | 0.2744 | 761 |
| goal_diff | gini_z | -0.0644 | 0.0877 | 0.2455 | 761 |
| goal_diff | opponent_strength | -0.5980 | 0.0734 | 0.2056 | 761 |
| points | level_mean_z | -0.0357 | 0.0552 | 0.1544 | 761 |
| points | gini_z | -0.0503 | 0.0573 | 0.1605 | 761 |
| points | opponent_strength | -0.3535 | 0.0456 | 0.1276 | 761 |
| xg_diff_h2 | h1_mean_z | -0.0538 | 0.0323 | 0.0904 | 761 |
| xg_diff_h2 | xg_diff_h1 | 0.1301 | 0.0583 | 0.1633 | 761 |
| xg_diff_h2 | opponent_strength | -0.2223 | 0.0333 | 0.0932 | 761 |
| failed_pass_rate | level_mean_z | 0.0038 | 0.0026 | 0.0072 | 761 |


## possession_distribution (8 rows)

| outcome | regressor | estimate | low | high | p | n | clusters | fixed_effects | spec |
|---|---|---|---|---|---|---|---|---|---|
| shot | two_plus | 0.0678 | 0.0570 | 0.0781 | 0.0000 | 47765 | 417 | stratum | two or more creative contributors, given max creativity |
| shot | max_z_good_surprise | 0.0333 | 0.0301 | 0.0367 | 0.0000 | 47765 | 417 | stratum | two or more creative contributors, given max creativity |
| shot | top_share | -0.0357 | -0.0403 | -0.0310 | 0.0000 | 42559 | 417 | stratum | concentration (top player's share), given max creativity |
| shot | max_z_good_surprise | 0.0428 | 0.0382 | 0.0475 | 0.0000 | 42559 | 417 | stratum | concentration (top player's share), given max creativity |
| goal | two_plus | 0.0109 | 0.0067 | 0.0148 | 0.0000 | 47765 | 417 | stratum | two or more creative contributors, given max creativity |
| goal | max_z_good_surprise | 0.0038 | 0.0025 | 0.0051 | 0.0000 | 47765 | 417 | stratum | two or more creative contributors, given max creativity |
| goal | top_share | -0.0063 | -0.0079 | -0.0048 | 0.0000 | 42559 | 417 | stratum | concentration (top player's share), given max creativity |
| goal | max_z_good_surprise | 0.0043 | 0.0024 | 0.0062 | 0.0000 | 42559 | 417 | stratum | concentration (top player's share), given max creativity |

