# Step 2: robustness, creative opportunity, enabling

536223 decisions (469733 in scope); primary measure `z_good_surprise`; 95% match-cluster bootstrap (1000); q = Benjamini-Hochberg over the tested claims.

## Claims

| claim | estimate | low | high | p | unit | q_bh |
|---|---|---|---|---|---|---|
| B1 base: z_good_surprise -> shot | 0.0389 | 0.0362 | 0.0416 | 0.0000 | probability per SD | 0.0000 |
| B1 base: z_good_surprise -> goal | 0.0050 | 0.0039 | 0.0061 | 0.0000 | probability per SD | 0.0000 |
| B1 excluding the last 2 decision(s) -> shot | 0.0570 | 0.0533 | 0.0604 | 0.0000 | probability per SD | 0.0000 |
| B1 excluding the last 2 decision(s) -> goal | 0.0080 | 0.0065 | 0.0095 | 0.0000 | probability per SD | 0.0000 |
| B1 excluding last 2, originality and value jointly -> shot (rarity coefficient) | 0.0180 | 0.0143 | 0.0215 | 0.0000 | probability per SD | 0.0000 |
| B3 goal: scoring team: creativity change | -0.0607 | -0.0777 | -0.0454 | 0.0000 | SD | 0.0000 |
| B3 red card: team sent off: creativity change | -0.1057 | -0.1828 | -0.0206 | 0.0120 | SD | 0.0150 |
| B3 tactical shift: creativity change | 0.0227 | 0.0089 | 0.0364 | 0.0000 | SD | 0.0000 |
| B3 tactical substitution: creativity change | 0.0115 | -0.0017 | 0.0251 | 0.1060 | SD | 0.1136 |
| C2 goal: scoring team: z_opportunity | -0.0244 | -0.0487 | -0.0003 | 0.0480 | SD | 0.0554 |
| C2 goal: scoring team: z_use | -0.0567 | -0.0728 | -0.0418 | 0.0000 | SD | 0.0000 |
| B4 substitutes minus starters after 60' (within position) | -0.0068 | -0.0284 | 0.0156 | 0.5340 | SD | 0.5340 |
| B8 surprise premium (fine strata) | 0.0019 | 0.0017 | 0.0021 | 0.0000 | goal probability per unit rarity | 0.0000 |
| C3 creative action -> next teammate's opportunity | 0.0038 | 0.0025 | 0.0051 | 0.0000 | nats per SD | 0.0000 |
| C3 creative action -> next teammate's space (nearest opponent) | -0.1898 | -0.2085 | -0.1708 | 0.0000 | yd per SD | 0.0000 |
| A1 good_surprise vs. FIFA vision (within context x position) | 0.1709 | 0.0816 | 0.2664 |  | Spearman rho |  |


## Sections

- [Build-up creativity -> shots and goals: robustness](B1_buildup/summary.md)
- [Hidden contributions: creativity by role, credited vs. uncredited](B2_hidden_contributions/summary.md)
- [Event studies: windows, pre-trends, splits, opportunity vs. use (B3-B5, C2)](B3_event_studies/summary.md)
- [Substitutes vs. starters after 60', within position group](B4_fresh_legs/summary.md)
- [Power of team-level nulls; distribution of creativity within possessions](B6_power_distribution/summary.md)
- [Surprise premium: robustness](B8_surprise_premium/summary.md)
- [Creative opportunity and its use](C1_opportunity/summary.md)
- [Enabling: effect of a creative action on the next teammate's opportunity and space](C3_enabling/summary.md)
- [A1 external ratings](A1_external_ratings/report.md)