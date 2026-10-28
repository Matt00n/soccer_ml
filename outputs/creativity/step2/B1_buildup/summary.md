# Build-up creativity -> shots and goals: robustness

Linear probability models; possession creativity = max over the possession team's in-scope, non-shot decisions (z across possessions); strata = start value decile x third x play pattern x length; 95% match-cluster bootstrap.

## buildup_specifications (57 rows)

| outcome | regressor | estimate | low | high | p | n | clusters | fixed_effects | spec | sample | base_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| shot | max_z_good_surprise | 0.0389 | 0.0362 | 0.0416 | 0.0000 | 58890 | 417 | stratum | base: z_good_surprise | cross-fitted | 0.1442 |
| goal | max_z_good_surprise | 0.0050 | 0.0039 | 0.0061 | 0.0000 | 58890 | 417 | stratum | base: z_good_surprise | cross-fitted | 0.0172 |
| xg | max_z_good_surprise | 0.0052 | 0.0047 | 0.0058 | 0.0000 | 58890 | 417 | stratum | base: z_good_surprise | cross-fitted | 0.0172 |
| shot | max_z_good_rarity | 0.0484 | 0.0456 | 0.0512 | 0.0000 | 58890 | 417 | stratum | base: z_good_rarity | cross-fitted | 0.1442 |
| goal | max_z_good_rarity | 0.0071 | 0.0060 | 0.0083 | 0.0000 | 58890 | 417 | stratum | base: z_good_rarity | cross-fitted | 0.0172 |
| xg | max_z_good_rarity | 0.0069 | 0.0064 | 0.0075 | 0.0000 | 58890 | 417 | stratum | base: z_good_rarity | cross-fitted | 0.0172 |
| shot | max_z_c51_tail_rarity | 0.0743 | 0.0713 | 0.0776 | 0.0000 | 58890 | 417 | stratum | base: z_c51_tail_rarity | cross-fitted | 0.1442 |
| goal | max_z_c51_tail_rarity | 0.0098 | 0.0084 | 0.0112 | 0.0000 | 58890 | 417 | stratum | base: z_c51_tail_rarity | cross-fitted | 0.0172 |
| xg | max_z_c51_tail_rarity | 0.0100 | 0.0094 | 0.0107 | 0.0000 | 58890 | 417 | stratum | base: z_c51_tail_rarity | cross-fitted | 0.0172 |
| shot | max_z_rarity | 0.0151 | 0.0124 | 0.0177 | 0.0000 | 58890 | 417 | stratum | base: z_rarity | cross-fitted | 0.1442 |
| goal | max_z_rarity | 0.0023 | 0.0013 | 0.0033 | 0.0000 | 58890 | 417 | stratum | base: z_rarity | cross-fitted | 0.0172 |
| xg | max_z_rarity | 0.0018 | 0.0012 | 0.0023 | 0.0000 | 58890 | 417 | stratum | base: z_rarity | cross-fitted | 0.0172 |
| shot | max_z_advantage | 0.1049 | 0.0993 | 0.1107 | 0.0000 | 58890 | 417 | stratum | base: z_advantage | cross-fitted | 0.1442 |
| goal | max_z_advantage | 0.0249 | 0.0219 | 0.0278 | 0.0000 | 58890 | 417 | stratum | base: z_advantage | cross-fitted | 0.0172 |
| xg | max_z_advantage | 0.0216 | 0.0199 | 0.0233 | 0.0000 | 58890 | 417 | stratum | base: z_advantage | cross-fitted | 0.0172 |
| shot | max_z_good_surprise | 0.0389 | 0.0361 | 0.0416 | 0.0000 | 58890 | 417 | stratum+context | base + team-season FE | cross-fitted | 0.1442 |
| goal | max_z_good_surprise | 0.0049 | 0.0038 | 0.0060 | 0.0000 | 58890 | 417 | stratum+context | base + team-season FE | cross-fitted | 0.0172 |
| xg | max_z_good_surprise | 0.0052 | 0.0046 | 0.0057 | 0.0000 | 58890 | 417 | stratum+context | base + team-season FE | cross-fitted | 0.0172 |
| shot | max_z_good_surprise | 0.0386 | 0.0358 | 0.0414 | 0.0000 | 58890 | 417 | stratum+match_id | base + match FE | cross-fitted | 0.1442 |
| goal | max_z_good_surprise | 0.0049 | 0.0038 | 0.0061 | 0.0000 | 58890 | 417 | stratum+match_id | base + match FE | cross-fitted | 0.0172 |
| xg | max_z_good_surprise | 0.0052 | 0.0046 | 0.0057 | 0.0000 | 58890 | 417 | stratum+match_id | base + match FE | cross-fitted | 0.0172 |
| shot | max_z_good_surprise | 0.0102 | 0.0076 | 0.0128 | 0.0000 | 58890 | 417 | stratum | base + field progression | cross-fitted | 0.1442 |
| shot | max_x | 0.1809 | 0.1764 | 0.1858 | 0.0000 | 58890 | 417 | stratum | base + field progression | cross-fitted | 0.1442 |
| goal | max_z_good_surprise | 0.0008 | -0.0003 | 0.0019 | 0.1760 | 58890 | 417 | stratum | base + field progression | cross-fitted | 0.0172 |
| goal | max_x | 0.0265 | 0.0243 | 0.0287 | 0.0000 | 58890 | 417 | stratum | base + field progression | cross-fitted | 0.0172 |
| xg | max_z_good_surprise | 0.0014 | 0.0008 | 0.0019 | 0.0000 | 58890 | 417 | stratum | base + field progression | cross-fitted | 0.0172 |
| xg | max_x | 0.0244 | 0.0232 | 0.0255 | 0.0000 | 58890 | 417 | stratum | base + field progression | cross-fitted | 0.0172 |
| shot | max_raw_rarity | 0.0044 | 0.0016 | 0.0071 | 0.0000 | 58890 | 417 | stratum | originality and value jointly | cross-fitted | 0.1442 |
| shot | max_raw_advantage | 0.1045 | 0.0989 | 0.1105 | 0.0000 | 58890 | 417 | stratum | originality and value jointly | cross-fitted | 0.1442 |
| goal | max_raw_rarity | -0.0003 | -0.0013 | 0.0007 | 0.6060 | 58890 | 417 | stratum | originality and value jointly | cross-fitted | 0.0172 |
| goal | max_raw_advantage | 0.0250 | 0.0219 | 0.0279 | 0.0000 | 58890 | 417 | stratum | originality and value jointly | cross-fitted | 0.0172 |
| xg | max_raw_rarity | -0.0004 | -0.0010 | 0.0001 | 0.1000 | 58890 | 417 | stratum | originality and value jointly | cross-fitted | 0.0172 |
| xg | max_raw_advantage | 0.0217 | 0.0200 | 0.0233 | 0.0000 | 58890 | 417 | stratum | originality and value jointly | cross-fitted | 0.0172 |
| shot | max_z_good_surprise | 0.0530 | 0.0497 | 0.0561 | 0.0000 | 47732 | 417 | stratum | excluding the last 1 decision(s) | cross-fitted | 0.1606 |
| goal | max_z_good_surprise | 0.0068 | 0.0055 | 0.0081 | 0.0000 | 47732 | 417 | stratum | excluding the last 1 decision(s) | cross-fitted | 0.0187 |
| xg | max_z_good_surprise | 0.0069 | 0.0063 | 0.0076 | 0.0000 | 47732 | 417 | stratum | excluding the last 1 decision(s) | cross-fitted | 0.0184 |
| shot | max_z_good_surprise | 0.0570 | 0.0533 | 0.0604 | 0.0000 | 40896 | 417 | stratum | excluding the last 2 decision(s) | cross-fitted | 0.1707 |
| goal | max_z_good_surprise | 0.0080 | 0.0065 | 0.0095 | 0.0000 | 40896 | 417 | stratum | excluding the last 2 decision(s) | cross-fitted | 0.0203 |
| xg | max_z_good_surprise | 0.0079 | 0.0071 | 0.0087 | 0.0000 | 40896 | 417 | stratum | excluding the last 2 decision(s) | cross-fitted | 0.0196 |
| shot | max_raw_rarity | 0.0180 | 0.0143 | 0.0215 | 0.0000 | 40896 | 417 | stratum | excluding last 2, originality and value jointly | cross-fitted | 0.1707 |
| shot | max_raw_advantage | 0.0989 | 0.0925 | 0.1053 | 0.0000 | 40896 | 417 | stratum | excluding last 2, originality and value jointly | cross-fitted | 0.1707 |
| goal | max_raw_rarity | 0.0026 | 0.0010 | 0.0040 | 0.0020 | 40896 | 417 | stratum | excluding last 2, originality and value jointly | cross-fitted | 0.0203 |
| goal | max_raw_advantage | 0.0175 | 0.0145 | 0.0207 | 0.0000 | 40896 | 417 | stratum | excluding last 2, originality and value jointly | cross-fitted | 0.0203 |
| xg | max_raw_rarity | 0.0019 | 0.0011 | 0.0025 | 0.0000 | 40896 | 417 | stratum | excluding last 2, originality and value jointly | cross-fitted | 0.0196 |
| xg | max_raw_advantage | 0.0200 | 0.0180 | 0.0221 | 0.0000 | 40896 | 417 | stratum | excluding last 2, originality and value jointly | cross-fitted | 0.0196 |
| shot | max_z_good_surprise | 0.0563 | 0.0526 | 0.0599 | 0.0000 | 40896 | 417 | stratum+match_id | excluding last 2 + match FE | cross-fitted | 0.1707 |
| goal | max_z_good_surprise | 0.0080 | 0.0064 | 0.0095 | 0.0000 | 40896 | 417 | stratum+match_id | excluding last 2 + match FE | cross-fitted | 0.0203 |
| xg | max_z_good_surprise | 0.0078 | 0.0070 | 0.0086 | 0.0000 | 40896 | 417 | stratum+match_id | excluding last 2 + match FE | cross-fitted | 0.0196 |
| shot | max_z_good_surprise | 0.0538 | 0.0499 | 0.0576 | 0.0000 | 34985 | 417 | stratum | excluding the last 3 decision(s) | cross-fitted | 0.1808 |
| goal | max_z_good_surprise | 0.0074 | 0.0057 | 0.0091 | 0.0000 | 34985 | 417 | stratum | excluding the last 3 decision(s) | cross-fitted | 0.0213 |
| xg | max_z_good_surprise | 0.0074 | 0.0065 | 0.0083 | 0.0000 | 34985 | 417 | stratum | excluding the last 3 decision(s) | cross-fitted | 0.0207 |
| shot | max_z_good_surprise | 0.0356 | 0.0303 | 0.0406 | 0.0000 | 18836 | 133 | stratum | base, reference yardstick (club and Argentina matches) | reference | 0.1502 |
| goal | max_z_good_surprise | 0.0061 | 0.0037 | 0.0084 | 0.0000 | 18836 | 133 | stratum | base, reference yardstick (club and Argentina matches) | reference | 0.0203 |
| xg | max_z_good_surprise | 0.0060 | 0.0046 | 0.0073 | 0.0000 | 18836 | 133 | stratum | base, reference yardstick (club and Argentina matches) | reference | 0.0193 |
| shot | max_z_good_surprise | 0.0376 | 0.0323 | 0.0430 | 0.0000 | 18836 | 133 | stratum | base, cross-fitted on the same matches | cross-fitted (same matches) | 0.1502 |
| goal | max_z_good_surprise | 0.0057 | 0.0032 | 0.0083 | 0.0000 | 18836 | 133 | stratum | base, cross-fitted on the same matches | cross-fitted (same matches) | 0.0203 |
| xg | max_z_good_surprise | 0.0064 | 0.0051 | 0.0078 | 0.0000 | 18836 | 133 | stratum | base, cross-fitted on the same matches | cross-fitted (same matches) | 0.0193 |

