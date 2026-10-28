# Event studies: windows, pre-trends, splits, opportunity vs. use (B3-B5, C2)

Effect = change after vs. before the event minus the same change in 20 placebo windows per event (other matches, same period, clock +-3 min). Values: z-scores (SD) except quadrant_share, chose_quadrant_s (rates) and pressure_5yd (opponents within 5 yd).

## event_studies (257 rows, first 60 shown)

| event | value | variant | group | effect | low | high | p | events | split |
|---|---|---|---|---|---|---|---|---|---|
| goal: scoring team | z_good_surprise | 10 min | all | -0.0607 | -0.0777 | -0.0454 | 0.0000 | 1140 |  |
| goal: scoring team | z_good_surprise | 10 min | level | -0.0742 | -0.1145 | -0.0348 | 0.0000 | 175 | state_after |
| goal: scoring team | z_good_surprise | 10 min | scorer behind | 0.0222 | -0.0337 | 0.0753 | 0.4860 | 75 | state_after |
| goal: scoring team | z_good_surprise | 10 min | scorer leads | -0.0651 | -0.0823 | -0.0488 | 0.0000 | 890 | state_after |
| goal: scoring team | z_good_surprise | 10 min | 60' or later | -0.0552 | -0.0818 | -0.0290 | 0.0000 | 439 | time |
| goal: scoring team | z_good_surprise | 10 min | before 60' | -0.0642 | -0.0835 | -0.0459 | 0.0000 | 701 | time |
| goal: scoring team | z_good_surprise | 5 min | all | -0.0937 | -0.1139 | -0.0729 | 0.0000 | 1075 |  |
| goal: scoring team | z_good_surprise | 15 min | all | -0.0551 | -0.0695 | -0.0412 | 0.0000 | 1139 |  |
| goal: scoring team | z_good_surprise | pre-trend (-20..-10 vs -10..0) | all | 0.0215 | 0.0076 | 0.0359 | 0.0000 | 960 |  |
| goal: scoring team | z_good_surprise | skip 2 min after | all | -0.0478 | -0.0614 | -0.0329 | 0.0000 | 1101 |  |
| goal: conceding team | z_good_surprise | 10 min | all | -0.0129 | -0.0304 | 0.0022 | 0.1040 | 1106 |  |
| goal: conceding team | z_good_surprise | 10 min | level | -0.0112 | -0.0563 | 0.0343 | 0.6360 | 166 | state_after |
| goal: conceding team | z_good_surprise | 10 min | scorer behind | 0.0271 | -0.0269 | 0.0838 | 0.3540 | 73 | state_after |
| goal: conceding team | z_good_surprise | 10 min | scorer leads | -0.0166 | -0.0347 | 0.0015 | 0.0640 | 867 | state_after |
| goal: conceding team | z_good_surprise | 10 min | 60' or later | -0.0082 | -0.0336 | 0.0159 | 0.5320 | 435 | time |
| goal: conceding team | z_good_surprise | 10 min | before 60' | -0.0160 | -0.0367 | 0.0062 | 0.1580 | 671 | time |
| goal: conceding team | z_good_surprise | 5 min | all | -0.0070 | -0.0311 | 0.0173 | 0.5480 | 976 |  |
| goal: conceding team | z_good_surprise | 15 min | all | -0.0117 | -0.0262 | 0.0019 | 0.0880 | 1110 |  |
| goal: conceding team | z_good_surprise | pre-trend (-20..-10 vs -10..0) | all | -0.0109 | -0.0284 | 0.0066 | 0.2320 | 943 |  |
| goal: conceding team | z_good_surprise | skip 2 min after | all | -0.0187 | -0.0341 | -0.0032 | 0.0240 | 1071 |  |
| red card: team sent off | z_good_surprise | 10 min | all | -0.1057 | -0.1828 | -0.0206 | 0.0120 | 40 |  |
| red card: team sent off | z_good_surprise | 5 min | all | -0.1706 | -0.2670 | -0.0671 | 0.0000 | 22 |  |
| red card: team sent off | z_good_surprise | 15 min | all | -0.0980 | -0.1593 | -0.0373 | 0.0020 | 41 |  |
| red card: team sent off | z_good_surprise | pre-trend (-20..-10 vs -10..0) | all | -0.0090 | -0.0780 | 0.0607 | 0.7460 | 45 |  |
| red card: opponent | z_good_surprise | 10 min | all | 0.0294 | -0.0401 | 0.0982 | 0.4360 | 47 |  |
| red card: opponent | z_good_surprise | 5 min | all | 0.0026 | -0.0958 | 0.1087 | 0.9140 | 37 |  |
| red card: opponent | z_good_surprise | 15 min | all | 0.0284 | -0.0294 | 0.0875 | 0.3600 | 47 |  |
| red card: opponent | z_good_surprise | pre-trend (-20..-10 vs -10..0) | all | -0.0469 | -0.1233 | 0.0263 | 0.2160 | 47 |  |
| tactical shift | z_good_surprise | 10 min | all | 0.0227 | 0.0089 | 0.0364 | 0.0000 | 1203 |  |
| tactical shift | z_good_surprise | 10 min | False | 0.0236 | 0.0011 | 0.0475 | 0.0360 | 451 | with_substitution |
| tactical shift | z_good_surprise | 10 min | True | 0.0221 | 0.0048 | 0.0388 | 0.0100 | 752 | with_substitution |
| tactical shift | z_good_surprise | 10 min | leading | 0.0304 | 0.0054 | 0.0580 | 0.0140 | 356 | state |
| tactical shift | z_good_surprise | 10 min | level | 0.0111 | -0.0144 | 0.0380 | 0.4420 | 330 | state |
| tactical shift | z_good_surprise | 10 min | trailing | 0.0247 | 0.0036 | 0.0471 | 0.0180 | 517 | state |
| tactical shift | z_good_surprise | 5 min | all | 0.0148 | -0.0050 | 0.0361 | 0.1520 | 1097 |  |
| tactical shift | z_good_surprise | 15 min | all | 0.0276 | 0.0158 | 0.0386 | 0.0000 | 1203 |  |
| tactical shift | z_good_surprise | pre-trend (-20..-10 vs -10..0) | all | -0.0099 | -0.0236 | 0.0042 | 0.1480 | 1128 |  |
| tactical substitution | z_good_surprise | 10 min | all | 0.0115 | -0.0017 | 0.0251 | 0.1060 | 2354 |  |
| tactical substitution | z_good_surprise | 5 min | all | 0.0095 | -0.0069 | 0.0251 | 0.2140 | 2773 |  |
| tactical substitution | z_good_surprise | 15 min | all | -0.0178 | -0.0390 | 0.0032 | 0.1140 | 1039 |  |
| tactical substitution | z_good_surprise | pre-trend (-20..-10 vs -10..0) | all | 0.0385 | -0.0029 | 0.0825 | 0.0580 | 271 |  |
| goal: scoring team | z_good_rarity | 10 min | all | -0.0215 | -0.0373 | -0.0068 | 0.0080 | 1140 |  |
| goal: scoring team | z_good_rarity | 10 min | level | -0.0183 | -0.0539 | 0.0185 | 0.3200 | 175 | state_after |
| goal: scoring team | z_good_rarity | 10 min | scorer behind | 0.0800 | 0.0148 | 0.1367 | 0.0180 | 75 | state_after |
| goal: scoring team | z_good_rarity | 10 min | scorer leads | -0.0307 | -0.0487 | -0.0138 | 0.0000 | 890 | state_after |
| goal: scoring team | z_good_rarity | 10 min | 60' or later | -0.0052 | -0.0332 | 0.0207 | 0.6140 | 439 | time |
| goal: scoring team | z_good_rarity | 10 min | before 60' | -0.0317 | -0.0508 | -0.0123 | 0.0000 | 701 | time |
| goal: scoring team | z_good_rarity | pre-trend (-20..-10 vs -10..0) | all | 0.0272 | 0.0123 | 0.0410 | 0.0000 | 960 |  |
| goal: conceding team | z_good_rarity | 10 min | all | -0.0264 | -0.0439 | -0.0094 | 0.0020 | 1106 |  |
| goal: conceding team | z_good_rarity | 10 min | level | -0.0387 | -0.0903 | 0.0114 | 0.1320 | 166 | state_after |
| goal: conceding team | z_good_rarity | 10 min | scorer behind | 0.0179 | -0.0365 | 0.0741 | 0.5280 | 73 | state_after |
| goal: conceding team | z_good_rarity | 10 min | scorer leads | -0.0278 | -0.0462 | -0.0082 | 0.0060 | 867 | state_after |
| goal: conceding team | z_good_rarity | 10 min | 60' or later | -0.0293 | -0.0587 | -0.0040 | 0.0120 | 435 | time |
| goal: conceding team | z_good_rarity | 10 min | before 60' | -0.0246 | -0.0454 | -0.0032 | 0.0240 | 671 | time |
| goal: conceding team | z_good_rarity | pre-trend (-20..-10 vs -10..0) | all | -0.0111 | -0.0302 | 0.0076 | 0.2280 | 943 |  |
| red card: team sent off | z_good_rarity | 10 min | all | -0.0701 | -0.1536 | 0.0143 | 0.1200 | 40 |  |
| red card: team sent off | z_good_rarity | pre-trend (-20..-10 vs -10..0) | all | 0.0213 | -0.0584 | 0.0994 | 0.6080 | 45 |  |
| red card: opponent | z_good_rarity | 10 min | all | 0.0787 | -0.0147 | 0.1813 | 0.1080 | 47 |  |
| red card: opponent | z_good_rarity | pre-trend (-20..-10 vs -10..0) | all | -0.0231 | -0.0925 | 0.0420 | 0.5180 | 47 |  |
| tactical shift | z_good_rarity | 10 min | all | 0.0204 | 0.0065 | 0.0346 | 0.0040 | 1203 |  |


## events_goal_scoring_team (1217 rows, first 60 shown)

| match_id | period | clock | team | conceding_team | own_goal | player | possession | play_pattern | score_for_after | score_against_after | state_after | time | delta | placebo | effect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3764440 | 2 | 47.0833 | Barcelona | Elche | False | Lionel Andrés Messi Cuccittini | 70 | From Free Kick | 1 | 0 | scorer leads | before 60' | -0.2851 | 0.1337 | -0.4189 |
| 3764440 | 2 | 68.0000 | Barcelona | Elche | False | Lionel Andrés Messi Cuccittini | 97 | From Counter | 2 | 0 | scorer leads | 60' or later | -0.0537 | -0.0012 | -0.0525 |
| 3764440 | 2 | 72.4000 | Barcelona | Elche | False | Jordi Alba Ramos | 106 | From Free Kick | 3 | 0 | scorer leads | 60' or later | -0.0056 | -0.0537 | 0.0481 |
| 3764661 | 1 | 2.7667 | Athletic Club | Barcelona | False | Iñaki Williams Arthuer | 7 | From Counter | 1 | 0 | scorer leads | before 60' | -0.2864 | -0.0182 | -0.2682 |
| 3764661 | 1 | 13.3500 | Barcelona | Athletic Club | False | Pedro González López | 33 | From Corner | 1 | 1 | level | before 60' | -0.2698 | -0.0088 | -0.2610 |
| 3764661 | 1 | 37.6000 | Barcelona | Athletic Club | False | Lionel Andrés Messi Cuccittini | 84 | From Free Kick | 2 | 1 | scorer leads | before 60' | -0.0115 | 0.0388 | -0.0504 |
| 3764661 | 2 | 61.1333 | Barcelona | Athletic Club | False | Lionel Andrés Messi Cuccittini | 127 | From Throw In | 3 | 1 | scorer leads | 60' or later | -0.1790 | -0.0015 | -0.1774 |
| 3764661 | 2 | 89.5500 | Athletic Club | Barcelona | False | Iker Muniain Goñi | 186 | Regular Play | 2 | 3 | scorer behind | 60' or later | 0.0115 | 0.0097 | 0.0019 |
| 3773369 | 1 | 12.8167 | Barcelona | Huesca | False | Lionel Andrés Messi Cuccittini | 35 | Regular Play | 1 | 0 | scorer leads | before 60' | 0.0665 | -0.0118 | 0.0783 |
| 3773369 | 1 | 34.7333 | Barcelona | Huesca | False | Antoine Griezmann | 73 | Regular Play | 2 | 0 | scorer leads | before 60' | -0.1180 | 0.0362 | -0.1542 |
| 3773369 | 1 | 48.2000 | Huesca | Barcelona | False | Rafael Mir Vicente | 88 | Other | 1 | 2 | scorer behind | before 60' |  | -0.0117 |  |
| 3773369 | 2 | 52.8000 | Barcelona | Huesca | False | Óscar Mingueza García | 101 | From Corner | 3 | 1 | scorer leads | before 60' | -0.1404 | 0.0135 | -0.1539 |
| 3773369 | 2 | 89.6833 | Barcelona | Huesca | False | Lionel Andrés Messi Cuccittini | 196 | Regular Play | 4 | 1 | scorer leads | 60' or later | -0.3191 | 0.0406 | -0.3597 |
| 3773377 | 1 | 28.8667 | Valencia | Barcelona | False | Mouctar Diakhaby | 52 | From Corner | 1 | 0 | scorer leads | before 60' | 0.1099 | 0.0137 | 0.0963 |
| 3773377 | 1 | 48.6500 | Barcelona | Valencia | False | Lionel Andrés Messi Cuccittini | 77 | Other | 1 | 1 | level | before 60' | 0.7700 | 0.1390 | 0.6310 |
| 3773377 | 2 | 51.4167 | Barcelona | Valencia | False | Ronald Federico Araújo da Silva | 96 | From Corner | 2 | 1 | scorer leads | before 60' | -0.2530 | 0.0805 | -0.3334 |
| 3773377 | 2 | 68.1500 | Valencia | Barcelona | False | Maximiliano Gómez González | 117 | Regular Play | 2 | 2 | level | 60' or later | 0.4448 | 0.0529 | 0.3919 |
| 3773386 | 1 | 30.1000 | Deportivo Alavés | Barcelona | False | Luis Jesús Rioja González | 52 | Regular Play | 1 | 0 | scorer leads | before 60' | -0.6640 | 0.0395 | -0.7036 |
| 3773386 | 2 | 62.9500 | Barcelona | Deportivo Alavés | False | Antoine Griezmann | 104 | From Free Kick | 1 | 1 | level | 60' or later | 0.0351 | 0.0445 | -0.0094 |
| 3773387 | 1 | 25.0000 | Barcelona | Levante UD | False | Lionel Andrés Messi Cuccittini | 46 | Regular Play | 1 | 0 | scorer leads | before 60' | 0.0123 | 0.0679 | -0.0556 |
| 3773387 | 1 | 33.5833 | Barcelona | Levante UD | False | Pedro González López | 60 | From Counter | 2 | 0 | scorer leads | before 60' | -0.2933 | -0.0481 | -0.2453 |
| 3773387 | 2 | 56.2833 | Levante UD | Barcelona | False | Gonzalo Julián Melero Manzanares | 96 | From Goal Kick | 1 | 2 | scorer behind | before 60' | -0.2290 | -0.1231 | -0.1060 |
| 3773387 | 2 | 59.0000 | Levante UD | Barcelona | False | José Luis Morales Nogales | 98 | Regular Play | 2 | 2 | level | before 60' | -0.1545 | 0.0374 | -0.1919 |
| 3773387 | 2 | 63.1000 | Barcelona | Levante UD | False | Ousmane Dembélé | 112 | Regular Play | 3 | 2 | scorer leads | 60' or later | -0.1133 | 0.0211 | -0.1344 |
| 3773387 | 2 | 82.6000 | Levante UD | Barcelona | False | Sergio León Limones | 141 | Regular Play | 3 | 3 | level | 60' or later | -0.2897 | -0.0364 | -0.2533 |
| 3773403 | 1 | 19.4667 | Barcelona | Athletic Club | False | Lionel Andrés Messi Cuccittini | 43 | From Free Kick | 1 | 0 | scorer leads | before 60' | -0.0726 | 0.0414 | -0.1140 |
| 3773403 | 2 | 48.4333 | Athletic Club | Barcelona | True | Jordi Alba Ramos | 98 | From Free Kick | 1 | 1 | level | before 60' | -0.1077 | 0.1137 | -0.2213 |
| 3773403 | 2 | 73.9500 | Barcelona | Athletic Club | False | Antoine Griezmann | 148 | Regular Play | 2 | 1 | scorer leads | 60' or later | -0.0662 | 0.0095 | -0.0757 |
| 3773415 | 1 | 31.3833 | Barcelona | Cádiz | False | Lionel Andrés Messi Cuccittini | 58 | Other | 1 | 0 | scorer leads | before 60' | -0.1744 | 0.0160 | -0.1905 |
| 3773415 | 2 | 88.2667 | Cádiz | Barcelona | False | Alejandro Fernández Iglesias | 158 | Other | 1 | 1 | level | 60' or later | -0.9779 | 0.0028 | -0.9807 |
| 3773428 | 1 | 7.8833 | Cádiz | Barcelona | False | Álvaro Giménez Candela | 14 | From Corner | 1 | 0 | scorer leads | before 60' | 0.1340 | 0.0956 | 0.0385 |
| 3773428 | 2 | 56.9333 | Barcelona | Cádiz | True | Pedro Alcalá Guirado | 100 | Regular Play | 1 | 1 | level | before 60' | 0.0973 | -0.0399 | 0.1372 |
| 3773428 | 2 | 62.2333 | Cádiz | Barcelona | False | Álvaro Negredo Sánchez | 108 | From Throw In | 2 | 1 | scorer leads | 60' or later | 0.7192 | 0.0281 | 0.6911 |
| 3773457 | 1 | 27.1000 | Barcelona | Celta Vigo | False | Lionel Andrés Messi Cuccittini | 54 | Regular Play | 1 | 0 | scorer leads | before 60' | 0.3112 | -0.0423 | 0.3535 |
| 3773457 | 1 | 37.2333 | Celta Vigo | Barcelona | False | Santiago Mina Lorenzo | 76 | From Throw In | 1 | 1 | level | before 60' | 0.0507 | 0.0403 | 0.0104 |
| 3773457 | 2 | 88.4667 | Celta Vigo | Barcelona | False | Santiago Mina Lorenzo | 195 | Regular Play | 2 | 1 | scorer leads | 60' or later | -0.1647 | -0.0366 | -0.1281 |
| 3773466 | 1 | 10.1333 | Barcelona | Celta Vigo | False | Anssumane Fati | 20 | From Free Kick | 1 | 0 | scorer leads | before 60' | -0.3808 | -0.0667 | -0.3141 |
| 3773466 | 2 | 50.3333 | Barcelona | Celta Vigo | True | Lucas René Olaza Catrofe | 98 | From Throw In | 2 | 0 | scorer leads | before 60' | -0.0258 | 0.0958 | -0.1216 |
| 3773466 | 2 | 94.0667 | Barcelona | Celta Vigo | False | Sergi Roberto Carnicer | 164 | Regular Play | 3 | 0 | scorer leads | 60' or later |  | -0.0823 |  |
| 3773474 | 2 | 89.6000 | Barcelona | Real Valladolid | False | Ousmane Dembélé | 189 | Regular Play | 1 | 0 | scorer leads | 60' or later | -0.0070 | 0.0112 | -0.0182 |
| 3773477 | 1 | 21.6833 | Barcelona | Real Betis | False | Ousmane Dembélé | 45 | From Keeper | 1 | 0 | scorer leads | before 60' | -0.0520 | 0.0252 | -0.0772 |
| 3773477 | 1 | 46.9500 | Real Betis | Barcelona | False | Arnaldo Antonio Sanabria Ayala | 82 | Regular Play | 1 | 1 | level | before 60' |  | 0.0891 |  |
| 3773477 | 2 | 48.2333 | Barcelona | Real Betis | False | Antoine Griezmann | 84 | From Corner | 2 | 1 | scorer leads | before 60' | -0.0479 | -0.0311 | -0.0168 |
| 3773477 | 2 | 60.9500 | Barcelona | Real Betis | False | Lionel Andrés Messi Cuccittini | 106 | Other | 3 | 1 | scorer leads | 60' or later | -0.1420 | 0.0619 | -0.2039 |
| 3773477 | 2 | 72.9333 | Real Betis | Barcelona | False | Lorenzo Jesús Morón García | 126 | From Throw In | 2 | 3 | scorer behind | 60' or later | 0.1738 | -0.0720 | 0.2458 |
| 3773477 | 2 | 81.4333 | Barcelona | Real Betis | False | Lionel Andrés Messi Cuccittini | 139 | From Corner | 4 | 2 | scorer leads | 60' or later | 0.2198 | 0.0080 | 0.2119 |
| 3773477 | 2 | 89.1000 | Barcelona | Real Betis | False | Pedro González López | 150 | From Throw In | 5 | 2 | scorer leads | 60' or later | 0.1542 | -0.0056 | 0.1598 |
| 3773497 | 1 | 12.9667 | Real Madrid | Barcelona | False | Karim Benzema | 31 | Regular Play | 1 | 0 | scorer leads | before 60' | -0.0860 | -0.0338 | -0.0521 |
| 3773497 | 1 | 27.1000 | Real Madrid | Barcelona | False | Toni Kroos | 57 | From Free Kick | 2 | 0 | scorer leads | before 60' | 0.1175 | 0.0040 | 0.1135 |
| 3773497 | 2 | 59.1500 | Barcelona | Real Madrid | False | Óscar Mingueza García | 124 | Regular Play | 1 | 2 | scorer behind | before 60' | -0.1078 | 0.0479 | -0.1558 |
| 3773523 | 1 | 26.9500 | Real Sociedad | Barcelona | False | Willian José da Silva | 48 | From Corner | 1 | 0 | scorer leads | before 60' | -0.2728 | 0.0515 | -0.3243 |
| 3773523 | 1 | 30.6833 | Barcelona | Real Sociedad | False | Jordi Alba Ramos | 53 | Regular Play | 1 | 1 | level | before 60' | -0.0761 | 0.0441 | -0.1202 |
| 3773523 | 1 | 42.5000 | Barcelona | Real Sociedad | False | Frenkie de Jong | 74 | From Throw In | 2 | 1 | scorer leads | before 60' | -0.5841 | -0.0035 | -0.5807 |
| 3773526 | 1 | 28.9000 | Barcelona | Deportivo Alavés | False | Francisco António Machado Mota de Castro Trincão | 43 | From Throw In | 1 | 0 | scorer leads | before 60' | -0.0480 | 0.0189 | -0.0670 |
| 3773526 | 1 | 45.8333 | Barcelona | Deportivo Alavés | False | Lionel Andrés Messi Cuccittini | 65 | From Throw In | 2 | 0 | scorer leads | before 60' |  | -0.0073 |  |
| 3773526 | 2 | 56.7167 | Deportivo Alavés | Barcelona | False | Luis Jesús Rioja González | 91 | From Free Kick | 1 | 2 | scorer behind | before 60' | -0.1091 | -0.0105 | -0.0986 |
| 3773526 | 2 | 73.4500 | Barcelona | Deportivo Alavés | False | Francisco António Machado Mota de Castro Trincão | 112 | From Throw In | 3 | 1 | scorer leads | 60' or later | -0.2065 | -0.0448 | -0.1616 |
| 3773526 | 2 | 74.7833 | Barcelona | Deportivo Alavés | False | Lionel Andrés Messi Cuccittini | 114 | Regular Play | 4 | 1 | scorer leads | 60' or later | -0.2793 | -0.0283 | -0.2511 |
| 3773526 | 2 | 79.1833 | Barcelona | Deportivo Alavés | False | Héctor Junior Firpo Adames | 119 | Regular Play | 5 | 1 | scorer leads | 60' or later | -0.0016 | 0.0347 | -0.0362 |
| 3773547 | 1 | 28.9667 | Barcelona | Osasuna | False | Martin Braithwaite Christensen | 56 | From Free Kick | 1 | 0 | scorer leads | before 60' | -0.2608 | -0.0129 | -0.2479 |


## events_goal_conceding_team (1217 rows, first 60 shown)

| match_id | period | clock | team | conceding_team | own_goal | player | possession | play_pattern | score_for_after | score_against_after | state_after | time | delta | placebo | effect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3764440 | 2 | 47.0833 | Barcelona | Elche | False | Lionel Andrés Messi Cuccittini | 70 | From Free Kick | 1 | 0 | scorer leads | before 60' |  | 0.1337 |  |
| 3764440 | 2 | 68.0000 | Barcelona | Elche | False | Lionel Andrés Messi Cuccittini | 97 | From Counter | 2 | 0 | scorer leads | 60' or later | -0.0092 | -0.0012 | -0.0080 |
| 3764440 | 2 | 72.4000 | Barcelona | Elche | False | Jordi Alba Ramos | 106 | From Free Kick | 3 | 0 | scorer leads | 60' or later | -0.2329 | -0.0537 | -0.1792 |
| 3764661 | 1 | 2.7667 | Athletic Club | Barcelona | False | Iñaki Williams Arthuer | 7 | From Counter | 1 | 0 | scorer leads | before 60' | 0.2201 | -0.0182 | 0.2383 |
| 3764661 | 1 | 13.3500 | Barcelona | Athletic Club | False | Pedro González López | 33 | From Corner | 1 | 1 | level | before 60' | -0.2220 | -0.0088 | -0.2132 |
| 3764661 | 1 | 37.6000 | Barcelona | Athletic Club | False | Lionel Andrés Messi Cuccittini | 84 | From Free Kick | 2 | 1 | scorer leads | before 60' | -0.0065 | 0.0388 | -0.0453 |
| 3764661 | 2 | 61.1333 | Barcelona | Athletic Club | False | Lionel Andrés Messi Cuccittini | 127 | From Throw In | 3 | 1 | scorer leads | 60' or later | -0.0629 | -0.0015 | -0.0614 |
| 3764661 | 2 | 89.5500 | Athletic Club | Barcelona | False | Iker Muniain Goñi | 186 | Regular Play | 2 | 3 | scorer behind | 60' or later | -0.4670 | 0.0097 | -0.4767 |
| 3773369 | 1 | 12.8167 | Barcelona | Huesca | False | Lionel Andrés Messi Cuccittini | 35 | Regular Play | 1 | 0 | scorer leads | before 60' | -0.4711 | -0.0118 | -0.4594 |
| 3773369 | 1 | 34.7333 | Barcelona | Huesca | False | Antoine Griezmann | 73 | Regular Play | 2 | 0 | scorer leads | before 60' | -0.2076 | 0.0362 | -0.2438 |
| 3773369 | 1 | 48.2000 | Huesca | Barcelona | False | Rafael Mir Vicente | 88 | Other | 1 | 2 | scorer behind | before 60' |  | -0.0117 |  |
| 3773369 | 2 | 52.8000 | Barcelona | Huesca | False | Óscar Mingueza García | 101 | From Corner | 3 | 1 | scorer leads | before 60' | -0.5083 | 0.0135 | -0.5218 |
| 3773369 | 2 | 89.6833 | Barcelona | Huesca | False | Lionel Andrés Messi Cuccittini | 196 | Regular Play | 4 | 1 | scorer leads | 60' or later | -0.0454 | 0.0406 | -0.0860 |
| 3773377 | 1 | 28.8667 | Valencia | Barcelona | False | Mouctar Diakhaby | 52 | From Corner | 1 | 0 | scorer leads | before 60' | 0.0271 | 0.0137 | 0.0134 |
| 3773377 | 1 | 48.6500 | Barcelona | Valencia | False | Lionel Andrés Messi Cuccittini | 77 | Other | 1 | 1 | level | before 60' |  | 0.1390 |  |
| 3773377 | 2 | 51.4167 | Barcelona | Valencia | False | Ronald Federico Araújo da Silva | 96 | From Corner | 2 | 1 | scorer leads | before 60' | -0.1236 | 0.0805 | -0.2040 |
| 3773377 | 2 | 68.1500 | Valencia | Barcelona | False | Maximiliano Gómez González | 117 | Regular Play | 2 | 2 | level | 60' or later | 0.1948 | 0.0529 | 0.1419 |
| 3773386 | 1 | 30.1000 | Deportivo Alavés | Barcelona | False | Luis Jesús Rioja González | 52 | Regular Play | 1 | 0 | scorer leads | before 60' | -0.2393 | 0.0395 | -0.2789 |
| 3773386 | 2 | 62.9500 | Barcelona | Deportivo Alavés | False | Antoine Griezmann | 104 | From Free Kick | 1 | 1 | level | 60' or later | -0.3666 | 0.0445 | -0.4110 |
| 3773387 | 1 | 25.0000 | Barcelona | Levante UD | False | Lionel Andrés Messi Cuccittini | 46 | Regular Play | 1 | 0 | scorer leads | before 60' | -0.1389 | 0.0679 | -0.2069 |
| 3773387 | 1 | 33.5833 | Barcelona | Levante UD | False | Pedro González López | 60 | From Counter | 2 | 0 | scorer leads | before 60' | -0.2422 | -0.0481 | -0.1941 |
| 3773387 | 2 | 56.2833 | Levante UD | Barcelona | False | Gonzalo Julián Melero Manzanares | 96 | From Goal Kick | 1 | 2 | scorer behind | before 60' | 0.3334 | -0.1231 | 0.4565 |
| 3773387 | 2 | 59.0000 | Levante UD | Barcelona | False | José Luis Morales Nogales | 98 | Regular Play | 2 | 2 | level | before 60' | 0.1440 | 0.0374 | 0.1065 |
| 3773387 | 2 | 63.1000 | Barcelona | Levante UD | False | Ousmane Dembélé | 112 | Regular Play | 3 | 2 | scorer leads | 60' or later | 0.0673 | 0.0211 | 0.0462 |
| 3773387 | 2 | 82.6000 | Levante UD | Barcelona | False | Sergio León Limones | 141 | Regular Play | 3 | 3 | level | 60' or later | 0.2861 | -0.0364 | 0.3225 |
| 3773403 | 1 | 19.4667 | Barcelona | Athletic Club | False | Lionel Andrés Messi Cuccittini | 43 | From Free Kick | 1 | 0 | scorer leads | before 60' | -0.0077 | 0.0414 | -0.0490 |
| 3773403 | 2 | 48.4333 | Athletic Club | Barcelona | True | Jordi Alba Ramos | 98 | From Free Kick | 1 | 1 | level | before 60' | 0.2073 | 0.1137 | 0.0936 |
| 3773403 | 2 | 73.9500 | Barcelona | Athletic Club | False | Antoine Griezmann | 148 | Regular Play | 2 | 1 | scorer leads | 60' or later | 0.1515 | 0.0095 | 0.1420 |
| 3773415 | 1 | 31.3833 | Barcelona | Cádiz | False | Lionel Andrés Messi Cuccittini | 58 | Other | 1 | 0 | scorer leads | before 60' | 1.0329 | 0.0160 | 1.0169 |
| 3773415 | 2 | 88.2667 | Cádiz | Barcelona | False | Alejandro Fernández Iglesias | 158 | Other | 1 | 1 | level | 60' or later | 0.2328 | 0.0028 | 0.2300 |
| 3773428 | 1 | 7.8833 | Cádiz | Barcelona | False | Álvaro Giménez Candela | 14 | From Corner | 1 | 0 | scorer leads | before 60' | 0.0177 | 0.0956 | -0.0779 |
| 3773428 | 2 | 56.9333 | Barcelona | Cádiz | True | Pedro Alcalá Guirado | 100 | Regular Play | 1 | 1 | level | before 60' | 0.1953 | -0.0399 | 0.2352 |
| 3773428 | 2 | 62.2333 | Cádiz | Barcelona | False | Álvaro Negredo Sánchez | 108 | From Throw In | 2 | 1 | scorer leads | 60' or later | -0.0379 | 0.0281 | -0.0660 |
| 3773457 | 1 | 27.1000 | Barcelona | Celta Vigo | False | Lionel Andrés Messi Cuccittini | 54 | Regular Play | 1 | 0 | scorer leads | before 60' | -0.3548 | -0.0423 | -0.3125 |
| 3773457 | 1 | 37.2333 | Celta Vigo | Barcelona | False | Santiago Mina Lorenzo | 76 | From Throw In | 1 | 1 | level | before 60' | -0.1354 | 0.0403 | -0.1757 |
| 3773457 | 2 | 88.4667 | Celta Vigo | Barcelona | False | Santiago Mina Lorenzo | 195 | Regular Play | 2 | 1 | scorer leads | 60' or later | 0.0190 | -0.0366 | 0.0556 |
| 3773466 | 1 | 10.1333 | Barcelona | Celta Vigo | False | Anssumane Fati | 20 | From Free Kick | 1 | 0 | scorer leads | before 60' | 0.1561 | -0.0667 | 0.2227 |
| 3773466 | 2 | 50.3333 | Barcelona | Celta Vigo | True | Lucas René Olaza Catrofe | 98 | From Throw In | 2 | 0 | scorer leads | before 60' | -0.1228 | 0.0958 | -0.2187 |
| 3773466 | 2 | 94.0667 | Barcelona | Celta Vigo | False | Sergi Roberto Carnicer | 164 | Regular Play | 3 | 0 | scorer leads | 60' or later |  | -0.0823 |  |
| 3773474 | 2 | 89.6000 | Barcelona | Real Valladolid | False | Ousmane Dembélé | 189 | Regular Play | 1 | 0 | scorer leads | 60' or later | 0.2391 | 0.0112 | 0.2279 |
| 3773477 | 1 | 21.6833 | Barcelona | Real Betis | False | Ousmane Dembélé | 45 | From Keeper | 1 | 0 | scorer leads | before 60' | -0.0816 | 0.0252 | -0.1069 |
| 3773477 | 1 | 46.9500 | Real Betis | Barcelona | False | Arnaldo Antonio Sanabria Ayala | 82 | Regular Play | 1 | 1 | level | before 60' |  | 0.0891 |  |
| 3773477 | 2 | 48.2333 | Barcelona | Real Betis | False | Antoine Griezmann | 84 | From Corner | 2 | 1 | scorer leads | before 60' |  | -0.0311 |  |
| 3773477 | 2 | 60.9500 | Barcelona | Real Betis | False | Lionel Andrés Messi Cuccittini | 106 | Other | 3 | 1 | scorer leads | 60' or later | -0.4038 | 0.0619 | -0.4657 |
| 3773477 | 2 | 72.9333 | Real Betis | Barcelona | False | Lorenzo Jesús Morón García | 126 | From Throw In | 2 | 3 | scorer behind | 60' or later | -0.0825 | -0.0720 | -0.0105 |
| 3773477 | 2 | 81.4333 | Barcelona | Real Betis | False | Lionel Andrés Messi Cuccittini | 139 | From Corner | 4 | 2 | scorer leads | 60' or later | -0.1665 | 0.0080 | -0.1744 |
| 3773477 | 2 | 89.1000 | Barcelona | Real Betis | False | Pedro González López | 150 | From Throw In | 5 | 2 | scorer leads | 60' or later | 0.2060 | -0.0056 | 0.2116 |
| 3773497 | 1 | 12.9667 | Real Madrid | Barcelona | False | Karim Benzema | 31 | Regular Play | 1 | 0 | scorer leads | before 60' | -0.0351 | -0.0338 | -0.0013 |
| 3773497 | 1 | 27.1000 | Real Madrid | Barcelona | False | Toni Kroos | 57 | From Free Kick | 2 | 0 | scorer leads | before 60' | -0.1778 | 0.0040 | -0.1818 |
| 3773497 | 2 | 59.1500 | Barcelona | Real Madrid | False | Óscar Mingueza García | 124 | Regular Play | 1 | 2 | scorer behind | before 60' | 0.2142 | 0.0479 | 0.1663 |
| 3773523 | 1 | 26.9500 | Real Sociedad | Barcelona | False | Willian José da Silva | 48 | From Corner | 1 | 0 | scorer leads | before 60' | 0.0926 | 0.0515 | 0.0410 |
| 3773523 | 1 | 30.6833 | Barcelona | Real Sociedad | False | Jordi Alba Ramos | 53 | Regular Play | 1 | 1 | level | before 60' | -0.1746 | 0.0441 | -0.2187 |
| 3773523 | 1 | 42.5000 | Barcelona | Real Sociedad | False | Frenkie de Jong | 74 | From Throw In | 2 | 1 | scorer leads | before 60' | 0.2785 | -0.0035 | 0.2820 |
| 3773526 | 1 | 28.9000 | Barcelona | Deportivo Alavés | False | Francisco António Machado Mota de Castro Trincão | 43 | From Throw In | 1 | 0 | scorer leads | before 60' | -0.2762 | 0.0189 | -0.2952 |
| 3773526 | 1 | 45.8333 | Barcelona | Deportivo Alavés | False | Lionel Andrés Messi Cuccittini | 65 | From Throw In | 2 | 0 | scorer leads | before 60' |  | -0.0073 |  |
| 3773526 | 2 | 56.7167 | Deportivo Alavés | Barcelona | False | Luis Jesús Rioja González | 91 | From Free Kick | 1 | 2 | scorer behind | before 60' | 0.0635 | -0.0105 | 0.0740 |
| 3773526 | 2 | 73.4500 | Barcelona | Deportivo Alavés | False | Francisco António Machado Mota de Castro Trincão | 112 | From Throw In | 3 | 1 | scorer leads | 60' or later | -0.4715 | -0.0448 | -0.4267 |
| 3773526 | 2 | 74.7833 | Barcelona | Deportivo Alavés | False | Lionel Andrés Messi Cuccittini | 114 | Regular Play | 4 | 1 | scorer leads | 60' or later | -0.5555 | -0.0283 | -0.5273 |
| 3773526 | 2 | 79.1833 | Barcelona | Deportivo Alavés | False | Héctor Junior Firpo Adames | 119 | Regular Play | 5 | 1 | scorer leads | 60' or later | 0.0082 | 0.0347 | -0.0265 |
| 3773547 | 1 | 28.9667 | Barcelona | Osasuna | False | Martin Braithwaite Christensen | 56 | From Free Kick | 1 | 0 | scorer leads | before 60' | -0.3520 | -0.0129 | -0.3391 |


## events_red_card_team_sent_off (53 rows)

| match_id | period | clock | team | player_id | player | card | goals_for | goals_against | delta | placebo | effect |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3773386 | 2 | 61.5500 | Deportivo Alavés | 9462 | José Ignacio Peleteiro Ramallo | Second Yellow | 1 | 0 | -0.3625 | 0.0458 | -0.4082 |
| 3773457 | 2 | 82.0833 | Barcelona | 6826 | Clément Lenglet | Second Yellow | 1 | 1 | 0.2291 | -0.0226 | 0.2517 |
| 3773466 | 1 | 41.5333 | Barcelona | 6826 | Clément Lenglet | Second Yellow | 1 | 0 | -0.3825 | 0.2374 | -0.6200 |
| 3773474 | 2 | 78.4667 | Real Valladolid | 11296 | Óscar Plano Pedreño | Red Card | 0 | 0 | -0.2297 | 0.0095 | -0.2392 |
| 3773477 | 2 | 57.0833 | Real Betis | 6648 | Aïssa Mandi | Red Card | 1 | 2 | -0.3392 | -0.0531 | -0.2861 |
| 3773497 | 2 | 89.3833 | Real Madrid | 5539 | Carlos Henrique Casimiro | Second Yellow | 2 | 1 |  | 0.0508 |  |
| 3773565 | 2 | 77.5000 | Granada | 6893 | Jesús Vallejo Lázaro | Red Card | 0 | 4 | -0.4284 | 0.0421 | -0.4705 |
| 3773689 | 2 | 64.5167 | Villarreal | 6771 | Manuel Trigueros Muñoz | Red Card | 1 | 2 | 0.0225 | 0.0763 | -0.0538 |
| 3788749 | 2 | 61.5333 | Poland | 3637 | Grzegorz Krychowiak | Second Yellow | 1 | 1 | 0.1379 | 0.1363 | 0.0017 |
| 3788766 | 2 | 54.7167 | Wales | 4934 | Ethan Ampadu | Red Card | 0 | 1 | 0.0563 | 0.0403 | 0.0159 |
| 3794689 | 2 | 89.7500 | Wales | 9540 | Harry Wilson | Red Card | 0 | 3 | -0.2318 | 0.0283 | -0.2601 |
| 3794690 | 2 | 51.5000 | Netherlands | 8069 | Matthijs de Ligt | Red Card | 0 | 0 | -0.3495 | -0.0153 | -0.3341 |
| 3794692 | 3 | 97.0667 | Sweden | 26875 | Marcus Andreas Danielsson | Red Card | 1 | 1 |  | 0.0770 |  |
| 3795108 | 2 | 76.3833 | Switzerland | 6983 | Remo Freuler | Red Card | 1 | 1 | 0.0526 | -0.0731 | 0.1257 |
| 3802676 | 2 | 56.1167 | Lens | 11820 | Kevin Danso | Second Yellow | 0 | 0 | 0.0761 | 0.0802 | -0.0042 |
| 3802690 | 2 | 84.2833 | Paris Saint-Germain | 5201 | Sergio Ramos García | Second Yellow | 0 | 1 | 0.1274 | 0.0032 | 0.1242 |
| 3802837 | 1 | 44.2833 | Saint-Étienne | 10476 | Timothée Kolodziejczak | Red Card | 1 | 0 |  | -0.0740 |  |
| 3802902 | 2 | 53.9000 | Paris Saint-Germain | 5245 | Achraf Hakimi Mouh | Red Card | 0 | 0 | -0.7109 | 0.0120 | -0.7229 |
| 3802948 | 2 | 56.1167 | Metz | 58878 | Boubacar Traore | Second Yellow | 0 | 4 | 0.3456 | -0.0652 | 0.4108 |
| 3802984 | 2 | 64.3833 | Paris Saint-Germain | 5597 | Keylor Navas Gamboa | Red Card | 1 | 0 | -0.1494 | 0.0313 | -0.1806 |
| 3835322 | 2 | 93.8500 | Denmark Women's | 276443 | Kathrine Møller Kühl | Second Yellow | 0 | 4 |  | 0.2149 |  |
| 3835333 | 2 | 85.7500 | Belgium Women's | 276907 | Amber Tysiak | Second Yellow | 1 | 2 | -0.0371 | 0.1179 | -0.1551 |
| 3837695 | 1 | 23.4667 | Nantes | 18239 | Fábio Pereira da Silva | Red Card | 0 | 1 | -0.2579 | 0.0053 | -0.2632 |
| 3837747 | 2 | 71.5833 | Marseille | 43945 | Samuel Gigot | Red Card | 0 | 1 | 0.2545 | 0.0567 | 0.1978 |
| 3837839 | 2 | 54.7000 | Paris Saint-Germain | 3166 | Marco Verratti | Red Card | 1 | 0 | 0.0577 | -0.0003 | 0.0580 |
| 3837947 | 1 | 18.3333 | Lens | 31029 | Salis Abdul Samed | Red Card | 0 | 0 | -0.1474 | 0.0214 | -0.1688 |
| 3837963 | 1 | 19.5333 | Paris Saint-Germain | 5245 | Achraf Hakimi Mouh | Second Yellow | 0 | 1 | -0.1623 | -0.0824 | -0.0800 |
| 3837987 | 2 | 76.9167 | Paris Saint-Germain | 5245 | Achraf Hakimi Mouh | Red Card | 5 | 0 | 0.3443 | 0.1002 | 0.2441 |
| 3837987 | 2 | 79.6000 | AC Ajaccio | 3185 | Thomas Mangani | Red Card | 0 | 5 | 0.2209 | 0.0240 | 0.1969 |
| 3857273 | 2 | 83.7167 | Wales | 3488 | Wayne Hennessey | Red Card | 0 | 0 | -0.1859 | -0.0459 | -0.1400 |
| 3857280 | 2 | 92.9000 | Cameroon | 11174 | Vincent Paté Aboubakar | Second Yellow | 1 | 0 |  | 0.0790 |  |
| 3869486 | 2 | 92.1500 | Morocco | 296254 | Walid Cheddira | Second Yellow | 1 | 0 | -0.3182 | 0.0453 | -0.3634 |
| 3893790 | 2 | 95.5167 | Nigeria Women's | 401723 | Deborah Ajibola Abiodun | Red Card | 0 | 0 | -0.0971 | 0.0249 | -0.1221 |
| 3893793 | 2 | 96.5333 | Zambia W | 131629 | Catherine Musonda | Second Yellow | 0 | 4 |  | -0.0397 |  |
| 3893798 | 2 | 90.9833 | Jamaica Women's | 25554 | Khadija Monifa Shaw | Second Yellow | 0 | 0 | -0.0016 | -0.0254 | 0.0238 |
| 3893813 | 1 | 27.0833 | China PR Women's | 25490 | Rui Zhang | Red Card | 0 | 0 | 0.1706 | 0.0207 | 0.1499 |
| 3893820 | 2 | 62.8000 | Philippines Women's | 324160 | Sofia Nicole Dador Harrison | Red Card | 0 | 5 | -0.2458 | -0.0246 | -0.2211 |
| 3895250 | 2 | 77.9667 | FSV Mainz 05 | 40707 | Jessic Ngankam | Red Card | 1 | 2 | 0.0209 | -0.0438 | 0.0647 |
| 3895258 | 1 | 11.8333 | FC Köln | 33335 | Jan Thielmann | Red Card | 0 | 0 | -0.0226 | -0.0154 | -0.0072 |
| 3895292 | 1 | 47.5000 | Union Berlin | 6985 | Robin Gosens | Second Yellow | 0 | 0 |  | 0.0358 |  |
| 3895340 | 1 | 14.0000 | Bochum | 8241 | Felix Passlack | Red Card | 0 | 0 | -0.2214 | 0.0526 | -0.2741 |
| 3901735 | 2 | 86.3500 | England Women's | 31531 | Lauren James | Red Card | 0 | 0 | -0.1381 | -0.0009 | -0.1372 |
| 3930158 | 1 | 41.1167 | Scotland | 18797 | Ryan Porteous | Red Card | 0 | 2 | -0.0132 | 0.0664 | -0.0796 |
| 3930184 | 1 | 19.2833 | Czech Republic | 7801 | Antonín Barák | Second Yellow | 0 | 0 | 0.0234 | 0.0066 | 0.0168 |
| 3930184 | 2 | 97.6333 | Czech Republic | 21226 | Tomáš Chorý | Red Card | 1 | 2 |  | -0.0817 |  |
| 3942226 | 4 | 124.9167 | Spain | 5721 | Daniel Carvajal Ramos | Second Yellow | 2 | 1 |  | -0.3137 |  |
| 3942382 | 2 | 95.3500 | Turkey | 133168 | Bertuğ Özgür Yıldırım | Red Card | 1 | 2 |  | -0.0384 |  |
| 3998836 | 2 | 56.9333 | Iceland Women's | 32469 | Hildur Antonsdóttir | Second Yellow | 0 | 0 | 0.3618 | 0.0625 | 0.2993 |
| 3998846 | 2 | 94.7833 | Portugal Women's | 10141 | Ana Borges | Second Yellow | 1 | 1 |  | 0.0222 |  |
| 3998853 | 2 | 91.0833 | Norway Women's | 401647 | Marit Bratberg Lund | Second Yellow | 4 | 2 |  | 0.1106 |  |
| 3998856 | 1 | 30.6000 | Germany Women's | 46545 | Carlotta Wamser | Red Card | 1 | 2 | -0.5187 | 0.0427 | -0.5614 |
| 4018356 | 2 | 90.5833 | Switzerland Women's | 10404 | Noelle Maritz | Red Card | 0 | 2 | -0.1743 | 0.0817 | -0.2559 |
| 4018357 | 1 | 10.4667 | Germany Women's | 10399 | Kathrin Julia Hendrich | Red Card | 0 | 0 |  | 0.0457 |  |


## events_red_card_opponent (53 rows)

| match_id | period | clock | team | player_id | player | card | goals_for | goals_against | team_other | delta | placebo | effect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3773386 | 2 | 61.5500 | Deportivo Alavés | 9462 | José Ignacio Peleteiro Ramallo | Second Yellow | 1 | 0 | Barcelona | -0.0440 | 0.0458 | -0.0897 |
| 3773457 | 2 | 82.0833 | Barcelona | 6826 | Clément Lenglet | Second Yellow | 1 | 1 | Celta Vigo | 0.0448 | -0.0226 | 0.0674 |
| 3773466 | 1 | 41.5333 | Barcelona | 6826 | Clément Lenglet | Second Yellow | 1 | 0 | Celta Vigo | -0.1293 | 0.2374 | -0.3667 |
| 3773474 | 2 | 78.4667 | Real Valladolid | 11296 | Óscar Plano Pedreño | Red Card | 0 | 0 | Barcelona | -0.1509 | 0.0095 | -0.1604 |
| 3773477 | 2 | 57.0833 | Real Betis | 6648 | Aïssa Mandi | Red Card | 1 | 2 | Barcelona | -0.1792 | -0.0531 | -0.1261 |
| 3773497 | 2 | 89.3833 | Real Madrid | 5539 | Carlos Henrique Casimiro | Second Yellow | 2 | 1 | Barcelona | 0.0581 | 0.0508 | 0.0073 |
| 3773565 | 2 | 77.5000 | Granada | 6893 | Jesús Vallejo Lázaro | Red Card | 0 | 4 | Barcelona | 0.0329 | 0.0421 | -0.0092 |
| 3773689 | 2 | 64.5167 | Villarreal | 6771 | Manuel Trigueros Muñoz | Red Card | 1 | 2 | Barcelona | 0.1755 | 0.0763 | 0.0992 |
| 3788749 | 2 | 61.5333 | Poland | 3637 | Grzegorz Krychowiak | Second Yellow | 1 | 1 | Slovakia | -0.2331 | 0.1363 | -0.3693 |
| 3788766 | 2 | 54.7167 | Wales | 4934 | Ethan Ampadu | Red Card | 0 | 1 | Italy | -0.0109 | 0.0403 | -0.0512 |
| 3794689 | 2 | 89.7500 | Wales | 9540 | Harry Wilson | Red Card | 0 | 3 | Denmark | -0.0393 | 0.0283 | -0.0676 |
| 3794690 | 2 | 51.5000 | Netherlands | 8069 | Matthijs de Ligt | Red Card | 0 | 0 | Czech Republic | -0.0608 | -0.0153 | -0.0455 |
| 3794692 | 3 | 97.0667 | Sweden | 26875 | Marcus Andreas Danielsson | Red Card | 1 | 1 | Ukraine | -0.1860 | 0.0770 | -0.2630 |
| 3795108 | 2 | 76.3833 | Switzerland | 6983 | Remo Freuler | Red Card | 1 | 1 | Spain | -0.1773 | -0.0731 | -0.1042 |
| 3802676 | 2 | 56.1167 | Lens | 11820 | Kevin Danso | Second Yellow | 0 | 0 | Paris Saint-Germain | 0.0157 | 0.0802 | -0.0645 |
| 3802690 | 2 | 84.2833 | Paris Saint-Germain | 5201 | Sergio Ramos García | Second Yellow | 0 | 1 | Lorient | 0.4895 | 0.0032 | 0.4862 |
| 3802837 | 1 | 44.2833 | Saint-Étienne | 10476 | Timothée Kolodziejczak | Red Card | 1 | 0 | Paris Saint-Germain | 0.0954 | -0.0740 | 0.1695 |
| 3802902 | 2 | 53.9000 | Paris Saint-Germain | 5245 | Achraf Hakimi Mouh | Red Card | 0 | 0 | Olympique de Marseille | -0.2964 | 0.0120 | -0.3084 |
| 3802948 | 2 | 56.1167 | Metz | 58878 | Boubacar Traore | Second Yellow | 0 | 4 | Paris Saint-Germain | 0.1993 | -0.0652 | 0.2645 |
| 3802984 | 2 | 64.3833 | Paris Saint-Germain | 5597 | Keylor Navas Gamboa | Red Card | 1 | 0 | Nantes | -0.0737 | 0.0313 | -0.1050 |
| 3835322 | 2 | 93.8500 | Denmark Women's | 276443 | Kathrine Møller Kühl | Second Yellow | 0 | 4 | Germany Women's |  | 0.2149 |  |
| 3835333 | 2 | 85.7500 | Belgium Women's | 276907 | Amber Tysiak | Second Yellow | 1 | 2 | France Women's | -0.0034 | 0.1179 | -0.1213 |
| 3837695 | 1 | 23.4667 | Nantes | 18239 | Fábio Pereira da Silva | Red Card | 0 | 1 | Paris Saint-Germain | -0.3533 | 0.0053 | -0.3586 |
| 3837747 | 2 | 71.5833 | Marseille | 43945 | Samuel Gigot | Red Card | 0 | 1 | Paris Saint-Germain | -0.1342 | 0.0567 | -0.1909 |
| 3837839 | 2 | 54.7000 | Paris Saint-Germain | 3166 | Marco Verratti | Red Card | 1 | 0 | Stade de Reims | 0.4704 | -0.0003 | 0.4708 |
| 3837947 | 1 | 18.3333 | Lens | 31029 | Salis Abdul Samed | Red Card | 0 | 0 | Paris Saint-Germain | 0.1214 | 0.0214 | 0.0999 |
| 3837963 | 1 | 19.5333 | Paris Saint-Germain | 5245 | Achraf Hakimi Mouh | Second Yellow | 0 | 1 | Lorient | 0.2889 | -0.0824 | 0.3712 |
| 3837987 | 2 | 76.9167 | Paris Saint-Germain | 5245 | Achraf Hakimi Mouh | Red Card | 5 | 0 | AC Ajaccio | -0.0986 | 0.1002 | -0.1988 |
| 3837987 | 2 | 79.6000 | AC Ajaccio | 3185 | Thomas Mangani | Red Card | 0 | 5 | Paris Saint-Germain | 0.2198 | 0.0240 | 0.1958 |
| 3857273 | 2 | 83.7167 | Wales | 3488 | Wayne Hennessey | Red Card | 0 | 0 | Iran | -0.2769 | -0.0459 | -0.2310 |
| 3857280 | 2 | 92.9000 | Cameroon | 11174 | Vincent Paté Aboubakar | Second Yellow | 1 | 0 | Brazil | -0.0971 | 0.0790 | -0.1761 |
| 3869486 | 2 | 92.1500 | Morocco | 296254 | Walid Cheddira | Second Yellow | 1 | 0 | Portugal | 0.3162 | 0.0453 | 0.2710 |
| 3893790 | 2 | 95.5167 | Nigeria Women's | 401723 | Deborah Ajibola Abiodun | Red Card | 0 | 0 | Canada Women's | 0.6732 | 0.0249 | 0.6483 |
| 3893793 | 2 | 96.5333 | Zambia W | 131629 | Catherine Musonda | Second Yellow | 0 | 4 | Japan Women's |  | -0.0397 |  |
| 3893798 | 2 | 90.9833 | Jamaica Women's | 25554 | Khadija Monifa Shaw | Second Yellow | 0 | 0 | France Women's | 0.2516 | -0.0254 | 0.2770 |
| 3893813 | 1 | 27.0833 | China PR Women's | 25490 | Rui Zhang | Red Card | 0 | 0 | Haiti Women's | -0.1903 | 0.0207 | -0.2110 |
| 3893820 | 2 | 62.8000 | Philippines Women's | 324160 | Sofia Nicole Dador Harrison | Red Card | 0 | 5 | Norway Women's | 0.2257 | -0.0246 | 0.2504 |
| 3895250 | 2 | 77.9667 | FSV Mainz 05 | 40707 | Jessic Ngankam | Red Card | 1 | 2 | Bayer Leverkusen | -0.2574 | -0.0438 | -0.2137 |
| 3895258 | 1 | 11.8333 | FC Köln | 33335 | Jan Thielmann | Red Card | 0 | 0 | Bayer Leverkusen | 0.1585 | -0.0154 | 0.1739 |
| 3895292 | 1 | 47.5000 | Union Berlin | 6985 | Robin Gosens | Second Yellow | 0 | 0 | Bayer Leverkusen | 0.2007 | 0.0358 | 0.1648 |
| 3895340 | 1 | 14.0000 | Bochum | 8241 | Felix Passlack | Red Card | 0 | 0 | Bayer Leverkusen | 0.1076 | 0.0526 | 0.0549 |
| 3901735 | 2 | 86.3500 | England Women's | 31531 | Lauren James | Red Card | 0 | 0 | Nigeria Women's | -0.0843 | -0.0009 | -0.0834 |
| 3930158 | 1 | 41.1167 | Scotland | 18797 | Ryan Porteous | Red Card | 0 | 2 | Germany | 0.6867 | 0.0664 | 0.6203 |
| 3930184 | 1 | 19.2833 | Czech Republic | 7801 | Antonín Barák | Second Yellow | 0 | 0 | Turkey | 0.2481 | 0.0066 | 0.2415 |
| 3930184 | 2 | 97.6333 | Czech Republic | 21226 | Tomáš Chorý | Red Card | 1 | 2 | Turkey |  | -0.0817 |  |
| 3942226 | 4 | 124.9167 | Spain | 5721 | Daniel Carvajal Ramos | Second Yellow | 2 | 1 | Germany |  | -0.3137 |  |
| 3942382 | 2 | 95.3500 | Turkey | 133168 | Bertuğ Özgür Yıldırım | Red Card | 1 | 2 | Netherlands |  | -0.0384 |  |
| 3998836 | 2 | 56.9333 | Iceland Women's | 32469 | Hildur Antonsdóttir | Second Yellow | 0 | 0 | WNT Finland | 0.1814 | 0.0625 | 0.1189 |
| 3998846 | 2 | 94.7833 | Portugal Women's | 10141 | Ana Borges | Second Yellow | 1 | 1 | Italy Women's | 0.1336 | 0.0222 | 0.1114 |
| 3998853 | 2 | 91.0833 | Norway Women's | 401647 | Marit Bratberg Lund | Second Yellow | 4 | 2 | Iceland Women's |  | 0.1106 |  |
| 3998856 | 1 | 30.6000 | Germany Women's | 46545 | Carlotta Wamser | Red Card | 1 | 2 | Sweden Women's | -0.2205 | 0.0427 | -0.2631 |
| 4018356 | 2 | 90.5833 | Switzerland Women's | 10404 | Noelle Maritz | Red Card | 0 | 2 | Spain Women's | 0.1952 | 0.0817 | 0.1135 |
| 4018357 | 1 | 10.4667 | Germany Women's | 10399 | Kathrin Julia Hendrich | Red Card | 0 | 0 | France Women's | 0.3279 | 0.0457 | 0.2822 |


## events_tactical_shift (1404 rows, first 60 shown)

| match_id | period | clock | team | formation | goals_for | goals_against | with_substitution | state | delta | placebo | effect |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3764440 | 2 | 45.0000 | Barcelona | 4231 | 0 | 0 | True | level |  | 0.0660 |  |
| 3764440 | 2 | 65.0500 | Elche | 343 | 0 | 1 | False | trailing | 0.0046 | 0.0386 | -0.0341 |
| 3773369 | 2 | 67.4833 | Huesca | 3511 | 1 | 3 | False | trailing | 0.4301 | -0.0787 | 0.5087 |
| 3773369 | 2 | 69.3167 | Barcelona | 3421 | 3 | 1 | True | leading | -0.0535 | -0.0392 | -0.0143 |
| 3773369 | 2 | 77.9000 | Huesca | 3511 | 1 | 3 | True | trailing | -0.3784 | 0.0065 | -0.3849 |
| 3773369 | 2 | 78.3333 | Huesca | 3511 | 1 | 3 | True | trailing | -0.2868 | -0.0221 | -0.2647 |
| 3773369 | 2 | 79.2333 | Huesca | 3511 | 1 | 3 | False | trailing | -0.2599 | 0.0671 | -0.3270 |
| 3773372 | 2 | 76.4333 | Barcelona | 433 | 0 | 0 | False | level | 0.3062 | 0.0230 | 0.2831 |
| 3773377 | 2 | 45.0000 | Barcelona | 4231 | 1 | 1 | True | level |  | 0.0930 |  |
| 3773377 | 2 | 72.8000 | Barcelona | 4231 | 2 | 2 | True | level | 0.1343 | -0.0325 | 0.1668 |
| 3773377 | 2 | 78.3667 | Barcelona | 343 | 2 | 2 | True | level | 0.0678 | -0.0127 | 0.0805 |
| 3773386 | 2 | 45.0000 | Barcelona | 4231 | 0 | 1 | True | trailing |  | -0.0671 |  |
| 3773387 | 1 | 3.5667 | Barcelona | 352 | 0 | 0 | False | level | 0.0366 | 0.0119 | 0.0248 |
| 3773387 | 2 | 60.8833 | Levante UD | 442 | 2 | 2 | True | level | 0.0471 | 0.0250 | 0.0221 |
| 3773403 | 2 | 83.6833 | Barcelona | 3421 | 2 | 1 | True | leading | -0.1038 | -0.0703 | -0.0335 |
| 3773415 | 2 | 58.3667 | Cádiz | 4141 | 0 | 1 | True | trailing | 0.3712 | -0.0417 | 0.4130 |
| 3773415 | 2 | 61.6333 | Barcelona | 433 | 1 | 0 | False | leading | -0.0533 | 0.0697 | -0.1230 |
| 3773415 | 2 | 64.6500 | Barcelona | 433 | 1 | 0 | True | leading | 0.0589 | 0.0231 | 0.0358 |
| 3773415 | 2 | 74.2333 | Barcelona | 4231 | 1 | 0 | True | leading | -0.0738 | -0.0902 | 0.0164 |
| 3773415 | 2 | 75.2833 | Cádiz | 352 | 0 | 1 | True | trailing | 0.0615 | 0.0191 | 0.0424 |
| 3773415 | 2 | 80.8833 | Barcelona | 433 | 1 | 0 | False | leading | -0.0381 | 0.0031 | -0.0412 |
| 3773428 | 2 | 45.0500 | Cádiz | 4141 | 1 | 0 | False | leading |  | 0.0089 |  |
| 3773428 | 2 | 47.3000 | Barcelona | 4231 | 0 | 1 | False | trailing | 0.2006 | -0.0630 | 0.2636 |
| 3773457 | 2 | 70.9667 | Barcelona | 352 | 1 | 1 | True | level | -0.2502 | 0.0152 | -0.2654 |
| 3773466 | 1 | 32.0167 | Celta Vigo | 352 | 0 | 1 | True | trailing | 0.1386 | 0.0756 | 0.0630 |
| 3773466 | 1 | 32.2333 | Celta Vigo | 352 | 0 | 1 | True | trailing | 0.1386 | -0.1156 | 0.2542 |
| 3773466 | 1 | 35.7000 | Celta Vigo | 352 | 0 | 1 | False | trailing | 0.1189 | -0.0019 | 0.1208 |
| 3773466 | 1 | 37.2667 | Celta Vigo | 352 | 0 | 1 | False | trailing | 0.1514 | 0.0944 | 0.0571 |
| 3773466 | 1 | 48.1500 | Barcelona | 4231 | 1 | 0 | False | leading |  | -0.0433 |  |
| 3773466 | 2 | 47.8167 | Celta Vigo | 352 | 0 | 1 | False | trailing | -0.0207 | 0.0512 | -0.0719 |
| 3773466 | 2 | 59.7167 | Celta Vigo | 4231 | 0 | 2 | True | trailing | 0.0707 | 0.0821 | -0.0114 |
| 3773466 | 2 | 64.6000 | Celta Vigo | 4231 | 0 | 2 | False | trailing | -0.0373 | 0.0294 | -0.0667 |
| 3773466 | 2 | 67.6500 | Celta Vigo | 4231 | 0 | 2 | False | trailing | -0.0325 | -0.0232 | -0.0093 |
| 3773466 | 2 | 71.8333 | Celta Vigo | 4231 | 0 | 2 | False | trailing | 0.0572 | -0.0571 | 0.1143 |
| 3773466 | 2 | 73.4333 | Barcelona | 4231 | 2 | 0 | False | leading | -0.0128 | 0.0255 | -0.0383 |
| 3773474 | 2 | 45.0000 | Barcelona | 4141 | 0 | 0 | False | level |  | 0.0526 |  |
| 3773474 | 2 | 63.6333 | Barcelona | 433 | 0 | 0 | True | level | -0.4620 | -0.0307 | -0.4313 |
| 3773477 | 2 | 45.0000 | Barcelona | 442 | 1 | 1 | True | level |  | 0.1004 |  |
| 3773477 | 2 | 62.3500 | Real Betis | 4231 | 1 | 3 | True | trailing | -0.3019 | -0.0519 | -0.2499 |
| 3773477 | 2 | 72.2000 | Real Betis | 4231 | 1 | 3 | False | trailing | 0.3584 | 0.0234 | 0.3350 |
| 3773497 | 2 | 49.5167 | Barcelona | 352 | 0 | 2 | False | trailing | 0.1336 | 0.0287 | 0.1049 |
| 3773497 | 2 | 80.7000 | Barcelona | 352 | 1 | 2 | True | trailing | 0.1770 | 0.0462 | 0.1307 |
| 3773523 | 2 | 56.9667 | Real Sociedad | 433 | 1 | 2 | False | trailing | -0.0719 | -0.0312 | -0.0407 |
| 3773523 | 2 | 75.8833 | Barcelona | 433 | 2 | 1 | False | leading | -0.0020 | 0.0547 | -0.0567 |
| 3773526 | 2 | 45.0000 | Barcelona | 433 | 2 | 0 | True | leading |  | -0.0128 |  |
| 3773526 | 2 | 46.9833 | Deportivo Alavés | 442 | 0 | 2 | False | trailing | 0.1919 | 0.0732 | 0.1187 |
| 3773547 | 2 | 61.7333 | Barcelona | 433 | 3 | 0 | True | leading | -0.0120 | 0.0518 | -0.0638 |
| 3773547 | 2 | 66.6167 | Barcelona | 4231 | 3 | 0 | True | leading | 0.3842 | -0.1287 | 0.5129 |
| 3773552 | 2 | 77.1833 | Huesca | 433 | 0 | 1 | False | trailing | -0.1262 | 0.0481 | -0.1743 |
| 3773552 | 2 | 81.1667 | Barcelona | 433 | 1 | 0 | True | leading | 0.2209 | 0.1238 | 0.0971 |
| 3773565 | 2 | 45.1333 | Granada | 4231 | 0 | 3 | True | trailing |  | 0.0844 |  |
| 3773565 | 2 | 75.4667 | Barcelona | 433 | 4 | 0 | False | leading | -0.0676 | 0.0539 | -0.1215 |
| 3773565 | 2 | 77.8833 | Granada | 4231 | 0 | 4 | False | trailing | -0.4284 | 0.0168 | -0.4452 |
| 3773571 | 2 | 78.1667 | Real Valladolid | 442 | 0 | 3 | True | trailing | 0.1032 | 0.0254 | 0.0778 |
| 3773585 | 2 | 89.2500 | Barcelona | 352 | 1 | 2 | False | trailing | 0.2564 | -0.0301 | 0.2865 |
| 3773586 | 2 | 73.1000 | Barcelona | 4141 | 1 | 1 | False | level | 0.2019 | 0.0162 | 0.1857 |
| 3773586 | 2 | 80.3500 | Barcelona | 4231 | 1 | 2 | True | trailing | 0.0796 | -0.0259 | 0.1055 |
| 3773587 | 2 | 79.7000 | Barcelona | 4231 | 0 | 1 | True | trailing | -0.0193 | -0.0168 | -0.0025 |
| 3773587 | 2 | 92.6667 | Getafe | 442 | 1 | 0 | False | leading | 0.0610 | -0.0496 | 0.1107 |
| 3773593 | 2 | 47.3667 | Villarreal | 433 | 0 | 4 | False | trailing | 0.2451 | 0.0917 | 0.1534 |


## events_tactical_substitution (3381 rows, first 60 shown)

| match_id | period | clock | team | player_off_id | player_off | player_on_id | player_on | reason | goals_for | goals_against | delta | placebo | effect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3764440 | 2 | 45.0000 | Barcelona | 6947 | Miralem Pjanić | 5477 | Ousmane Dembélé | Tactical | 0 | 0 |  | 0.0326 |  |
| 3764440 | 2 | 59.3333 | Elche | 7897 | Miguel Ángel Garrido Cifuentes | 23825 | Fidel Chaves de la Torre | Tactical | 0 | 1 | -0.0527 | 0.0805 | -0.1332 |
| 3764440 | 2 | 59.4167 | Elche | 15880 | Emiliano Ariel Rigoni | 24251 | José Antonio Morente Oliva | Tactical | 0 | 1 | -0.0753 | -0.0235 | -0.0519 |
| 3764440 | 2 | 65.8167 | Barcelona | 22390 | Francisco António Machado Mota de Castro Trincão | 5203 | Sergio Busquets i Burgos | Tactical | 1 | 0 | 0.0642 | -0.8371 | 0.9013 |
| 3764440 | 2 | 71.3833 | Elche | 7064 | Lucas Ariel Boyé | 3246 | Guido Marcelo Carrillo | Tactical | 0 | 2 | -0.1752 |  |  |
| 3764440 | 2 | 71.6167 | Elche | 12072 | Pere Milla Peña | 24166 | Juan Francisco Martínez Modesto | Tactical | 0 | 2 | -0.2445 |  |  |
| 3764440 | 2 | 75.3000 | Barcelona | 4447 | Martin Braithwaite Christensen | 5487 | Antoine Griezmann | Tactical | 3 | 0 | 0.0504 | 0.0021 | 0.0483 |
| 3764440 | 2 | 75.5500 | Barcelona | 5213 | Gerard Piqué Bernabéu | 6826 | Clément Lenglet | Tactical | 3 | 0 | 0.0312 | 0.0659 | -0.0347 |
| 3764440 | 2 | 76.1333 | Barcelona | 5211 | Jordi Alba Ramos | 21881 | Sergino Dest | Tactical | 3 | 0 | 0.0056 | 0.2066 | -0.2010 |
| 3764661 | 1 | 30.9833 | Athletic Club | 3023 | Yuri Berchiche Izeta | 6789 | Mikel Balenziaga Oruesagasti | Tactical | 1 | 1 | -0.1485 | 0.0258 | -0.1742 |
| 3764661 | 2 | 64.5833 | Athletic Club | 6667 | Mikel Vesga Arruti | 6775 | Daniel García Carrillo | Tactical | 1 | 3 | 0.1676 | -0.1563 | 0.3238 |
| 3764661 | 2 | 64.5833 | Athletic Club | 6649 | Óscar de Marcos Arana | 7478 | Alejandro Berenguer Remiro | Tactical | 1 | 3 | 0.1676 | 0.0172 | 0.1504 |
| 3764661 | 2 | 74.3500 | Barcelona | 21881 | Sergino Dest | 43728 | Óscar Mingueza García | Tactical | 3 | 1 | -0.0336 | -0.4888 | 0.4552 |
| 3764661 | 2 | 74.4333 | Barcelona | 5477 | Ousmane Dembélé | 4447 | Martin Braithwaite Christensen | Tactical | 3 | 1 | -0.0147 |  |  |
| 3764661 | 2 | 79.3167 | Athletic Club | 6391 | Raúl García Escudero | 47226 | Jon Morcillo Conesa | Tactical | 1 | 3 | -0.1160 |  |  |
| 3773369 | 2 | 59.3000 | Huesca | 24681 | Daniel Escriche Romero | 22199 | Sergio Gómez Martín | Tactical | 1 | 3 | 0.0389 | 0.0935 | -0.0546 |
| 3773369 | 2 | 59.5667 | Huesca | 23307 | Idrissa Doumbia | 30483 | Jaime Seoane Valenciano | Tactical | 1 | 3 | 0.0915 | -0.0706 | 0.1621 |
| 3773369 | 2 | 62.2667 | Barcelona | 8118 | Frenkie de Jong | 32480 | Ronald Federico Araújo da Silva | Tactical | 3 | 1 | 0.0555 | 0.1383 | -0.0828 |
| 3773369 | 2 | 68.5667 | Barcelona | 5487 | Antoine Griezmann | 39073 | Moriba Kourouma Kourouma | Tactical | 3 | 1 | 0.0242 | 0.0880 | -0.0639 |
| 3773369 | 2 | 68.5833 | Barcelona | 21881 | Sergino Dest | 22390 | Francisco António Machado Mota de Castro Trincão | Tactical | 3 | 1 | 0.0242 | 0.1099 | -0.0857 |
| 3773369 | 2 | 77.2833 | Huesca | 21399 | Pablo Ínsua Blanco | 16298 | Denis Vavro | Tactical | 1 | 3 | -0.3774 | -0.3434 | -0.0340 |
| 3773369 | 2 | 77.3167 | Huesca | 6564 | Pablo Maffeo Becerra | 6787 | Pedro López Muñoz | Tactical | 1 | 3 | -0.3774 | -0.1516 | -0.2258 |
| 3773369 | 2 | 77.3500 | Huesca | 11670 | David Ferreiro Quiroga | 20063 | Javier Ontiveros Parra | Tactical | 1 | 3 | -0.3774 |  |  |
| 3773369 | 2 | 84.3500 | Barcelona | 5477 | Ousmane Dembélé | 4447 | Martin Braithwaite Christensen | Tactical | 3 | 1 | 0.0544 | -0.4060 | 0.4604 |
| 3773369 | 2 | 84.4833 | Barcelona | 30486 | Pedro González López | 24841 | Ricard Puig Martí | Tactical | 3 | 1 | 0.0383 | 0.3726 | -0.3343 |
| 3773372 | 1 | 12.9500 | Atlético Madrid | 3245 | Thomas Lemar | 6381 | Saúl Ñíguez Esclapez | Tactical | 0 | 0 | -0.1236 | -0.0991 | -0.0245 |
| 3773372 | 2 | 45.0000 | Barcelona | 43728 | Óscar Mingueza García | 32480 | Ronald Federico Araújo da Silva | Tactical | 0 | 0 |  | -0.1122 |  |
| 3773372 | 2 | 66.9167 | Atlético Madrid | 6381 | Saúl Ñíguez Esclapez | 12041 | João Félix Sequeira | Tactical | 0 | 0 | 0.1891 | 0.0720 | 0.1172 |
| 3773372 | 2 | 72.2833 | Atlético Madrid | 6377 | Ángel Martín Correa | 6589 | Geoffrey Kondogbia | Tactical | 0 | 0 | 0.0916 |  |  |
| 3773372 | 2 | 74.7167 | Barcelona | 21881 | Sergino Dest | 5477 | Ousmane Dembélé | Tactical | 0 | 0 | 0.2693 |  |  |
| 3773372 | 2 | 74.7667 | Barcelona | 30486 | Pedro González López | 6379 | Sergi Roberto Carnicer | Tactical | 0 | 0 | 0.2608 | -0.0642 | 0.3250 |
| 3773377 | 2 | 45.0000 | Barcelona | 5203 | Sergio Busquets i Burgos | 8118 | Frenkie de Jong | Tactical | 1 | 1 |  | 0.0622 |  |
| 3773377 | 2 | 72.7667 | Barcelona | 5487 | Antoine Griezmann | 22390 | Francisco António Machado Mota de Castro Trincão | Tactical | 2 | 2 | 0.1343 | 0.0685 | 0.0658 |
| 3773377 | 2 | 78.3500 | Barcelona | 3501 | Philippe Coutinho Correia | 6826 | Clément Lenglet | Tactical | 2 | 2 | 0.0678 | 0.2429 | -0.1751 |
| 3773377 | 2 | 85.0833 | Barcelona | 30486 | Pedro González López | 6947 | Miralem Pjanić | Tactical | 2 | 2 | 0.1095 | 0.2604 | -0.1509 |
| 3773377 | 2 | 86.6667 | Valencia | 4367 | Gonçalo Manuel Ganchinho Guedes | 23978 | Manuel Javier Vallejo Galván | Tactical | 2 | 2 | 0.1024 | 0.2001 | -0.0977 |
| 3773377 | 2 | 89.5500 | Valencia | 5186 | Denis Cheryshev | 22740 | Kang-In Lee | Tactical | 2 | 2 | 0.1776 | 0.0658 | 0.1118 |
| 3773386 | 2 | 45.0000 | Barcelona | 6826 | Clément Lenglet | 30486 | Pedro González López | Tactical | 0 | 1 |  | 0.0595 |  |
| 3773386 | 2 | 45.0000 | Barcelona | 5203 | Sergio Busquets i Burgos | 6947 | Miralem Pjanić | Tactical | 0 | 1 |  | -0.0762 |  |
| 3773386 | 2 | 45.0000 | Barcelona | 5477 | Ousmane Dembélé | 22390 | Francisco António Machado Mota de Castro Trincão | Tactical | 0 | 1 |  | 0.1518 |  |
| 3773386 | 2 | 56.6333 | Deportivo Alavés | 25134 | Deyverson Brum Silva Acosta | 3265 | José Luis Sanmartín Mato | Tactical | 1 | 0 | -0.0791 | -0.0029 | -0.0762 |
| 3773386 | 2 | 56.6500 | Deportivo Alavés | 26387 | Edgar Antonio Méndez Ortega | 31779 | Tomas Franco Tavares | Tactical | 1 | 0 | -0.0791 | -0.0161 | -0.0630 |
| 3773386 | 2 | 68.2500 | Barcelona | 5211 | Jordi Alba Ramos | 21881 | Sergino Dest | Tactical | 1 | 1 | 0.2159 | -0.0946 | 0.3105 |
| 3773386 | 2 | 71.5833 | Deportivo Alavés | 6622 | Tomás Pina Isla | 6935 | Adrián Marín Gómez | Tactical | 1 | 1 | 0.3846 |  |  |
| 3773386 | 2 | 71.7833 | Deportivo Alavés | 24049 | Luis Jesús Rioja González | 6632 | Manuel Alejandro García Sánchez | Tactical | 1 | 1 | 0.3846 | -0.0105 | 0.3951 |
| 3773386 | 2 | 77.3000 | Barcelona | 30756 | Anssumane Fati | 4447 | Martin Braithwaite Christensen | Tactical | 1 | 1 | -0.0315 | 0.1713 | -0.2028 |
| 3773387 | 2 | 45.0000 | Barcelona | 32480 | Ronald Federico Araújo da Silva | 6379 | Sergi Roberto Carnicer | Tactical | 2 | 0 |  | -0.0490 |  |
| 3773387 | 2 | 60.4167 | Levante UD | 6782 | Enis Bardhi | 30593 | Daniel Gómez Alcón | Tactical | 2 | 2 | 0.0552 | 0.0310 | 0.0242 |
| 3773387 | 2 | 60.8167 | Levante UD | 5598 | Óscar Esau Duarte Gaitán | 23966 | Mickaël Malsa | Tactical | 2 | 2 | 0.0464 | 0.1432 | -0.0968 |
| 3773387 | 2 | 73.2167 | Barcelona | 30486 | Pedro González López | 43728 | Óscar Mingueza García | Tactical | 3 | 2 | -0.1863 | -0.1468 | -0.0395 |
| 3773387 | 2 | 74.4667 | Levante UD | 11676 | Jorge Miramón Santagertrudis | 6672 | Jorge Andújar Moreno | Tactical | 2 | 3 | 0.2882 |  |  |
| 3773387 | 2 | 74.6833 | Levante UD | 6684 | Roger Martí Salvador | 6657 | Sergio León Limones | Tactical | 2 | 3 | 0.2882 | -0.5812 | 0.8694 |
| 3773387 | 2 | 80.8667 | Barcelona | 5477 | Ousmane Dembélé | 21881 | Sergino Dest | Tactical | 3 | 2 | 0.1284 | 0.2320 | -0.1037 |
| 3773387 | 2 | 80.8833 | Barcelona | 5487 | Antoine Griezmann | 4447 | Martin Braithwaite Christensen | Tactical | 3 | 2 | 0.1284 | -0.2479 | 0.3763 |
| 3773387 | 2 | 84.3000 | Barcelona | 6379 | Sergi Roberto Carnicer | 24841 | Ricard Puig Martí | Tactical | 3 | 3 | 0.2691 | 0.3233 | -0.0542 |
| 3773387 | 2 | 89.2500 | Levante UD | 6668 | José Luis Morales Nogales | 41202 | Giorgi Kochorashvili | Tactical | 3 | 3 | 0.2141 | 0.0767 | 0.1375 |
| 3773403 | 2 | 65.9167 | Athletic Club | 35473 | Unai Vencedor Paris | 6667 | Mikel Vesga Arruti | Tactical | 1 | 1 | -0.1209 | 0.1167 | -0.2376 |
| 3773403 | 2 | 66.1167 | Athletic Club | 6649 | Óscar de Marcos Arana | 7478 | Alejandro Berenguer Remiro | Tactical | 1 | 1 | -0.1209 | -0.1326 | 0.0117 |
| 3773403 | 2 | 66.2667 | Barcelona | 6947 | Miralem Pjanić | 6379 | Sergi Roberto Carnicer | Tactical | 1 | 1 | -0.1262 | -0.3767 | 0.2505 |
| 3773403 | 2 | 83.1000 | Athletic Club | 6393 | Iñaki Williams Arthuer | 32953 | Asier Villalibre Molina | Tactical | 1 | 2 | -0.0561 |  |  |

