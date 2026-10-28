# Hidden contributions: creativity by role, credited vs. uncredited

Creative = top 10% of good_surprise among in-scope decisions. Credited = shot, assist (key pass).

## roles_goal (5 rows)

| role | decisions | mean_z | creative_rate | creative | creativity_sum | share_of_creative | share_of_creativity |
|---|---|---|---|---|---|---|---|
| assist | 704 | 0.7017 | 0.2472 | 174 | 1950.0139 | 0.1540 | 0.1324 |
| earlier | 5064 | 0.0358 | 0.1062 | 538 | 6995.4192 | 0.4761 | 0.4748 |
| other (no credited pass) | 2801 | 0.1069 | 0.1103 | 309 | 4284.3701 | 0.2735 | 0.2908 |
| pre-assist | 607 | 0.4468 | 0.1796 | 109 | 1358.7179 | 0.0965 | 0.0922 |
| shot | 1283 | -0.5726 | 0.0000 | 0 | 144.5857 | 0.0000 | 0.0098 |


## uncredited_share (2 rows)

| possessions | creative_decisions | uncredited_share | low | high | uncredited_share_of_creativity |
|---|---|---|---|---|---|
| goal possessions | 1130 | 0.8460 | 0.8274 | 0.8651 | 0.8578 |
| shot possessions | 9678 | 0.8655 | 0.8578 | 0.8723 | 0.8674 |


## roles_shot (5 rows)

| role | decisions | mean_z | creative_rate | creative | creativity_sum | share_of_creative | share_of_creativity |
|---|---|---|---|---|---|---|---|
| earlier | 39287 | 0.0285 | 0.1048 | 4118 | 53671.9011 | 0.4255 | 0.4240 |
| key pass | 6156 | 0.4942 | 0.2083 | 1282 | 14388.0124 | 0.1325 | 0.1137 |
| other (no credited pass) | 30580 | 0.1051 | 0.1141 | 3488 | 46661.8623 | 0.3604 | 0.3686 |
| pre-key pass | 4805 | 0.3184 | 0.1602 | 770 | 9469.2250 | 0.0796 | 0.0748 |
| shot | 10197 | -0.5138 | 0.0020 | 20 | 2399.7566 | 0.0021 | 0.0190 |


## player_creativity_vs_ga (1 rows)

| players | spearman_within_position | low | high | share_top_quartile_creators_with_below_median_ga |
|---|---|---|---|---|
| 261 | -0.0208 | -0.1446 | 0.0996 | 0.4730 |


## hidden_creators (35 rows)

| player | team | position | women | decisions | minutes | goals | assists | ga_per90 | creativity | creativity_pct | ga_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Jule Brand | Germany Women's | Right Wing | True | 414 | 1030.4667 | 2.0000 | 2.0000 | 0.3494 | 0.1618 | 1.0000 | 0.3333 |
| Andreia Alexandra Norton | Portugal Women's | Right Center Midfield | True | 314 | 652.8833 | 0.0000 | 0.0000 | 0.0000 | 0.1038 | 1.0000 | 0.0909 |
| Vladimír Coufal | Czech Republic | Right Back | False | 406 | 760.7667 | 0.0000 | 1.0000 | 0.1183 | 0.1376 | 1.0000 | 0.3226 |
| Lieke Martens | Netherlands Women's | Left Center Forward | True | 402 | 756.4333 | 1.0000 | 2.0000 | 0.3569 | 0.0452 | 1.0000 | 0.2500 |
| Ana Borges | Portugal Women's | Right Back | True | 326 | 698.1833 | 0.0000 | 0.0000 | 0.0000 | 0.2678 | 1.0000 | 0.1471 |
| Toni Kroos | Germany | Left Defensive Midfield | False | 1118 | 1036.5667 | 1.0000 | 0.0000 | 0.0868 | 0.1829 | 1.0000 | 0.3462 |
| Pernille Mosegaard Harder | Denmark Women's | Center Attacking Midfield | True | 320 | 876.1667 | 2.0000 | 1.0000 | 0.3082 | 0.1817 | 1.0000 | 0.2500 |
| Fabian Lukas Schär | Switzerland | Right Center Back | False | 509 | 841.3333 | 0.0000 | 0.0000 | 0.0000 | 0.1332 | 0.9815 | 0.2500 |
| Leah Williamson | England Women's | Left Center Back | True | 929 | 1114.3167 | 0.0000 | 0.0000 | 0.0000 | 0.0870 | 0.9643 | 0.2500 |
| Sofyan Amrabat | Morocco | Center Defensive Midfield | False | 354 | 698.3833 | 0.0000 | 0.0000 | 0.0000 | 0.1488 | 0.9615 | 0.1346 |
| Hakim Ziyech | Morocco | Right Wing | False | 336 | 662.6333 | 1.0000 | 1.0000 | 0.2716 | 0.1645 | 0.9444 | 0.2222 |
| Kyle Walker | England | Right Back | False | 1111 | 1577.0500 | 0.0000 | 0.0000 | 0.0000 | 0.0061 | 0.9355 | 0.1129 |
| Gun Nathalie Björn | Sweden Women's | Right Center Back | True | 805 | 1332.3500 | 0.0000 | 0.0000 | 0.0000 | 0.0527 | 0.9286 | 0.2500 |
| Joachim Andersen | Denmark | Right Center Back | False | 630 | 747.5333 | 0.0000 | 0.0000 | 0.0000 | 0.0835 | 0.9259 | 0.2500 |
| Johanna Rytting-Kaneryd | Sweden Women's | Right Wing | True | 470 | 1086.8333 | 0.0000 | 2.0000 | 0.1656 | 0.1017 | 0.9167 | 0.0833 |
| Harry Kane | England | Center Forward | False | 480 | 1872.5333 | 10.0000 | 3.0000 | 0.6248 | 0.1180 | 0.9091 | 0.3636 |
| Kevin De Bruyne | Belgium | Center Attacking Midfield | False | 578 | 949.2333 | 2.0000 | 2.0000 | 0.3793 | 0.1375 | 0.9091 | 0.4545 |
| Abdou Diallo | Paris Saint-Germain | Left Center Back | False | 526 | 717.3833 | 0.0000 | 0.0000 | 0.0000 | 0.0601 | 0.9074 | 0.2500 |
| Mateo Kovačić | Croatia | Left Center Midfield | False | 1033 | 1226.7500 | 0.0000 | 1.0000 | 0.0734 | 0.1102 | 0.8889 | 0.3333 |
| Milos Veljkovic | Serbia | Right Center Back | False | 497 | 697.3167 | 0.0000 | 0.0000 | 0.0000 | 0.0569 | 0.8889 | 0.2500 |
| Marco Verratti | Paris Saint-Germain | Center Defensive Midfield | False | 4496 | 3751.4667 | 1.0000 | 4.0000 | 0.1200 | 0.0952 | 0.8846 | 0.4231 |
| Ingrid Syrstad Engen | Norway Women's | Left Defensive Midfield | True | 382 | 806.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0748 | 0.8750 | 0.1875 |
| João Pedro Cavaco Cancelo | Portugal | Right Back | False | 542 | 692.6833 | 0.0000 | 0.0000 | 0.0000 | -0.0051 | 0.8710 | 0.1129 |
| Sergio Busquets i Burgos | Barcelona | Center Defensive Midfield | False | 3433 | 3390.7000 | 0.0000 | 5.0000 | 0.1327 | 0.0709 | 0.8462 | 0.5000 |
| Ousmane Dembélé | Barcelona | Right Wing | False | 1761 | 2462.7667 | 5.0000 | 5.0000 | 0.3654 | 0.1031 | 0.8333 | 0.4444 |
| Aaron Ramsey | Wales | Right Center Midfield | False | 449 | 790.3000 | 1.0000 | 0.0000 | 0.1139 | 0.1099 | 0.8333 | 0.4444 |
| Aurélien Djani Tchouaméni | France | Right Defensive Midfield | False | 1100 | 1240.7333 | 1.0000 | 0.0000 | 0.0725 | 0.0600 | 0.8077 | 0.3077 |
| Yannick Ferreira Carrasco | Belgium | Left Wing Back | False | 343 | 668.9833 | 1.0000 | 0.0000 | 0.1345 | -0.0119 | 0.8065 | 0.4194 |
| Mykola Matvienko | Ukraine | Left Center Back | False | 750 | 781.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0173 | 0.7963 | 0.2500 |
| Magdalena Lilly Eriksson | Sweden Women's | Left Center Back | True | 722 | 1259.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0352 | 0.7857 | 0.2500 |
| Azzedine Ounahi | Morocco | Right Center Midfield | False | 347 | 601.0833 | 0.0000 | 0.0000 | 0.0000 | 0.0540 | 0.7778 | 0.1389 |
| Granit Xhaka | Bayer Leverkusen | Left Defensive Midfield | False | 4342 | 3952.1667 | 3.0000 | 1.0000 | 0.0911 | 0.0483 | 0.7692 | 0.3846 |
| Ricardo Iván Rodríguez Araya | Switzerland | Left Center Back | False | 734 | 1368.5500 | 0.0000 | 0.0000 | 0.0000 | 0.0015 | 0.7593 | 0.2500 |
| Ramona Bachmann | Switzerland Women's | Center Forward | True | 330 | 606.5333 | 2.0000 | 1.0000 | 0.4452 | 0.0072 | 0.7500 | 0.5000 |
| Teresa Abelleira Dueñas | Spain Women's | Center Defensive Midfield | True | 643 | 706.2500 | 1.0000 | 0.0000 | 0.1274 | 0.0630 | 0.7500 | 0.5000 |


## players_creativity_ga (261 rows, first 60 shown)

| player_id | player | creativity | decisions | position | team | women | minutes | goals | assists | ga_per90 | group | creativity_pct | ga_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2935 | Nordi Mukiele Mulere | -0.1205 | 495 | Right Center Back | Paris Saint-Germain | False | 894.6333 | 0.0000 | 3.0000 | 0.3018 | CB | 0.0926 | 1.0000 |
| 2954 | Youri Tielemans | 0.0294 | 398 | Right Defensive Midfield | Belgium | False | 611.1167 | 1.0000 | 0.0000 | 0.1473 | DM | 0.6538 | 0.6154 |
| 2988 | Memphis Depay | -0.0208 | 488 | Left Center Forward | Netherlands | False | 1117.3667 | 4.0000 | 2.0000 | 0.4833 | FW | 0.6364 | 0.1818 |
| 2995 | Ángel Fabián Di María Hernández | 0.0938 | 1337 | Right Wing | Paris Saint-Germain | False | 1717.3167 | 5.0000 | 5.0000 | 0.5241 | W | 0.7778 | 0.6111 |
| 2999 | Presnel Kimpembe | -0.0873 | 2242 | Left Center Back | Paris Saint-Germain | False | 2821.9833 | 1.0000 | 0.0000 | 0.0319 | CB | 0.2407 | 0.5000 |
| 3009 | Kylian Mbappé Lottin | 0.0440 | 3542 | Center Forward | Paris Saint-Germain | False | 5991.1833 | 58.0000 | 20.0000 | 1.1717 | FW | 0.8182 | 0.9091 |
| 3026 | Adrien Rabiot | -0.0184 | 833 | Left Defensive Midfield | France | False | 1224.6833 | 1.0000 | 1.0000 | 0.1470 | DM | 0.4231 | 0.5769 |
| 3034 | Kamil Glik | -0.2723 | 301 | Right Center Back | Poland | False | 674.1667 | 0.0000 | 0.0000 | 0.0000 | CB | 0.0185 | 0.2500 |
| 3043 | Christian Dannemann Eriksen | 0.0153 | 473 | Left Defensive Midfield | Denmark | False | 693.0500 | 1.0000 | 0.0000 | 0.1299 | DM | 0.5000 | 0.4615 |
| 3053 | Leroy Sané | 0.0109 | 399 | Right Wing | Germany | False | 672.1500 | 0.0000 | 1.0000 | 0.1339 | W | 0.4444 | 0.1111 |
| 3077 | Jan Vertonghen | -0.0365 | 785 | Left Center Back | Belgium | False | 929.5167 | 0.0000 | 0.0000 | 0.0000 | CB | 0.3889 | 0.2500 |
| 3089 | Kevin De Bruyne | 0.1375 | 578 | Center Attacking Midfield | Belgium | False | 949.2333 | 2.0000 | 2.0000 | 0.3793 | AM | 0.9091 | 0.4545 |
| 3090 | Nicolás Hernán Otamendi | -0.1022 | 654 | Left Center Back | Argentina | False | 733.9000 | 0.0000 | 1.0000 | 0.1226 | CB | 0.1296 | 0.7593 |
| 3166 | Marco Verratti | 0.0952 | 4496 | Center Defensive Midfield | Paris Saint-Germain | False | 3751.4667 | 1.0000 | 4.0000 | 0.1200 | DM | 0.8846 | 0.4231 |
| 3167 | Antonio Rüdiger | -0.0308 | 1048 | Right Center Back | Germany | False | 1177.2167 | 0.0000 | 0.0000 | 0.0000 | CB | 0.4259 | 0.2500 |
| 3193 | Bernardo Mota Veiga de Carvalho e Silva | -0.0172 | 773 | Right Center Midfield | Portugal | False | 1055.7000 | 1.0000 | 0.0000 | 0.0853 | CM | 0.3889 | 0.3889 |
| 3205 | Kyle Walker | 0.0061 | 1111 | Right Back | England | False | 1577.0500 | 0.0000 | 0.0000 | 0.0000 | FB | 0.9355 | 0.1129 |
| 3233 | Raheem Sterling | 0.0225 | 376 | Left Wing | England | False | 812.7333 | 4.0000 | 0.0000 | 0.4429 | W | 0.5000 | 0.5556 |
| 3244 | John Stones | -0.0218 | 1875 | Right Center Back | England | False | 1870.8833 | 0.0000 | 0.0000 | 0.0000 | CB | 0.5741 | 0.2500 |
| 3306 | Nathan Aké | -0.1006 | 736 | Left Center Back | Netherlands | False | 1052.4333 | 0.0000 | 2.0000 | 0.1710 | CB | 0.1481 | 0.8889 |
| 3308 | Kieran Trippier | -0.0654 | 732 | Left Back | England | False | 1173.4167 | 0.0000 | 1.0000 | 0.0767 | FB | 0.4839 | 0.2258 |
| 3311 | Daley Blind | 0.1033 | 627 | Left Center Back | Netherlands | False | 757.0833 | 1.0000 | 1.0000 | 0.2378 | CB | 0.9630 | 0.9630 |
| 3336 | Harry Maguire | 0.0351 | 973 | Left Center Back | England | False | 975.3167 | 1.0000 | 1.0000 | 0.1846 | CB | 0.8519 | 0.9074 |
| 3382 | Luke Shaw | -0.0583 | 885 | Left Back | England | False | 1232.8167 | 1.0000 | 4.0000 | 0.3650 | FB | 0.5484 | 0.8387 |
| 3436 | Idrissa Gana Gueye | -0.0361 | 1387 | Left Center Midfield | Paris Saint-Germain | False | 1694.1833 | 0.0000 | 1.0000 | 0.0531 | CM | 0.3333 | 0.2778 |
| 3471 | Dejan Lovren | -0.0155 | 668 | Right Center Back | Croatia | False | 905.4833 | 0.0000 | 1.0000 | 0.0994 | CB | 0.6481 | 0.6852 |
| 3477 | Álvaro Borja Morata Martín | -0.1361 | 356 | Center Forward | Spain | False | 1108.0000 | 7.0000 | 1.0000 | 0.6498 | FW | 0.1818 | 0.5455 |
| 3500 | Granit Xhaka | 0.0483 | 4342 | Left Defensive Midfield | Bayer Leverkusen | False | 3952.1667 | 3.0000 | 1.0000 | 0.0911 | DM | 0.7692 | 0.3846 |
| 3501 | Philippe Coutinho Correia | -0.0106 | 658 | Center Attacking Midfield | Barcelona | False | 651.3500 | 2.0000 | 2.0000 | 0.5527 | AM | 0.2727 | 0.7273 |
| 3517 | Aaron Ramsey | 0.1099 | 449 | Right Center Midfield | Wales | False | 790.3000 | 1.0000 | 0.0000 | 0.1139 | CM | 0.8333 | 0.4444 |
| 3533 | Xherdan Shaqiri | 0.1001 | 343 | Center Attacking Midfield | Switzerland | False | 752.2500 | 5.0000 | 2.0000 | 0.8375 | AM | 0.7273 | 1.0000 |
| 3567 | Georginio Wijnaldum | -0.0969 | 617 | Right Center Midfield | Paris Saint-Germain | False | 1448.0000 | 4.0000 | 1.0000 | 0.3108 | CM | 0.1111 | 0.7778 |
| 3570 | Pierre-Emile Højbjerg | 0.0432 | 1051 | Right Defensive Midfield | Denmark | False | 1254.2500 | 0.0000 | 3.0000 | 0.2153 | DM | 0.7308 | 0.6923 |
| 3575 | Oleksandr Zinchenko | -0.0889 | 478 | Left Center Midfield | Ukraine | False | 723.6833 | 1.0000 | 2.0000 | 0.3731 | CM | 0.1667 | 0.9444 |
| 3593 | Renato Júnior Luz Sanches | 0.0163 | 1061 | Left Center Midfield | Paris Saint-Germain | False | 1050.7167 | 2.0000 | 0.0000 | 0.1713 | CM | 0.5556 | 0.5556 |
| 3669 | Virgil van Dijk | -0.0312 | 771 | Left Center Back | Netherlands | False | 1082.6833 | 0.0000 | 0.0000 | 0.0000 | CB | 0.4074 | 0.2500 |
| 3943 | Declan Rice | -0.0318 | 1414 | Left Defensive Midfield | England | False | 1705.8333 | 0.0000 | 1.0000 | 0.0528 | DM | 0.2692 | 0.2692 |
| 3959 | Andreas Christensen | -0.0657 | 873 | Left Center Back | Denmark | False | 1191.7500 | 2.0000 | 0.0000 | 0.1510 | CB | 0.2963 | 0.8519 |
| 3961 | N'Golo Kanté | 0.0255 | 617 | Right Center Midfield | France | False | 938.5000 | 0.0000 | 0.0000 | 0.0000 | CM | 0.6111 | 0.1389 |
| 4320 | Neymar da Silva Santos Junior | 0.1702 | 2650 | Left Wing | Paris Saint-Germain | False | 3052.9333 | 24.0000 | 13.0000 | 1.0908 | W | 1.0000 | 1.0000 |
| 4353 | Aymeric Laporte | -0.0302 | 1991 | Left Center Back | Spain | False | 1515.1333 | 1.0000 | 0.0000 | 0.0594 | CB | 0.4444 | 0.5741 |
| 4354 | Phil Foden | -0.0180 | 475 | Left Wing | England | False | 1062.7000 | 1.0000 | 0.0000 | 0.0847 | W | 0.3333 | 0.0556 |
| 4372 | Marcos Aoás Corrêa | -0.0464 | 4272 | Right Center Back | Paris Saint-Germain | False | 4945.5333 | 5.0000 | 0.0000 | 0.0910 | CB | 0.3519 | 0.6667 |
| 4445 | Jules Koundé | -0.1281 | 822 | Right Back | France | False | 1398.2500 | 0.0000 | 0.0000 | 0.0000 | FB | 0.0968 | 0.1129 |
| 4447 | Martin Braithwaite Christensen | -0.0920 | 566 | Left Wing | Barcelona | False | 1736.3667 | 3.0000 | 2.0000 | 0.2592 | W | 0.1111 | 0.1667 |
| 4633 | Magdalena Lilly Eriksson | 0.0352 | 722 | Left Center Back | Sweden Women's | True | 1259.1000 | 0.0000 | 0.0000 | 0.0000 | CB | 0.7857 | 0.2500 |
| 4642 | Millie Bright | 0.0308 | 1063 | Right Center Back | England Women's | True | 1187.9167 | 0.0000 | 1.0000 | 0.0758 | CB | 0.7500 | 0.5714 |
| 4643 | Georgia Stanway | -0.0745 | 838 | Right Defensive Midfield | England Women's | True | 1606.8000 | 5.0000 | 0.0000 | 0.2801 | DM | 0.0625 | 0.8750 |
| 4658 | Keira Walsh | 0.0568 | 985 | Left Defensive Midfield | England Women's | True | 1675.6500 | 1.0000 | 2.0000 | 0.1611 | DM | 0.6875 | 0.8125 |
| 4659 | Ramona Bachmann | 0.0072 | 330 | Center Forward | Switzerland Women's | True | 606.5333 | 2.0000 | 1.0000 | 0.4452 | FW | 0.7500 | 0.5000 |
| 4706 | Kalvin Phillips | -0.0662 | 410 | Right Defensive Midfield | England | False | 732.9333 | 0.0000 | 2.0000 | 0.2456 | DM | 0.1154 | 0.7692 |
| 4870 | Tomáš Souček | 0.0398 | 355 | Left Center Midfield | Czech Republic | False | 760.7667 | 1.0000 | 0.0000 | 0.1183 | CM | 0.7222 | 0.5000 |
| 4979 | Katrina Gorry | 0.1212 | 385 | Right Defensive Midfield | Australia Women's | True | 663.5500 | 0.0000 | 1.0000 | 0.1356 | DM | 1.0000 | 0.6875 |
| 5000 | Stephanie-Elise Catley | 0.0013 | 322 | Left Back | Australia Women's | True | 710.9667 | 2.0000 | 0.0000 | 0.2532 | FB | 0.4706 | 0.7059 |
| 5058 | Rachel Daly | -0.0557 | 609 | Left Back | England Women's | True | 984.8000 | 1.0000 | 1.0000 | 0.1828 | FB | 0.2353 | 0.5294 |
| 5095 | Ellie Madison Carpenter | 0.0403 | 318 | Right Back | Australia Women's | True | 688.4000 | 0.0000 | 0.0000 | 0.0000 | FB | 0.6471 | 0.1471 |
| 5199 | Jorge Resurrección Merodio | -0.0123 | 669 | Right Center Midfield | Spain | False | 738.6500 | 0.0000 | 0.0000 | 0.0000 | CM | 0.4444 | 0.1389 |
| 5201 | Sergio Ramos García | 0.0562 | 2954 | Right Center Back | Paris Saint-Germain | False | 3076.3333 | 4.0000 | 1.0000 | 0.1463 | CB | 0.8704 | 0.8333 |
| 5203 | Sergio Busquets i Burgos | 0.0709 | 3433 | Center Defensive Midfield | Barcelona | False | 3390.7000 | 0.0000 | 5.0000 | 0.1327 | DM | 0.8462 | 0.5000 |
| 5204 | Bruno Miguel Borges Fernandes | 0.1424 | 645 | Center Attacking Midfield | Portugal | False | 984.9333 | 3.0000 | 3.0000 | 0.5483 | AM | 1.0000 | 0.6364 |

