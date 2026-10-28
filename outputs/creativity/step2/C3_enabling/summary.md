# Enabling: effect of a creative action on the next teammate's opportunity and space

Pairs: in-scope decision -> next decision by a different player of the same team. Strata: end cell (10 x 10 yd) x action type x value decile. estimate = change of the outcome per SD of creativity.

## enabling_effects (12 rows)

| outcome | regressor | estimate | low | high | p | n | clusters | fixed_effects | outcome_mean | outcome_sd | estimate_in_outcome_sd |
|---|---|---|---|---|---|---|---|---|---|---|---|
| next_opportunity_expected | creativity_z | 0.0045 | 0.0032 | 0.0057 | 0.0000 | 313166 | 417 | stratum | 1.4265 | 0.3794 | 0.0118 |
| next_opportunity_expected | creativity_z | 0.0038 | 0.0025 | 0.0051 | 0.0000 | 313166 | 417 | stratum+match_id | 1.4265 | 0.3794 | 0.0101 |
| next_opportunity_quadrant | creativity_z | -0.0100 | -0.0104 | -0.0097 | 0.0000 | 313166 | 417 | stratum | 0.4544 | 0.1047 | -0.0955 |
| next_opportunity_quadrant | creativity_z | -0.0099 | -0.0102 | -0.0095 | 0.0000 | 313166 | 417 | stratum+match_id | 0.4544 | 0.1047 | -0.0942 |
| next_nearest_opponent | creativity_z | -0.1971 | -0.2154 | -0.1772 | 0.0000 | 313054 | 417 | stratum | 7.9147 | 5.2349 | -0.0376 |
| next_nearest_opponent | creativity_z | -0.1898 | -0.2085 | -0.1708 | 0.0000 | 313054 | 417 | stratum+match_id | 7.9147 | 5.2349 | -0.0363 |
| next_opponents_within_5 | creativity_z | 0.0208 | 0.0180 | 0.0234 | 0.0000 | 313166 | 417 | stratum | 0.4476 | 0.7126 | 0.0292 |
| next_opponents_within_5 | creativity_z | 0.0197 | 0.0169 | 0.0223 | 0.0000 | 313166 | 417 | stratum+match_id | 0.4476 | 0.7126 | 0.0277 |
| next_raw_good_surprise | creativity_z | 0.0051 | -0.0042 | 0.0134 | 0.2780 | 281498 | 417 | stratum | 1.3501 | 2.1019 | 0.0024 |
| next_raw_good_surprise | creativity_z | 0.0014 | -0.0079 | 0.0098 | 0.8160 | 281498 | 417 | stratum+match_id | 1.3501 | 2.1019 | 0.0006 |
| next_v | creativity_z | 0.0013 | 0.0012 | 0.0014 | 0.0000 | 313166 | 417 | stratum | 0.0106 | 0.0258 | 0.0494 |
| next_v | creativity_z | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 313166 | 417 | stratum+match_id | 0.0106 | 0.0258 | 0.0489 |


## enabling_players (330 rows, first 60 shown)

| player_id | player | enabling | actions | creativity | team | position |
|---|---|---|---|---|---|---|
| 10650 | Jill Roord | 0.0963 | 347 | 0.9148 | Netherlands Women's | Right Center Midfield |
| 10167 | Tatiana Vanessa Ferreira Pinto | 0.0805 | 221 | 1.4681 | Portugal Women's | Left Center Midfield |
| 62710 | Johanna Rytting-Kaneryd | 0.0774 | 254 | 1.2329 | Sweden Women's | Right Wing |
| 10658 | Danielle van de Donk | 0.0763 | 271 | 1.1932 | Netherlands Women's | Left Center Midfield |
| 131586 | Kyra Lillee Cooney-Cross | 0.0739 | 222 | 1.3399 | Australia Women's | Left Defensive Midfield |
| 5058 | Rachel Daly | 0.0734 | 412 | 1.1657 | England Women's | Left Back |
| 5095 | Ellie Madison Carpenter | 0.0689 | 201 | 1.4252 | Australia Women's | Right Back |
| 13865 | Natalia Kuikka | 0.0684 | 220 | 1.6821 | WNT Finland | Right Center Back |
| 62962 | Esmee Brugts | 0.0679 | 227 | 1.3568 | Netherlands Women's | Left Wing Back |
| 10221 | Elin Ingrid Johanna Rubensson | 0.0660 | 204 | 1.1351 | Sweden Women's | Left Defensive Midfield |
| 25770 | Lisa Boattin | 0.0655 | 296 | 1.4920 | Italy Women's | Left Back |
| 24374 | Ingrid Syrstad Engen | 0.0639 | 277 | 1.4819 | Norway Women's | Left Defensive Midfield |
| 15789 | Rikke Læntver Sevecke | 0.0638 | 224 | 1.3022 | Denmark Women's | Right Center Back |
| 5000 | Stephanie-Elise Catley | 0.0604 | 210 | 1.3961 | Australia Women's | Left Back |
| 10370 | Cecilia Salvai | 0.0599 | 254 | 1.5048 | Italy Women's | Center Back |
| 10088 | Katrine Veje | 0.0588 | 378 | 1.1555 | Denmark Women's | Left Back |
| 15555 | Lauren Hemp | 0.0576 | 309 | 1.1265 | England Women's | Left Wing |
| 10215 | Fridolina Rolfö | 0.0536 | 289 | 1.0964 | Sweden Women's | Left Wing |
| 31534 | Ella Toone | 0.0527 | 339 | 1.1173 | England Women's | Center Attacking Midfield |
| 10649 | Lieke Martens | 0.0526 | 229 | 1.3039 | Netherlands Women's | Left Center Forward |
| 46987 | Jule Brand | 0.0510 | 208 | 1.4789 | Germany Women's | Right Wing |
| 62666 | Ingrid Filippa Angeldal | 0.0505 | 417 | 1.3874 | Sweden Women's | Right Defensive Midfield |
| 31531 | Lauren James | 0.0495 | 265 | 1.3270 | England Women's | Right Wing |
| 10141 | Ana Borges | 0.0492 | 210 | 1.8746 | Portugal Women's | Right Back |
| 25632 | Yui Hasegawa | 0.0492 | 200 | 1.4395 | Japan Women's | Right Defensive Midfield |
| 10374 | Svenja Huth | 0.0487 | 274 | 1.3123 | Germany Women's | Right Back |
| 10386 | Caroline Graham Hansen | 0.0465 | 211 | 1.1877 | Norway Women's | Center Attacking Midfield |
| 4979 | Katrina Gorry | 0.0444 | 228 | 1.5184 | Australia Women's | Right Defensive Midfield |
| 25478 | Lena Oberdorf | 0.0442 | 210 | 1.4386 | Germany Women's | Right Defensive Midfield |
| 4870 | Tomáš Souček | 0.0440 | 218 | 1.4622 | Czech Republic | Left Center Midfield |
| 63788 | Victoria Pelova | 0.0431 | 283 | 1.1538 | Netherlands Women's | Right Wing Back |
| 8518 | Marcel Sabitzer | 0.0431 | 286 | 1.3259 | Austria | Center Attacking Midfield |
| 5562 | Thomas Müller | 0.0424 | 207 | 1.1007 | Germany | Center Attacking Midfield |
| 4658 | Keira Walsh | 0.0422 | 761 | 1.4700 | England Women's | Left Defensive Midfield |
| 10137 | Kadidiatou Diani | 0.0414 | 243 | 1.2830 | France Women's | Right Wing |
| 46885 | Rebecca Knaak | 0.0402 | 219 | 1.2734 | Germany Women's | Left Center Back |
| 10220 | Kosovare Asllani | 0.0394 | 256 | 1.2920 | Sweden Women's | Center Attacking Midfield |
| 10117 | Sakina Karchaoui | 0.0389 | 531 | 1.3419 | France Women's | Left Back |
| 4643 | Georgia Stanway | 0.0389 | 561 | 1.0960 | England Women's | Right Defensive Midfield |
| 6798 | Stanislav Lobotka | 0.0385 | 233 | 1.2631 | Slovakia | Center Defensive Midfield |
| 25729 | Klara Bühl | 0.0374 | 256 | 1.0491 | Germany Women's | Left Wing |
| 191850 | Eveliina Summanen | 0.0373 | 203 | 1.3972 | WNT Finland | Left Defensive Midfield |
| 5544 | Ricardo Iván Rodríguez Araya | 0.0362 | 495 | 1.3416 | Switzerland | Left Center Back |
| 16554 | Joakim Mæhle | 0.0342 | 451 | 1.1031 | Denmark | Left Wing Back |
| 5207 | Cristiano Ronaldo dos Santos Aveiro | 0.0337 | 260 | 0.7653 | Portugal | Center Forward |
| 10228 | Manuela Giugliano | 0.0336 | 251 | 1.3477 | Italy Women's | Center Defensive Midfield |
| 31900 | Oleksandr Karavaev | 0.0335 | 205 | 1.5761 | Ukraine | Right Back |
| 11396 | Florian Grillitsch | 0.0335 | 292 | 1.3515 | Austria | Left Defensive Midfield |
| 6983 | Remo Freuler | 0.0333 | 500 | 1.0485 | Switzerland | Right Defensive Midfield |
| 3176 | Thomas Meunier | 0.0333 | 222 | 1.3806 | Belgium | Right Wing Back |
| 46881 | Janina Minge | 0.0321 | 225 | 1.1822 | Germany Women's | Right Center Back |
| 5632 | Thorgan Hazard | 0.0317 | 231 | 1.3235 | Belgium | Left Wing Back |
| 10336 | Alejandro Grimaldo García | 0.0313 | 1294 | 1.0477 | Bayer Leverkusen | Left Wing Back |
| 27886 | Alexis Mac Allister | 0.0310 | 232 | 1.1528 | Argentina | Left Center Midfield |
| 30971 | Sandie Toletti | 0.0303 | 354 | 1.2112 | France Women's | Left Defensive Midfield |
| 276906 | Sari Kees | 0.0303 | 257 | 1.3121 | Belgium Women's | Right Center Back |
| 21294 | Mykola Shaparenko | 0.0300 | 287 | 1.2724 | Ukraine | Right Center Midfield |
| 32143 | Diana Catarina Ribeiro Gomes | 0.0295 | 220 | 1.4339 | Portugal Women's | Right Center Back |
| 10222 | Jonna Ann-Charlotte Andersson | 0.0293 | 309 | 1.2897 | Sweden Women's | Left Back |
| 29163 | Josip Juranović | 0.0293 | 388 | 1.1614 | Croatia | Right Back |


## enabling_summary (1 rows)

| pairs | players | spearman_enabling_vs_own_creativity |
|---|---|---|
| 313166 | 330 | 0.0761 |

