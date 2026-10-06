# Measuring creativity in football decisions: preliminary results

Status: 6 October 2026. Data: StatsBomb open data with 360 freeze frames. Every number below is
out-of-sample: each decision is scored by models that never saw its match.

## Key results

- **Models.**
  - A next-action model (policy π) predicts the type, end zone and height of every on-ball decision
    over 1,771 possible actions (joint NLL 4.34 vs. 5.55 for a zone-frequency baseline).
  - An action-value model (Q) estimates each action's value. On the un-xPass protocol it matches
    VAEP and xT (AUC 0.77–0.78 vs. 0.780/0.782).
- **Primary measure.** `good_surprise` is the surprise of the chosen action among the actions with
  positive advantage (Q > V).
  - Known creative players rank higher than other players in the same position (AUC 0.650
    [0.566, 0.731]).
  - It beats value alone (+0.097 [0.020, 0.179]) and is nearly position-neutral (η² = 0.09).
- **Component checks.**
  - The policy is calibrated down to probabilities of 10⁻⁵, and rarity is uniform across all
    competitions and both genders.
  - Values are not inflated for rare actions on average.
  - Through balls score as creative (AUC 0.82), and creativity correlates with opponents bypassed
    (ρ = 0.28).
- **Yardstick.** Cross-fitted models learn the styles of the single-team club seasons, which
  lowers those players' scores (35th vs. 53rd percentile). A reference model that never saw these
  clubs fixes this. On it, Messi is his team's most creative player in all five of his contexts.
- **Downstream analyses.** Eight of 22 main hypotheses survive the Benjamini–Hochberg correction:
  - creative build-up predicts shots and goals;
  - originality pays beyond the predicted value;
  - teams become less creative after scoring;
  - red cards lower creativity; tactical shifts and substitutions raise it;
  - substitutes are less creative than starters late in matches.

  Team creativity does not predict match outcomes.
- **Open.**
  - The blinded expert rating is prepared (62 clips) but not yet done.
  - No external player ratings (e.g. EA FC) have been used yet.
  - The known-players list is reputation-based and was written by an AI assistant (pre-registered by commit).

---

## 1 Methodology

### 1.1 Data

StatsBomb open data, events plus 360 freeze frames: 426 matches, of which 418 are usable (the rest
have corrupt or unlinked 360 files). In total there are 536,223 decisions:

| Decision type | Count |
|---|---|
| Passes | 372,802 |
| Carries of at least 4.6 yd (one grid cell) | 126,803 |
| Clearances | 15,061 |
| Dribbles | 11,194 |
| Shots | 10,363 |

| Competition | Matches | Decisions | Note |
|---|---|---|---|
| FIFA World Cup 2022 | 64 | 82,307 | 7 Argentina matches |
| Women's World Cup 2023 | 64 | 75,411 | |
| UEFA Euro 2020 / 2024 | 51 / 51 | 66,329 / 65,796 | |
| UEFA Women's Euro 2022 / 2025 | 31 / 31 | 36,341 / 34,964 | |
| La Liga 2020/21 | 35 | 52,355 | Barcelona matches only |
| Ligue 1 2021/22 / 2022/23 | 26 / 32 | 34,061 / 42,343 | PSG matches only |
| 1. Bundesliga 2023/24 | 34 | 42,637 | Leverkusen matches only |
| MLS 2023 | 6 | 3,678 | Inter Miami matches only |
| AFCON 2023 | 1 | <50 | |

There are 389,507 men's and 146,716 women's decisions. Messi plays in 106 matches. The club seasons
are single-team releases, which drives the yardstick design in §1.4.

### 1.2 Next-action model (policy π)

- **Decision:** the moment a player chooses an on-ball action.
  - Possession spells are anchored at the gain event (receipt, recovery, interception, duel).
  - After a carry, another carry is masked out.
  - The target's type, height and body part are never inputs. Opponent events at the same
    timestamp are excluded from the history.
- **Output:** factorized into type (5 classes), then end zone (26×17 grid, for passes and carries),
  then pass height (ground, low, high). Together this gives a flat space of 1,771 actions.
- **Inputs:**
  - the 360 freeze frame: positions; teammate, actor and keeper flags; the visible area;
  - velocities, from a gated Hungarian matching to the frame 0.2–3 s earlier (at most 11 yd/s);
  - the event history: up to 16 events of the possession, encoded by a GRU;
  - the situation: restart type, pressure, minute, goal difference, home/away, women's competition.
- **Architecture:** a GATv2 graph network with reflection-equivariant group convolution (y-flip)
  and frame averaging, so zone predictions mirror with the pitch; a sequence-to-spatial encoder;
  2.0 M parameters.

| Model | Train / val matches | Joint NLL | Type acc. at spell start | Pass-zone NLL | Pass top-1 / top-5 | Carry-zone NLL | Height NLL |
|---|---|---|---|---|---|---|---|
| Full data | 368 / 49 | **4.337** | 70.0% | 4.016 | 10.7% / 35.2% | 3.446 | 0.365 |
| Cross-fit fold 0 | 210 / 49 | 4.479 | 69.5% | 4.167 | 8.5% / 30.4% | 3.503 | 0.388 |
| Cross-fit fold 1 | 208 / 49 | 4.499 | 68.8% | 4.204 | 7.6% / 28.7% | 3.487 | 0.394 |
| Reference model (tournament validation) | 256 / 29 | 4.539 | 68.7% | 4.266 | 7.7% / 28.4% | 3.490 | 0.401 |

On the full-data validation set, the zone-frequency baseline has joint NLL 5.55 and the marginal
baseline 6.85. Type accuracy after a carry is 96–97%; spell starts (65% of decisions) are the hard
case.

### 1.3 Action-value model (Q)

- **Head:** decomposed, Q = P(success)·V_success + (1 − P)·V_failure, with score and concede
  components (VAEP-style). Net value = score − concede.
- **Target:** Monte Carlo, 1 − Π(1 − xG) over the team's (or opponent's) shots within the next 10
  SPADL actions.
- **Training:** warm start from the policy's encoder (encoder learning rate ×0.1).
- **Distribution head (C51):** an auxiliary categorical distribution of the net discounted return
  (γ = 0.9 per decision). It uses 51 atoms on a symmetric support in [−1, 1] that is dense near 0,
  a distributional TD target with a Polyak-averaged target network, and loss weight 0.1.

| Value model | Protocol AUC | Protocol log-loss | Return loss |
|---|---|---|---|
| MC (full data) | 0.774 | 0.0477 | 0.1054 |
| Expected SARSA + C51 (full data) | 0.780 | 0.0497 | 0.1031 |
| λ-return, λ = 0.8 (full data) | 0.779 | 0.0495 | **0.1027** |
| VAEP / VAEP-xG / xT (same 35,860 passes) | 0.780 / 0.799 / 0.782 | 0.0474 / 0.0467 / 0.0486 | – |
| MC + C51, cross-fit folds 0 / 1 (own monitoring matches) | 0.781 / 0.769 | 0.0478 / 0.0541 | 0.1036 / 0.1102 |

The protocol is un-xPass Table 3: completed open-play passes, label = goal within 10 actions.
C51 calibration (CRPS) is 0.0164 / 0.0176 for the two folds; a trivial predictor scores 0.0191.

### 1.4 Out-of-sample scoring and yardsticks

- **Cross-fitting:** two folds split by match. Each fold's models (policy 16 epochs, value 3
  epochs) score the other fold, giving 536,223 out-of-sample decisions in 418 matches. Used for
  validation and for within-team and within-match analyses.
- **Reference model:** trained without the four club seasons and Argentina's World Cup 2022 matches
  (256 training and 29 validation matches; policy 10 epochs, value 3 epochs). It scores the 133
  held-out matches (184,243 decisions). Used for comparisons across teams and contexts, such as
  Messi at Barcelona, PSG, Argentina and Inter Miami, or Leverkusen over its season.

### 1.5 Creativity measures

Notation, for a decision in situation *s* with chosen action *a*:

| Term | Definition |
|---|---|
| π(a) | Probability of action *a* under the policy |
| Surprise | −log π(a) |
| Rarity | π(actions more likely than *a*) + π(a)/2; ranges 0–1, uniform under a calibrated policy |
| V(s) | Σ π·Q |
| Advantage A | Q(a) − V(s) |

A **set measure** restricts the options to a set S, renormalizes π on S, and scores the chosen
action's rarity or surprise within S. It is 0 if *a* is not in S.

| Role | Measure | Definition |
|---|---|---|
| **Primary** | `good_surprise` | log π(S) − log π(a), with S = {a′ : A(a′) > 0} |
| Robustness | `good_rarity` | Rarity within the same S |
| Robustness | `c51_tail_rarity` | Rarity within S = actions whose 90% return quantile exceeds that of the situation's return distribution |
| Robustness | `unxpass` | Passes only: rarity among passes if A > 0 (un-xPass definition) |
| Baseline | `rarity` / `advantage` / `difficulty` | Originality only / value only / 1 − P(success) of completed actions |
| Not recommended | `rarity_x_adv`, `surprise_x_adv`, `rarity_x_adv_completed`, `rarity_x_upside`, `rarity_x_td`, `tail_rarity` | Products of originality and value; rarity within actions whose success outcome beats V (see §2.5 for why) |

- **Scope ("intended targets"):** incomplete and deflected passes are dropped, because their end
  point is where the ball was intercepted, not where it was aimed. This leaves 469,733 decisions.
- **Aggregation:**
  - Player score: the mean over in-scope decisions, for players with at least 200 (539 players).
  - Downstream analyses use z-scores at decision level (SD units).

### 1.6 Main design choices

| Choice | Adopted | Alternatives tested | Reason |
|---|---|---|---|
| Prediction target | Type → zone → height (1,771 actions) | Receiver; pass zone only | Covers all decision types; early pass-target models were beaten by a closest-teammate heuristic |
| Leakage control | Anchor at gain events; carry mask; no target attributes as inputs | Naive event alignment | StatsBomb carry conventions had made type accuracy trivially 95% |
| Value target | Monte Carlo, decomposed head | SARSA, expected SARSA, IQL, λ-return, quantile, direct head, joint training | MC is stable; expected SARSA drifted after epoch 1; λ = 0.8 has the best return loss but worse protocol log-loss |
| Originality | Rarity and surprise within an option set | Raw surprise | Raw surprise rises with the number of options |
| Combining originality and value | Rarity among valuable options | Products | Products are mostly position (η² = 0.70) |
| Scope | Intended targets | All actions | Failed passes have inflated rarity (0.66 vs. 0.51) and made miscommunications look creative |
| Camera view | All targets | In view only | Out-of-view completed passes carry signal (§2.4) |
| Calibration | None (T = 1) | Temperature scaling | Fitted T ≈ 1.025; NLL gain 0.001; tail worse |
| Yardstick | Cross-fit, plus reference model for contexts | One model for all | Cross-fit absorbs club styles (§2.6) |
| Validation list | Known players committed before any ranking existed | – | Pre-registration |

### 1.7 Statistics

- **Intervals:** 95% match-clustered bootstrap, 1,000 draws (2,000 for known-groups). p-values come
  from the bootstrap distribution.
- **Multiple testing:** Benjamini–Hochberg over the 22 main downstream hypotheses.
- **Player-level validity:** comparisons within 8 position groups (GK, CB, FB, DM, CM, AM, W, FW),
  as AUCs of within-group percentiles. Measures are compared by paired bootstrap.
- **Fixed effects:** strata or teams are demeaned before slopes are estimated.
- **Event studies:**
  - Compare 10-minute windows before and after an event, within the same period.
  - Subtract the same change in 20 placebo windows from other matches: same period, clock within
    ±3 minutes, no such event within 10 minutes.
  - Require at least 5 decisions per window.
- **Interpretation caveat:** the models condition on goal difference and minute. Game-state effects
  are therefore relative to what is typical in that state.

---

## 2 Validation of the creativity measures

Unless stated otherwise: cross-fitted MC + C51 tables, intended scope.

### 2.1 Model components

**Policy calibration (observed / expected choices per probability band):**

| log₁₀ π band | ≤ −6 | −6…−5 | −5…−4 | −4…−3 | −3…−2.5 | −2.5…−2 | −2…−1.5 | −1.5…−1 | −1…−0.5 | > −0.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| All | 0.85 | **0.72** | 0.99 | 1.12 | 1.07 | 1.00 | 0.96 | 0.99 | 0.97 | 0.99 |
| Men / women | 0.80 / 1.02 | 0.76 / 0.61 | 1.02 / 0.94 | 1.12 / 1.12 | 1.06 / 1.07 | 1.00 / 1.01 | 0.96 / 0.96 | 0.99 / 0.98 | 0.97 / 0.95 | 0.99 / 0.99 |

The policy is calibrated to within ±12% for all probabilities above 10⁻⁵. The weak band 10⁻⁶–10⁻⁵
holds about 0.3% of the expected choices, and those actions have rarity close to 1 regardless.

**Rarity uniformity:** mean 0.511 and KS distance 0.015 overall.

| Group | Mean rarity | KS distance |
|---|---|---|
| Men / women | 0.510 / 0.512 | – |
| Each competition-season | 0.501–0.518 (MLS: 0.524) | 0.006–0.027 (MLS: 0.043, 3,678 decisions) |
| Each decision context | 0.499–0.514 | – |

By action type, rarity averages 0.54 for passes and 0.31 for shots and dribbles. That is not a
calibration failure, since it conditions on the outcome, but it means the measures favour passes.

**Temperature scaling:** no gain, so T = 1 is kept.
- Fold 0 (fitted on 49 matches, tested on 80): T = 1.025 for all heads; NLL 4.5058 → 4.5047; KS
  0.020 → 0.008; the 10⁻⁶–10⁻⁵ band gets worse (0.89 → 0.74).
- Reference model (fitted on tournaments, tested on 80 club matches): T ≈ 1; NLL 4.5314 → 4.5310.

**Value calibration by rarity decile:** predicted / realized xG label is 0.93–1.11, and the bias
(Q − realized return) is −0.0022 to −0.0007, with no trend toward rare actions.

**Winner's curse (top 1% of each measure):**

| Measure | Predicted A | Realized A, one-step | Realized A, Monte Carlo |
|---|---|---|---|
| `advantage` | 0.061 | 0.041 | 0.053 |
| `rarity_x_adv` | 0.056 | 0.037 | 0.046 |
| `good_surprise` | 0.009 | 0.010 | 0.013 |
| `c51_tail_rarity` | 0.012 | 0.020 | 0.023 |

Value-ranked tops are inflated by 15–35%. The set measures are not inflated.

### 2.2 Known creative players (known-groups validity)

The list has 48 players (30 men, 18 women), compiled from reputation by an AI assistant at the
author's request. It was committed (git) at 19:36 on 5 October 2026; the first rankings existed at
20:15. 42 players have at least 200 in-scope decisions, out of 539 in total. The AUC is computed on
percentiles within position groups.

| Measure | AUC [95% CI] | Δ vs. `advantage` | Δ vs. `rarity` | Known players in their position's top 10% |
|---|---|---|---|---|
| **`good_surprise`** | **0.650 [0.566, 0.731]** | **+0.097 [+0.020, +0.179]** | +0.063 [−0.002, +0.126] | 19% |
| `good_rarity` | 0.645 [0.562, 0.726] | **+0.091 [+0.026, +0.158]** | +0.058 [−0.017, +0.129] | 21% |
| `unxpass` | 0.645 [0.557, 0.728] | +0.090 [−0.001, +0.182] | +0.057 [−0.013, +0.126] | 24% |
| `c51_tail_rarity` | 0.634 [0.539, 0.727] | +0.079 [−0.000, +0.157] | +0.046 [−0.060, +0.141] | 26% |
| `rarity_x_td` | 0.599 [0.512, 0.684] | +0.044 | +0.011 | 17% |
| `tail_rarity` | 0.598 [0.512, 0.681] | +0.045 | +0.011 | 12% |
| `rarity_x_adv_completed` | 0.589 [0.499, 0.676] | +0.034 | +0.001 | 14% |
| `rarity` (baseline) | 0.588 [0.499, 0.680] | +0.033 | – | 10% |
| `surprise_x_adv` / `rarity_x_adv` | 0.583 / 0.582 | +0.029 / +0.028 | ≈ 0 | 14% / 12% |
| `rarity_x_upside` | 0.564 [0.473, 0.653] | +0.010 | −0.024 | 12% |
| `advantage` (baseline) | 0.554 [0.470, 0.640] | – | −0.033 | 10% |
| `difficulty` (baseline) | 0.478 [0.389, 0.571] | −0.076 | −0.110 | 12% |

**Sensitivity to scope and camera view** (known-groups AUC; Δ vs. `advantage` in parentheses):

| Measure | All actions | Intended (default) | Targets in view | Targets out of view (31 players) |
|---|---|---|---|---|
| `good_surprise` | 0.644 (+0.043) | 0.650 (+0.097) | 0.644 (+0.048) | 0.641 (+0.113) |
| `good_rarity` | 0.644 (+0.043) | 0.645 (+0.091) | 0.638 (+0.041) | 0.640 (+0.112) |
| `c51_tail_rarity` | 0.655 (+0.053) | 0.634 (+0.079) | 0.650 (+0.053) | 0.709 (+0.180) |
| `unxpass` | 0.615 (+0.013) | 0.645 (+0.090) | 0.655 (+0.059) | 0.677 (+0.149) |
| `advantage` | 0.602 | 0.554 | 0.598 | 0.530 |

**Reference model, held-out club and Argentina matches:** 8 known players, so this is indicative
only.

| Measure | AUC [95% CI] |
|---|---|
| `unxpass` | 0.808 [0.680, 0.926] |
| `rarity` | 0.798 [0.670, 0.909] |
| `good_rarity` | 0.774 [0.593, 0.930] |
| `good_surprise` | 0.768 [0.600, 0.918] |
| `advantage` | 0.657 [0.515, 0.799] |

### 2.3 Convergent evidence: StatsBomb annotations and line-breaking

These annotations are never model inputs for the decision itself. The table gives the AUC within
each action type; values above 0.5 mean annotated actions score higher.

| Annotation (n) | Expected | `good_surprise` | `good_rarity` | `c51_tail_rarity` | `unxpass` | `rarity` | `advantage` |
|---|---|---|---|---|---|---|---|
| Through ball (702) | + | 0.817 | 0.824 | 0.803 | 0.812 | 0.829 | 0.804 |
| Flair shot: backheel, lob, overhead (176) | + | 0.495 | 0.623 | 0.591 | – | 0.629 | 0.662 |
| Dummy, "no touch" (315) | + | 0.513 | 0.511 | 0.505 | 0.508 | 0.510 | 0.462 |
| Nutmeg, dribbles (823) | + | 0.487 | 0.488 | 0.487 | – | 0.481 | 0.485 |
| Goal kick (1,059) | − | 0.490 | 0.485 | 0.452 | 0.490 | 0.417 | 0.559 |
| Throw-in (7,018) | − | 0.495 | 0.494 | 0.477 | 0.497 | 0.438 | 0.515 |
| Switch of play (5,677) | ? | 0.638 | 0.639 | 0.614 | 0.640 | 0.865 | 0.577 |
| Cut-back (362) | ? | 0.579 | 0.603 | 0.596 | 0.581 | 0.568 | 0.530 |

- **Nutmegs and dummies are uninformative.** Dribbles have no target zone, and technique is not
  modelled.
- **Why the intended scope matters:** with all actions, miscommunications (0.58–0.61) and
  deflections (about 0.55) scored as creative.
- **Line-breaking:** Spearman ρ with opponents bypassed on forward passes is 0.28 for
  `good_surprise`, `good_rarity`, `c51_tail_rarity` and `unxpass`, 0.26 for rarity and 0.21 for
  advantage. Top-1% decisions bypass 4.4–5.3 opponents, against 2.8 on average.

### 2.4 Artefacts

Overall, 17% of decisions end outside the 360 camera view, 3.4% are restarts and 3.8% are
goalkeeper decisions. Among each measure's top 1%:

| Measure | Out of view | Restart | Goalkeeper |
|---|---|---|---|
| `good_surprise` | 53% | 1.6% | 3.4% |
| `good_rarity` | 47% | 1.9% | 2.0% |
| `c51_tail_rarity` | 44% | 2.5% | 1.6% |
| `rarity_x_adv` | 36% | 1.4% | 0.3% |
| `advantage` | 24% | 1.3% | 0.2% |

- **Out-of-view targets** are completed passes to a teammate who was off camera (long switches,
  balls in behind). Restricting to targets in view does not improve validity, and the out-of-view
  subset discriminates known players best (§2.2). Reported as a robustness split, not removed.
- **Failed passes:** mean rarity 0.66 vs. 0.51 for completed passes, so the intended scope drops them.

### 2.5 Discriminant validity, reliability, robustness

| Measure | η² position | η² team | ρ with final-third share | Split-half reliability (Spearman–Brown) | Stability across tournaments (mean ρ, 12 pairs) | Player agreement, C51 vs. MC value model | Player agreement, reference vs. cross-fit |
|---|---|---|---|---|---|---|---|
| `good_surprise` | **0.09** | 0.21 | 0.14 | 0.58 | 0.37 | 0.94 | 0.71 |
| `good_rarity` | 0.29 | 0.15 | 0.38 | 0.64 | 0.39 | 0.93 | 0.69 |
| `c51_tail_rarity` | 0.50 | 0.08 | 0.61 | 0.76 | **0.62** | – | – |
| `unxpass` | **0.05** | 0.24 | −0.18 | 0.56 | 0.24 | 0.93 | 0.73 |
| `rarity_x_adv` | 0.71 | 0.08 | 0.86 | 0.89 | 0.76 | 0.99 | 0.90 |
| `rarity` | 0.18 | 0.21 | 0.20 | 0.58 | 0.36 | 1.00 | 0.93 |
| `advantage` | 0.28 | 0.10 | 0.26 | 0.54 | 0.40 | 0.97 | 0.74 |

- **The products are mostly position.** They are reliable and stable because position is.
- **The set measures are nearly position-neutral** but less reliable: about 0.6 at 200 decisions.
- **Model dependence:**
  - Player scores are robust to the value model (ρ = 0.93–0.99).
  - Individual top actions are model-dependent: action-level top-1% overlap is 0.79–0.86 between
    value models, and 0.27–0.45 between independently trained model pairs.
  - The yardstick matters: reference vs. cross-fit agreement is 0.69–0.73 for the set measures.

### 2.6 Yardstick and measurement invariance

- **Club bias under cross-fitting** (all-actions MC tables): players with mostly club data sit at
  the **35th** percentile of their position on `good_surprise` (85 players), against **53rd** for
  tournament players (555).
  - By team: Barcelona 26th, Leverkusen 33rd, PSG 35th.
  - Cause: each fold model trained on the other half of the same club's matches.
  - Stars still rank high because they deviate from their own team's patterns.
- **Reference model transfer:** it never saw club football and 45% of its training matches are
  women's, yet mean rarity is 0.489–0.506 per held-out competition (KS ≤ 0.017). Its 10⁻⁶–10⁻⁵ band
  ratio is 0.57.
- **Gender:** among tournament players, women average 6 percentile points higher on the
  value-containing measures (`good_surprise` 56.5th vs. 50.7th), but lower on rarity (50.4th vs.
  54.7th). Rankings are therefore computed within gender.

### 2.7 Face validity (within gender and position, `good_surprise`, players with ≥ 200 decisions)

| Group | Top | Bottom |
|---|---|---|
| Men, central midfield | Modrić, Zieliński, Kovačić, Ramsey, Shaparenko | Wijnaldum, Moriba |
| Men, attacking midfield | Baumgartner, Bruno Fernandes, De Bruyne, Wirtz, Shaqiri | Müller, Hložek, Adli |
| Men, wingers | Malinovskiy, Leão, Doku, Neymar, Ziyech | – |
| Men, forwards | Messi, Kane, Güler, Mbappé, Embolo | Ronaldo, Morata, Schick |
| Men, centre-backs | Saïss, Fadlalla, Bonucci, Schär, Alaba | Glik, Demiral, Araújo |
| Women, central midfield | Clara Mateo, Horan, Andreia Norton, Hermoso, Bonmatí | – |
| Women, attacking midfield | Kika Nazareth, Harder, Lauren James | – |
| Women, forwards | – | Esther González, Blackstenius, Russo |

- **Plausible:** centre-backs (ball-playing vs. stoppers), central and attacking midfielders,
  wingers and forwards.
- **Weak:** full-backs (the top are attacking wing-backs) and defensive midfielders (ball-winners
  such as Oberdorf and Casemiro rank high, probably because quick passes right after recoveries are
  rare and valuable).

**Reference yardstick: players compared with their teammates in the same matches**
(`good_surprise`, raw units = nats):

| Player, context | Player − teammates [95% CI] | Rank in team |
|---|---|---|
| Messi, Barcelona 20/21 (35 matches) | +0.49 [0.41, 0.58] | 1st of 24 |
| Messi, PSG 21/22 (26) | +0.51 [0.37, 0.64] | 1st of 21 |
| Messi, PSG 22/23 (31) | +0.51 [0.41, 0.61] | 1st of 20 |
| Messi, Argentina WC 2022 (7) | +0.75 [0.47, 1.05] | 1st of 14 |
| Messi, Inter Miami 2023 (3) | +0.84 [0.52, 0.97] | 1st of 9 |
| Wirtz, Leverkusen 23/24 | +0.33 [0.22, 0.42] | 1st |
| Neymar, PSG 21/22 | +0.41 [0.29, 0.53] | 1st |
| Pedri, Barcelona 20/21 | +0.10 [0.02, 0.19] (cross-fit: −0.04) | 71st percentile |
| Grimaldo, Leverkusen 23/24 | −0.28 [−0.38, −0.18] | 11th percentile |

**Teams vs. their opponents in the same matches (reference yardstick):**

| Team | Difference [95% CI] |
|---|---|
| Leverkusen 23/24 | +0.056 [0.009, 0.103] |
| Barcelona 20/21 | −0.099 [−0.152, −0.049] |
| PSG 21/22 | −0.023 (n.s.) |
| PSG 22/23 | +0.013 (n.s.) |
| Argentina 2022 | +0.023 (n.s.) |

**Season trends:**
- Leverkusen: none (p = 0.62; relative to opponents, p = 0.78).
- Barcelona: a decline (p = 0.017; second minus first half −0.088 [−0.150, −0.032]), but not
  relative to opponents (p = 0.10).

### 2.8 Expert review (prepared, not yet done)

The blinded rating kit has 62 clips:

| Clips | Selection |
|---|---|
| 3 per measure | Drawn from that measure's top 1% |
| 10 | Matched on value, low originality |
| 8 | Matched on originality, value ≤ 0 |
| 8 | Random decisions |

- **Rating:** the rater scores creativity, originality and effectiveness (1–5) from a 2D scene with
  player and match hidden, plus data-issue flags. An optional video rating follows after revealing
  the match.
- **Planned analysis:** Spearman and AUC of each measure against the ratings, paired contrasts
  (originality and value components), the issue taxonomy, and 2D vs. video agreement.
- **Status:** one rater only. This is a first check, not a confirmatory study.

### 2.9 Assessment

- **Supported:**
  - Policy calibration and invariance across competitions, genders and contexts.
  - No average inflation of value for rare actions.
  - Through-ball and line-breaking convergence.
  - Known-groups validity (AUC about 0.65), with added value over value alone for
    `good_surprise` and `good_rarity`.
  - Robustness to the value model.
  - Face validity in most outfield positions.
- **Not yet supported:**
  - Agreement with human judgement (expert ratings pending).
  - Independence of the known-player list (reputation-based, written by an AI).
  - Added value over originality alone (borderline: +0.063 [−0.002, +0.126]).
  - Reliability of the set measures (about 0.6).
  - Validity for dribbles and technique.
  - Full-backs and defensive midfielders.
- **Recommendation:**
  - Primary measure: `good_surprise` on intended targets.
  - Robustness: `good_rarity` and `c51_tail_rarity` (most stable across tournaments, but
    position-dependent).
  - Comparisons across contexts on the reference yardstick only.

---

## 3 Downstream analyses

- **Design:** cross-fitted tables, primary measure `good_surprise` (z-scored, decision-level SD).
- **Scale:** 469,733 in-scope decisions, 1,217 goals, 3,619 substitutions, 53 red cards, 1,404
  tactical shifts, 58,890 possessions, 836 team-matches.
- **Outputs:** everything is stored per analysis (`outputs/creativity/second_order/`).

### 3.1 Main hypotheses (BH-corrected over 22)

| # | Hypothesis | Estimate [95% CI] | q | n |
|---|---|---|---|---|
| H01 | Realized minus predicted value rises with rarity (within Q deciles) | **+0.0016 [0.0014, 0.0018]** per unit rarity | <0.001 | 469,733 decisions |
| H02 | Build-up creativity → possession ends in a shot (same start value, zone, play pattern, length) | **+3.9 pp per SD [3.6, 4.2]** (base rate 14.4%) | <0.001 | 58,890 possessions |
| H03 | … → possession ends in a goal | **+0.50 pp per SD [0.39, 0.61]** (base rate 1.7%) | <0.001 | 58,890 |
| H04 | Creative flow: creative decisions follow creative ones beyond the possession level | +0.001 [−0.002, 0.004] (correlation) | 0.63 | 378,110 pairs |
| H05 | Team creativity at level score → match xG difference (team fixed effects) | −0.014 [−0.117, 0.106] | 0.86 | 761 team-matches |
| H06 | Concentration of team creativity (Gini) → xG difference | −0.033 [−0.135, 0.081] | 0.74 | 761 |
| H07 | First-half creativity → second-half xG difference | −0.054 [−0.129, −0.002] | 0.11 | 761 |
| H08 | Change after conceding (vs. placebo windows) | −0.013 [−0.030, 0.002] SD | 0.21 | 1,106 goals |
| H09 | Change after scoring | **−0.061 [−0.078, −0.045] SD** | <0.001 | 1,140 goals |
| H10 | The top outfield creator's share of decisions rises after conceding | +0.001 [−0.003, 0.005] | 0.63 | 1,106 |
| H11 | Creativity while trailing → comeback (draw level or win) | −0.018 [−0.066, 0.035] per SD | 0.62 | 429 team-matches (113 comebacks) |
| H12 | Late (75'+) creativity when leading minus when level, same team and match | −0.061 [−0.129, 0.008] SD | 0.16 | 99 |
| H13 | Team change after a substitution vs. how much more creative the incoming player is | +0.007 [−0.011, 0.026] per SD | 0.62 | 968 |
| H14 | Teammates more creative while a creative player is on the pitch (on/off) | −0.002 [−0.006, 0.002] | 0.37 | 3,594 splits |
| H15 | Change after a red card (team sent off) | **−0.106 [−0.183, −0.021] SD** | 0.033 | 40 |
| H16 | Change after a tactical shift | **+0.023 [0.009, 0.036] SD** | <0.001 | 1,203 |
| H17 | After 60', substitutes minus starters (same team and match) | **−0.019 [−0.034, −0.004] SD** | 0.015 | 791 team-matches |
| H18 | Knockout vs. group matches (team fixed effects) | +0.001 [−0.010, 0.015] SD | 0.86 | 582 |
| H19 | A player's change between consecutive tournaments | −0.009 [−0.022, 0.005] SD | 0.37 | 149 transitions |
| H20 | Team creativity vs. opponent strength | −0.001 [−0.003, 0.003] per goal | 0.86 | 812 |
| H21 | Under pressure minus not under pressure (within player) | +0.006 [−0.008, 0.020] SD | 0.55 | 568 players |
| H22 | Novelty wears off: repeated vs. first creative pattern in a match (difference-in-differences of realized minus predicted value) | **+0.0010 [0.0003, 0.0018]** (opposite of the hypothesis) | 0.031 | 417 matches |

### 3.2 Robustness and details

- **Is H02/H03 just the value component?** No: originality alone (`rarity`) also predicts shots
  (+1.5 pp per SD [1.2, 1.8]) and goals (+0.23 pp [0.13, 0.33]). Value alone predicts more (+10.5
  pp and +2.5 pp), as expected.
- **Chain position:** in goal possessions, creativity peaks at the assist and pre-assist (+0.48 and
  +0.46 SD) and is low for the shot itself.

  | Possession outcome | Assist position, `good_surprise` | Assist position, `rarity` |
  |---|---|---|
  | Goal | +0.48 | +0.33 |
  | Shot, no goal | +0.41 | +0.21 |
  | No shot | +0.13 | +0.05 |

- **Surprise premium (H01):**
  - Monte Carlo residual: +0.0018 [0.0012, 0.0024].
  - Positive-advantage decisions: +0.0024 (one-step), +0.0029 (Monte Carlo).
  - Size: about 18% of a mean action's value across the rarity range.
  - Alternative explanation: the value model shrinks the values of rare actions toward the mean.
- **Reactions to goals (H09):**
  - The drop is −0.065 when the scorer now leads; +0.022 (n.s.) when still behind.
  - The conceding team shows no change on `good_surprise`, but −0.026 [−0.044, −0.010] on
    `good_rarity`, so the measures differ here.
- **Within-match profile:**
  - Trailing teams after 75' are +0.025 [0.014, 0.034] SD more creative than typical for that
    state.
  - Second-half stoppage time: +0.029 [0.015, 0.044]. The end of extra time: +0.149 [0.049, 0.280]
    (486 decisions).
  - The first 15 minutes: −0.013 [−0.020, −0.006].
- **Personnel:**
  - Tactical substitutions: team +0.017 [0.004, 0.033] SD; teammates of the swapped players +0.018
    [0.003, 0.033].
  - Injury substitutions: n.s.
  - The opponent after a red card: +0.029 (n.s.).
  - Star absence (reference yardstick): no consistent effect. Neymar −0.005 and −0.016; Verratti
    +0.027 (n.s.; cross-fit +0.056 [0.023, 0.093]).
- **Team creativity:** no association with xG, goal difference or points. Its association with the
  failed-pass rate is +0.004 (p = 0.07).

### 3.3 Descriptive results

- **Creativity allowed to opponents** (opponent-adjusted, ≥ 5 matches):
  - Least: Ukraine 2020 (−0.050), England 2020 (−0.044), Australia (women) 2023 (−0.042), Morocco
    2022 (−0.033), Brazil 2022 (−0.032).
  - Most: Portugal 2024 (+0.044), Switzerland 2020, Spain (women) 2025, England (women) 2023 (+0.03).
- **Player types** (k-means, 5 clusters, 331 players with ≥ 300 decisions):

  | Type | Players | Examples |
  |---|---|---|
  | High-value creators (high value and risk, carry-heavy) | 61 | Messi, Mbappé, Neymar, Wirtz, Dembélé, Di María |
  | Passing creators | 91 | Verratti, Xhaka, Busquets, Pedri, Griezmann |
  | Carriers | 78 | de Jong, Hakimi, Stones |
  | Conservative passers | 91 | Marquinhos, Alba, Kimpembe |
  | Artefact cluster (dribble-dominated, few creative actions) | 10 | – |

- **Enablers** (recipients of creative passes, relative to their team): target strikers (Lukaku,
  Hegerberg, Mitrović, Schick, Embolo).
- **Duos:** the strongest pair-specific creativity is in long-ball links (Laporte→Olmo,
  Tapsoba→Boniface, Xhaka→Boniface, Walsh→Hemp, Wirtz→Schick).
- **Creative hubs:** by total creativity (volume), a team's top creator is often a centre-back
  (Škriniar, Söyüncü, Blind, Rúben Dias, Spitse).

### 3.4 Caveats

- **Relative effects:** game-state and time effects are relative to the typical behaviour in that
  state.
- **Observational design:** substitutions, cards and shifts are not random.
- **Small samples:** red cards (40) and star absences (3–16 matches).
- **Sensitivity:** some effects differ between `good_surprise` and `good_rarity`.

---

## Appendix A: Further insights from the models

1. **The hardest decision is at the start of a possession spell.** Type accuracy is 70% at spell
   starts (65% of decisions) and 96–97% after carries.
   - Pass targets: top-1 zone 10.7%, top-5 35%, within one zone 42%, median error 10 yd; the
     receiver is identified 52% of the time.
   - Carry targets: top-1 12%, top-5 44%.
2. **Data size matters.** Each cross-fit fold has about 57% of the training data and is about 0.15
   nats worse than the full-data model. The fold models were still improving at epoch 16.
3. **Shot and dribble decisions look predictable** (mean rarity 0.31), passes less so (0.54). The
   creativity measures mainly capture passing and carrying decisions.
4. **Failed passes look original** because their end points are interception points (rarity 0.66
   vs. 0.51).
5. **About one in six decisions ends outside the camera view;** among top-creative decisions it is
   36–53%. These carry real signal (§2.2, §2.4).
6. **Single-team releases bias cross-fitted measures** (§2.6). Within-team comparisons are robust;
   comparisons across teams need the reference model.
7. **Value models:**
   - Distributional (C51) training does not change player rankings (ρ ≥ 0.93).
   - λ-returns gave the best return loss.
   - Expected SARSA overfits after epoch 1 (temporal-difference drift).
   - Value tops are inflated by 15–35% (winner's curse), but on average there is no inflation for
     rare actions; if anything they realize more value than predicted.
8. **Women's and men's football:** the policy is equally calibrated for both, but the
   value-containing measures are higher for women (+6 percentile points).

## Appendix B: What we tried

| Area | Variants | Outcome |
|---|---|---|
| Prediction target | Receiver (pointer / closest teammate), pass zone (26×17), full next action (type → zone → height) | Full next action adopted. An early pass-target model was beaten by a closest-teammate heuristic (top-1 7.6% vs. 8.7%) |
| Policy architecture | Spatial GATv2; sequence-to-spatial (GRU history plus GATv2); reflection group convolution; frame averaging | Sequence-to-spatial with y-flip symmetry adopted |
| Input features | Game state, velocities, distance edges (v3); actor role | v3 adopted; actor role implemented but off by default |
| Leakage | Naive alignment vs. anchoring at gain events, carry mask, same-time opponent events removed | Without anchoring, type accuracy was a trivial 95% |
| Value targets | Monte Carlo, SARSA, expected SARSA, IQL (expectile 0.7), λ-return (λ = 0.8) | MC adopted (stable); expected SARSA drifts after epoch 1; λ-return has the best return loss but worse protocol log-loss |
| Value heads and training | Direct vs. decomposed; warm start vs. joint training; frozen vs. fine-tuned encoder | Decomposed, warm start, encoder learning rate ×0.1 |
| Distributions | Expectation, quantile regression, C51 | C51 as auxiliary head. Fixes needed: exactly symmetric support, peaked initialization, v_max 1 and dense atoms near 0 (CRPS 0.0198 → 0.0165, below the trivial 0.0191) |
| Value baselines | VAEP, VAEP-xG (socceraction), xT (16×12, 26×17, 12×8) | On par with VAEP and xT; VAEP-xG best on the protocol |
| Creativity measures | 13 (§1.5) | Set measures best; products are position-driven; `difficulty` fails |
| C51 tail set | Upper quantile > V(s) vs. > the situation's upper quantile | The first covered 99.9% of options, so the second was adopted |
| Scope and view | All vs. intended; in / out of camera view | Intended adopted; view kept as a robustness split |
| Calibration | Temperature scaling, one temperature per head | No gain; T = 1 |
| Yardstick | Cross-fitting (2 folds), reference model, main split | Cross-fit for validation, reference for contexts |
| Expert validation | Blinded 2D kit with reveal, matched controls, scoring script | Ready, not yet rated |

## Appendix C: Outlook (full research project)

1. **Validation:**
   - A blinded multi-rater expert study (consensual assessment technique): at least 5 raters, about
     200 clips, separate development and confirmation clips, inter-rater reliability, video and 2D
     conditions.
   - External ratings (EA FC or Football Manager vision/flair), with discriminant attributes.
   - Recall of famous creative moments from external lists.
   - Pre-registration (OSF).
2. **Measurement:**
   - A model of the intended target of failed passes.
   - Player- and team-conditioned policies, separating personal from population originality.
   - Epistemic uncertainty (ensembles, seeds) to control the winner's curse.
   - Leave-one-team-out reference models for every team, and 5-fold cross-fitting.
   - Position-normalized products; a sequence-level creativity model (multi-action plans,
     combinations).
3. **Data:**
   - Larger 360 coverage (more seasons, leagues, women's football).
   - Broadcast or optical tracking (full-pitch positions remove the camera-view problem).
   - Longitudinal player careers (development, transfers).
4. **Analyses:**
   - Quasi-experimental designs: substitutions, red cards, injuries and coach changes as natural
     experiments.
   - Team creativity vs. success with adequate power.
   - Opponent adaptation within and across matches.
   - Creativity–risk trade-offs.
   - Age curves; comparisons between the women's and men's games.
5. **Theory and application:**
   - Ground the measure in creativity research (standard definition; Simonton's multiplicative
     definition; tactical creativity, Memmert).
   - Applications in scouting, talent identification, training feedback and opposition analysis.

## Appendix D: Risks

| Risk | Current evidence | Mitigation |
|---|---|---|
| Construct validity: no ground truth | Known-groups AUC 0.65; expert ratings pending | Multi-rater expert study; external ratings; pre-registration |
| Halo and selection bias in the validation list | List is reputation-based and written by an AI | Independent external lists; within-position tests; blinded raters |
| Model dependence | Player level robust (ρ 0.93–0.99 across value models); action level less so (top-1% overlap 0.27–0.45 between model instances) | Report player-level claims; ensembles; uncertainty intervals |
| Yardstick dependence | Cross-fit vs. reference ρ 0.69–0.73 (set measures); club bias | Reference models per context; within-team designs |
| Values extrapolated for rare actions | Tops inflated by 15–35%; no average inflation | Realized-value checks; epistemic uncertainty; set measures (not inflated) |
| Camera view (360 data) | 36–53% of top decisions out of view | View split as robustness; tracking data |
| Measurement invariance | Women +6 percentile points on value measures; position effects | Within-gender and within-position analyses |
| Event annotation errors and intent | Failed and deflected passes | Intended scope; intent model |
| Data coverage | 418 matches; single-team club seasons; few red cards and absences | More data; reference models; avoid low-power claims |
| Multiple testing and observational inference | 22 main hypotheses, 8 significant | Benjamini–Hochberg; pre-registered confirmatory set; quasi-experiments |
| Novelty vs. un-xPass (Robberechts et al.) and related work | We extend to all actions, value-gated option sets, distributional tails, out-of-sample and a validation framework | Explicit comparison (`unxpass` is implemented as a baseline) |
| Reproducibility | Creativity code not yet committed; data are StatsBomb open data | Commit code and record data versions; manifests with input checksums exist |
| Ethics | Rankings of identifiable players; gender comparisons | Aggregate reporting; uncertainty intervals; no individual claims without intervals |

## Appendix E: Files

| Content | Location |
|---|---|
| Decision tables, cross-fitted (MC + C51; MC) | `outputs/creativity/crossfit_c51_fold{0,1}.csv.gz`, `crossfit_fold{0,1}.csv.gz` |
| Decision table, reference model | `outputs/creativity/reference_holdout.csv.gz` |
| Main validation report | `outputs/creativity/validation_c51/report.md` |
| Validation variants | `validation_c51_{in_view,out_view,all_actions}/`, `validation_reference/` |
| Team and context analyses | `outputs/creativity/team_analyses/report.md` |
| Temperature check | `outputs/creativity/calibration/*/report.md` |
| Downstream analyses | `outputs/creativity/second_order/index.md`, `hypotheses.csv`, `manifest.json` |
| Match context tables | `outputs/creativity/context/` |
| Expert review kit | `outputs/creativity/review_kit1/review.html` (key: `key_DO_NOT_OPEN.csv`) |
| Code | `soccer_ml/creativity/`, `scripts/creativity/` |
| How to run | `CREATIVITY_VALIDATION_README.md` |
