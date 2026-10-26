# Technical Appendix

In this appendix, we describe the data we use and how we prepare it (A.1), our surprise and usefulness models
(A.2), how we derive our creativity measure and the creative opportunity from these models (A.3), how we validate
the measure (A.4), and the analyses underlying the preliminary results reported in the proposal (A.5). Table A2
lists all estimates cited in the proposal.

## A.1 Data

**Source.** We use StatsBomb open data (StatsBomb, 2024), which combines event data with 360 freeze frames. The
event data record every on-ball action with its type, location, end location and outcome, as well as further
annotations such as pass height, body part or whether a pass was a through ball. For each event, the freeze frame
records the positions of all players visible in the broadcast image at that moment, flags the acting player, their
teammates and the goalkeepers, and delimits the visible area of the pitch. We explicitly note that freeze frames are
snapshots at event times rather than continuous tracking data, and that they only cover the players visible in the
broadcast image. The data comprise all 426 matches for which 360 data is available, of which 418 are usable, as the
remaining matches have corrupt 360 files or 360 frames that cannot be linked to the events. Table A1 gives an
overview. We remark that the club data each contain only the matches of a single team, which we account for in our
modeling (A.2).

**Table A1: Data.**

| Competition | Matches | Decisions | Note |
|---|---|---|---|
| FIFA World Cup 2022 | 64 | 82,307 | |
| FIFA Women's World Cup 2023 | 64 | 75,411 | |
| UEFA Euro 2020 / 2024 | 51 / 51 | 66,329 / 65,796 | |
| UEFA Women's Euro 2022 / 2025 | 31 / 31 | 36,341 / 34,964 | |
| La Liga 2020/21 | 35 | 52,355 | FC Barcelona matches only |
| Ligue 1 2021/22, 2022/23 | 26 / 32 | 34,061 / 42,343 | Paris Saint-Germain matches only |
| Bundesliga 2023/24 | 34 | 42,637 | Bayer Leverkusen matches only |
| MLS 2023 | 6 | 3,678 | Inter Miami matches only |
| Total | 418 usable (426) | 536,223 | 389,507 men's, 146,716 women's |

**Decisions.** We define a decision as the moment a player chooses an on-ball action. We consider five action
types, namely passes, carries of at least 4.6 yd (the length of one zone, see below), shots, dribbles and
clearances, and treat shorter carries as part of the preceding ball control. A naive alignment of decisions with
events leaks information about the chosen action, since StatsBomb annotates a carry between the reception of the
ball and the subsequent action. We therefore anchor each possession spell at the event in which the player gains
control of the ball, i.e., a reception, recovery, interception or duel, and exclude carries as an option directly
after a carry. Further, the chosen action's type, height and body part are never used as inputs, and opponent events
with the same timestamp as the decision are excluded from the event history.

**State.** For each decision $i$, the state $s_i$ consists of the freeze frame, estimated player velocities, the
event history and a set of situation features. The freeze frame contributes the positions of up to 22 visible
players together with their actor, teammate and goalkeeper flags as well as the visible area. As freeze frames do
not include velocities, we estimate them by matching each team's players to the most recent previous freeze frame
recorded 0.2 to 3 s earlier. We solve this matching as an assignment problem with the Hungarian method (Kuhn, 1955),
admitting only pairs that are reachable at a maximum speed of 11 yd/s plus a tolerance of 2 yd. The event history
comprises up to 16 preceding events of the current possession with their type, outcome, body part, locations and
time gaps. Finally, the situation features encode the restart type, pressure on the actor, minute and period, goal
difference, home team, whether the competition is a women's competition, and the share of the pitch that is visible.
For the score and the further context needed in our analyses, we reconstruct the course of every match from the
event stream. This comprises the score before each decision, the number of players on the pitch per team, all goals
(1,217, excluding penalty shoot-outs), substitutions (3,619), red cards (53) and tactical shifts (1,404), the outcome
of each possession in terms of shots, expected goals (xG) and goals, and the final results. The reconstructed final
scores match the official results in all 418 matches.

**Action space.** We discretize the pitch of 120 × 80 yd into a grid of 26 × 17 zones. An action $a$ is either a
pass to one of the 442 zones at one of three heights (ground, low, high), a carry to one of the 442 zones, a shot, a
dribble, or a clearance. This yields a discrete action space $\mathcal{A}$ with $|\mathcal{A}| = 1{,}771$ actions.
For incomplete passes, however, the recorded end location is where the ball was intercepted or left the pitch rather
than the intended target, and the same holds for deflected passes. As our measure judges the choice of target, we
exclude incomplete and deflected passes from all creativity scores, which leaves 469,733 decisions. All
other action types are retained regardless of their outcome.

**External ratings.** For validation, we use player ratings from the FIFA video game series by EA Sports, which are
assessed per player and season by a network of data reviewers and scouts. We use the public sofifa-based dataset
(Leone, 2023), which contains the ratings of men from FIFA 15 to 23 at release and of women from FIFA 16 to 23
including all in-season updates. We match each player to the version that was current at the respective competition
(e.g., Euro 2020 to FIFA 21 and the World Cup 2022 to FIFA 23). To identify players, we restrict candidates to the
same team, i.e., the same nationality for national teams and the same club otherwise, and score them by the overlap
of name tokens with the StatsBomb name and nickname, requiring a unique best match. In this way, we match 87% of all
eligible player-contexts, 499 of them to contemporaneous ratings. As no contemporaneous ratings are available for
Euro 2024, the Bundesliga 2023/24 and the Women's Euro 2025, we use lagged ratings for these competitions only in a
sensitivity analysis.

## A.2 Models

We train two models. The surprise model estimates how likely each action is in a given state, and the usefulness
model estimates how valuable each action is.

**Surprise model.** The surprise model $\pi_\theta(a \mid s)$ factorizes the probability of an action into its type
$k$, its target zone $z$ for passes and carries, and its height $h$ for passes:

$$\pi_\theta(a \mid s) = \pi_\theta(k \mid s)\, \pi_\theta(z \mid k, s)\, \pi_\theta(h \mid z, s).$$

A graph attention network (GATv2; Brody, Alon, & Yahav, 2022) with 4 layers, 8 attention heads and a hidden dimension
of 128 encodes the visible players, where edges carry the team relation and the pairwise distance between players. A
recurrent encoder (GRU; Cho et al., 2014) summarizes the event history and conditions the graph representation. The
type head is a classifier, whereas the zone and height heads produce one output per zone on the 26 × 17 grid. Since
the game is symmetric with respect to the long axis of the pitch, we make the model equivariant to mirroring along
this axis. Specifically, we evaluate the model on the original and on the mirrored state and average the outputs
after mirroring them back (frame averaging; Puny et al., 2022). A pass to the left wing in a mirrored situation thus
receives exactly the same probability as the corresponding pass to the right wing. The model has 2.0 M parameters.

We train the surprise model by minimizing the negative log-likelihood of the observed actions with Adam (Kingma & Ba,
2015) for 16 epochs, using a learning rate of $3\cdot10^{-4}$ with a cosine schedule (Loshchilov & Hutter, 2017),
weight decay of $10^{-4}$, a batch size of 128 and gradient clipping at a norm of 1. On held-out matches, the models
we use for scoring (see out-of-sample scoring below) reach a joint negative log-likelihood of 4.48 to 4.50 nats,
compared to 5.62 nats for a baseline that predicts each zone by its frequency in the situation, while a model
trained on 368 matches reaches 4.34 nats. The action type is predicted correctly in 69% of decisions at the start of
a possession spell, which constitutes the hardest case, and in 96% of decisions following a carry.

**Usefulness model.** The usefulness model estimates the value $Q(s,a)$ of an action. We define this value as the
difference $Q(s,a) = Q^{+}(s,a) - Q^{-}(s,a)$ between the probability $Q^{+}(s,a)$ that the acting team scores and the
probability $Q^{-}(s,a)$ that it concedes within the next ten actions. Each of these two probabilities depends on
whether the action succeeds. We therefore decompose them using the success probability $p(s,a)$ of the action and the
scoring and conceding probabilities $Q^{\pm}_{\text{succ}}(s,a)$ after a successful and $Q^{\pm}_{\text{fail}}(s,a)$
after a failed action:

$$Q^{\pm}(s,a) = p(s,a)\, Q^{\pm}_{\text{succ}}(s,a) + \big(1-p(s,a)\big)\, Q^{\pm}_{\text{fail}}(s,a).$$

Following established action valuation (Decroos et al., 2019), the training targets are Monte Carlo labels over the
next ten actions, for which we use the expected goals of shots rather than goals to reduce noise. That is, the target
for scoring is $y^{+} = 1-\prod_j (1-\text{xG}_j)$ over the team's shots in this horizon, and the target for
conceding $y^{-}$ is defined analogously over the opponent's shots. The usefulness model shares the encoder of the
surprise model, is initialized from it and is trained for 3 epochs, with the encoder's learning rate scaled by 0.1.
For robustness analyses, an auxiliary head additionally predicts the full distribution of the discounted net return
of every action on a categorical support of 51 atoms in $[-1, 1]$ with a discount of 0.9 per decision. This head is
trained by distributional temporal-difference learning (Bellemare, Dabney, & Munos, 2017) with a target network
(Mnih et al., 2015) and a loss weight of 0.1. On the evaluation protocol of Robberechts et al. (2023), which uses
completed open-play passes and labels them by whether a goal follows within the next ten actions, the usefulness model
reaches an AUC of 0.77 to 0.78. This is on par with VAEP (Decroos et al., 2019) and expected threat (xT; Singh,
2018) with 0.78 each, while VAEP trained on xG labels reaches 0.80.

**Out-of-sample scoring.** All scores we report are out-of-sample. Following the idea of cross-fitting (Chernozhukov
et al., 2018), we randomly split the matches into two folds, train both models on one fold and score the decisions of
the other fold, and vice versa. In all cases, the final training epoch was also the best epoch on 49 monitoring
matches of the respective held-out fold, so model selection does not affect the scored data. Cross-fitting, however,
has a consequence for the club data. As each club season contains only the matches of a single team, a model trained
on the team's other matches learns the team's style and thus judges its players against their own team rather than
against all teams. For comparisons across teams and contexts, we therefore additionally train a reference model on
all matches except the four club seasons and Argentina's World Cup matches (256 training matches), which scores the
133 held-out matches against a common standard.

## A.3 Creativity Measure and Creative Opportunity

**Usefulness.** We define the value of a typical choice in state $s$ as the expected value under the surprise model,

$$V(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s)\, Q(s,a).$$

We call an action useful if its value exceeds that of the typical choice, i.e., if its usefulness
$U(s,a) = Q(s,a) - V(s)$ is positive. The useful options form the set $\mathcal{S}(s) = \{a : U(s,a) > 0\}$.

**Creativity.** Our creativity measure is the surprise of the chosen action $a_i$ among the useful options:

$$c_i = \begin{cases} -\log \pi(a_i \mid s_i,\, a \in \mathcal{S}) = \log \pi(\mathcal{S} \mid s_i) - \log \pi(a_i \mid s_i), & a_i \in \mathcal{S}(s_i),\\ 0, & \text{otherwise.}\end{cases}$$

Surprise is meant in the information-theoretic sense (Shannon, 1948) and corresponds to the originality criterion of
Simonton (2012). Usefulness enters as a threshold rather than as a weight, such that a useful action is the more
creative the less likely it was among the useful options, independent of how useful it is. We explicitly prefer this
over multiplying surprise and usefulness, as the product mainly reflects where on the pitch a player acts. Position
explains 70% of the variance in player scores for the product, compared to 9% for our measure.

**Creative opportunity and its use.** As the surprise model provides the full distribution over all options, we can
also determine the creativity that a typical player would produce in a given situation. We define the creative
opportunity of decision $i$ as the expected creativity under the surprise model,

$$o_i = \mathbb{E}_{a \sim \pi(\cdot \mid s_i)}[c] = \pi(\mathcal{S} \mid s_i)\, H\big(\pi(\cdot \mid s_i,\, a \in \mathcal{S})\big),$$

which equals the probability of choosing a useful option times the diversity of the useful options as measured by
their entropy $H$. The use of the opportunity is then given by $u_i = c_i - o_i$. Thus, each decision decomposes into
the creativity its situation offered and the creativity the player made of it. As a consistency check, average
creativity is close to average opportunity (1.31 vs. 1.38 nats), with the small gap stemming from the excluded
incomplete passes, which are disproportionately unusual. As an alternative definition of the creative opportunity,
we use the share of unusual options, i.e., the least likely 20% of the probability mass, that are useful, which
corresponds to the upper right quadrant in Figure 1.

**Robustness measures and aggregation.** We repeat key analyses with two variants of the measure. The first replaces
surprise by rarity among the useful options, where the rarity of an action is the probability mass of all more
likely options plus half of its own probability. The second uses the predicted return distributions and counts an
option as useful if the 90% quantile of its return distribution exceeds that of the situation. Player scores average
$c_i$ over a player's decisions. For decision-level analyses, we standardize $c_i$, $o_i$ and $u_i$ across all
in-scope decisions and report effects in standard deviations (SD).

## A.4 Validation

We first assess whether the surprise model's probabilities are calibrated, as the measure builds on the probabilities
of unlikely actions. Across all decisions, observed frequencies match the predicted probabilities within ±12% for all
probability levels above $10^{-5}$. Rarity is uniformly distributed with a mean of 0.511, as expected under
calibration, and remains so in every competition and for men and women alike, with means between 0.50 and 0.52.
Temperature scaling (Guo et al., 2017) does not improve calibration further.

We then compare player scores with the external FIFA ratings, ranking both within competition and position group to
remove differences in level and role. Creativity correlates with the vision rating (Spearman ρ = 0.17, 95% CI
[0.08, 0.27]), whereas usefulness alone does not (ρ = 0.02), with a difference of 0.15 [0.07, 0.24]. Creativity
further correlates with passing, ball control and dribbling (ρ between 0.12 and 0.19) but not with strength, jumping
or stamina (|ρ| < 0.08), and players labeled as playmakers in the game score higher than other players in their
position (AUC = 0.62).

As StatsBomb annotations are not used as model inputs for the decision itself, they provide independent controls.
Among passes, through balls score higher than other passes (AUC = 0.82), whereas goal kicks and throw-ins do not
(AUC between 0.49 and 0.50). Lastly, creativity correlates with the number of opponents a forward pass bypasses
(ρ = 0.28), is largely independent of position (9% of the variance in player scores), and player scores are robust to
the choice of usefulness model (ρ = 0.94 between models with and without the distribution head).

## A.5 Analyses

**Inference.** As decisions, possessions and events within a match are dependent, we compute all confidence intervals
by resampling matches (cluster bootstrap; Cameron, Gelbach, & Miller, 2008) with 1,000 resamples and derive two-sided
p-values from the bootstrap distribution. We control the false discovery rate across all claims reported in the
proposal (Benjamini & Hochberg, 1995). We remark that both models condition on score and time, such that the effects
of game events are to be read relative to what is typical in the same score and time.

**Build-up creativity and chances.** For each possession $p$ as defined by StatsBomb, we take the maximum creativity
$C_p$ over the decisions of the team in possession before the shot, standardized across 58,890 possessions. We
estimate linear probability models

$$y_p = \beta\, C_p + \alpha_{g(p)} + \varepsilon_p,$$

where $y_p$ indicates whether the possession ends in a shot (or a goal), and $\alpha_{g(p)}$ are fixed effects for
strata of possessions with the same starting value (decile of $V$), pitch third, play pattern and possession length.
To rule out that the effect only reflects the final pass, we repeat the estimation excluding the final one to three
decisions before the shot. We further add team-season or match fixed effects, control for the furthest field
position reached, which we regard as a likely mediator, enter originality (rarity) and usefulness jointly, and repeat
the estimation with scores from the reference model. Build-up creativity raises the probability of a shot by 3.9
percentage points per SD and that of a goal by 0.50 percentage points. Excluding the final two decisions, the effect
increases to 5.7 points for shots, such that it does not merely reflect the final pass. The estimates are unchanged
with team-season or match fixed effects (3.9 points) and similar with the reference model (3.6 points). Originality
predicts shots also when entered jointly with usefulness (1.8 points per SD, excluding the final two decisions).
Controlling for the field position reached reduces the effect on shots to 1.0 points and renders the effect on goals
insignificant, consistent with creative actions working through moving the ball forward.

**Hidden contributions.** In possessions ending in a goal, we assign each decision to one of four roles: the shot,
the assist as indicated by StatsBomb's goal-assist flag, the pre-assist as the decision preceding the assist, and
earlier decisions. We call a decision creative if its creativity lies in the top 10% of all decisions. We then compare
the rate of creative decisions across roles and determine the share of creative decisions that receive neither goal
nor assist credit. At the player level, we correlate players' average creativity with their goals plus assists per
90 minutes, both expressed as percentiles within gender and position group, for all players with at least 600
minutes and 300 decisions. Assists and pre-assists are creative in 24.7% and 18.0% of cases, i.e., 2.5 and 1.8 times the overall rate
of 10%. Yet, 85% of the creative decisions in goal possessions (95% CI [83, 87]) receive neither goal nor assist
credit, and players' creativity is unrelated to their goals and assists (Spearman ρ = −0.02 [−0.15, 0.10]).

**Game events.** We estimate the effect of an event on a team's creativity as the change in the team's mean
standardized creativity between the 10 minutes after and the 10 minutes before the event within the same period,
requiring at least five decisions in both windows. As creativity also varies systematically over the course of a
match, we subtract the same change for 20 placebo windows per event, which we draw from other matches at the same
period and match time (±3 minutes) without such an event nearby. The effect is the mean of these differences across
events. We study goals, separately for the scoring and the conceding team (1,140 and 1,106 events with sufficient
data), red cards for the team sent off (40), tactical shifts, i.e., formation changes (1,203), and tactical
substitutions (2,354). We repeat each analysis with 5- and 15-minute windows, test for pre-trends by comparing the
windows from −20 to −10 and from −10 to 0 minutes, and skip the first two minutes after goals to exclude the restart.
We additionally adjust creativity for pressure on the actor, measured by the number of opponents within 5 yd and the
distance to the nearest opponent, and estimate the effects separately for the creative opportunity $o_i$ and its use
$u_i$. After scoring, the scoring team's creativity drops by 0.061 SD. This holds for 5- and 15-minute windows (−0.094
and −0.055 SD), when skipping the restart (−0.048 SD) and after adjusting for pressure (−0.059 SD). We note that
creativity rises before goals (pre-trend of +0.022 SD), reflecting creative build-up, such that the drop relative to
earlier play is somewhat smaller. The drop is almost entirely a drop in use (−0.057 SD), while the creative
opportunity barely changes (−0.024 SD). The conceding team shows no significant change (−0.013 SD). After a red card,
the creativity of the team sent off drops by 0.106 SD without a pre-trend, and here the creative opportunity itself
shrinks (−0.122 SD). Tactical shifts raise creativity by 0.023 SD without a pre-trend, driven by use (+0.021 SD).
Tactical substitutions do not significantly change creativity (+0.011 SD) but enlarge the creative opportunity
(+0.046 SD).

**Distributed creativity.** To study how creativity is distributed within a possession, we extend the build-up model
for possessions with at least three decisions. Given the possession's maximum creativity, we add an indicator for
whether two or more players made a creative decision or, alternatively, the share of the possession's creativity
contributed by its most creative player. Possessions with two or more creative players end in a shot 6.8 percentage points and in a goal 1.1 percentage
points more often, beyond the effect of the single most creative decision. Conversely, a higher share of the most
creative player lowers the probability of a shot by 3.6 percentage points per SD.

**Opportunity and use across players.** We decompose the variance of player scores into the variance of the players'
average creative opportunity, the variance of their average use, and the covariance of both. The variance of use alone equals the variance of player scores (share 1.00), whereas the opportunity accounts
for a share of 0.13, offset by a negative covariance (−0.14). Differences between players thus stem almost entirely
from how they use their opportunities rather than from the opportunities they have.

**Surprise premium.** We test whether originality yields value beyond what the usefulness model predicts. For each
decision, we compute the realized one-step value $r_i + 0.9\, V(s_{i+1}) - V(s_i)$, where $r_i$ is the xG difference of
the shots until the next decision, and take its residual with respect to the predicted usefulness $U(s_i, a_i)$. We
regress this residual on the rarity of the chosen action within strata of the same predicted value (deciles of $Q$),
action type and pitch third. We estimate this separately for passes and carries, with the Monte Carlo return as an
alternative realized value, and with scores from the reference model. Across all decisions, the residual increases with rarity
(+0.0019 per unit rarity). The premium is concentrated in passes (+0.0024, about a third of the mean value of a pass),
holds with the Monte Carlo return and the reference model, but is negative for carries (−0.0016).

**Enabling.** We pair each decision with the next decision of a different player of the same team, yielding 313,166
pairs. We regress the second player's creative opportunity, their distance to the nearest opponent and the value of
their situation on the first player's creativity. The regressions include fixed effects for strata of the same target
area (10 × 10 yd), action type and starting value decile, as well as match fixed effects. A creative decision moves the next player into a more valuable situation (+0.05 SD) but also into a tighter one,
with the nearest opponent 0.19 yd closer per SD of creativity. The next player's creative opportunity increases only
marginally (+0.01 SD), and their creativity is unchanged.

**Table A2: Preliminary estimates cited in the proposal.** 95% confidence intervals from the match bootstrap; q-values
after Benjamini–Hochberg correction across the tested claims. Estimates without q-values are descriptive or not part
of the confirmatory set.

| Analysis | Estimate [95% CI] | q |
|---|---|---|
| Build-up creativity → shot, per SD (base rate 14.4%) | +3.9 pp [3.6, 4.2] | <0.001 |
| … excluding the final two decisions | +5.7 pp [5.3, 6.0] | <0.001 |
| Build-up creativity → goal, per SD (base rate 1.7%) | +0.50 pp [0.39, 0.61] | <0.001 |
| Originality given usefulness → shot, excluding the final two decisions, per SD | +1.8 pp [1.4, 2.2] | <0.001 |
| Creative decisions among assists / pre-assists (overall rate 10%) | 24.7% / 18.0% | – |
| Share of creative decisions in goal possessions without goal or assist credit | 85% [83, 87] | – |
| Player creativity vs. goals and assists per 90 (Spearman, within position) | −0.02 [−0.15, 0.10] | – |
| Scoring team after a goal: creativity (SD) | −0.061 [−0.078, −0.045] | <0.001 |
| … opportunity / use (SD) | −0.024 [−0.049, 0.000] / −0.057 [−0.073, −0.042] | 0.055 / <0.001 |
| Team after a red card: creativity / opportunity (SD) | −0.106 [−0.183, −0.021] / −0.122 [−0.239, −0.010] | 0.015 / – |
| After a tactical shift: creativity (SD) | +0.023 [0.009, 0.036] | <0.001 |
| After a tactical substitution: creativity / opportunity (SD) | +0.011 [−0.002, 0.025] / +0.046 [0.024, 0.068] | 0.114 / – |
| Two or more creative players in a possession → shot / goal | +6.8 pp [5.7, 7.8] / +1.1 pp [0.7, 1.5] | – |
| Share of variance in player creativity due to use / opportunity | 1.00 / 0.13 (covariance −0.14) | – |
| Surprise premium, passes (per unit rarity; mean pass value 0.007) | +0.0024 [0.0022, 0.0026] | <0.001 |
| Creativity vs. FIFA vision rating (Spearman, within position) | 0.17 [0.08, 0.27] | – |

## References

Bellemare, M. G., Dabney, W., & Munos, R. (2017). A distributional perspective on reinforcement learning. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 449–458).

Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300.

Brody, S., Alon, U., & Yahav, E. (2022). How attentive are graph attention networks? In *International Conference on Learning Representations*.

Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *Review of Economics and Statistics, 90*(3), 414–427.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal, 21*(1), C1–C68.

Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing* (pp. 1724–1734).

Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). Actions speak louder than goals: Valuing player actions in soccer. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 1851–1861).

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 1321–1330).

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *International Conference on Learning Representations*.

Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly, 2*(1–2), 83–97.

Leone, S. (2023). *FIFA 23 complete player dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/stefanoleone992/fifa-23-complete-player-dataset

Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. In *International Conference on Learning Representations*.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., … Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529–533.

Puny, O., Atzmon, M., Ben-Hamu, H., Misra, I., Grover, A., Smith, E. J., & Lipman, Y. (2022). Frame averaging for invariant and equivariant network design. In *International Conference on Learning Representations*.

Robberechts, P., Van Roy, M., & Davis, J. (2023). un-xPass: Measuring soccer player's creativity. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (pp. 4768–4777).

Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal, 27*(3), 379–423.

Simonton, D. K. (2012). Taking the U.S. Patent Office criteria seriously: A quantitative three-criterion creativity definition and its implications. *Creativity Research Journal, 24*(2–3), 97–106.

Singh, K. (2018). *Introducing expected threat (xT)*. https://karun.in/blog/expected-threat.html

StatsBomb. (2024). *StatsBomb open data* [Data set]. https://github.com/statsbomb/open-data
