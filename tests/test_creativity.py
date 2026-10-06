"""Tests for creativity statistics, measures, validation checks and the review kit."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from soccer_ml.creativity.analysis import compare_groups, player_in_contexts, season_trend, team_summary
from soccer_ml.creativity.measures import CREATIVITY_MEASURES, MEASURES, compute_measures, player_scores, scope_mask
from soccer_ml.creativity.review import build_key, clip_payload, review_page, sample_clips
from soccer_ml.creativity.scoring import NUM_CALIBRATION_BINS, decision_statistics, set_rarity
from soccer_ml.creativity.validation import (
    annotation_controls,
    artefact_rates,
    auc,
    discriminant,
    famous_moments,
    known_groups,
    line_breaking,
    model_agreement,
    pit_uniformity,
    player_rankings,
    policy_calibration,
    position_rankings,
    split_half_reliability,
    value_calibration_by_rarity,
    winners_curse,
)
from soccer_ml.data.next_action_dataset import ACTION_TYPES, NextActionDataset, collate_next_action
from soccer_ml.data.schema import STATSBOMB_NODE_LAYOUT
from soccer_ml.models.next_action import NextActionModel, action_log_likelihood
from soccer_ml.models.value import ValueModel

THREE_SIXTY = Path("data_toy_receiver/three_sixty_3764440.json")
EVENTS = Path("data_toy_receiver/events_3764440.json")
needs_data = pytest.mark.skipif(not THREE_SIXTY.exists(), reason="toy StatsBomb data not available")


def test_set_rarity_hand_example():
    pi = torch.tensor([[0.5, 0.3, 0.2]])
    result = set_rarity(pi, pi.log(), None, torch.tensor([1]))
    assert result["rarity"].item() == pytest.approx(0.5 + 0.15)
    assert result["surprise"].item() == pytest.approx(-math.log(0.3))
    member = torch.tensor([[False, True, True]])
    inside = set_rarity(pi, pi.log(), member, torch.tensor([2]))
    assert inside["mass"].item() == pytest.approx(0.5)
    assert inside["rarity"].item() == pytest.approx((0.3 + 0.1) / 0.5)
    assert inside["surprise"].item() == pytest.approx(-math.log(0.2 / 0.5))
    outside = set_rarity(pi, pi.log(), member, torch.tensor([0]))
    assert math.isnan(outside["rarity"].item()) and outside["inside"].item() == 0


def test_rarity_is_uniform_when_actions_follow_the_policy():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(4000, 50, generator=generator) * 2
    pi = torch.softmax(logits, -1)
    observed = torch.multinomial(pi, 1, generator=generator).squeeze(-1)
    result = set_rarity(pi, pi.log(), None, observed)
    randomized = result["above"] + torch.rand(4000, generator=generator) * result["p_obs"]
    assert stats.kstest(randomized.numpy(), "uniform").statistic < 0.03
    assert result["rarity"].mean().item() == pytest.approx(0.5, abs=0.02)


@pytest.fixture(scope="module")
def toy_batch():
    dataset = NextActionDataset(THREE_SIXTY, EVENTS)
    return collate_next_action([dataset[i] for i in range(0, len(dataset), max(1, len(dataset) // 64))])


@needs_data
def test_decision_statistics_on_toy_batch(toy_batch):
    torch.manual_seed(0)
    policy = NextActionModel(hidden_dim=32, num_layers=2, nb_heads=4, temporal_layers=1, dropout=0.0).eval()
    value = ValueModel(NextActionModel(hidden_dim=32, num_layers=2, nb_heads=4, temporal_layers=1, dropout=0.0),
                       head="decomposed", distribution="categorical").eval()
    with torch.no_grad():
        policy_outputs = policy(toy_batch)
        result = decision_statistics(policy_outputs, value(toy_batch), toy_batch, value, STATSBOMB_NODE_LAYOUT)
    b = toy_batch.batch_size
    assert all(tensor.shape == (b,) for tensor in result.values())
    assert torch.all((result["rarity"] >= 0) & (result["rarity"] <= 1))
    nll = action_log_likelihood(policy_outputs, toy_batch)
    known = nll["has_height"] | (toy_batch.action_type != ACTION_TYPES.index("pass"))
    torch.testing.assert_close(result["surprise"][known], nll["joint_nll"][known], atol=1e-3, rtol=1e-3)
    masses = torch.stack([result[f"calib_mass_{k}"] for k in range(NUM_CALIBRATION_BINS)], -1)
    torch.testing.assert_close(masses.sum(-1), torch.ones(b), atol=1e-4, rtol=1e-4)
    assert torch.equal(result["in_good"] > 0.5, result["advantage"] > 0)
    for name in ("good", "tail", "c51_tail", "pass"):
        assert torch.all((result[f"{name}_mass"] >= 0) & (result[f"{name}_mass"] <= 1 + 1e-5))
        inside = result[f"in_{name}"] > 0.5
        assert torch.all(result[f"{name}_rarity"][inside] <= 1 + 1e-5) and result[f"{name}_rarity"][~inside].isnan().all()
    is_pass = toy_batch.action_type == ACTION_TYPES.index("pass")
    assert torch.equal(result["in_pass"] > 0.5, is_pass)
    assert torch.all(result["c51_q10"] <= result["c51_q90"]) and torch.all(result["upside"] >= 0)
    bypassed = result["opponents_bypassed"]
    assert torch.all(bypassed[~bypassed.isnan()] >= 0)


def _synthetic_table(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    players = rng.integers(0, 40, n)
    matches = rng.integers(0, 30, n)
    action = rng.choice(["pass", "carry", "shot", "dribble"], n, p=[0.6, 0.3, 0.05, 0.05])
    rarity = rng.uniform(size=n)
    advantage = rng.normal(0, 0.01, n)
    table = pd.DataFrame({
        "event_uuid": [f"e{i}" for i in range(n)], "match_id": matches, "player_id": players,
        "player": [f"Player {p} Name" for p in players], "team": [f"T{p % 8}" for p in players],
        "position": rng.choice(["Center Midfield", "Center Back", "Left Wing"], n),
        "match_date": [f"2024-01-{m % 28 + 1:02d}" for m in matches],
        "competition": np.where(matches < 15, "Euro", "World Cup"), "season": np.where(matches < 15, "2020", "2022"),
        "women": matches % 3 == 0, "home_team": [f"Home{m}" for m in matches], "away_team": [f"Away{m}" for m in matches],
        "minute": rng.integers(0, 95, n), "second": rng.integers(0, 60, n), "period": rng.integers(1, 3, n),
        "action_type": action, "decision_context": rng.choice(["spell_start", "after_carry", "other"], n),
        "rarity": rarity, "rarity_random": rng.uniform(size=n), "surprise": -np.log(rng.uniform(0.001, 1, n)),
        "advantage": advantage, "td_advantage": advantage + rng.normal(0, 0.01, n),
        "mc_advantage": advantage + rng.normal(0, 0.05, n), "mc_return": rng.normal(0, 0.05, n),
        "q": rng.uniform(0, 0.05, n), "q_score": rng.uniform(0, 0.05, n), "v": rng.uniform(0, 0.05, n),
        "label_xg_scores": rng.uniform(0, 0.05, n), "p_success": rng.uniform(size=n),
        "success": rng.integers(0, 2, n).astype(float), "upside": rng.uniform(0, 0.02, n),
        "in_good": (advantage > 0).astype(float), "good_rarity": np.where(advantage > 0, rng.uniform(size=n), np.nan),
        "good_surprise": np.where(advantage > 0, rng.uniform(0, 5, n), np.nan),
        "in_tail": rng.integers(0, 2, n).astype(float), "tail_rarity": rng.uniform(size=n),
        "pass_rarity": np.where(action == "pass", rng.uniform(size=n), np.nan),
        "calib_bin": rng.integers(0, NUM_CALIBRATION_BINS, n),
        "end_zone_visible": rng.integers(0, 2, n).astype(float), "restart": rng.choice(["", "Goal Kick", "Throw-in"], n, p=[0.9, 0.05, 0.05]),
        "start_x": rng.uniform(0, 120, n), "end_x": rng.uniform(0, 120, n),
        "opponents_bypassed": rng.integers(0, 6, n).astype(float),
    })
    for k in range(NUM_CALIBRATION_BINS):
        table[f"calib_mass_{k}"] = 1.0 / NUM_CALIBRATION_BINS
    for flag in ("miscommunication", "deflected", "through_ball", "no_touch", "nutmeg", "flair_shot", "cut_back",
                 "switch", "first_time"):
        table[flag] = rng.uniform(size=n) < 0.05
    # A planted positive control: through balls are rarer choices.
    table.loc[table["through_ball"], "rarity"] = rng.uniform(0.8, 1.0, int(table["through_ball"].sum()))
    return table


def test_measures_and_validation_checks_run():
    table = _synthetic_table()
    scores = compute_measures(table, scope="all")
    assert set(scores.columns) == {m.name for m in MEASURES} - {"c51_tail_rarity"}
    assert scores["unxpass"][table["action_type"] != "pass"].isna().all()
    assert (scores["good_rarity"][table["advantage"] <= 0] == 0).all()
    assert policy_calibration(table)["ratio"].notna().all()
    assert pit_uniformity(table, "decision_context")["n"].sum() == len(table)
    assert len(value_calibration_by_rarity(table)) == 10
    assert set(winners_curse(table, scores)["measure"]) == set(scores.columns)
    assert len(artefact_rates(table, scores)) == len(scores.columns) + 1
    controls = annotation_controls(table, scores)
    through = controls[(controls["control"] == "through_ball") & (controls["measure"] == "rarity")]["auc"].iat[0]
    assert through > 0.75  # the planted control is recovered
    assert len(line_breaking(table, scores)) == len(scores.columns)
    assert discriminant(table, scores, 50)["eta2_team"].between(0, 1).all()
    reliability = split_half_reliability(table, scores, 50)
    assert (reliability["players"] > 10).all()
    rankings = player_rankings(table, scores, 50, top_n=5, bootstrap=20)
    assert (rankings["rank_p05"] <= rankings["rank_p95"]).all()
    players = player_scores(table, scores, 50)
    known = known_groups(players, {0, 1, 2}, list(CREATIVITY_MEASURES[:3]) + ["advantage", "rarity"], bootstrap=200)
    assert known["known_players"].eq(3).all() and (known["auc_low"] <= known["auc_high"]).all()
    assert known.loc[known["measure"] == CREATIVITY_MEASURES[0], "diff_vs_advantage"].notna().all()
    rankings_by_position = position_rankings(players, ["rarity_x_adv"], top=2, bottom=1)
    assert set(rankings_by_position["side"]) == {"top", "bottom"} and rankings_by_position["percentile"].between(0, 1).all()
    agreement = model_agreement(table, scores, table, scores)
    assert np.allclose(agreement["action_spearman"].dropna(), 1.0)
    row = table.iloc[0]
    moments = pd.DataFrame([{"description": "planted", "team_a": row["home_team"], "team_b": row["away_team"],
                             "minute": row["minute"], "player": row["player"]}])
    found = famous_moments(table, scores, moments)
    assert found["matches_found"].iat[0] >= 1 and 0 < found["pct_rarity"].iat[0] <= 1


def test_auc_matches_scipy():
    rng = np.random.default_rng(1)
    scores, labels = rng.normal(size=500), rng.uniform(size=500) < 0.3
    value, se, n1 = auc(scores, labels)
    u = stats.mannwhitneyu(scores[labels], scores[~labels]).statistic
    assert value == pytest.approx(u / (labels.sum() * (~labels).sum())) and se > 0 and n1 == labels.sum()


@needs_data
def test_review_kit_is_blinded():
    table = _synthetic_table(3000)
    events = json.loads(EVENTS.read_text())
    frames = {frame["event_uuid"] for frame in json.loads(THREE_SIXTY.read_text())}
    passes = [e for e in events if e["type"]["name"] == "Pass" and e["id"] in frames]
    scores = compute_measures(table)
    clips = sample_clips(table, scores, per_measure=2, originality_controls=4, value_controls=3, random_clips=4)
    assert clips["clip_id"].is_unique and clips["row"].is_unique
    paired = clips[clips["reason"].str.endswith("control")]
    assert len(paired) > 0 and paired["pair_clip"].isin(clips["clip_id"]).all()
    payloads = []
    for number, clip in enumerate(clips.itertuples()):
        event = passes[number % len(passes)]
        row = table.loc[clip.row].copy()
        row["event_uuid"], row["anchor_uuid"], row["action_type"] = event["id"], event["id"], "pass"
        payload = clip_payload(row, EVENTS, THREE_SIXTY)
        assert payload["players"] and payload["action"]["end"] is not None
        payloads.append({"clip_id": clip.clip_id, **payload})
    page = review_page(payloads, "test-kit")
    for secret in ["top:", "control", "random", *scores.columns]:
        assert f'"{secret}' not in page, secret
    key = build_key(table, scores, clips)
    assert {"reason", "pair_clip", "pct_rarity_x_adv"} <= set(key.columns) and len(key) == len(clips)


def test_scope_mask_drops_unintended_pass_targets():
    table = pd.DataFrame({
        "action_type": ["pass", "pass", "pass", "carry", "shot", "pass"],
        "success": [1.0, 0.0, 1.0, 0.0, 0.0, np.nan],
        "deflected": [False, False, True, False, False, False],
        "end_zone_visible": [1.0, 1.0, 0.0, 0.0, np.nan, 1.0],
    })
    assert scope_mask(table, "intended").tolist() == [True, False, False, True, True, False]
    assert scope_mask(table, "all").all()
    assert scope_mask(table, "all", "out").tolist() == [False, False, True, True, False, False]
    assert scope_mask(table, "intended", "in").tolist() == [True, False, False, False, True, False]


def _team_table(seed: int = 0) -> pd.DataFrame:
    """Two teams over 20 matches; player 10 of team A is more creative; team A trends upwards."""
    rng = np.random.default_rng(seed)
    rows = []
    for match in range(20):
        for team, players in (("A", range(10, 16)), ("B", range(20, 26))):
            for player in players:
                for _ in range(30):
                    value = rng.normal(0, 1) + (1.0 if player == 10 else 0.0) + (0.1 * match if team == "A" else 0.0)
                    rows.append({"match_id": match, "match_date": f"2024-01-{match + 1:02d}", "team": team,
                                 "competition": "League", "season": "2024", "player_id": player, "value": value})
    return pd.DataFrame(rows)


def test_team_analyses_recover_planted_effects():
    table = _team_table()
    scores = table[["value"]]
    result = compare_groups(scores, table["match_id"], table["player_id"] == 10,
                            (table["team"] == "A") & (table["player_id"] != 10), bootstrap=300)
    assert result["difference"].iat[0] == pytest.approx(1.0, abs=0.15) and result["low"].iat[0] > 0.6
    contexts = player_in_contexts(table, scores, 10, min_decisions=50, bootstrap=300)
    assert contexts["percentile_in_team"].iat[0] == 1.0 and contexts["teammates_ranked"].iat[0] == 5
    per_match, trend = season_trend(table, scores, "A | League 2024", permutations=2000, bootstrap=300)
    team = trend[trend["series"] == "team"].iloc[0]
    assert len(per_match) == 20 and team["slope_per_match"] == pytest.approx(0.1, abs=0.02) and team["permutation_p"] < 0.01
    flat = season_trend(table, scores, "B | League 2024", permutations=2000, bootstrap=300)[1]
    assert flat[flat["series"] == "team"]["permutation_p"].iat[0] > 0.01
    summary = team_summary(table, scores, min_matches=5, bootstrap=200)
    assert set(summary["context"]) == {"A | League 2024", "B | League 2024"}


def test_second_order_tools():
    from soccer_ml.creativity.second_order import Windows, cluster_mean, event_study, fe_slope

    rng = np.random.default_rng(0)
    n = 4000
    strata, clusters = rng.integers(0, 20, n), rng.integers(0, 100, n)
    x = rng.normal(size=n) + strata * 0.5
    y = 0.3 * x + strata * 2.0 + rng.normal(size=n)  # stratum effects must not leak into the slope
    estimate, low, high, p, count = fe_slope(pd.Series(y), pd.Series(x), strata, pd.Series(clusters), 300)
    assert estimate == pytest.approx(0.3, abs=0.05) and low < 0.3 < high and p < 0.01 and count == n
    mean, low, high, p, count = cluster_mean(pd.Series(rng.normal(0.5, 1, n)), pd.Series(clusters), 300)
    assert low < 0.5 < high and p < 0.01
    # Event study: one team's value jumps by 1 at minute 30 in every match; placebo windows see no jump.
    rows = []
    for match in range(40):
        for team in ("A", "B"):
            for minute in np.arange(0, 45, 0.5):
                jump = 1.0 if (team == "A" and minute >= 30 and match < 20) else 0.0
                rows.append({"match_id": match, "team": team, "period_ctx": 1, "clock": minute, "player_id": 1,
                             "value": jump + rng.normal(0, 0.1)})
    windows = Windows(pd.DataFrame(rows), "value")
    events = pd.DataFrame({"match_id": range(20), "period": 1, "clock": 30.0, "team": "A"})
    result = event_study(windows, events, "team", events, width=10, placebos=10)
    assert result["effect"].mean() == pytest.approx(1.0, abs=0.1)


def test_benjamini_hochberg_and_context(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("rso", "scripts/creativity/run_second_order.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q = module.benjamini_hochberg(pd.Series([0.01, 0.04, 0.03, 0.5]))
    assert q.round(4).tolist() == [0.04, 0.0533, 0.0533, 0.5]
    if EVENTS.exists():
        from soccer_ml.creativity.context import match_context

        tables = match_context(EVENTS, {"home_team": "Barcelona"})
        goals = pd.DataFrame(tables["goals"])
        assert goals["team"].value_counts().to_dict() == {"Barcelona": 3}  # Barcelona 3-0 Elche
        decisions = pd.DataFrame(tables["decisions"])
        assert decisions["players_own"].between(10, 11).all()
        assert (decisions["goals_for"] + decisions["goals_against"]).max() <= 3
