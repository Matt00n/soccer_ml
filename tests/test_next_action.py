"""Tests for next-action samples, model, likelihood and baselines."""

import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from soccer_ml.data.next_action_dataset import (
    ACTION_TYPES,
    DECISION_CONTEXTS,
    GAIN_EVENTS,
    GRAPH_FEATURES,
    IGNORE_INDEX,
    LOCATED_ACTIONS,
    NextActionDataset,
    collate_next_action,
)
from soccer_ml.data.pitch_grid import PitchGrid
from soccer_ml.data.schema import STATSBOMB_NODE_LAYOUT
from soccer_ml.data.temporal_receiver_dataset import transform_event_numeric
from soccer_ml.data.transforms import D2GroupTransform, flip_grid
from soccer_ml.models.next_action import NextActionModel, action_log_likelihood
from soccer_ml.training.next_action_trainer import NextActionMetrics, fit_next_action_priors

THREE_SIXTY = Path("data_toy_receiver/three_sixty_3764440.json")
EVENTS = Path("data_toy_receiver/events_3764440.json")
pytestmark = pytest.mark.skipif(not THREE_SIXTY.exists(), reason="toy StatsBomb data not available")
GRID = PitchGrid(26, 17)


@pytest.fixture(scope="module")
def dataset() -> NextActionDataset:
    return NextActionDataset(THREE_SIXTY, EVENTS)


@pytest.fixture(scope="module")
def batch(dataset):
    # A mix of every action type.
    by_type = {}
    for index in range(len(dataset)):
        by_type.setdefault(int(dataset[index].action_type), []).append(index)
    indices = [i for rows in by_type.values() for i in rows[:3]]
    return collate_next_action([dataset[i] for i in indices])


def test_samples_and_labels(dataset):
    events = {event["id"]: event for event in json.loads(EVENTS.read_text())}
    seen = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        action = ACTION_TYPES[int(sample.action_type)]
        seen.add(action)
        event = events[sample.metadata["event_uuid"]]
        if action in LOCATED_ACTIONS:
            end = event[action]["end_location"]
            assert int(sample.target_zone) == GRID.coords_to_zone(end[0], end[1])
            if action == "carry":
                assert math.dist(event["location"][:2], end[:2]) >= GRID.cell_length
        else:
            assert int(sample.target_zone) == IGNORE_INDEX
        assert (int(sample.pass_height) != IGNORE_INDEX) == (action == "pass" and "height" in event["pass"])
    assert seen == set(ACTION_TYPES)
    # Failed passes are included.
    assert any("outcome" in events[dataset[i].metadata["event_uuid"]].get("pass", {}) for i in range(len(dataset)))


def test_graph_features_do_not_reveal_the_action(dataset):
    graph = torch.stack([dataset[i].graph_features for i in range(len(dataset))])
    actions = torch.stack([dataset[i].action_type for i in range(len(dataset))])
    restart = [GRAPH_FEATURES.index(name) for name in GRAPH_FEATURES if name.startswith("restart_")]
    open_play = GRAPH_FEATURES.index("open_play")
    torch.testing.assert_close(graph[:, restart].sum(-1) + graph[:, open_play], torch.ones(len(dataset)))
    # Open-play situations contain every action type, so no feature encodes the type itself.
    assert set(actions[graph[:, open_play] == 1].tolist()) == set(range(len(ACTION_TYPES)))


def test_history_excludes_opponent_reactions(dataset):
    events = sorted(json.loads(EVENTS.read_text()), key=lambda event: event["index"])
    by_id = {event["id"]: position for position, event in enumerate(events)}
    checked = 0
    for index in range(len(dataset)):
        target = events[by_id[dataset[index].metadata["event_uuid"]]]
        history = dataset._history(events, by_id[dataset[index].metadata["anchor_uuid"]], target)
        for past in history:
            assert not (
                past["id"] in target.get("related_events", [])
                and past["team"]["id"] != target["team"]["id"]
                and past["timestamp"] == target["timestamp"]
            )
        checked += target["type"]["name"] == "Dribble"
        if target["type"]["name"] == "Dribble":
            assert all(past["type"]["name"] != "Dribbled Past" for past in history[-2:])
    assert checked > 0


def test_decisions_are_anchored_at_ball_gain(dataset):
    """No sample may read its type off StatsBomb's carry conventions."""
    events = sorted(json.loads(EVENTS.read_text()), key=lambda event: event["index"])
    by_id = {event["id"]: position for position, event in enumerate(events)}
    carry = ACTION_TYPES.index("carry")
    contexts = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        context = DECISION_CONTEXTS[int(sample.decision_context)]
        contexts.add(context)
        target = events[by_id[sample.metadata["event_uuid"]]]
        anchor_position = by_id[sample.metadata["anchor_uuid"]]
        anchor = events[anchor_position]
        if context == "after_carry":
            assert not sample.type_mask[carry] and int(sample.action_type) != carry
            assert anchor["id"] == target["id"]
        else:
            assert sample.type_mask.all()
            own_before = [
                event for event in events[:anchor_position]
                if event.get("player", {}).get("id") == target["player"]["id"]
                and event.get("possession") == target["possession"]
            ]
            assert not own_before or own_before[-1]["type"]["name"] != "Carry"
        if context == "spell_start":
            assert anchor["type"]["name"] in GAIN_EVENTS
            history = dataset._history(events, anchor_position, target)
            assert all(event["index"] < anchor["index"] for event in history)
    assert contexts == set(DECISION_CONTEXTS)


def test_collate_edges_and_visibility(batch):
    mask = batch.player_mask
    assert torch.equal(batch.adj_mat, (mask.unsqueeze(-1) & mask.unsqueeze(-2)).float())
    torch.testing.assert_close(batch.edge_features, batch.edge_features.transpose(1, 2))
    assert batch.edge_features[~mask].abs().sum() == 0
    assert batch.visible_zones.any(dim=-1).all() and (~batch.visible_zones).any(dim=-1).all()


def _model(**overrides):
    kwargs = dict(hidden_dim=32, num_layers=2, nb_heads=4, temporal_layers=1, dropout=0.0)
    kwargs.update(overrides)
    return NextActionModel(**kwargs)


@pytest.mark.parametrize(
    "overrides", [{}, {"use_history": False}, {"use_visibility": False}, {"use_mlp": True}, {"symmetry": "none"}]
)
def test_forward_backward(batch, overrides):
    model = _model(**overrides)
    outputs = model(batch)
    assert outputs["type_logits"].shape == (batch.batch_size, len(ACTION_TYPES))
    assert outputs["zone_logits"].shape == (batch.batch_size, len(LOCATED_ACTIONS), GRID.num_zones)
    assert outputs["height_logits"].shape == (batch.batch_size, 3, GRID.num_zones)
    nll = action_log_likelihood(outputs, batch)
    nll["joint_nll"].mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def _reflect(batch, action):
    visible = flip_grid(batch.visible_zones.view(-1, GRID.grid_m, GRID.grid_n), action).reshape(batch.batch_size, -1)
    return dataclasses.replace(
        batch,
        node_features=D2GroupTransform.from_layout(STATSBOMB_NODE_LAYOUT).transform_node_features(
            batch.node_features, action
        ),
        event_numeric=transform_event_numeric(batch.event_numeric, action),
        visible_zones=visible,
    )


@pytest.mark.parametrize("symmetry,actions", [("y_flip", (2,)), ("d2", (1, 2, 3))])
def test_heads_are_reflection_equivariant(batch, symmetry, actions):
    model = _model(symmetry=symmetry).eval()
    with torch.no_grad():
        outputs = model(batch)
        for action in actions:
            reflected = model(_reflect(batch, action))
            torch.testing.assert_close(reflected["type_logits"], outputs["type_logits"], atol=1e-4, rtol=1e-4)
            for key in ("zone_logits", "height_logits"):
                expected = flip_grid(outputs[key].view(*outputs[key].shape[:2], GRID.grid_m, GRID.grid_n), action)
                torch.testing.assert_close(reflected[key], expected.flatten(2), atol=1e-4, rtol=1e-4)


def test_impossible_actions_get_no_probability(batch):
    masked = batch.type_mask.clone()
    masked[: batch.batch_size // 2, ACTION_TYPES.index("carry")] = False
    masked_batch = dataclasses.replace(batch, type_mask=masked)
    with torch.no_grad():
        probabilities = _model().eval()(masked_batch)["type_logits"].softmax(dim=-1)
    assert torch.all(probabilities[~masked] < 1e-6)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(batch.batch_size))


def test_decision_slot_placeholders_are_ignored(batch):
    model = _model().eval()
    altered = dataclasses.replace(
        batch,
        event_type=batch.event_type.clone(),
        event_outcome=batch.event_outcome.clone(),
        event_subtype=batch.event_subtype.clone(),
        event_numeric=batch.event_numeric.clone(),
    )
    altered.event_type[:, -1] = 30
    altered.event_outcome[:, -1] = 2
    altered.event_subtype[:, -1] = 3
    altered.event_numeric[:, -1] = 5.0
    with torch.no_grad():
        for key, value in model(batch).items():
            torch.testing.assert_close(model(altered)[key], value)


def test_log_likelihood_bookkeeping(batch):
    uniform = {
        "type_logits": torch.zeros(batch.batch_size, len(ACTION_TYPES)),
        "zone_logits": torch.zeros(batch.batch_size, len(LOCATED_ACTIONS), GRID.num_zones),
        "height_logits": torch.zeros(batch.batch_size, 3, GRID.num_zones),
    }
    nll = action_log_likelihood(uniform, batch)
    torch.testing.assert_close(nll["type_nll"], torch.full((batch.batch_size,), math.log(len(ACTION_TYPES))))
    expected_zone = torch.where(nll["located"], math.log(GRID.num_zones), 0.0)
    torch.testing.assert_close(nll["zone_nll"], expected_zone)
    torch.testing.assert_close(nll["height_nll"], torch.where(nll["has_height"], math.log(3), 0.0))
    torch.testing.assert_close(nll["joint_nll"], nll["type_nll"] + nll["zone_nll"] + nll["height_nll"])
    assert nll["has_height"].sum() > 0 and (~nll["located"]).sum() > 0


def test_priors_are_normalized_and_metrics_run(dataset, batch):
    loader = DataLoader(dataset, batch_size=256, collate_fn=collate_next_action)
    priors = fit_next_action_priors(loader, GRID)
    for kind in ("marginal", "zone_prior"):
        outputs = priors.predict(batch, kind)
        for key, dim in (("type_logits", -1), ("zone_logits", -1), ("height_logits", 1)):
            torch.testing.assert_close(outputs[key].exp().sum(dim), torch.ones_like(outputs[key].sum(dim)))
        metrics = NextActionMetrics(GRID)
        metrics.update(outputs, batch, action_log_likelihood(outputs, batch))
        result = metrics.compute()
        assert result["joint_nll"] > 0 and "pass_nll" in result and "height_acc" in result
        shares = [result[f"share_{name}"] for name in DECISION_CONTEXTS if f"share_{name}" in result]
        assert abs(sum(shares) - 1.0) < 1e-6 and "type_acc_spell_start" in result
        mask = batch.type_mask
        assert torch.all(outputs["type_logits"][~mask].exp() < 1e-6)
