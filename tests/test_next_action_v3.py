"""Tests for the optional v3 inputs: actor role, game state, velocities, distance edges."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from soccer_ml.data.next_action_dataset import (
    ACTOR_ROLES,
    GRAPH_DIM,
    MATCH_SLACK,
    MAX_SPEED,
    VELOCITY_MIN_GAP,
    NextActionDataset,
    collate_next_action,
    graph_feature_names,
    match_velocities,
    node_layout_for,
)
from soccer_ml.data.schema import STATSBOMB_NODE_LAYOUT, STATSBOMB_VELOCITY_NODE_LAYOUT
from soccer_ml.data.temporal_receiver_dataset import transform_event_numeric
from soccer_ml.data.transforms import D2GroupTransform, flip_grid
from soccer_ml.models.next_action import NextActionModel, action_log_likelihood
from soccer_ml.models.value import ValueModel

THREE_SIXTY = Path("data_toy_receiver/three_sixty_3764440.json")
EVENTS = Path("data_toy_receiver/events_3764440.json")
V2_CHECKPOINT = Path("checkpoints/next_action_v2/best_model.pt")
pytestmark = pytest.mark.skipif(not THREE_SIXTY.exists(), reason="toy StatsBomb data not available")


@pytest.fixture(scope="module")
def v3_dataset() -> NextActionDataset:
    return NextActionDataset(THREE_SIXTY, EVENTS, actor_role=True, game_state=True, velocities=True)


@pytest.fixture(scope="module")
def v3_batch(v3_dataset):
    return next(iter(DataLoader(v3_dataset, batch_size=24, shuffle=True, generator=torch.Generator().manual_seed(1),
                                collate_fn=collate_next_action)))


def test_defaults_reproduce_v2_inputs():
    dataset = NextActionDataset(THREE_SIXTY, EVENTS, max_samples=50)
    sample = dataset[0]
    assert sample.graph_features.shape == (GRAPH_DIM,)
    assert sample.node_features.shape[-1] == STATSBOMB_NODE_LAYOUT.dim
    if V2_CHECKPOINT.exists():  # an existing checkpoint must keep working with the default dataset
        checkpoint = torch.load(V2_CHECKPOINT, map_location="cpu", weights_only=False)
        model = NextActionModel(**checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["model_state_dict"])
        batch = collate_next_action([dataset[i] for i in range(8)])
        with torch.no_grad():
            nll = action_log_likelihood(model.eval()(batch), batch)["joint_nll"]
        assert torch.isfinite(nll).all()


def test_match_velocities_on_a_known_shift():
    previous = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [80.0, 40.0]])
    previous_team = np.array([True, True, False, False])
    current = previous + np.array([[2.0, 0.0], [0.0, -1.0], [1.0, 1.0], [0.0, 0.0]])
    current[3] = [10.0, 70.0]  # jumped far beyond MAX_SPEED: must stay unmatched
    current_team = previous_team.copy()
    velocity, matched = match_velocities(current, current_team, previous, previous_team, dt=0.5)
    assert matched.tolist() == [True, True, True, False]
    np.testing.assert_allclose(velocity[:3], [[4.0, 0.0], [0.0, -2.0], [2.0, 2.0]])
    assert np.linalg.norm(current[3] - previous[3]) > MAX_SPEED * 0.5 + MATCH_SLACK


def test_feature_contents(v3_dataset):
    names = v3_dataset.graph_feature_names
    assert names == graph_feature_names(True, True)
    graph = v3_dataset._tensors["graph_features"]
    roles = graph[:, [names.index(f"role_{role}") for role in ACTOR_ROLES]]
    assert torch.all(roles.sum(-1) == 1)
    events = {event["id"]: event for event in json.loads(EVENTS.read_text())}
    # Barcelona won 3-0: their goal difference is never negative, Elche's never positive.
    difference = graph[:, names.index("goal_difference_over_3")]
    for index in range(len(v3_dataset)):
        team = events[v3_dataset[index].metadata["event_uuid"]]["team"]["name"]
        assert (difference[index] >= 0) if team == "Barcelona" else (difference[index] <= 0)
    assert difference.max() == 1.0 and difference.min() == -1.0
    nodes, mask = v3_dataset._tensors["node_features"], v3_dataset._tensors["player_mask"]
    known = nodes[..., STATSBOMB_VELOCITY_NODE_LAYOUT.dim - 1][mask]
    assert known.mean() > 0.6
    layout = STATSBOMB_VELOCITY_NODE_LAYOUT
    speed = torch.sqrt((nodes[..., layout.vx] * 12) ** 2 + (nodes[..., layout.vy] * 8) ** 2)[mask][known > 0.5]
    # Matches are limited to MAX_SPEED * dt + MATCH_SLACK, so speeds stay below MAX_SPEED + MATCH_SLACK / dt.
    assert speed.median() < 6.0 and speed.max() <= MAX_SPEED + MATCH_SLACK / VELOCITY_MIN_GAP


def _v3_model(**overrides):
    kwargs = dict(
        in_node_dim=STATSBOMB_VELOCITY_NODE_LAYOUT.dim,
        in_graph_dim=len(graph_feature_names(True, True)),
        node_layout=STATSBOMB_VELOCITY_NODE_LAYOUT.name,
        edge_distance=True,
        hidden_dim=32, num_layers=2, nb_heads=4, temporal_layers=1, dropout=0.0,
    )
    kwargs.update(overrides)
    return NextActionModel(**kwargs)


def _reflect(batch, action):
    visible = flip_grid(batch.visible_zones.view(-1, 26, 17), action).reshape(batch.batch_size, -1)
    return dataclasses.replace(
        batch,
        node_features=D2GroupTransform.from_layout(node_layout_for(True)).transform_node_features(
            batch.node_features, action
        ),
        event_numeric=transform_event_numeric(batch.event_numeric, action),
        visible_zones=visible,
    )


@pytest.mark.parametrize("symmetry,actions", [("y_flip", (2,)), ("d2", (1, 2, 3))])
def test_v3_model_is_reflection_equivariant(v3_batch, symmetry, actions):
    model = _v3_model(symmetry=symmetry).eval()
    with torch.no_grad():
        outputs = model(v3_batch)
        for action in actions:
            reflected = model(_reflect(v3_batch, action))
            torch.testing.assert_close(reflected["type_logits"], outputs["type_logits"], atol=1e-4, rtol=1e-4)
            for key in ("zone_logits", "height_logits"):
                expected = flip_grid(outputs[key].view(*outputs[key].shape[:2], 26, 17), action).flatten(2)
                torch.testing.assert_close(reflected[key], expected, atol=1e-4, rtol=1e-4)


def test_distance_edges_and_value_model_train(v3_batch):
    model = _v3_model()
    distance = model.pairwise_distance(v3_batch)
    assert torch.allclose(distance, distance.transpose(1, 2)) and torch.all(distance[v3_batch.adj_mat == 0] == 0)
    value = ValueModel(model, head="decomposed")
    outputs = value(v3_batch)
    loss = outputs["q"].mean() + action_log_likelihood(model(v3_batch), v3_batch)["joint_nll"].mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in value.parameters() if p.grad is not None)
