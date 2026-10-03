"""Tests for value targets, value models and their training losses."""

import dataclasses
import itertools
import json
import random
from pathlib import Path

import pytest
import torch

from soccer_ml.data.next_action_dataset import (
    ACTION_TYPES,
    REWARDS,
    VALUE_LABELS,
    NextActionDataset,
    TransitionDataset,
    collate_next_action,
    collate_transitions,
)
from soccer_ml.data.schema import STATSBOMB_NODE_LAYOUT
from soccer_ml.data.temporal_receiver_dataset import transform_event_numeric
from soccer_ml.data.transforms import D2GroupTransform, flip_grid
from soccer_ml.models.next_action import NextActionModel, action_log_likelihood
from soccer_ml.models.value import (
    PASS_HEIGHTS,
    ValueModel,
    action_space_size,
    gather_action,
    observed_action_index,
    policy_action_log_probs,
    swap_components,
)
from soccer_ml.training.value_trainer import (
    ValueConfig,
    ValueLearner,
    bootstrap,
    expectile_loss,
    quantile_huber_loss,
    validate_configuration,
)

THREE_SIXTY = Path("data_toy_receiver/three_sixty_3764440.json")
EVENTS = Path("data_toy_receiver/events_3764440.json")
pytestmark = pytest.mark.skipif(not THREE_SIXTY.exists(), reason="toy StatsBomb data not available")
Z = 26 * 17


@pytest.fixture(scope="module")
def dataset() -> NextActionDataset:
    rng = random.Random(0)
    ids = [event["id"] for event in json.loads(EVENTS.read_text())]
    labels = {i: (float(rng.random() < 0.02), float(rng.random() < 0.01), rng.random() * 0.1, rng.random() * 0.05)
              for i in ids}
    return NextActionDataset(THREE_SIXTY, EVENTS, value_labels=labels)


@pytest.fixture(scope="module")
def batches(dataset):
    pairs = TransitionDataset(dataset)
    by_type = {}
    for index in range(len(dataset)):
        by_type.setdefault(int(dataset[index].action_type), []).append(index)
    indices = [i for rows in by_type.values() for i in rows[:3]]
    return collate_transitions([pairs[i] for i in indices])


def _policy():
    return NextActionModel(hidden_dim=32, num_layers=2, nb_heads=4, temporal_layers=1, dropout=0.0)


def test_transitions_and_rewards(dataset):
    events = sorted(json.loads(EVENTS.read_text()), key=lambda event: event["index"])
    total_xg = sum(e["shot"].get("statsbomb_xg", 0.0) for e in events if e["type"]["name"] == "Shot")
    rewards = torch.stack([dataset[i].reward for i in range(len(dataset))])
    collected = float(rewards[:, REWARDS.index("our_xg")].sum() + rewards[:, REWARDS.index("their_xg")].sum())
    # Reward windows partition the events after the first decision of each period.
    assert 0.8 * total_xg <= collected <= total_xg + 1e-4
    for index in range(len(dataset)):
        sample = dataset[index]
        following = int(sample.next_index)
        if following < 0:
            assert bool(sample.done)
            continue
        current, nxt = sample.metadata, dataset[following].metadata
        assert nxt["anchor_position"] > current["anchor_position"] and nxt["period"] == current["period"]
        assert bool(sample.next_flip) == (nxt["team_id"] != current["team_id"])
        if ACTION_TYPES[int(sample.action_type)] == "shot":
            xg = events[current["target_position"]]["shot"].get("statsbomb_xg", 0.0)
            assert float(sample.reward[REWARDS.index("our_xg")]) >= xg - 1e-6


def test_flat_action_space_matches_policy_likelihood(batches):
    batch, _ = batches
    policy = _policy().eval()
    with torch.no_grad():
        outputs = policy(batch)
    log_probs = policy_action_log_probs(outputs)
    assert log_probs.shape == (batch.batch_size, action_space_size(Z))
    torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones(batch.batch_size), atol=1e-4, rtol=1e-4)
    observed = gather_action(log_probs, observed_action_index(batch, Z))
    nll = action_log_likelihood(outputs, batch)
    has_height = nll["has_height"] | (batch.action_type != ACTION_TYPES.index("pass"))
    torch.testing.assert_close(observed[has_height], -nll["joint_nll"][has_height], atol=1e-3, rtol=1e-3)


def test_bootstrap_swaps_and_terminates(batches):
    batch, _ = batches
    values = torch.rand(batch.batch_size, 2)
    result = bootstrap(values, batch)
    for i in range(batch.batch_size):
        expected = torch.zeros(2) if batch.done[i] else (values[i].flip(0) if batch.next_flip[i] else values[i])
        torch.testing.assert_close(result[i], expected)
    torch.testing.assert_close(swap_components(values, torch.ones(batch.batch_size, dtype=torch.bool)), values.flip(1))


def test_expectile_and_quantile_losses():
    diff = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    torch.testing.assert_close(expectile_loss(diff, 0.5), torch.tensor([1.0, 1.0]))
    loss = expectile_loss(diff, 0.9)  # score up-weights positive errors, concede negative ones
    torch.testing.assert_close(loss, torch.tensor([0.9 + 0.1, 0.1 + 0.9]))
    # The quantile loss is minimized at the empirical quantiles.
    samples = torch.linspace(0, 1, 101).view(1, 1, -1)
    good = torch.tensor([[[0.1, 0.3, 0.5, 0.7, 0.9]]])
    worse = good + 0.1
    assert quantile_huber_loss(good, samples, 0.01) < quantile_huber_loss(worse, samples, 0.01)


VALID = [
    (target, head, distribution, training)
    for target, head, distribution, training in itertools.product(
        ("mc", "sarsa", "expected_sarsa", "iql"), ("direct", "decomposed"), ("expectation", "quantile"), ("warm", "joint")
    )
    if distribution == "expectation" or (head == "direct" and target in ("mc", "sarsa"))
]


def test_unsupported_combinations_raise():
    for target, head, distribution in (
        ("mc", "decomposed", "quantile"), ("expected_sarsa", "direct", "quantile"), ("iql", "direct", "quantile")
    ):
        with pytest.raises(ValueError):
            validate_configuration(ValueConfig(target=target, head=head, distribution=distribution))
    assert len(VALID) == 20


@pytest.mark.parametrize("target,head,distribution,training", VALID)
def test_value_configurations_train(batches, target, head, distribution, training):
    batch, next_batch = batches
    torch.manual_seed(0)
    model = ValueModel(_policy(), head=head, distribution=distribution, num_quantiles=5,
                       state_value=target == "iql", joint=training == "joint")
    config = ValueConfig(target=target, head=head, distribution=distribution, training=training)
    learner = ValueLearner(model, config, policy=_policy().eval() if training == "warm" else None)
    targets = learner.targets(batch, next_batch)
    assert torch.isfinite(targets["y"]).all()
    outputs = model(batch)
    assert outputs["q"].shape == (batch.batch_size, 2, action_space_size(Z))
    terms = learner.loss(outputs, batch, targets)
    assert torch.isfinite(terms["total"])
    terms["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    learner.update_target()


def _reflect(batch, action):
    visible = flip_grid(batch.visible_zones.view(-1, 26, 17), action).reshape(batch.batch_size, -1)
    return dataclasses.replace(
        batch,
        node_features=D2GroupTransform.from_layout(STATSBOMB_NODE_LAYOUT).transform_node_features(
            batch.node_features, action
        ),
        event_numeric=transform_event_numeric(batch.event_numeric, action),
        visible_zones=visible,
    )


@pytest.mark.parametrize("head", ["direct", "decomposed"])
def test_value_surfaces_are_reflection_equivariant(batches, head):
    batch, _ = batches
    model = ValueModel(_policy(), head=head).eval()
    heights = len(PASS_HEIGHTS)
    with torch.no_grad():
        q = model(batch)["q"]
        reflected = model(_reflect(batch, 2))["q"]

    def grids(values):  # passes and carries as (B, 2, heights + 1, M, N), scalars separately
        located = values[..., : (heights + 1) * Z].reshape(values.shape[0], 2, heights + 1, 26, 17)
        return located, values[..., (heights + 1) * Z :]

    located, scalars = grids(q)
    located_reflected, scalars_reflected = grids(reflected)
    torch.testing.assert_close(located_reflected, flip_grid(located, 2), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(scalars_reflected, scalars, atol=1e-5, rtol=1e-4)
