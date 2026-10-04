"""Tests for λ-return targets and the C51 distributional value head."""

import dataclasses
import json
import random
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from soccer_ml.data.next_action_dataset import (
    NextActionDataset,
    TransitionDataset,
    collate_transitions,
    lambda_returns,
)
from soccer_ml.data.schema import STATSBOMB_NODE_LAYOUT
from soccer_ml.data.temporal_receiver_dataset import transform_event_numeric
from soccer_ml.data.transforms import D2GroupTransform, flip_grid
from soccer_ml.models.next_action import NextActionModel
from soccer_ml.models.value import (
    PASS_HEIGHTS,
    ValueModel,
    categorical_support,
    distribution_logits,
    project_categorical,
)
from soccer_ml.training.value_trainer import ValueConfig, ValueEvaluation, ValueLearner, validate_configuration

THREE_SIXTY = Path("data_toy_receiver/three_sixty_3764440.json")
EVENTS = Path("data_toy_receiver/events_3764440.json")
Z = 26 * 17


class _Chain:
    """Minimal stand-in for a NextActionDataset: decisions 0 -> 1 -> 2 (period end), 3 alone."""

    def __init__(self):
        rewards = torch.tensor([[0.1, 0.0, 0, 0], [0.0, 0.2, 0, 0], [0.3, 0.0, 0, 0], [0.05, 0.0, 0, 0]])
        self._tensors = {
            "reward": rewards,
            "next_index": torch.tensor([1, 2, -1, -1]),
            "next_flip": torch.tensor([False, True, False, False]),
            "done": torch.tensor([False, False, True, True]),
        }
        self._metadata = [{"anchor_position": p} for p in (10, 20, 30, 40)]

    def __len__(self):
        return 4


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_lambda_returns_match_hand_computation(lam):
    chain, gamma = _Chain(), 0.9
    bootstrap = torch.tensor([[0.5, 0.1], [0.4, 0.2], [0.3, 0.3], [0.2, 0.2]])
    returns = lambda_returns(chain, gamma, lam, bootstrap)
    g2 = torch.tensor([0.3, 0.0])
    g1 = torch.tensor([0.0, 0.2]) + gamma * ((1 - lam) * bootstrap[2] + lam * g2).flip(0)  # flip: other team next
    g0 = torch.tensor([0.1, 0.0]) + gamma * ((1 - lam) * bootstrap[1] + lam * g1)
    torch.testing.assert_close(returns[2], g2)
    torch.testing.assert_close(returns[1], g1)
    torch.testing.assert_close(returns[0], g0)
    torch.testing.assert_close(returns[3], torch.tensor([0.05, 0.0]))


def test_lambda_one_needs_no_bootstrap_and_lambda_below_one_does():
    lambda_returns(_Chain(), 0.9, 1.0)
    with pytest.raises(ValueError):
        lambda_returns(_Chain(), 0.9, 0.8)


def test_categorical_projection():
    support = categorical_support(51, 1.0)
    assert torch.allclose(support, -support.flip(0)) and support[25] == 0 and torch.all(support.diff() > 0)
    probabilities = torch.softmax(torch.randn(4, 51), dim=-1)
    # Shifting by an exact atom offset at the centre maps atoms onto atoms.
    projected = project_categorical(probabilities, support.unsqueeze(0).expand(4, -1).clone(), support)
    torch.testing.assert_close(projected, probabilities)
    # Mass is conserved and the mean is preserved while inside the support.
    shifted = 0.05 + 0.9 * support.unsqueeze(0).expand(4, -1)
    projected = project_categorical(probabilities, shifted, support)
    torch.testing.assert_close(projected.sum(-1), torch.ones(4))
    torch.testing.assert_close((projected * support).sum(-1), (probabilities * shifted).sum(-1), atol=1e-5, rtol=1e-5)
    # Mirroring: the other team's net return is ours negated.
    mirrored = project_categorical(probabilities.flip(-1), -shifted.flip(-1), support)
    torch.testing.assert_close(mirrored, projected.flip(-1), atol=1e-6, rtol=1e-6)


pytestmark_data = pytest.mark.skipif(not THREE_SIXTY.exists(), reason="toy StatsBomb data not available")


@pytest.fixture(scope="module")
def dataset() -> NextActionDataset:
    rng = random.Random(0)
    ids = [event["id"] for event in json.loads(EVENTS.read_text())]
    labels = {i: (float(rng.random() < 0.02), float(rng.random() < 0.01), rng.random() * 0.1, rng.random() * 0.05)
              for i in ids}
    return NextActionDataset(THREE_SIXTY, EVENTS, value_labels=labels)


@pytest.fixture(scope="module")
def batches(dataset):
    loader = DataLoader(TransitionDataset(dataset), batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(0), collate_fn=collate_transitions)
    return next(iter(loader))


def _policy():
    return NextActionModel(hidden_dim=32, num_layers=2, nb_heads=4, temporal_layers=1, dropout=0.0)


CONFIGS = [
    ("lambda_return", "decomposed", "expectation", 0.8),
    ("lambda_return", "direct", "expectation", 1.0),
    ("mc", "decomposed", "categorical", 0.8),
    ("expected_sarsa", "decomposed", "categorical", 0.8),
    ("lambda_return", "decomposed", "categorical", 0.8),
]


@pytestmark_data
@pytest.mark.parametrize("target,head,distribution,lam", CONFIGS)
def test_new_configurations_train_and_evaluate(dataset, batches, target, head, distribution, lam):
    batch, next_batch, indices = batches
    torch.manual_seed(0)
    model = ValueModel(_policy(), head=head, distribution=distribution)
    config = ValueConfig(target=target, head=head, distribution=distribution, lam=lam)
    validate_configuration(config)
    learner = ValueLearner(model, config, policy=_policy().eval())
    if target == "lambda_return":
        learner.refresh_lambda_targets(dataset, torch.device("cpu"))
        assert learner.lambda_targets.shape == (len(dataset), 2)
    targets = learner.targets(batch, next_batch, indices)
    if distribution == "categorical":
        torch.testing.assert_close(targets["distribution_target"].sum(-1), torch.ones(batch.batch_size))
    terms = learner.loss(model(batch), batch, targets)
    assert torch.isfinite(terms["total"])
    terms["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    evaluation = ValueEvaluation(model, lambda_returns(dataset, 0.9, 1.0))
    with torch.no_grad():
        evaluation.update(model(batch), batch, indices)
    metrics = evaluation.compute()
    assert "return_selection" in metrics
    if distribution == "categorical":
        assert 0 <= metrics["distribution_crps"] and "distribution_pit_below_0.5" in metrics


@pytestmark_data
def test_categorical_distributions_are_reflection_equivariant(batches):
    batch = batches[0]
    model = ValueModel(_policy(), head="decomposed", distribution="categorical").eval()
    visible = flip_grid(batch.visible_zones.view(-1, 26, 17), 2).reshape(batch.batch_size, -1)
    reflected = dataclasses.replace(
        batch,
        node_features=D2GroupTransform.from_layout(STATSBOMB_NODE_LAYOUT).transform_node_features(batch.node_features, 2),
        event_numeric=transform_event_numeric(batch.event_numeric, 2),
        visible_zones=visible,
    )
    with torch.no_grad():
        logits = distribution_logits(model, model(batch))  # (B, A, K)
        logits_reflected = distribution_logits(model, model(reflected))
    located = (len(PASS_HEIGHTS) + 1) * Z

    def grid(values):
        return values[:, :located].reshape(values.shape[0], len(PASS_HEIGHTS) + 1, 26, 17, -1).movedim(-1, 1)

    torch.testing.assert_close(grid(logits_reflected), flip_grid(grid(logits), 2), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(logits_reflected[:, located:], logits[:, located:], atol=1e-4, rtol=1e-4)


def test_unsupported_lambda_quantile():
    with pytest.raises(ValueError):
        validate_configuration(ValueConfig(target="lambda_return", distribution="quantile"))
