"""Tests for pitch discretization, zone baselines, and zone metrics."""

import math

import pytest
import torch

from soccer_ml.data.pitch_grid import (
    PitchGrid,
    ZonePriorBaseline,
    closest_teammate_zone_scores,
    passer_zones,
)
from soccer_ml.data.schema import STATSBOMB_NODE_LAYOUT, SoccerGraphBatch
from soccer_ml.data.transforms import normalize_pitch_coordinates
from soccer_ml.training.metrics import ZoneMetricAccumulator


@pytest.fixture
def grid() -> PitchGrid:
    return PitchGrid(grid_m=26, grid_n=17)


def test_default_grid_is_26_by_17():
    grid = PitchGrid()
    assert (grid.grid_m, grid.grid_n, grid.num_zones) == (26, 17, 442)


def test_coords_zone_round_trip(grid):
    for zone in range(grid.num_zones):
        x, y = grid.zone_to_coords(zone)
        assert grid.coords_to_zone(x, y) == zone
    centers = grid.zone_centers()
    torch.testing.assert_close(grid.batch_coords_to_zones(centers), torch.arange(grid.num_zones))


def test_boundaries_and_out_of_bounds(grid):
    assert grid.coords_to_zone(0.0, 0.0) == 0
    assert grid.coords_to_zone(120.0, 80.0) == grid.num_zones - 1
    assert grid.coords_to_zone(-5.0, -5.0) == 0
    assert grid.coords_to_zone(130.0, 90.0) == grid.num_zones - 1
    assert grid.coords_to_zone(0.0, 80.0) == grid.grid_n - 1
    assert grid.coords_to_zone(120.0, 0.0) == (grid.grid_m - 1) * grid.grid_n
    coords = torch.tensor([[0.0, 0.0], [120.0, 80.0], [-5.0, 90.0], [60.0, 40.0]])
    expected = [grid.coords_to_zone(float(x), float(y)) for x, y in coords]
    assert grid.batch_coords_to_zones(coords).tolist() == expected


def test_zone_index_layout_matches_row_major_grid(grid):
    # Zone k = bin_x * grid_n + bin_y, so reshaping to (M, N) puts x on dim 0.
    zone = grid.coords_to_zone(119.0, 1.0)
    assert divmod(zone, grid.grid_n) == (grid.grid_m - 1, 0)


def test_distance_units(grid):
    zone = torch.tensor([grid.coords_to_zone(60.0, 40.0)])
    center = grid.batch_zones_to_coords(zone)
    target = center + torch.tensor([[12.0, 0.0]])
    assert grid.compute_distance(zone, target).item() == pytest.approx(12.0)
    assert grid.compute_distance(zone, target, metric=True).item() == pytest.approx(12.0 * 105 / 120)


def test_gaussian_soft_targets_peak_at_true_zone(grid):
    end = torch.tensor([[33.0, 12.0]])
    soft = grid.gaussian_soft_targets(end, sigma=3.0)
    assert soft.shape == (1, grid.num_zones)
    assert soft.sum().item() == pytest.approx(1.0)
    assert soft.argmax().item() == grid.coords_to_zone(33.0, 12.0)


def _graph_batch(positions, teammate, actor_index=0):
    n = len(positions)
    layout = STATSBOMB_NODE_LAYOUT
    node_features = torch.zeros(1, n, layout.dim)
    for index, (x, y) in enumerate(positions):
        nx, ny = normalize_pitch_coordinates(torch.tensor(x), torch.tensor(y), 120.0, 80.0, 10.0)
        node_features[0, index, :2] = torch.stack([nx, ny])
        node_features[0, index, layout.teammate] = float(teammate[index])
    node_features[0, actor_index, layout.actor] = 1.0
    return SoccerGraphBatch(
        node_features=node_features,
        edge_features=torch.zeros(1, n, n, 1),
        graph_features=torch.zeros(1, 7),
        adj_mat=torch.ones(1, n, n),
        player_mask=torch.ones(1, n, dtype=torch.bool),
    )


def test_closest_teammate_zone_ranking(grid):
    positions = [(50.0, 40.0), (70.0, 40.0), (55.0, 45.0), (52.0, 41.0)]
    teammate = [True, True, True, False]  # the nearest player is an opponent
    batch = _graph_batch(positions, teammate)
    scores = closest_teammate_zone_scores(batch, grid)
    ranking = scores.topk(2).indices[0].tolist()
    assert ranking == [grid.coords_to_zone(55.0, 45.0), grid.coords_to_zone(70.0, 40.0)]
    assert passer_zones(batch, grid).item() == grid.coords_to_zone(50.0, 40.0)


def test_zone_prior_conditional_backoff(grid):
    prior = ZonePriorBaseline(grid)
    prior.update(torch.tensor([0, 0, 0, 5]), torch.tensor([10, 10, 11, 200]))
    assert prior.most_frequent_zone == 10
    unseen = 5 * grid.grid_n + 2  # bin (5, 2); +10 y-bins stays on the pitch
    log_probs = prior.conditional_scores(torch.tensor([0, 5, unseen]))
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(3))
    assert log_probs[0].argmax().item() == 10
    assert log_probs[1].argmax().item() == 200
    # An unseen passer zone backs off to the pass-vector prior: the observed
    # 0 -> 10 displacement (+10 y-bins), shifted to the unseen passer zone.
    assert log_probs[2].argmax().item() == unseen + 10


def test_pass_vector_prior_is_translation_equivariant(grid):
    prior = ZonePriorBaseline(grid)
    start = grid.coords_to_zone(30.0, 40.0)
    prior.update(torch.tensor([start]), torch.tensor([start + 2 * grid.grid_n]))  # two bins forward
    other = grid.coords_to_zone(70.0, 20.0)
    log_probs = prior.pass_vector_scores(torch.tensor([start, other]))
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(2))
    assert log_probs.argmax(dim=-1).tolist() == [start + 2 * grid.grid_n, other + 2 * grid.grid_n]


def test_zone_metric_accumulator(grid):
    accumulator = ZoneMetricAccumulator(grid)
    target = torch.tensor([grid.coords_to_zone(30.0, 30.0), grid.coords_to_zone(90.0, 60.0)])
    end = torch.tensor([[30.0, 30.0], [90.0, 60.0]])
    scores = torch.full((2, grid.num_zones), -10.0)
    scores[0, target[0]] = 0.0  # exact hit
    scores[1, target[1] + 1] = 0.0  # neighbouring zone (one bin along y)
    scores[1, target[1]] = -1.0  # second choice
    log_probs = torch.log_softmax(scores, dim=-1)
    accumulator.update(log_probs, target, end, log_probs=log_probs)
    metrics = accumulator.compute()
    assert metrics["top_1_acc"] == pytest.approx(0.5)
    assert metrics["top_3_acc"] == pytest.approx(1.0)
    assert metrics["within_1_zone_acc"] == pytest.approx(1.0)
    assert metrics["nll"] > 0
    assert metrics["median_dist_yd"] < math.hypot(grid.cell_length, grid.cell_width)


def test_teammate_kernel_baseline_is_normalized_and_prefers_close_teammates(grid):
    from soccer_ml.data.pitch_grid import TeammateKernelBaseline

    positions = [(50.0, 40.0), (56.0, 40.0), (90.0, 70.0), (52.0, 41.0)]
    batch = _graph_batch(positions, [True, True, True, False])
    batch.target_zone_label = torch.tensor([grid.coords_to_zone(56.0, 40.0)])
    kernel = TeammateKernelBaseline(grid)
    kernel.update(batch)
    tau, sigma, eps = kernel.finalize()
    log_probs = kernel.log_probs(batch)
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(1), atol=1e-5)
    assert log_probs.argmax().item() == grid.coords_to_zone(56.0, 40.0)  # nearest teammate, not the opponent


def test_receiver_accuracy_metric(grid):
    from soccer_ml.data.pitch_grid import receiver_candidates

    positions = [(50.0, 40.0), (70.0, 40.0), (55.0, 60.0), (52.0, 41.0)]
    batch = _graph_batch(positions, [True, True, True, False])
    batch.receiver_label = torch.tensor([2])
    receivers = receiver_candidates(batch, grid)
    scores = torch.zeros(1, grid.num_zones)
    scores[0, grid.coords_to_zone(57.0, 63.0)] = 1.0  # near teammate 2
    accumulator = ZoneMetricAccumulator(grid)
    accumulator.update(scores, torch.tensor([0]), torch.tensor([[55.0, 60.0]]), receivers=receivers)
    assert accumulator.compute()["receiver_acc"] == 1.0


def test_probabilistic_binary_metrics_match_hand_computation():
    from soccer_ml.training.metrics import probabilistic_binary_metrics

    p = torch.tensor([0.1, 0.4, 0.35, 0.8, 0.4])
    y = torch.tensor([0, 0, 1, 1, 1])
    metrics = probabilistic_binary_metrics(p, y)
    # Positive/negative pairs: (0.35 vs 0.1) win, (0.35 vs 0.4) loss, (0.8 vs both) wins,
    # (0.4 vs 0.1) win, (0.4 vs 0.4) tie -> (1 + 0 + 2 + 1 + 0.5) / 6.
    assert metrics["auc"] == pytest.approx(4.5 / 6)
    assert metrics["brier"] == pytest.approx(float(((p - y) ** 2).mean()))
    assert metrics["logloss"] == pytest.approx(float(torch.nn.functional.binary_cross_entropy(p, y.float())))
