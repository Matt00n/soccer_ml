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
from soccer_ml.data.schema import SoccerGraphBatch
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
    node_features = torch.zeros(1, n, 9)
    for index, (x, y) in enumerate(positions):
        nx, ny = normalize_pitch_coordinates(torch.tensor(x), torch.tensor(y), 120.0, 80.0, 10.0)
        node_features[0, index, :2] = torch.stack([nx, ny])
        node_features[0, index, 7] = float(teammate[index])
    node_features[0, actor_index, 6] = 1.0
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
    accumulator.update(torch.log_softmax(scores, dim=-1), target, end, scores_are_log_probs=True)
    metrics = accumulator.compute()
    assert metrics["top_1_acc"] == pytest.approx(0.5)
    assert metrics["top_3_acc"] == pytest.approx(1.0)
    assert metrics["within_1_zone_acc"] == pytest.approx(1.0)
    assert metrics["nll"] > 0
    assert metrics["median_dist_yd"] < math.hypot(grid.cell_length, grid.cell_width)
