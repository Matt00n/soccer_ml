"""Pitch discretization and spatial grid utilities for pass target location prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from soccer_ml.data.schema import STATSBOMB_NODE_LAYOUT, SoccerGraphBatch

# Default grid: 26 x 17 zones, consistent with prior pass-destination work
# (~4 m cells on a 105 x 68 m pitch; 4.6 x 4.7 yd on the 120 x 80 StatsBomb pitch).
DEFAULT_GRID_M = 26
DEFAULT_GRID_N = 17

# StatsBomb units are yards on a nominal 120 x 80 pitch.  Metric distances are
# reported after rescaling each axis onto a standard 105 x 68 m pitch.
METRIC_PITCH_LENGTH = 105.0
METRIC_PITCH_WIDTH = 68.0


class PitchGrid:
    """Discretizes a soccer pitch into an M x N grid of spatial zones.

    Standard StatsBomb coordinates run from x in [0, pitch_length] (default 120.0)
    and y in [0, pitch_width] (default 80.0).

    The grid divides:
      - x into grid_m bins along pitch length (0 <= bin_x < grid_m)
      - y into grid_n bins along pitch width (0 <= bin_y < grid_n)
    Zone index: k = bin_x * grid_n + bin_y in [0, grid_m * grid_n - 1], so a
    (B, M * N) logit tensor reshapes row-major into a (B, M, N) pitch image.
    Bins are uniform and therefore symmetric under the D2 pitch reflections.
    """

    def __init__(
        self,
        grid_m: int = DEFAULT_GRID_M,
        grid_n: int = DEFAULT_GRID_N,
        pitch_length: float = 120.0,
        pitch_width: float = 80.0,
    ):
        if grid_m <= 0 or grid_n <= 0:
            raise ValueError(f"grid_m and grid_n must be positive, got ({grid_m}, {grid_n})")
        self.grid_m = grid_m
        self.grid_n = grid_n
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.num_zones = grid_m * grid_n

        self.cell_length = pitch_length / grid_m
        self.cell_width = pitch_width / grid_n

    def coords_to_zone(self, x: float, y: float) -> int:
        """Convert continuous pitch coordinates (x, y) to a discrete zone index.

        Out-of-bounds coordinates are clamped onto the nearest border zone.
        """
        bin_x = min(max(int(math.floor(float(x) / self.cell_length)), 0), self.grid_m - 1)
        bin_y = min(max(int(math.floor(float(y) / self.cell_width)), 0), self.grid_n - 1)
        return bin_x * self.grid_n + bin_y

    def batch_coords_to_zones(self, coords: torch.Tensor) -> torch.Tensor:
        """Convert (..., 2) pitch coordinates to (...,) long zone indices."""
        bin_x = torch.floor(coords[..., 0] / self.cell_length).long().clamp(0, self.grid_m - 1)
        bin_y = torch.floor(coords[..., 1] / self.cell_width).long().clamp(0, self.grid_n - 1)
        return bin_x * self.grid_n + bin_y

    def zone_to_coords(self, zone_idx: int) -> Tuple[float, float]:
        """Convert a discrete zone index to its continuous center coordinates (x, y)."""
        zone_idx = min(max(int(zone_idx), 0), self.num_zones - 1)
        bin_x = zone_idx // self.grid_n
        bin_y = zone_idx % self.grid_n

        center_x = (bin_x + 0.5) * self.cell_length
        center_y = (bin_y + 0.5) * self.cell_width
        return center_x, center_y

    def batch_zones_to_coords(self, zones: torch.Tensor) -> torch.Tensor:
        """Convert a tensor of zone indices (...,) to continuous center coordinates (..., 2)."""
        clamped_zones = zones.clamp(0, self.num_zones - 1)
        bin_x = torch.div(clamped_zones, self.grid_n, rounding_mode="floor").float()
        bin_y = (clamped_zones % self.grid_n).float()

        center_x = (bin_x + 0.5) * self.cell_length
        center_y = (bin_y + 0.5) * self.cell_width
        return torch.stack([center_x, center_y], dim=-1)

    def zone_centers(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Return (M * N, 2) centers of all zones in zone-index order."""
        return self.batch_zones_to_coords(torch.arange(self.num_zones, device=device))

    def zone_bins(self, zones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split zone indices into (bin_x, bin_y)."""
        return torch.div(zones, self.grid_n, rounding_mode="floor"), zones % self.grid_n

    def compute_distance(
        self,
        predicted_zones: torch.Tensor,
        target_coords: torch.Tensor,
        metric: bool = False,
    ) -> torch.Tensor:
        """Euclidean distance between predicted zone centers and target coords.

        Args:
            predicted_zones: (B,) long tensor of zone indices.
            target_coords: (B, 2) float tensor of true (x, y) coordinates.
            metric: If True, rescale onto a 105 x 68 m pitch and return meters;
                otherwise return StatsBomb pitch units (yards).

        Returns:
            distances: (B,) float tensor.
        """
        pred_coords = self.batch_zones_to_coords(predicted_zones.to(target_coords.device))
        delta = pred_coords - target_coords
        if metric:
            scale = torch.tensor(
                [METRIC_PITCH_LENGTH / self.pitch_length, METRIC_PITCH_WIDTH / self.pitch_width],
                device=delta.device,
                dtype=delta.dtype,
            )
            delta = delta * scale
        return torch.norm(delta, dim=-1)

    def normalized_to_pitch(self, xy: torch.Tensor, target_scale: float = 10.0) -> torch.Tensor:
        """Invert :func:`normalize_pitch_coordinates` for (..., 2) node coordinates."""
        size = torch.tensor([self.pitch_length, self.pitch_width], device=xy.device, dtype=xy.dtype)
        return (xy / target_scale + 0.5) * size

    def gaussian_soft_targets(self, target_coords: torch.Tensor, sigma: float) -> torch.Tensor:
        """Spatially smoothed zone targets: softmax(-d^2 / (2 sigma^2)) over zone centers.

        Args:
            target_coords: (B, 2) pass end locations in pitch units.
            sigma: Standard deviation in pitch units (yards).

        Returns:
            (B, M * N) target distribution.
        """
        centers = self.zone_centers(device=target_coords.device).to(target_coords.dtype)
        squared = (target_coords.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=-1)
        return torch.softmax(-squared / (2.0 * sigma**2), dim=-1)


# ============================================================================
# Pitch Target Location Baselines
#
# Every baseline returns a (B, M * N) score tensor, so the top-k and distance
# metrics are computed identically for baselines and learned models.
# ============================================================================


def _actor_indices(batch: SoccerGraphBatch) -> torch.Tensor:
    """Index of the passer (actor flag) in every graph."""
    return batch.node_features[..., STATSBOMB_NODE_LAYOUT.actor].argmax(dim=-1)


def passer_zones(batch: SoccerGraphBatch, pitch_grid: PitchGrid, target_scale: float = 10.0) -> torch.Tensor:
    """Zone containing the passer of every target graph, shape (B,)."""
    rows = torch.arange(batch.batch_size, device=batch.node_features.device)
    passer_xy = batch.node_features[rows, _actor_indices(batch), :2]
    return pitch_grid.batch_coords_to_zones(pitch_grid.normalized_to_pitch(passer_xy, target_scale))


def _teammate_candidates(
    batch: SoccerGraphBatch, pitch_grid: PitchGrid, target_scale: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Player positions in pitch units (B, N, 2), visible non-passer teammates (B, N), passer index (B,)."""
    B, N, _ = batch.node_features.shape
    device = batch.node_features.device
    xy = pitch_grid.normalized_to_pitch(batch.node_features[..., :2], target_scale)
    actor = _actor_indices(batch)
    mask = batch.player_mask if batch.player_mask is not None else torch.ones(B, N, dtype=torch.bool, device=device)
    valid = mask & (batch.node_features[..., STATSBOMB_NODE_LAYOUT.teammate] > 0.5)
    valid[torch.arange(B, device=device), actor] = False
    return xy, valid, actor


@dataclass
class ReceiverCandidates:
    """Visible teammates of the passer, for un-xPass-style receiver accuracy.

    ``receiver`` is the visible teammate closest to the true end location
    (the dataset's receiver label); ``nearest`` maps a predicted location to
    the visible teammate closest to it.
    """

    xy: torch.Tensor  # (B, N, 2) pitch units
    candidates: torch.Tensor  # (B, N) bool
    receiver: torch.Tensor  # (B,)

    @property
    def has_candidate(self) -> torch.Tensor:
        return self.candidates.any(dim=-1)

    def nearest(self, locations: torch.Tensor) -> torch.Tensor:
        distances = torch.norm(self.xy - locations.to(self.xy.device).unsqueeze(1), dim=-1)
        return distances.masked_fill(~self.candidates, float("inf")).argmin(dim=-1)


def receiver_candidates(
    batch: SoccerGraphBatch, pitch_grid: PitchGrid, target_scale: float = 10.0
) -> ReceiverCandidates:
    xy, valid, _ = _teammate_candidates(batch, pitch_grid, target_scale)
    return ReceiverCandidates(xy=xy, candidates=valid, receiver=batch.receiver_label.to(xy.device))


def closest_teammate_zone_scores(
    batch: SoccerGraphBatch,
    pitch_grid: PitchGrid,
    target_scale: float = 10.0,
) -> torch.Tensor:
    """Baseline: rank zones by the distance of the teammate they contain to the passer.

    The top-1 prediction is the zone of the closest visible teammate, top-2 the
    zone of the second closest teammate, and so on.  Without any visible
    teammate the passer's own zone is predicted.  Distances are measured in
    pitch units (the normalized coordinates scale x and y differently).
    """
    B = batch.batch_size
    xy, valid, actor = _teammate_candidates(batch, pitch_grid, target_scale)
    rows = torch.arange(B, device=xy.device)
    distances = torch.norm(xy - xy[rows, actor].unsqueeze(1), dim=-1)

    player_zones = pitch_grid.batch_coords_to_zones(xy)
    # Closer teammates get higher scores; invalid players contribute nothing.
    player_scores = torch.where(valid, -distances, torch.full_like(distances, -1e9))
    scores = torch.full((B, pitch_grid.num_zones), -1e9, device=xy.device)
    scores.scatter_reduce_(1, player_zones, player_scores, reduce="amax")

    fallback = ~valid.any(dim=-1)
    if fallback.any():
        own_zone = player_zones[rows, actor]
        scores[fallback, own_zone[fallback]] = 0.0
    return scores


class TeammateKernelBaseline:
    """Probabilistic closest-teammate baseline, so it has an NLL.

    p(z) = (1 - eps) * sum_i w_i K_sigma(z | teammate_i) + eps / Z, with
    w = softmax(-distance_to_passer / tau) over visible teammates (the passer
    itself when none is visible) and K_sigma a Gaussian over zone centers,
    normalized over the grid.  (tau, sigma, eps) are chosen by grid search on
    the training NLL; with three parameters this is not prone to overfitting.
    """

    TAUS = (2.0, 5.0, 10.0, 20.0, 50.0)
    SIGMAS = (2.0, 4.0, 8.0)
    EPSILONS = (0.05, 0.2, 0.5)

    def __init__(self, pitch_grid: PitchGrid, target_scale: float = 10.0):
        self.pitch_grid = pitch_grid
        self.target_scale = target_scale
        self.nll_sums = torch.zeros(len(self.TAUS), len(self.SIGMAS), len(self.EPSILONS), dtype=torch.float64)
        self.count = 0
        self.params: Optional[Tuple[float, float, float]] = None

    def _components(self, batch: SoccerGraphBatch, taus, sigmas) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mixture weights (T, B, N) and per-player zone kernels (S, B, N, Z)."""
        xy, valid, actor = _teammate_candidates(batch, self.pitch_grid, self.target_scale)
        rows = torch.arange(batch.batch_size, device=xy.device)
        no_teammate = ~valid.any(dim=-1)
        valid[no_teammate, actor[no_teammate]] = True
        distances = torch.norm(xy - xy[rows, actor].unsqueeze(1), dim=-1)
        weights = torch.stack(
            [torch.softmax((-distances / tau).masked_fill(~valid, float("-inf")), dim=-1) for tau in taus]
        )
        centers = self.pitch_grid.zone_centers(device=xy.device)
        squared = (xy.unsqueeze(2) - centers.view(1, 1, -1, 2)).pow(2).sum(dim=-1)  # (B, N, Z)
        kernels = torch.stack([torch.softmax(-squared / (2.0 * sigma**2), dim=-1) for sigma in sigmas])
        return weights, kernels

    @torch.no_grad()
    def update(self, batch: SoccerGraphBatch) -> None:
        """Accumulate the training NLL of every (tau, sigma, eps) combination."""
        weights, kernels = self._components(batch, self.TAUS, self.SIGMAS)
        target = batch.target_zone_label.to(kernels.device)
        index = target.view(1, -1, 1, 1).expand(kernels.shape[0], -1, kernels.shape[2], 1)
        kernel_at_target = kernels.gather(-1, index).squeeze(-1)  # (S, B, N)
        mixture = torch.einsum("tbn,sbn->tsb", weights, kernel_at_target)  # (T, S, B)
        z = self.pitch_grid.num_zones
        for e, eps in enumerate(self.EPSILONS):
            nll = -torch.log((1.0 - eps) * mixture + eps / z)
            self.nll_sums[:, :, e] += nll.sum(dim=-1).cpu().double()
        self.count += batch.batch_size

    def finalize(self) -> Tuple[float, float, float]:
        t, rest = divmod(int(self.nll_sums.argmin()), len(self.SIGMAS) * len(self.EPSILONS))
        s, e = divmod(rest, len(self.EPSILONS))
        self.params = (self.TAUS[t], self.SIGMAS[s], self.EPSILONS[e])
        return self.params

    @torch.no_grad()
    def log_probs(self, batch: SoccerGraphBatch) -> torch.Tensor:
        if self.params is None:
            self.finalize()
        tau, sigma, eps = self.params
        weights, kernels = self._components(batch, (tau,), (sigma,))
        mixture = torch.einsum("bn,bnz->bz", weights[0], kernels[0])
        return torch.log((1.0 - eps) * mixture + eps / self.pitch_grid.num_zones)


def random_zone_scores(
    batch: SoccerGraphBatch,
    pitch_grid: PitchGrid,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Baseline: a uniformly random ranking of all zones (its NLL is log(M * N))."""
    scores = torch.rand(batch.batch_size, pitch_grid.num_zones, generator=generator)
    return scores.to(batch.node_features.device)


class ZonePriorBaseline:
    """Empirical target-zone priors fitted on the training split.

    * ``marginal``: the training frequency of every target zone (the "most
      frequent zone" baseline, extended to a full ranking).
    * ``pass_vector``: the distribution of pass displacements in grid cells,
      shifted to the passer's zone and renormalized over on-pitch zones.  It
      ignores *where* on the pitch a pass starts, so it needs little data.
    * ``conditional``: a zone-transition table, the frequency of target zones
      given the passer's zone, smoothed towards the pass-vector prior with
      pseudo-count weight ``backoff_weight``.  A GNN should clearly beat this
      to be learning more than "where passes from here usually go".

    All return log-probabilities, so their log-loss is directly comparable to
    a model's cross-entropy.
    """

    def __init__(self, pitch_grid: PitchGrid, alpha: float = 0.5, backoff_weight: float = 20.0):
        self.pitch_grid = pitch_grid
        self.alpha = alpha
        self.backoff_weight = backoff_weight
        m, n, z = pitch_grid.grid_m, pitch_grid.grid_n, pitch_grid.num_zones
        self.marginal_counts = torch.zeros(z)
        self.transition_counts = torch.zeros(z, z)
        self.offset_counts = torch.zeros(2 * m - 1, 2 * n - 1)
        # offset_index[p, t] indexes the flattened offset histogram for a pass p -> t.
        bin_x, bin_y = pitch_grid.zone_bins(torch.arange(z))
        dx = bin_x.unsqueeze(0) - bin_x.unsqueeze(1) + (m - 1)
        dy = bin_y.unsqueeze(0) - bin_y.unsqueeze(1) + (n - 1)
        self._offset_index = dx * (2 * n - 1) + dy

    def update(self, passer_zone: torch.Tensor, target_zone: torch.Tensor) -> None:
        passer_zone, target_zone = passer_zone.cpu(), target_zone.cpu()
        z = self.pitch_grid.num_zones
        self.marginal_counts += torch.bincount(target_zone, minlength=z).float()
        self.transition_counts += torch.bincount(passer_zone * z + target_zone, minlength=z * z).float().view(z, z)
        offsets = self._offset_index[passer_zone, target_zone]
        self.offset_counts += torch.bincount(offsets, minlength=self.offset_counts.numel()).float().view_as(
            self.offset_counts
        )

    @property
    def most_frequent_zone(self) -> int:
        return int(self.marginal_counts.argmax().item())

    def marginal_log_probs(self) -> torch.Tensor:
        smoothed = self.marginal_counts + self.alpha
        return (smoothed / smoothed.sum()).log()

    def marginal_scores(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.marginal_log_probs().to(device).unsqueeze(0).expand(batch_size, -1)

    def pass_vector_scores(self, passer_zone: torch.Tensor) -> torch.Tensor:
        index = self._offset_index.to(passer_zone.device)[passer_zone]
        counts = self.offset_counts.flatten().to(passer_zone.device)[index] + self.alpha
        return (counts / counts.sum(dim=-1, keepdim=True)).log()

    def conditional_scores(self, passer_zone: torch.Tensor) -> torch.Tensor:
        prior = self.pass_vector_scores(passer_zone).exp()
        counts = self.transition_counts.to(passer_zone.device)[passer_zone]
        smoothed = counts + self.backoff_weight * prior
        return (smoothed / smoothed.sum(dim=-1, keepdim=True)).log()
