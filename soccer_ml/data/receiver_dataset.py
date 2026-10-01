"""Receiver prediction dataset and baselines for pass reception task."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from soccer_ml.data.pitch_grid import DEFAULT_GRID_M, DEFAULT_GRID_N, PitchGrid
from soccer_ml.data.schema import SoccerGraphBatch, SoccerSample
from soccer_ml.data.statsbomb import StatsBomb360Parser
from soccer_ml.data.transforms import normalize_pitch_coordinates


class ReceiverParser(StatsBomb360Parser):
    """Extended StatsBomb parser that extracts pass target locations and receiver information."""

    def __init__(
        self,
        pitch_length: float = 120.0,
        pitch_width: float = 80.0,
        target_scale: float = 10.0,
        max_players: int = 22,
        min_players: int = 4,
        grid_m: int = DEFAULT_GRID_M,
        grid_n: int = DEFAULT_GRID_N,
        pitch_grid: Optional[PitchGrid] = None,
    ):
        super().__init__(
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            target_scale=target_scale,
            max_players=max_players,
            min_players=min_players,
        )
        self.pitch_grid = pitch_grid or PitchGrid(
            grid_m=grid_m,
            grid_n=grid_n,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )

    def parse_pass_frame(
        self,
        frame_dict: Dict[str, Any],
        event_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[SoccerSample]:
        """Parse a freeze-frame + completed pass event, extracting pitch grid target location.

        Returns None if:
            - No pass event
            - Event is not a completed pass
            - Missing pass end location
            - Insufficient players in frame
        """
        # Only process pass events
        if event_dict is None or event_dict.get("type", {}).get("name") != "Pass":
            return None

        pass_info = event_dict.get("pass", {})

        # Only completed passes (StatsBomb completed passes have outcome=None)
        if pass_info.get("outcome") is not None:
            return None

        pass_end_loc = pass_info.get("end_location")
        if pass_end_loc is None or len(pass_end_loc) < 2:
            return None

        freezeframe = frame_dict.get("freeze_frame", [])
        num_players = len(freezeframe)

        if num_players < self.min_players:
            return None

        # Truncate or pad up to max_players
        n = min(num_players, self.max_players)
        selected_players = freezeframe[:n]

        node_features_list = []
        is_opposing_list = []
        receiver_idx: Optional[int] = None
        best_receiver_dist: float = float("inf")

        for idx, p in enumerate(selected_players):
            loc = p.get("location", [self.pitch_length / 2.0, self.pitch_width / 2.0])
            raw_x, raw_y = loc[0], loc[1]

            # Coordinate normalization
            norm_x, norm_y = normalize_pitch_coordinates(
                torch.tensor(raw_x, dtype=torch.float32),
                torch.tensor(raw_y, dtype=torch.float32),
                pitch_length=self.pitch_length,
                pitch_width=self.pitch_width,
                target_scale=self.target_scale,
            )

            v_x = 0.0
            v_y = 0.0
            height = 1.80
            weight = 0.75
            is_actor = 1.0 if p.get("actor", False) else 0.0
            is_teammate = 1.0 if p.get("teammate", False) else 0.0
            is_keeper = 1.0 if p.get("keeper", False) else 0.0

            node_features_list.append(
                [
                    norm_x.item(),
                    norm_y.item(),
                    v_x,
                    v_y,
                    height,
                    weight,
                    is_actor,
                    is_teammate,
                    is_keeper,
                ]
            )

            is_opponent = not p.get("teammate", False)
            is_opposing_list.append(is_opponent)

            # Optional receiver heuristic: closest teammate to pass end location
            if p.get("teammate", False) and not p.get("actor", False):
                dist = (
                    (raw_x - pass_end_loc[0]) ** 2 + (raw_y - pass_end_loc[1]) ** 2
                ) ** 0.5
                if receiver_idx is None or dist < best_receiver_dist:
                    receiver_idx = idx
                    best_receiver_dist = dist

        # Convert to tensors: (N_active, 9)
        node_fts = torch.tensor(node_features_list, dtype=torch.float32)
        padded_node_fts = torch.zeros(
            self.max_players, node_fts.shape[-1], dtype=torch.float32
        )
        padded_node_fts[:n] = node_fts

        player_mask = torch.zeros(self.max_players, dtype=torch.bool)
        player_mask[:n] = True

        opp_tensor = torch.tensor(is_opposing_list, dtype=torch.bool)
        edge_matrix = (
            (opp_tensor.unsqueeze(0) != opp_tensor.unsqueeze(1)).float().unsqueeze(-1)
        )

        padded_edge_fts = torch.zeros(
            self.max_players, self.max_players, 1, dtype=torch.float32
        )
        padded_edge_fts[:n, :n] = edge_matrix

        padded_adj_mat = torch.zeros(
            self.max_players, self.max_players, dtype=torch.float32
        )
        padded_adj_mat[:n, :n] = 1.0

        # Global features: (7,)
        # index 0: is_pass, index 1: is_shot, index 2: period, index 3: minute / 90.0,
        # index 4: under_pressure, index 5: pass_height, index 6: pass_body_part
        graph_fts = torch.zeros(7, dtype=torch.float32)
        graph_fts[0] = 1.0  # is_pass
        graph_fts[2] = float(event_dict.get("period", 1))
        graph_fts[3] = float(event_dict.get("minute", 0)) / 90.0
        graph_fts[4] = 1.0 if event_dict.get("under_pressure", False) else 0.0

        height_name = pass_info.get("height", {}).get("name", "")
        if height_name == "Ground Pass":
            graph_fts[5] = 0.33
        elif height_name == "Low Pass":
            graph_fts[5] = 0.66
        elif height_name == "High Pass":
            graph_fts[5] = 1.0

        bp_name = pass_info.get("body_part", {}).get("name", "")
        if bp_name == "Right Foot":
            graph_fts[6] = 0.25
        elif bp_name == "Left Foot":
            graph_fts[6] = 0.50
        elif bp_name == "Head":
            graph_fts[6] = 0.75
        elif bp_name:
            graph_fts[6] = 1.0

        shot_label = torch.tensor([0.0], dtype=torch.float32)
        creativity_label = torch.zeros(self.max_players, dtype=torch.float32)

        # Discretize pass destination into target zone
        target_zone = self.pitch_grid.coords_to_zone(pass_end_loc[0], pass_end_loc[1])
        target_zone_label = torch.tensor(target_zone, dtype=torch.long)
        pass_end_location = torch.tensor(
            [float(pass_end_loc[0]), float(pass_end_loc[1])], dtype=torch.float32
        )

        target_receiver = torch.tensor(
            receiver_idx if receiver_idx is not None else 0, dtype=torch.long
        )

        recipient_info = pass_info.get("recipient", {})
        recipient_name = recipient_info.get("name") if isinstance(recipient_info, dict) else None

        return SoccerSample(
            node_features=padded_node_fts,
            edge_features=padded_edge_fts,
            graph_features=graph_fts,
            adj_mat=padded_adj_mat,
            receiver_label=target_receiver,
            target_zone_label=target_zone_label,
            pass_end_location=pass_end_location,
            shot_label=shot_label,
            creativity_label=creativity_label,
            player_mask=player_mask,
            metadata={
                "event_uuid": frame_dict.get("event_uuid"),
                "num_real_players": n,
                "recipient_name": recipient_name,
                "pass_end_location": pass_end_loc,
                "target_zone": target_zone,
            },
        )


class ReceiverDataset(Dataset):
    """PyTorch Dataset for receiver prediction task using StatsBomb 360 data."""

    def __init__(
        self,
        three_sixty_path: Union[str, Path],
        events_path: Union[str, Path],
        max_samples: Optional[int] = None,
        parser: Optional[ReceiverParser] = None,
    ):
        super().__init__()
        self.three_sixty_path = Path(three_sixty_path)
        self.events_path = Path(events_path)
        self.parser = parser or ReceiverParser()

        self.samples: List[SoccerSample] = []
        self._load_reception_data(max_samples=max_samples)

    def _load_reception_data(self, max_samples: Optional[int] = None):
        """Load only pass frames with valid receiver annotations."""
        with open(self.three_sixty_path, "r", encoding="utf-8") as f:
            frames = json.load(f)

        events_map: Dict[str, Dict[str, Any]] = {}
        with open(self.events_path, "r", encoding="utf-8") as f:
            events_list = json.load(f)
        events_map = {e.get("id"): e for e in events_list if "id" in e}

        for frame in frames:
            ev_id = frame.get("event_uuid")
            event_dict = events_map.get(ev_id)

            sample = self.parser.parse_pass_frame(frame, event_dict)
            if sample is not None:
                self.samples.append(sample)
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SoccerSample:
        return self.samples[idx]


# ============================================================================
# Baseline Prediction Functions
# ============================================================================


def closest_teammate_baseline(batch: SoccerGraphBatch) -> torch.Tensor:
    """Baseline: predict closest teammate to ball-possessing player.

    Args:
        batch: SoccerGraphBatch with node_features shape (B, N, D_v)
               index 0-1: x, y coordinates; index 6: ball_possession flag

    Returns:
        predictions: (B,) long tensor with predicted receiver index for each sample
    """
    B, N, _ = batch.node_features.shape
    predictions = torch.zeros(B, dtype=torch.long, device=batch.node_features.device)

    for b in range(B):
        node_fts = batch.node_features[b]  # (N, D_v)
        mask = (
            batch.player_mask[b]
            if batch.player_mask is not None
            else torch.ones(N, dtype=torch.bool)
        )

        # Find ball carrier (highest possession value)
        poss_values = node_fts[:, 6]  # ball_possession channel
        ball_carrier_idx = torch.argmax(poss_values)

        # Find closest teammate (same team as carrier, excluding carrier)
        ball_carrier_pos = node_fts[ball_carrier_idx, :2]
        distances = torch.norm(node_fts[:, :2] - ball_carrier_pos, dim=1)

        # Mask out: ball carrier, opponents, invalid players
        edge_fts = batch.edge_features[b]  # (N, N, D_e)
        is_opponent = edge_fts[ball_carrier_idx, :, 0] > 0.5  # opponent edge feature

        distances[ball_carrier_idx] = float("inf")
        distances[is_opponent] = float("inf")
        distances[~mask] = float("inf")

        closest = torch.argmin(distances)
        predictions[b] = closest

    return predictions


def random_teammate_baseline(batch: SoccerGraphBatch) -> torch.Tensor:
    """Baseline: predict random teammate.

    Args:
        batch: SoccerGraphBatch

    Returns:
        predictions: (B,) long tensor
    """
    B, N, _ = batch.node_features.shape
    predictions = torch.zeros(B, dtype=torch.long, device=batch.node_features.device)

    for b in range(B):
        node_fts = batch.node_features[b]
        mask = (
            batch.player_mask[b]
            if batch.player_mask is not None
            else torch.ones(N, dtype=torch.bool)
        )

        # Find ball carrier
        poss_values = node_fts[:, 6]
        ball_carrier_idx = torch.argmax(poss_values)

        # Find teammates
        edge_fts = batch.edge_features[b]
        is_opponent = edge_fts[ball_carrier_idx, :, 0] > 0.5

        valid_teammates = mask & ~is_opponent
        valid_teammates[ball_carrier_idx] = False

        if valid_teammates.sum() > 0:
            valid_indices = torch.nonzero(valid_teammates, as_tuple=True)[0]
            rand_idx = torch.randperm(len(valid_indices))[0]
            predictions[b] = valid_indices[rand_idx]
        else:
            predictions[b] = 0

    return predictions


def most_frequent_receiver_baseline(
    batch: SoccerGraphBatch,
    frequency_map: Optional[Dict[int, int]] = None,
) -> torch.Tensor:
    """Baseline: predict most frequently occurring receiver (team bias).

    Args:
        batch: SoccerGraphBatch
        frequency_map: dict mapping receiver index to frequency (optional for cached stats)

    Returns:
        predictions: (B,) long tensor
    """
    B, N, _ = batch.node_features.shape
    predictions = torch.zeros(B, dtype=torch.long, device=batch.node_features.device)

    if frequency_map is None or len(frequency_map) == 0:
        # Default: biased toward positions farther from ball carrier
        # For now, just predict a random teammate (would need training data to compute properly)
        return random_teammate_baseline(batch)

    for b in range(B):
        best_idx = max(frequency_map.keys(), key=lambda k: frequency_map[k])
        predictions[b] = best_idx

    return predictions


# ============================================================================
# Collation Function (extends existing collate_soccer_graphs)
# ============================================================================


def collate_receiver_graphs(batch: List[SoccerSample]) -> SoccerGraphBatch:
    """Collates a list of SoccerSample instances into a batch (same as standard collation)."""
    from soccer_ml.data.dataset import collate_soccer_graphs

    return collate_soccer_graphs(batch)


