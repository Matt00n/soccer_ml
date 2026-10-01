"""Parser and PyTorch Dataset for StatsBomb 360 freeze-frame and event data."""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import torch
from torch.utils.data import Dataset

from soccer_ml.data.schema import SoccerSample
from soccer_ml.data.transforms import normalize_pitch_coordinates


class StatsBomb360Parser:
    """Parses StatsBomb open-data JSON files into structured SoccerSample graphs."""

    def __init__(
        self,
        pitch_length: float = 120.0,
        pitch_width: float = 80.0,
        target_scale: float = 10.0,
        max_players: int = 22,
        min_players: int = 4,
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.target_scale = target_scale
        self.max_players = max_players
        self.min_players = min_players

    def parse_frame(
        self,
        frame_dict: Dict[str, Any],
        event_dict: Optional[Dict[str, Any]] = None,
    ) -> Optional[SoccerSample]:
        """Convert a single 360 freeze-frame and its associated event into a SoccerSample."""
        freeze_frame = frame_dict.get("freeze_frame", [])
        num_players = len(freeze_frame)

        if num_players < self.min_players:
            return None

        # Truncate or pad up to max_players
        n = min(num_players, self.max_players)
        selected_players = freeze_frame[:n]

        node_features_list = []
        is_opposing_list = []
        receiver_idx: Optional[int] = None
        pass_end_loc = None

        if event_dict is not None and "pass" in event_dict:
            pass_info = event_dict["pass"]
            pass_end_loc = pass_info.get("end_location")

        for idx, p in enumerate(selected_players):
            loc = p.get("location", [self.pitch_length / 2.0, self.pitch_width / 2.0])
            raw_x, raw_y = loc[0], loc[1]

            # Coordinate normalization to zero-centered box
            norm_x, norm_y = normalize_pitch_coordinates(
                torch.tensor(raw_x, dtype=torch.float32),
                torch.tensor(raw_y, dtype=torch.float32),
                pitch_length=self.pitch_length,
                pitch_width=self.pitch_width,
                target_scale=self.target_scale,
            )

            # Velocities: default 0.0 for single static snapshot
            v_x = 0.0
            v_y = 0.0

            # Default normalized height & weight (matching TacticAI 180cm / 75kg downscaled)
            height = 1.80
            weight = 0.75

            # Ball possession / actor, teammate, keeper
            is_actor = 1.0 if p.get("actor", False) else 0.0
            is_teammate = 1.0 if p.get("teammate", False) else 0.0
            is_keeper = 1.0 if p.get("keeper", False) else 0.0

            node_features_list.append([
                norm_x.item(),
                norm_y.item(),
                v_x,
                v_y,
                height,
                weight,
                is_actor,
                is_teammate,
                is_keeper,
            ])

            # Teammate indicator: False means opponent of the actor
            is_opponent = not p.get("teammate", False)
            is_opposing_list.append(is_opponent)

            # Check if this player is closest to pass end location (if event is a pass)
            if pass_end_loc is not None and p.get("teammate", False) and not p.get("actor", False):
                dist = ((raw_x - pass_end_loc[0]) ** 2 + (raw_y - pass_end_loc[1]) ** 2) ** 0.5
                if dist < 4.0:  # within 4 meters
                    receiver_idx = idx

        # Convert to tensor: (N_active, 9)
        node_fts = torch.tensor(node_features_list, dtype=torch.float32)

        # Pad to fixed size max_players (default 22) for unified batch collation
        padded_node_fts = torch.zeros(self.max_players, node_fts.shape[-1], dtype=torch.float32)
        padded_node_fts[:n] = node_fts

        player_mask = torch.zeros(self.max_players, dtype=torch.bool)
        player_mask[:n] = True

        # Build edge features: (max_players, max_players, 1)
        # 1.0 if opposing team, 0.0 if same team
        opp_tensor = torch.tensor(is_opposing_list, dtype=torch.bool)
        edge_matrix = (opp_tensor.unsqueeze(0) != opp_tensor.unsqueeze(1)).float().unsqueeze(-1)  # (n, n, 1)

        padded_edge_fts = torch.zeros(self.max_players, self.max_players, 1, dtype=torch.float32)
        padded_edge_fts[:n, :n] = edge_matrix

        # Adjacency matrix: connected for active players
        padded_adj_mat = torch.zeros(self.max_players, self.max_players, dtype=torch.float32)
        padded_adj_mat[:n, :n] = 1.0

        # Global features: (7,)
        # index 0: is_pass, index 1: is_shot, index 2: period, index 3: minute / 90.0,
        # index 4: under_pressure, index 5: pass_height, index 6: pass_body_part
        graph_fts = torch.zeros(7, dtype=torch.float32)
        shot_label = torch.tensor([0.0], dtype=torch.float32)
        creativity_label = torch.zeros(self.max_players, dtype=torch.float32)

        if event_dict is not None:
            ev_type = event_dict.get("type", {}).get("name", "")
            if ev_type == "Pass":
                graph_fts[0] = 1.0
            elif ev_type == "Shot":
                graph_fts[1] = 1.0
                shot_label[0] = 1.0

            graph_fts[2] = float(event_dict.get("period", 1))
            graph_fts[3] = float(event_dict.get("minute", 0)) / 90.0
            graph_fts[4] = 1.0 if event_dict.get("under_pressure", False) else 0.0

            if "pass" in event_dict:
                p_info = event_dict["pass"]
                height_name = p_info.get("height", {}).get("name", "")
                if height_name == "Ground Pass":
                    graph_fts[5] = 0.33
                elif height_name == "Low Pass":
                    graph_fts[5] = 0.66
                elif height_name == "High Pass":
                    graph_fts[5] = 1.0

                bp_name = p_info.get("body_part", {}).get("name", "")
                if bp_name == "Right Foot":
                    graph_fts[6] = 0.25
                elif bp_name == "Left Foot":
                    graph_fts[6] = 0.50
                elif bp_name == "Head":
                    graph_fts[6] = 0.75
                elif bp_name:
                    graph_fts[6] = 1.0

            # Creativity proxy: pass length or xG
            if "shot" in event_dict:
                xg = float(event_dict["shot"].get("statsbomb_xg", 0.05))
                creativity_label[:n] = xg

        # Receiver label
        target_receiver = torch.tensor(receiver_idx if receiver_idx is not None else 0, dtype=torch.long)

        return SoccerSample(
            node_features=padded_node_fts,
            edge_features=padded_edge_fts,
            graph_features=graph_fts,
            adj_mat=padded_adj_mat,
            receiver_label=target_receiver,
            shot_label=shot_label,
            creativity_label=creativity_label,
            player_mask=player_mask,
            metadata={
                "event_uuid": frame_dict.get("event_uuid"),
                "num_real_players": n,
            },
        )


class StatsBomb360Dataset(Dataset):
    """PyTorch Dataset loading real StatsBomb 360 freeze-frame and event graphs."""

    def __init__(
        self,
        three_sixty_path: Union[str, Path],
        events_path: Optional[Union[str, Path]] = None,
        parser: Optional[StatsBomb360Parser] = None,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.three_sixty_path = Path(three_sixty_path)
        self.events_path = Path(events_path) if events_path else None
        self.parser = parser or StatsBomb360Parser()

        self.samples: List[SoccerSample] = []
        self._load_data(max_samples=max_samples)

    def _load_data(self, max_samples: Optional[int] = None):
        with open(self.three_sixty_path, "r", encoding="utf-8") as f:
            frames = json.load(f)

        events_map: Dict[str, Dict[str, Any]] = {}
        if self.events_path and self.events_path.exists():
            with open(self.events_path, "r", encoding="utf-8") as f:
                events_list = json.load(f)
            events_map = {e.get("id"): e for e in events_list if "id" in e}

        for frame in frames:
            ev_id = frame.get("event_uuid")
            event_dict = events_map.get(ev_id)

            sample = self.parser.parse_frame(frame, event_dict)
            if sample is not None:
                self.samples.append(sample)
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SoccerSample:
        return self.samples[idx]
