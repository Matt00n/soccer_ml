"""Causal event histories for StatsBomb 360 receiver prediction.

The existing :mod:`receiver_dataset` deliberately represents a pass as one
freeze-frame graph.  This module leaves that representation untouched and
builds a fixed-length, *causal* sequence around it:

``[previous 360 graphs ..., current pass graph]``.

StatsBomb 360 is event driven, so a history element is a previous event with
an available freeze frame rather than an imagined regularly sampled tracking
frame.  Player IDs are not present in the public 360 freeze frames.  We retain
the variable visible-player masks and expose coordinates/team/actor/known
recipient roles; temporal models can consequently learn soft geometric
alignment instead of assuming that node index ``i`` has a persistent identity.

StatsBomb stores every event, and its freeze frame, in the frame of the team
*performing* it (that team attacks left to right).  Events by the other team
(pressures, duels, clearances, ...) are therefore rotated by 180 degrees and
have inverted ``teammate`` flags relative to the target pass.  Every history
element is mapped into the frame of the target passer's team before use.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from soccer_ml.data.pitch_grid import DEFAULT_GRID_M, DEFAULT_GRID_N
from soccer_ml.data.receiver_dataset import ReceiverParser
from soccer_ml.data.schema import STATSBOMB_NODE_LAYOUT, SoccerGraphBatch, SoccerSample
from soccer_ml.data.statsbomb import StatsBomb360Parser


# StatsBomb type IDs are stable within the open-data schema.  Reserving 0 for
# an unknown/current target event prevents the receiver model from consuming a
# current-event outcome or other post-event information.
NUM_EVENT_TYPES = 64
NUM_OUTCOMES = 32
NUM_BODY_PARTS = 16
ROLE_FEATURE_DIM = 3  # actor, known historical recipient, teammate of target passer

# Layout of the continuous per-event features (all in the target team's frame).
EVENT_DELTA_T_IDX = 0  # log time since previous included event
EVENT_DURATION_IDX = 1  # log event duration
EVENT_LENGTH_IDX = 2  # start -> end distance / 120 (passes, carries, shots)
EVENT_SIN_IDX = 3  # sin of start -> end direction
EVENT_COS_IDX = 4  # cos of start -> end direction
EVENT_PRESSURE_IDX = 5  # under_pressure flag
EVENT_X_IDX = 6  # start x / 120
EVENT_Y_IDX = 7  # start y / 80
EVENT_END_X_IDX = 8  # end x / 120 (start x when the event has no end location)
EVENT_END_Y_IDX = 9  # end y / 80
EVENT_SAME_TEAM_IDX = 10  # 1 if performed by the target passer's team
EVENT_NUMERIC_DIM = 11

# Graph feature channels describing *how* a pass is played (height, body
# part).  For the target pass these are part of the decision being predicted
# and unknown at next-action time, so they are zeroed unless requested.
TARGET_PASS_ATTRIBUTE_INDICES = (5, 6)

_OUTCOME_NAMES = {
    "": 0,
    "Complete": 1,
    "Incomplete": 2,
    "Success": 3,
    "Success In Play": 4,
    "Success Out": 5,
    "Goal": 6,
    "Saved": 7,
    "Blocked": 8,
    "Off T": 9,
    "Wayward": 10,
    "Out": 11,
    "Won": 12,
    "Lost": 13,
    "Lost In Play": 14,
    "Lost Out": 15,
    "Pass Offside": 16,
    "No Touch": 17,
    "Claim": 18,
    "In Play Danger": 19,
    "Touched In": 20,
    "Unknown": 21,
}

_BODY_PART_NAMES = {
    "": 0,
    "Right Foot": 1,
    "Left Foot": 2,
    "Head": 3,
    "Chest": 4,
    "Other": 5,
    "Both Hands": 6,
    "Keeper Arm": 7,
    "Left Hand": 8,
    "Right Hand": 9,
    "Drop Kick": 10,
}


def _timestamp_seconds(event: Dict[str, Any]) -> float:
    """Return seconds elapsed within a period for a StatsBomb event."""
    timestamp = str(event.get("timestamp", "00:00:00.000"))
    try:
        hours, minutes, seconds = timestamp.split(":")
        return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)
    except (TypeError, ValueError):
        # ``minute`` and ``second`` are less precise but still preserve order.
        return 60.0 * float(event.get("minute", 0.0)) + float(event.get("second", 0.0))


def _nested_outcome(event: Dict[str, Any]) -> str:
    """Extract a single outcome label from event-type-specific dictionaries."""
    event_type = str(event.get("type", {}).get("name", "")).lower().replace("*", "")
    keys = [event_type.replace(" ", "_"), "pass", "shot", "dribble", "duel", "goalkeeper"]
    for key in keys:
        details = event.get(key)
        if not isinstance(details, dict):
            continue
        outcome = details.get("outcome")
        if isinstance(outcome, dict):
            return str(outcome.get("name", ""))
        # A pass with no outcome is a completed pass, unlike event types where
        # absence simply means that no outcome field is defined.
        if key == "pass" and outcome is None:
            return "Complete"
    return ""


def _nested_body_part(event: Dict[str, Any]) -> str:
    for key in ("pass", "shot", "carry", "goalkeeper"):
        details = event.get(key)
        if isinstance(details, dict) and isinstance(details.get("body_part"), dict):
            return str(details["body_part"].get("name", ""))
    return ""


def _event_categoricals(event: Dict[str, Any]) -> tuple[int, int, int]:
    type_id = int(event.get("type", {}).get("id", 0) or 0)
    type_id = min(max(type_id, 1), NUM_EVENT_TYPES - 1)
    outcome_id = _OUTCOME_NAMES.get(_nested_outcome(event), 0)
    body_part_id = _BODY_PART_NAMES.get(_nested_body_part(event), 0)
    return type_id, min(outcome_id, NUM_OUTCOMES - 1), min(body_part_id, NUM_BODY_PARTS - 1)


def _event_end_location(event: Dict[str, Any]) -> Optional[List[float]]:
    for key in ("pass", "carry", "shot"):
        details = event.get(key)
        if isinstance(details, dict):
            end = details.get("end_location")
            if isinstance(end, list) and len(end) >= 2:
                return [float(end[0]), float(end[1])]
    return None


def _event_numeric_features(
    event: Dict[str, Any],
    delta_seconds: float,
    same_team: bool,
    pitch_length: float = 120.0,
    pitch_width: float = 80.0,
) -> torch.Tensor:
    """Build continuous, pre-event-safe context for an historical event.

    Outcome is represented separately as a categorical label.  Only previous
    events call this function in a temporal sample; values for the target pass
    are zeroed in :class:`TemporalReceiverDataset`.  Events by the opposing
    team (``same_team=False``) are rotated into the target team's frame.
    """
    location = event.get("location")
    if not isinstance(location, list) or len(location) < 2:
        location = [pitch_length / 2.0, pitch_width / 2.0]
    start = [float(location[0]), float(location[1])]
    end = _event_end_location(event) or list(start)
    if not same_team:
        start = [pitch_length - start[0], pitch_width - start[1]]
        end = [pitch_length - end[0], pitch_width - end[1]]
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    # StatsBomb's pass angle is atan2(dy, dx) in pitch coordinates; the same
    # definition is used for carries and shots.  Stationary events get (0, 0).
    sin_angle, cos_angle = (dy / length, dx / length) if length > 1e-6 else (0.0, 0.0)
    duration = float(event.get("duration", 0.0) or 0.0)
    features = torch.zeros(EVENT_NUMERIC_DIM, dtype=torch.float32)
    features[EVENT_DELTA_T_IDX] = math.log1p(max(delta_seconds, 0.0)) / math.log1p(120.0)
    features[EVENT_DURATION_IDX] = math.log1p(max(duration, 0.0)) / math.log1p(30.0)
    features[EVENT_LENGTH_IDX] = length / pitch_length
    features[EVENT_SIN_IDX] = sin_angle
    features[EVENT_COS_IDX] = cos_angle
    features[EVENT_PRESSURE_IDX] = float(bool(event.get("under_pressure", False)))
    features[EVENT_X_IDX] = start[0] / pitch_length
    features[EVENT_Y_IDX] = start[1] / pitch_width
    features[EVENT_END_X_IDX] = end[0] / pitch_length
    features[EVENT_END_Y_IDX] = end[1] / pitch_width
    features[EVENT_SAME_TEAM_IDX] = float(same_team)
    return features


def transform_event_numeric(event_numeric: torch.Tensor, action: int) -> torch.Tensor:
    """Apply a D2 pitch reflection to (..., EVENT_NUMERIC_DIM) event features.

    Matches :class:`~soccer_ml.data.transforms.D2GroupTransform`: action 1
    flips the pitch length (x), action 2 the width (y), action 3 both.
    """
    if action == 0:
        return event_numeric
    out = event_numeric.clone()
    if action in (1, 3):
        for index in (EVENT_X_IDX, EVENT_END_X_IDX):
            out[..., index] = 1.0 - out[..., index]
        out[..., EVENT_COS_IDX] = -out[..., EVENT_COS_IDX]
    if action in (2, 3):
        for index in (EVENT_Y_IDX, EVENT_END_Y_IDX):
            out[..., index] = 1.0 - out[..., index]
        out[..., EVENT_SIN_IDX] = -out[..., EVENT_SIN_IDX]
    return out


def _rotate_to_target_frame(graph: SoccerSample, roles: torch.Tensor) -> tuple[SoccerSample, torch.Tensor]:
    """Rotate an opposing-team frame by 180 degrees and invert team flags.

    Normalized coordinates are zero-centered, so the rotation negates x and y.
    Teammate flags become relative to the target passer's team; the
    opposing-team edge feature is symmetric and unchanged.
    """
    layout = STATSBOMB_NODE_LAYOUT
    valid = graph.player_mask.to(graph.node_features.dtype)
    node_features = graph.node_features.clone()
    node_features[:, [layout.x, layout.y]] = -node_features[:, [layout.x, layout.y]]
    node_features[:, layout.teammate] = (1.0 - node_features[:, layout.teammate]) * valid
    rotated_roles = roles.clone()
    rotated_roles[:, 2] = (1.0 - rotated_roles[:, 2]) * valid
    rotated = SoccerSample(
        node_features=node_features,
        edge_features=graph.edge_features,
        graph_features=graph.graph_features,
        adj_mat=graph.adj_mat,
        player_mask=graph.player_mask,
    )
    return rotated, rotated_roles


def _historical_roles(
    frame: Dict[str, Any], event: Dict[str, Any], max_players: int
) -> torch.Tensor:
    """Return actor, recipient-candidate, and teammate flags for a frame.

    A public freeze frame has no player IDs.  For an observed historical pass,
    the recipient candidate uses the same robust proxy as the existing receiver
    dataset: the closest non-actor teammate to the recorded pass end location.
    It is never populated for the current target pass, so it cannot leak its
    receiver label into the prediction.
    """
    roles = torch.zeros(max_players, ROLE_FEATURE_DIM, dtype=torch.float32)
    players = frame.get("freeze_frame", [])[:max_players]
    end_location = None
    pass_info = event.get("pass")
    if isinstance(pass_info, dict) and pass_info.get("recipient") is not None:
        end_location = pass_info.get("end_location")

    receiver_idx: Optional[int] = None
    if isinstance(end_location, list) and len(end_location) >= 2:
        best_distance = float("inf")
        for idx, player in enumerate(players):
            if not player.get("teammate", False) or player.get("actor", False):
                continue
            loc = player.get("location", [])
            if len(loc) < 2:
                continue
            distance = math.dist((float(loc[0]), float(loc[1])), (float(end_location[0]), float(end_location[1])))
            if distance < best_distance:
                best_distance = distance
                receiver_idx = idx

    for idx, player in enumerate(players):
        roles[idx, 0] = float(bool(player.get("actor", False)))
        roles[idx, 2] = float(bool(player.get("teammate", False)))
    if receiver_idx is not None:
        roles[receiver_idx, 1] = 1.0
    return roles


@dataclass
class TemporalReceiverSample:
    """One receiver/target location sample and its left-padded causal graph history."""

    node_features: torch.Tensor  # (T, N, D)
    edge_features: torch.Tensor  # (T, N, N, E)
    graph_features: torch.Tensor  # (T, G)
    adj_mat: torch.Tensor  # (T, N, N)
    player_mask: torch.Tensor  # (T, N)
    role_features: torch.Tensor  # (T, N, 3)
    event_type: torch.Tensor  # (T,)
    event_outcome: torch.Tensor  # (T,)
    event_body_part: torch.Tensor  # (T,)
    event_numeric: torch.Tensor  # (T, EVENT_NUMERIC_DIM)
    time_delta: torch.Tensor  # (T,), seconds since previous included event
    sequence_mask: torch.Tensor  # (T,), true for graph timesteps
    receiver_label: torch.Tensor  # scalar
    target_zone_label: Optional[torch.Tensor] = None  # scalar
    pass_end_location: Optional[torch.Tensor] = None  # (2,) float tensor
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TemporalReceiverBatch:
    """A collated temporal receiver batch with a view of its current graphs."""

    node_features: torch.Tensor
    edge_features: torch.Tensor
    graph_features: torch.Tensor
    adj_mat: torch.Tensor
    player_mask: torch.Tensor
    role_features: torch.Tensor
    event_type: torch.Tensor
    event_outcome: torch.Tensor
    event_body_part: torch.Tensor
    event_numeric: torch.Tensor
    time_delta: torch.Tensor
    sequence_mask: torch.Tensor
    receiver_label: torch.Tensor
    target_zone_label: Optional[torch.Tensor] = None
    pass_end_location: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "TemporalReceiverBatch":
        return TemporalReceiverBatch(
            node_features=self.node_features.to(device),
            edge_features=self.edge_features.to(device),
            graph_features=self.graph_features.to(device),
            adj_mat=self.adj_mat.to(device),
            player_mask=self.player_mask.to(device),
            role_features=self.role_features.to(device),
            event_type=self.event_type.to(device),
            event_outcome=self.event_outcome.to(device),
            event_body_part=self.event_body_part.to(device),
            event_numeric=self.event_numeric.to(device),
            time_delta=self.time_delta.to(device),
            sequence_mask=self.sequence_mask.to(device),
            receiver_label=self.receiver_label.to(device),
            target_zone_label=(
                self.target_zone_label.to(device) if self.target_zone_label is not None else None
            ),
            pass_end_location=(
                self.pass_end_location.to(device) if self.pass_end_location is not None else None
            ),
        )

    @property
    def batch_size(self) -> int:
        return int(self.node_features.shape[0])

    def current_graph_batch(self) -> SoccerGraphBatch:
        """Expose the final target graph for existing naive baselines."""
        return SoccerGraphBatch(
            node_features=self.node_features[:, -1],
            edge_features=self.edge_features[:, -1],
            graph_features=self.graph_features[:, -1],
            adj_mat=self.adj_mat[:, -1],
            receiver_label=self.receiver_label,
            target_zone_label=self.target_zone_label,
            pass_end_location=self.pass_end_location,
            player_mask=self.player_mask[:, -1],
        )


def collate_temporal_receiver_graphs(
    batch: Sequence[TemporalReceiverSample],
) -> TemporalReceiverBatch:
    """Collate fixed-length temporal graph samples."""
    if not batch:
        raise ValueError("Cannot collate an empty temporal receiver batch.")

    target_zone_label = (
        torch.stack([sample.target_zone_label for sample in batch]).long()
        if batch[0].target_zone_label is not None
        else None
    )
    pass_end_location = (
        torch.stack([sample.pass_end_location for sample in batch]).float()
        if batch[0].pass_end_location is not None
        else None
    )

    return TemporalReceiverBatch(
        node_features=torch.stack([sample.node_features for sample in batch]),
        edge_features=torch.stack([sample.edge_features for sample in batch]),
        graph_features=torch.stack([sample.graph_features for sample in batch]),
        adj_mat=torch.stack([sample.adj_mat for sample in batch]),
        player_mask=torch.stack([sample.player_mask for sample in batch]),
        role_features=torch.stack([sample.role_features for sample in batch]),
        event_type=torch.stack([sample.event_type for sample in batch]),
        event_outcome=torch.stack([sample.event_outcome for sample in batch]),
        event_body_part=torch.stack([sample.event_body_part for sample in batch]),
        event_numeric=torch.stack([sample.event_numeric for sample in batch]),
        time_delta=torch.stack([sample.time_delta for sample in batch]),
        sequence_mask=torch.stack([sample.sequence_mask for sample in batch]),
        receiver_label=torch.stack([sample.receiver_label for sample in batch]).long(),
        target_zone_label=target_zone_label,
        pass_end_location=pass_end_location,
    )


@dataclass
class _FrameRecord:
    event: Dict[str, Any]
    frame: Dict[str, Any]
    graph: SoccerSample
    roles: torch.Tensor
    order: int
    timestamp_seconds: float
    team_id: Optional[int]


class TemporalReceiverDataset(Dataset):
    """Receiver passes with preceding event-driven StatsBomb 360 graphs.

    Args:
        context_length: Number of preceding snapshots.  The returned sequence
            always has ``context_length + 1`` elements; final position is the
            target pass graph and leading absent elements are masked.
        context_scope: ``"possession"`` (default) resets history at possession
            and period boundaries. ``"period"`` retains prior possessions in a
            half; ``"match"`` retains all previous available snapshots.
        max_samples: Applies to valid target passes in frame-file order, matching
            the existing :class:`ReceiverDataset` semantics.
        grid_m, grid_n: Pass target grid, used when no ``parser`` is supplied.
        include_target_pass_attributes: Keep the target pass's height and body
            part in its graph features.  Off by default: they describe the
            action being predicted, not the situation.
    """

    def __init__(
        self,
        three_sixty_path: Union[str, Path],
        events_path: Union[str, Path],
        context_length: int = 8,
        context_scope: Literal["possession", "period", "match"] = "possession",
        max_samples: Optional[int] = None,
        parser: Optional[ReceiverParser] = None,
        grid_m: int = DEFAULT_GRID_M,
        grid_n: int = DEFAULT_GRID_N,
        include_target_pass_attributes: bool = False,
    ):
        if context_length < 0:
            raise ValueError("context_length must be non-negative.")
        if context_scope not in {"possession", "period", "match"}:
            raise ValueError("context_scope must be 'possession', 'period', or 'match'.")

        self.three_sixty_path = Path(three_sixty_path)
        self.events_path = Path(events_path)
        self.context_length = context_length
        self.sequence_length = context_length + 1
        self.context_scope = context_scope
        self.include_target_pass_attributes = include_target_pass_attributes
        self.receiver_parser = parser or ReceiverParser(grid_m=grid_m, grid_n=grid_n)
        self.frame_parser = StatsBomb360Parser(
            pitch_length=self.receiver_parser.pitch_length,
            pitch_width=self.receiver_parser.pitch_width,
            target_scale=self.receiver_parser.target_scale,
            max_players=self.receiver_parser.max_players,
            min_players=self.receiver_parser.min_players,
        )
        self.samples: List[TemporalReceiverSample] = []
        self._load(max_samples=max_samples)

    def _load(self, max_samples: Optional[int]) -> None:
        with self.three_sixty_path.open("r", encoding="utf-8") as file:
            frames = json.load(file)
        with self.events_path.open("r", encoding="utf-8") as file:
            events = json.load(file)

        events_by_id = {event.get("id"): event for event in events if event.get("id")}
        records: List[_FrameRecord] = []
        targets_in_frame_order: List[tuple[str, SoccerSample]] = []

        # Keep target selection byte-for-byte conceptually aligned with the
        # existing ReceiverDataset: frame order, valid completed passes only,
        # and the same ReceiverParser instance.
        for frame in frames:
            event_id = frame.get("event_uuid")
            event = events_by_id.get(event_id)
            if event is None:
                continue
            graph = self.frame_parser.parse_frame(frame, event)
            if graph is not None:
                records.append(
                    _FrameRecord(
                        event=event,
                        frame=frame,
                        graph=graph,
                        roles=_historical_roles(frame, event, self.receiver_parser.max_players),
                        order=int(event.get("index", len(records))),
                        timestamp_seconds=_timestamp_seconds(event),
                        team_id=(event.get("team") or {}).get("id"),
                    )
                )
            target = self.receiver_parser.parse_pass_frame(frame, event)
            if target is not None:
                targets_in_frame_order.append((str(event_id), target))
                if max_samples is not None and len(targets_in_frame_order) >= max_samples:
                    break

        records.sort(key=lambda record: record.order)
        record_by_event_id = {str(record.event.get("id")): (idx, record) for idx, record in enumerate(records)}

        for event_id, target_graph in targets_in_frame_order:
            record_info = record_by_event_id.get(event_id)
            if record_info is None:
                continue
            target_index, target_record = record_info
            history_indices = self._history_indices(records, target_index)
            self.samples.append(
                self._make_sample(records, history_indices, target_record, target_graph)
            )

    def _history_indices(self, records: Sequence[_FrameRecord], target_index: int) -> List[int]:
        target = records[target_index].event
        selected: List[int] = []
        for index in range(target_index - 1, -1, -1):
            candidate = records[index].event
            if self.context_scope != "match" and candidate.get("period") != target.get("period"):
                break
            if self.context_scope == "possession" and candidate.get("possession") != target.get("possession"):
                break
            selected.append(index)
            if len(selected) >= self.context_length:
                break
        selected.reverse()
        return selected

    def _make_sample(
        self,
        records: Sequence[_FrameRecord],
        history_indices: Sequence[int],
        target_record: _FrameRecord,
        target_graph: SoccerSample,
    ) -> TemporalReceiverSample:
        n_players = self.receiver_parser.max_players
        node_dim = target_graph.node_features.shape[-1]
        edge_dim = target_graph.edge_features.shape[-1]
        graph_dim = target_graph.graph_features.shape[-1]
        sequence_length = self.sequence_length

        node_features = torch.zeros(sequence_length, n_players, node_dim)
        edge_features = torch.zeros(sequence_length, n_players, n_players, edge_dim)
        graph_features = torch.zeros(sequence_length, graph_dim)
        adj_mat = torch.zeros(sequence_length, n_players, n_players)
        player_mask = torch.zeros(sequence_length, n_players, dtype=torch.bool)
        role_features = torch.zeros(sequence_length, n_players, ROLE_FEATURE_DIM)
        event_type = torch.zeros(sequence_length, dtype=torch.long)
        event_outcome = torch.zeros(sequence_length, dtype=torch.long)
        event_body_part = torch.zeros(sequence_length, dtype=torch.long)
        event_numeric = torch.zeros(sequence_length, EVENT_NUMERIC_DIM)
        time_delta = torch.zeros(sequence_length)
        sequence_mask = torch.zeros(sequence_length, dtype=torch.bool)

        history_records = [records[index] for index in history_indices]
        entries: List[tuple[_FrameRecord, SoccerSample, torch.Tensor, bool, bool]] = []
        for record in history_records:
            same_team = record.team_id == target_record.team_id
            graph, roles = record.graph, record.roles
            if not same_team:
                graph, roles = _rotate_to_target_frame(graph, roles)
            entries.append((record, graph, roles, True, same_team))
        # Current pass graph comes from ReceiverParser to guarantee exact parity
        # with the single-frame receiver task.  Its event labels remain unknown.
        entries.append((target_record, target_graph, target_record.roles, False, True))
        start = sequence_length - len(entries)

        previous_timestamp: Optional[float] = None
        for local_idx, (record, graph, roles, labels_are_visible, same_team) in enumerate(entries):
            idx = start + local_idx
            node_features[idx] = graph.node_features
            edge_features[idx] = graph.edge_features
            graph_features[idx] = graph.graph_features
            adj_mat[idx] = graph.adj_mat
            player_mask[idx] = graph.player_mask
            role_features[idx] = roles
            # The target-pass recipient proxy is explicitly removed.  The actor
            # and teammate flags remain part of the observed current graph.
            if not labels_are_visible:
                role_features[idx, :, 1] = 0.0

            sequence_mask[idx] = True
            delta = 0.0 if previous_timestamp is None else max(record.timestamp_seconds - previous_timestamp, 0.0)
            # The first included graph can follow an omitted context event.  A
            # zero delta makes that boundary explicit rather than guessing.
            time_delta[idx] = delta
            previous_timestamp = record.timestamp_seconds

            if labels_are_visible:
                type_id, outcome_id, body_part_id = _event_categoricals(record.event)
                event_type[idx] = type_id
                event_outcome[idx] = outcome_id
                event_body_part[idx] = body_part_id
                event_numeric[idx] = _event_numeric_features(
                    record.event,
                    delta,
                    same_team,
                    self.receiver_parser.pitch_length,
                    self.receiver_parser.pitch_width,
                )

        if not self.include_target_pass_attributes:
            graph_features[-1, list(TARGET_PASS_ATTRIBUTE_INDICES)] = 0.0

        return TemporalReceiverSample(
            node_features=node_features,
            edge_features=edge_features,
            graph_features=graph_features,
            adj_mat=adj_mat,
            player_mask=player_mask,
            role_features=role_features,
            event_type=event_type,
            event_outcome=event_outcome,
            event_body_part=event_body_part,
            event_numeric=event_numeric,
            time_delta=time_delta,
            sequence_mask=sequence_mask,
            receiver_label=target_graph.receiver_label,
            target_zone_label=target_graph.target_zone_label,
            pass_end_location=target_graph.pass_end_location,
            metadata={
                "event_uuid": target_record.event.get("id"),
                "event_index": target_record.order,
                "history_events": len(history_records),
                "history_opponent_events": sum(not entry[4] for entry in entries),
                "context_scope": self.context_scope,
            },
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TemporalReceiverSample:
        return self.samples[index]

