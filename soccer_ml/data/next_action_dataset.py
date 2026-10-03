"""Next-action prediction samples from StatsBomb 360 freeze frames and event streams.

One sample is an on-ball decision: every pass (completed or failed), carry of
at least ``carry_min_length`` (one grid cell by default), shot, dribble and
clearance whose event has a 360 freeze frame.  The label factorizes as

    action type  ->  end zone (passes and carries)  ->  height (passes)

A sample stores only what the event-sequence-to-GATv2 model consumes: the
freeze frame at the decision moment (in the actor's team frame, as StatsBomb
provides it) and the preceding events of the possession from the *full*
event stream, so history events need no freeze frame.  Edge features and
adjacency are derived in :func:`collate_next_action`.

Decision moments.  StatsBomb starts every carry at the exact timestamp the
player gained the ball and never follows a carry with another carry of the
same player.  Sampling every action at its own timestamp would let the model
read the type off the annotation ("my own carry just ended, so this cannot be
a carry").  Decisions are therefore aligned:

* Every decision that starts a spell on the ball is anchored at the event in
  which the player gained it (ball receipt, recovery, interception, ...):
  long carries, one-touch actions, and release actions entered through the
  player's own *short* carry (a touch).  Frame, history and timing come from
  that moment, so all of them compete under identical information.
* A release action after the player's own *long* carry is a second decision at
  release time; "carry" is structurally impossible there and is removed from
  the type distribution via ``type_mask``.

``decision_context`` records which case applies (see :data:`DECISION_CONTEXTS`).

Leakage rules:

* The target frame's graph features describe the situation only (game time,
  pressure, restart type, competition, visibility); never the action type,
  height or body part.
* The target event's own categoricals and numerics are never encoded; its
  history slot carries only the time since the previous event.
* History excludes the target's (and anchor's) related events by the *other*
  team at the same timestamp: StatsBomb indexes reactions such as "Dribbled
  Past" or duels before the action they react to, which would reveal the label.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import ConcatDataset, Dataset

from soccer_ml.data.pitch_grid import DEFAULT_GRID_M, DEFAULT_GRID_N, PitchGrid
from soccer_ml.data.schema import (
    STATSBOMB_NODE_LAYOUT,
    STATSBOMB_VELOCITY_NODE_LAYOUT,
    NodeLayout,
    SoccerGraphBatch,
)
from soccer_ml.data.temporal_receiver_dataset import (
    EVENT_NUMERIC_DIM,
    _event_categoricals,
    _event_numeric_features,
    _timestamp_seconds,
)

ACTION_TYPES = ("pass", "carry", "shot", "dribble", "clearance")
ACTION_INDEX = {name: index for index, name in enumerate(ACTION_TYPES)}
_STATSBOMB_ACTIONS = {"Pass": "pass", "Carry": "carry", "Shot": "shot", "Dribble": "dribble", "Clearance": "clearance"}
# Action types with an end-zone head, in head order.
LOCATED_ACTIONS = ("pass", "carry")
PASS_HEIGHTS = ("Ground Pass", "Low Pass", "High Pass")
IGNORE_INDEX = -100

# spell_start:  anchored at the event in which the player gained the ball
# after_carry:  release after the player's own long carry; "carry" is masked out
# other:        no gain event to anchor at (first-time clearances, actions after an own dribble, ...)
DECISION_CONTEXTS = ("spell_start", "after_carry", "other")
# Events in which a player gains control of the ball; carries start at their timestamp.
GAIN_EVENTS = {"Ball Receipt*", "Ball Recovery", "Interception", "Duel", "50/50", "Goal Keeper"}

# Restarts are fixed by the preceding stoppage, so they are known before the decision.
RESTART_TYPES = ("Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in", "Penalty")
# Pass sub-types of *history* events.  "Recovery"/"Interception" describe how a
# pass was played, so they are only used for past events, never for the target.
EVENT_SUBTYPES = ("", "Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in", "Recovery", "Interception")
NUM_EVENT_SUBTYPES = len(EVENT_SUBTYPES)

# Target-frame graph features.
GRAPH_FEATURES = (
    "second_half",
    "extra_time",
    "minute_over_90",
    "under_pressure",
    "open_play",
    *(f"restart_{name.lower().replace(' ', '_').replace('-', '_')}" for name in RESTART_TYPES),
    "womens_competition",
    "visible_zone_fraction",
    "visible_teammates_over_11",
    "visible_opponents_over_11",
)
GRAPH_DIM = len(GRAPH_FEATURES)
EDGE_DIM = 1

# -- Optional features (off by default, so older checkpoints keep their inputs) --
# Role of the player on the ball from StatsBomb's position, side-agnostic so it
# is invariant under the wing reflection.
ACTOR_ROLES = (
    "goalkeeper", "center_back", "full_back", "defensive_midfield", "central_midfield",
    "attacking_midfield", "wide", "forward", "unknown_role",
)
_POSITION_ROLES = {
    "Goalkeeper": "goalkeeper",
    "Center Back": "center_back", "Left Center Back": "center_back", "Right Center Back": "center_back",
    "Left Back": "full_back", "Right Back": "full_back", "Left Wing Back": "full_back", "Right Wing Back": "full_back",
    "Center Defensive Midfield": "defensive_midfield", "Left Defensive Midfield": "defensive_midfield",
    "Right Defensive Midfield": "defensive_midfield",
    "Center Midfield": "central_midfield", "Left Center Midfield": "central_midfield",
    "Right Center Midfield": "central_midfield",
    "Center Attacking Midfield": "attacking_midfield", "Left Attacking Midfield": "attacking_midfield",
    "Right Attacking Midfield": "attacking_midfield",
    "Left Wing": "wide", "Right Wing": "wide", "Left Midfield": "wide", "Right Midfield": "wide",
    "Center Forward": "forward", "Left Center Forward": "forward", "Right Center Forward": "forward",
    "Secondary Striker": "forward",
}
# Score state before the decision, from the actor's team's perspective.
GAME_STATE_FEATURES = ("goal_difference_over_3", "is_home")
# Velocity estimation: players of each team are matched to the previous freeze
# frame at least VELOCITY_MIN_GAP and at most VELOCITY_MAX_GAP seconds earlier;
# matches implying more than MAX_SPEED yd/s (plus MATCH_SLACK yd of position noise) are rejected.
VELOCITY_MIN_GAP = 0.2
VELOCITY_MAX_GAP = 3.0
MAX_SPEED = 11.0
MATCH_SLACK = 2.0


def graph_feature_names(actor_role: bool = False, game_state: bool = False) -> tuple:
    names = GRAPH_FEATURES
    if actor_role:
        names = names + tuple(f"role_{role}" for role in ACTOR_ROLES)
    if game_state:
        names = names + GAME_STATE_FEATURES
    return names


def node_layout_for(velocities: bool = False) -> NodeLayout:
    return STATSBOMB_VELOCITY_NODE_LAYOUT if velocities else STATSBOMB_NODE_LAYOUT


def match_velocities(
    current_xy: np.ndarray,
    current_team: np.ndarray,
    previous_xy: np.ndarray,
    previous_team: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocities (pitch units / s) of current players matched to a previous frame.

    Players are matched within each team by minimum total squared distance
    (Hungarian algorithm) over the physically possible pairs: pairs further
    apart than MAX_SPEED * dt + MATCH_SLACK get a prohibitive cost *before* the
    assignment (otherwise one outlier, e.g. a player leaving the camera view,
    can pull correct pairs apart) and stay unmatched.  Returns (n, 2)
    velocities and an (n,) matched mask.
    """
    velocities = np.zeros((len(current_xy), 2))
    matched = np.zeros(len(current_xy), dtype=bool)
    limit = MAX_SPEED * dt + MATCH_SLACK
    for team in (True, False):
        current = np.flatnonzero(current_team == team)
        previous = np.flatnonzero(previous_team == team)
        if len(current) == 0 or len(previous) == 0:
            continue
        distance = np.linalg.norm(current_xy[current, None] - previous_xy[None, previous], axis=-1)
        rows, columns = linear_sum_assignment(np.where(distance <= limit, distance**2, 1e9))
        for row, column in zip(rows, columns):
            if distance[row, column] <= limit:
                velocities[current[row]] = (current_xy[current[row]] - previous_xy[previous[column]]) / dt
                matched[current[row]] = True
    return velocities, matched

# Value targets (see ``scripts/value/vaep_baseline.py``): goal and xG-sequence
# labels of the acting team scoring / conceding within the next 10 SPADL actions.
VALUE_LABELS = ("scores", "concedes", "xg_scores", "xg_concedes")
# Rewards between a decision and the next one: xG and goals of shots by the
# acting team ("our") and by the other team ("their").
REWARDS = ("our_xg", "their_xg", "our_goals", "their_goals")

# Freeze frames with fewer visible players are skipped.
_MIN_PLAYERS = 4


@lru_cache(maxsize=8)
def _match_index(matches_dir: str) -> Dict[str, tuple]:
    """Match id -> (home team id, women's competition), from StatsBomb ``matches/*/*.json``."""
    index = {}
    for path in Path(matches_dir).glob("*/*.json"):
        for match in json.loads(path.read_text()):
            home = match.get("home_team") or {}
            index[str(match["match_id"])] = (home.get("home_team_id"), home.get("home_team_gender") == "female")
    return index


def _match_info(events_path: Path) -> Optional[tuple]:
    matches_dir = events_path.parent.parent / "matches"
    return _match_index(str(matches_dir)).get(events_path.stem) if matches_dir.is_dir() else None


def _visible_zones(polygon: Sequence[float], centers: np.ndarray) -> np.ndarray:
    """Zone centers inside a flat [x1, y1, x2, y2, ...] polygon (vectorized ray casting)."""
    if len(polygon) < 6:
        return np.zeros(len(centers), dtype=bool)
    xs, ys = np.asarray(polygon[0::2], dtype=float), np.asarray(polygon[1::2], dtype=float)
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
    px, py = centers[:, :1], centers[:, 1:]
    crosses = (ys > py) != (y2 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_y = (x2 - xs) * (py - ys) / (y2 - ys) + xs
    return ((crosses & (px < x_at_y)).sum(axis=1) % 2) == 1


def _restart_type(event: Dict[str, Any]) -> str:
    for key in ("pass", "shot"):
        details = event.get(key)
        if isinstance(details, dict):
            name = (details.get("type") or {}).get("name", "")
            if name in RESTART_TYPES:
                return name
    return ""


def _event_subtype(event: Dict[str, Any]) -> int:
    details = event.get("pass")
    name = (details.get("type") or {}).get("name", "") if isinstance(details, dict) else ""
    return EVENT_SUBTYPES.index(name) if name in EVENT_SUBTYPES else 0


@dataclass
class NextActionSample:
    node_features: torch.Tensor  # (N, STATSBOMB_NODE_LAYOUT.dim)
    player_mask: torch.Tensor  # (N,) bool
    graph_features: torch.Tensor  # (GRAPH_DIM,)
    visible_zones: torch.Tensor  # (M * N,) bool
    event_type: torch.Tensor  # (T,) long, final slot = target (unknown)
    event_outcome: torch.Tensor  # (T,)
    event_body_part: torch.Tensor  # (T,)
    event_subtype: torch.Tensor  # (T,)
    event_numeric: torch.Tensor  # (T, EVENT_NUMERIC_DIM)
    time_delta: torch.Tensor  # (T,)
    sequence_mask: torch.Tensor  # (T,) bool
    action_type: torch.Tensor  # scalar long
    type_mask: torch.Tensor  # (len(ACTION_TYPES),) bool, actions possible at this decision
    decision_context: torch.Tensor  # scalar long, index into DECISION_CONTEXTS
    target_zone: torch.Tensor  # scalar long, IGNORE_INDEX if the action has no end zone
    pass_height: torch.Tensor  # scalar long, IGNORE_INDEX unless a pass
    end_location: torch.Tensor  # (2,) pitch units, zeros if no end zone
    receiver_label: torch.Tensor  # scalar long, closest visible teammate to a pass's end; IGNORE_INDEX otherwise
    # -- value targets ---------------------------------------------------------
    value_label: torch.Tensor  # (len(VALUE_LABELS),) Monte Carlo labels, NaN if unavailable
    has_value_label: torch.Tensor  # scalar bool
    success: torch.Tensor  # scalar long: pass completed / shot scored / dribble won / own team acts next; IGNORE_INDEX if unknown
    reward: torch.Tensor  # (len(REWARDS),) shots between this action and the next decision
    next_index: torch.Tensor  # scalar long, index of the next decision in this match (-1 at the end of a period)
    next_flip: torch.Tensor  # scalar bool, the next decision is the other team's
    done: torch.Tensor  # scalar bool, no further decision in this period
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class NextActionBatch:
    node_features: torch.Tensor
    edge_features: torch.Tensor
    adj_mat: torch.Tensor
    player_mask: torch.Tensor
    graph_features: torch.Tensor
    visible_zones: torch.Tensor
    event_type: torch.Tensor
    event_outcome: torch.Tensor
    event_body_part: torch.Tensor
    event_subtype: torch.Tensor
    event_numeric: torch.Tensor
    time_delta: torch.Tensor
    sequence_mask: torch.Tensor
    action_type: torch.Tensor
    type_mask: torch.Tensor
    decision_context: torch.Tensor
    target_zone: torch.Tensor
    pass_height: torch.Tensor
    end_location: torch.Tensor
    receiver_label: torch.Tensor
    value_label: torch.Tensor
    has_value_label: torch.Tensor
    success: torch.Tensor
    reward: torch.Tensor
    next_index: torch.Tensor
    next_flip: torch.Tensor
    done: torch.Tensor

    def to(self, device: torch.device) -> "NextActionBatch":
        return NextActionBatch(**{name: value.to(device) for name, value in self.__dict__.items()})

    @property
    def batch_size(self) -> int:
        return int(self.node_features.shape[0])

    def current_graph_batch(self) -> SoccerGraphBatch:
        """The target frames as a plain graph batch (for zone baselines and metrics)."""
        return SoccerGraphBatch(
            node_features=self.node_features,
            edge_features=self.edge_features,
            graph_features=self.graph_features,
            adj_mat=self.adj_mat,
            receiver_label=self.receiver_label,
            target_zone_label=self.target_zone,
            pass_end_location=self.end_location,
            player_mask=self.player_mask,
        )


def collate_next_action(samples: Sequence[NextActionSample]) -> NextActionBatch:
    """Stack samples and derive the opposing-team edge feature and adjacency."""
    if not samples:
        raise ValueError("Cannot collate an empty next-action batch.")
    stacked = {
        name: torch.stack([getattr(sample, name) for sample in samples])
        for name in NextActionSample.__dataclass_fields__
        if name != "metadata"
    }
    mask = stacked["player_mask"]
    pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    teammate = stacked["node_features"][..., STATSBOMB_NODE_LAYOUT.teammate] > 0.5
    opposing = (teammate.unsqueeze(-1) != teammate.unsqueeze(-2)) & pair_mask
    return NextActionBatch(
        edge_features=opposing.unsqueeze(-1).float(),
        adj_mat=pair_mask.float(),
        **stacked,
    )


class NextActionDataset(Dataset):
    """Next-action samples of one match.

    Args:
        context_length: Number of preceding events encoded; sequences have
            ``context_length + 1`` slots, the last one being the decision.
        context_scope: ``"possession"`` (default) or ``"period"``.
        carry_min_length: Carries shorter than this (pitch units) are not
            decisions; they stay in the history.  Defaults to one grid cell.
        womens_competition: Override the women's-competition flag; by default
            it is looked up in StatsBomb's ``matches`` folder next to ``events``.
        actor_role: Append a one-hot of the acting player's role (:data:`ACTOR_ROLES`).
        game_state: Append goal difference and home/away (:data:`GAME_STATE_FEATURES`);
            home/away is 0.5 when the ``matches`` folder is unavailable.
        velocities: Estimate player velocities by matching each team's players to
            the previous freeze frame (:func:`match_velocities`); node features
            then follow ``STATSBOMB_VELOCITY_NODE_LAYOUT``.
        value_labels: Optional event id -> :data:`VALUE_LABELS` mapping
            (:func:`load_value_labels`) for Monte Carlo value targets.
    """

    def __init__(
        self,
        three_sixty_path: Union[str, Path],
        events_path: Union[str, Path],
        context_length: int = 16,
        context_scope: Literal["possession", "period"] = "possession",
        grid_m: int = DEFAULT_GRID_M,
        grid_n: int = DEFAULT_GRID_N,
        carry_min_length: Optional[float] = None,
        max_samples: Optional[int] = None,
        womens_competition: Optional[bool] = None,
        pitch_length: float = 120.0,
        pitch_width: float = 80.0,
        target_scale: float = 10.0,
        max_players: int = 22,
        value_labels: Optional[Mapping[str, Sequence[float]]] = None,
        actor_role: bool = False,
        game_state: bool = False,
        velocities: bool = False,
    ):
        if context_scope not in {"possession", "period"}:
            raise ValueError("context_scope must be 'possession' or 'period'.")
        self.three_sixty_path = Path(three_sixty_path)
        self.events_path = Path(events_path)
        self.context_length = context_length
        self.context_scope = context_scope
        self.grid = PitchGrid(grid_m, grid_n, pitch_length, pitch_width)
        self.carry_min_length = self.grid.cell_length if carry_min_length is None else carry_min_length
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.target_scale = target_scale
        self.max_players = max_players
        info = _match_info(self.events_path)
        self.home_team_id = info[0] if info else None
        self.womens_competition = bool(info and info[1]) if womens_competition is None else womens_competition
        self.actor_role = actor_role
        self.game_state = game_state
        self.velocities = velocities
        self.node_layout = node_layout_for(velocities)
        self.graph_feature_names = graph_feature_names(actor_role, game_state)
        self._zone_centers = self.grid.zone_centers().numpy()
        self.value_labels = value_labels
        self._load(max_samples)

    # -- parsing --------------------------------------------------------------

    def _is_target(self, event: Dict[str, Any]) -> bool:
        action = _STATSBOMB_ACTIONS.get((event.get("type") or {}).get("name", ""))
        if action is None or not isinstance(event.get("location"), list):
            return False
        if action == "pass":
            details = event.get("pass") or {}
            outcome = (details.get("outcome") or {}).get("name", "")
            return isinstance(details.get("end_location"), list) and outcome != "Injury Clearance"
        if action == "carry":
            end = (event.get("carry") or {}).get("end_location")
            return isinstance(end, list) and math.dist(event["location"][:2], end[:2]) >= self.carry_min_length
        return True

    def _load(self, max_samples: Optional[int]) -> None:
        frames = {frame["event_uuid"]: frame for frame in json.loads(self.three_sixty_path.read_text())}
        events = sorted(json.loads(self.events_path.read_text()), key=lambda event: event.get("index", 0))
        self._frame_positions = [i for i, event in enumerate(events) if event.get("id") in frames]
        self._score_before = self._score_states(events) if self.game_state else None
        samples: List[NextActionSample] = []
        for position, event in enumerate(events):
            frame = frames.get(event.get("id"))
            if frame is None or not self._is_target(event):
                continue
            sample = self._make_sample(events, position, event, frames)
            if sample is None:
                continue
            samples.append(sample)
            if max_samples is not None and len(samples) >= max_samples:
                break
        self._add_value_targets(events, samples)
        del self._frame_positions, self._score_before
        # Store one stacked tensor per field: per-sample tensor objects would
        # cost several KB of Python overhead each at ~650k samples.
        fields = [name for name in NextActionSample.__dataclass_fields__ if name != "metadata"]
        self._tensors: Dict[str, torch.Tensor] = (
            {name: torch.stack([getattr(sample, name) for sample in samples]) for name in fields} if samples else {}
        )
        self._metadata: List[Dict[str, Any]] = [sample.metadata or {} for sample in samples]

    def _add_value_targets(self, events: Sequence[Dict[str, Any]], samples: List[NextActionSample]) -> None:
        """Monte Carlo labels, success, rewards and links to the next decision of every sample."""
        by_anchor = sorted(range(len(samples)), key=lambda i: samples[i].metadata["anchor_position"])
        for rank, i in enumerate(by_anchor):
            sample, meta = samples[i], samples[i].metadata
            event = events[meta["target_position"]]
            following = by_anchor[rank + 1] if rank + 1 < len(by_anchor) else None
            if following is not None and samples[following].metadata["period"] != meta["period"]:
                following = None
            # Rewards: shots from this action up to the next decision (or the end of the period).
            end = samples[following].metadata["anchor_position"] if following is not None else len(events)
            reward = [0.0] * len(REWARDS)
            for position in range(meta["target_position"], end):
                shot = events[position]
                if shot.get("period") != meta["period"]:
                    break
                if (shot.get("type") or {}).get("name") != "Shot":
                    continue
                ours = (shot.get("team") or {}).get("id") == meta["team_id"]
                details = shot.get("shot") or {}
                reward[0 if ours else 1] += float(details.get("statsbomb_xg", 0.0) or 0.0)
                reward[2 if ours else 3] += float((details.get("outcome") or {}).get("name") == "Goal")
            sample.reward = torch.tensor(reward)
            sample.next_index = torch.tensor(following if following is not None else -1)
            sample.done = torch.tensor(following is None)
            flip = following is not None and samples[following].metadata["team_id"] != meta["team_id"]
            sample.next_flip = torch.tensor(flip)

            action = ACTION_TYPES[int(sample.action_type)]
            if action == "pass":
                success = int("outcome" not in (event.get("pass") or {}))
            elif action == "shot":
                success = int(((event.get("shot") or {}).get("outcome") or {}).get("name") == "Goal")
            elif action == "dribble":
                success = int(((event.get("dribble") or {}).get("outcome") or {}).get("name") == "Complete")
            else:  # carries and clearances succeed if the own team makes the next decision
                success = IGNORE_INDEX if following is None else int(not flip)
            sample.success = torch.tensor(success)

            label = self.value_labels.get(event["id"]) if self.value_labels is not None else None
            sample.value_label = torch.tensor(label if label is not None else [float("nan")] * len(VALUE_LABELS))
            sample.has_value_label = torch.tensor(label is not None)

    def _frame_tensors(self, frame: Dict[str, Any]) -> Optional[tuple]:
        players = frame.get("freeze_frame", [])[: self.max_players]
        if len(players) < _MIN_PLAYERS or not any(player.get("actor") for player in players):
            return None
        layout = self.node_layout
        nodes = torch.zeros(self.max_players, layout.dim)
        mask = torch.zeros(self.max_players, dtype=torch.bool)
        positions = []
        for index, player in enumerate(players):
            x, y = (player.get("location") or [self.pitch_length / 2, self.pitch_width / 2])[:2]
            positions.append((float(x), float(y)))
            nodes[index, layout.x] = (x - self.pitch_length / 2) / self.pitch_length * self.target_scale
            nodes[index, layout.y] = (y - self.pitch_width / 2) / self.pitch_width * self.target_scale
            nodes[index, layout.actor] = float(bool(player.get("actor")))
            nodes[index, layout.teammate] = float(bool(player.get("teammate")))
            nodes[index, layout.keeper] = float(bool(player.get("keeper")))
            mask[index] = True
        return nodes, mask, positions, players

    def _decision_anchor(self, events: Sequence[Dict[str, Any]], position: int) -> tuple[int, List[int], int, bool]:
        """Locate the decision moment of the target at ``position``.

        Returns the anchor position, the positions whose freeze frames may
        represent the decision (in order of preference), the decision context
        and whether "carry" is impossible.
        """
        target = events[position]
        player = (target.get("player") or {}).get("id")
        is_carry = target["type"]["name"] == "Carry"
        frame_positions = [position]
        for index in range(position - 1, -1, -1):
            event = events[index]
            if event.get("period") != target.get("period") or event.get("possession") != target.get("possession"):
                break
            if (event.get("player") or {}).get("id") != player:
                continue
            kind = (event.get("type") or {}).get("name")
            if kind == "Carry" and not is_carry:
                end = (event.get("carry") or {}).get("end_location")
                if isinstance(end, list) and math.dist(event["location"][:2], end[:2]) >= self.carry_min_length:
                    return position, [position], DECISION_CONTEXTS.index("after_carry"), True
                frame_positions.insert(0, index)  # a touch: keep looking for the gain event
                continue
            if kind in GAIN_EVENTS:
                return index, [index, *frame_positions], DECISION_CONTEXTS.index("spell_start"), False
            break
        # No gain event: anchor at a skipped touch if there is one, else at the action itself.
        return frame_positions[0], frame_positions, DECISION_CONTEXTS.index("other"), False

    def _history(
        self, events: Sequence[Dict[str, Any]], anchor_position: int, target: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        anchor = events[anchor_position]
        related = set(target.get("related_events", [])) | set(anchor.get("related_events", []))
        moments = {target.get("timestamp"), anchor.get("timestamp")}
        team = (target.get("team") or {}).get("id")
        selected: List[Dict[str, Any]] = []
        for index in range(anchor_position - 1, -1, -1):
            event = events[index]
            if event.get("period") != target.get("period"):
                break
            if self.context_scope == "possession" and event.get("possession") != target.get("possession"):
                break
            if not isinstance(event.get("location"), list):
                continue
            if (
                event.get("id") in related
                and (event.get("team") or {}).get("id") != team
                and event.get("timestamp") in moments
            ):
                continue
            selected.append(event)
            if len(selected) >= self.context_length:
                break
        selected.reverse()
        return selected

    @staticmethod
    def _score_states(events: Sequence[Dict[str, Any]]) -> List[Dict[Any, int]]:
        """Goals per team scored before each event (penalty shoot-outs excluded)."""
        score: Dict[Any, int] = {}
        states = []
        for event in events:
            states.append(dict(score))
            kind = (event.get("type") or {}).get("name")
            team = (event.get("team") or {}).get("id")
            if event.get("period", 1) > 4:
                continue
            if kind == "Shot" and ((event.get("shot") or {}).get("outcome") or {}).get("name") == "Goal":
                score[team] = score.get(team, 0) + 1
            elif kind == "Own Goal For":
                score[team] = score.get(team, 0) + 1
        return states

    def _add_velocities(
        self,
        nodes: torch.Tensor,
        positions: List[tuple],
        players: List[Dict[str, Any]],
        events: Sequence[Dict[str, Any]],
        frame_position: int,
        frames: Dict[str, Dict[str, Any]],
    ) -> None:
        """Fill the velocity channels from the most recent earlier freeze frame of the same period."""
        layout = self.node_layout
        current = events[frame_position]
        now = _timestamp_seconds(current)
        index = bisect.bisect_left(self._frame_positions, frame_position) - 1
        while index >= 0:
            previous = events[self._frame_positions[index]]
            if previous.get("period") != current.get("period"):
                return
            dt = now - _timestamp_seconds(previous)
            if dt > VELOCITY_MAX_GAP:
                return
            if dt >= VELOCITY_MIN_GAP:
                break
            index -= 1
        else:
            return
        previous_players = frames[previous["id"]].get("freeze_frame", [])
        if not previous_players:
            return
        # Frames are in the frame of the team performing their event: rotate if that differs.
        rotate = (previous.get("team") or {}).get("id") != (current.get("team") or {}).get("id")
        previous_xy = np.array([(p.get("location") or [60.0, 40.0])[:2] for p in previous_players], dtype=float)
        previous_team = np.array([bool(p.get("teammate")) for p in previous_players])
        if rotate:
            previous_xy = np.array([self.pitch_length, self.pitch_width]) - previous_xy
            previous_team = ~previous_team
        current_xy = np.array(positions, dtype=float)
        current_team = np.array([bool(p.get("teammate")) for p in players])
        velocity, matched = match_velocities(current_xy, current_team, previous_xy, previous_team, dt)
        count = len(players)
        # Same normalization as the coordinates: pitch units -> target_scale box.
        nodes[:count, layout.vx] = torch.from_numpy(velocity[:, 0] / self.pitch_length * self.target_scale).float()
        nodes[:count, layout.vy] = torch.from_numpy(velocity[:, 1] / self.pitch_width * self.target_scale).float()
        nodes[:count, layout.dim - 1] = torch.from_numpy(matched.astype(np.float32))

    def _graph_features(
        self,
        event: Dict[str, Any],
        anchor: Dict[str, Any],
        visible: np.ndarray,
        players: List[Dict[str, Any]],
        anchor_position: int,
    ) -> torch.Tensor:
        """Situation at the decision moment (``anchor``); the restart type is fixed by the stoppage."""
        features = dict.fromkeys(self.graph_feature_names, 0.0)
        period = int(anchor.get("period", 1))
        features["second_half"] = float(period >= 2)
        features["extra_time"] = float(period >= 3)
        features["minute_over_90"] = float(anchor.get("minute", 0)) / 90.0
        features["under_pressure"] = float(bool(anchor.get("under_pressure", False)))
        restart = _restart_type(event)
        if restart:
            features[f"restart_{restart.lower().replace(' ', '_').replace('-', '_')}"] = 1.0
        else:
            features["open_play"] = 1.0
        features["womens_competition"] = float(self.womens_competition)
        features["visible_zone_fraction"] = float(visible.mean())
        teammates = sum(bool(player.get("teammate")) for player in players)
        features["visible_teammates_over_11"] = teammates / 11.0
        features["visible_opponents_over_11"] = (len(players) - teammates) / 11.0
        if self.actor_role:
            role = _POSITION_ROLES.get((event.get("position") or {}).get("name"), "unknown_role")
            features[f"role_{role}"] = 1.0
        if self.game_state:
            team = (event.get("team") or {}).get("id")
            score = self._score_before[anchor_position]
            own = score.get(team, 0)
            other = sum(goals for side, goals in score.items() if side != team)
            features["goal_difference_over_3"] = max(-3, min(3, own - other)) / 3.0
            features["is_home"] = 0.5 if self.home_team_id is None else float(team == self.home_team_id)
        return torch.tensor([features[name] for name in self.graph_feature_names], dtype=torch.float32)

    def _make_sample(
        self, events: Sequence[Dict[str, Any]], position: int, event: Dict[str, Any], frames: Dict[str, Dict[str, Any]]
    ) -> Optional[NextActionSample]:
        anchor_position, frame_positions, context, carry_masked = self._decision_anchor(events, position)
        anchor = events[anchor_position]
        # The anchor's freeze frame, else one at the same moment (the touch), else the action's own.
        frame_position = next(i for i in frame_positions if events[i].get("id") in frames)
        frame = frames[events[frame_position]["id"]]
        parsed = self._frame_tensors(frame)
        if parsed is None:
            return None
        nodes, mask, positions, players = parsed
        if self.velocities:
            self._add_velocities(nodes, positions, players, events, frame_position, frames)
        visible = _visible_zones(frame.get("visible_area", []), self._zone_centers)

        action = _STATSBOMB_ACTIONS[event["type"]["name"]]
        target_zone, height, receiver = IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX
        end_location = torch.zeros(2)
        if action in LOCATED_ACTIONS:
            end = (event.get(action) or {})["end_location"]
            end_location = torch.tensor([float(end[0]), float(end[1])])
            target_zone = self.grid.coords_to_zone(end[0], end[1])
        if action == "pass":
            height_name = ((event.get("pass") or {}).get("height") or {}).get("name", "")
            height = PASS_HEIGHTS.index(height_name) if height_name in PASS_HEIGHTS else IGNORE_INDEX
            best = float("inf")
            for index, player in enumerate(players):
                if player.get("teammate") and not player.get("actor"):
                    distance = math.dist(positions[index], (end_location[0].item(), end_location[1].item()))
                    if distance < best:
                        best, receiver = distance, index

        steps = self.context_length + 1
        event_type = torch.zeros(steps, dtype=torch.long)
        event_outcome = torch.zeros(steps, dtype=torch.long)
        event_body_part = torch.zeros(steps, dtype=torch.long)
        event_subtype = torch.zeros(steps, dtype=torch.long)
        event_numeric = torch.zeros(steps, EVENT_NUMERIC_DIM)
        time_delta = torch.zeros(steps)
        sequence_mask = torch.zeros(steps, dtype=torch.bool)

        history = self._history(events, anchor_position, event)
        target_team = (event.get("team") or {}).get("id")
        start = steps - 1 - len(history)
        previous_timestamp: Optional[float] = None
        for offset, past in enumerate(history):
            slot = start + offset
            timestamp = _timestamp_seconds(past)
            delta = 0.0 if previous_timestamp is None else max(timestamp - previous_timestamp, 0.0)
            previous_timestamp = timestamp
            type_id, outcome_id, body_part_id = _event_categoricals(past)
            event_type[slot], event_outcome[slot], event_body_part[slot] = type_id, outcome_id, body_part_id
            event_subtype[slot] = _event_subtype(past)
            same_team = (past.get("team") or {}).get("id") == target_team
            event_numeric[slot] = _event_numeric_features(past, delta, same_team, self.pitch_length, self.pitch_width)
            time_delta[slot] = delta
            sequence_mask[slot] = True
        # Decision slot: only the time since the previous event is known.
        target_timestamp = _timestamp_seconds(anchor)
        time_delta[-1] = 0.0 if previous_timestamp is None else max(target_timestamp - previous_timestamp, 0.0)
        sequence_mask[-1] = True

        return NextActionSample(
            node_features=nodes,
            player_mask=mask,
            graph_features=self._graph_features(event, anchor, visible, players, anchor_position),
            visible_zones=torch.from_numpy(visible),
            event_type=event_type,
            event_outcome=event_outcome,
            event_body_part=event_body_part,
            event_subtype=event_subtype,
            event_numeric=event_numeric,
            time_delta=time_delta,
            sequence_mask=sequence_mask,
            action_type=torch.tensor(ACTION_INDEX[action]),
            type_mask=torch.tensor([not (carry_masked and name == "carry") for name in ACTION_TYPES]),
            decision_context=torch.tensor(context),
            target_zone=torch.tensor(target_zone),
            pass_height=torch.tensor(height),
            end_location=end_location,
            receiver_label=torch.tensor(receiver),
            # Value targets are filled in by _add_value_targets once all decisions are known.
            value_label=torch.zeros(len(VALUE_LABELS)),
            has_value_label=torch.tensor(False),
            success=torch.tensor(IGNORE_INDEX),
            reward=torch.zeros(len(REWARDS)),
            next_index=torch.tensor(-1),
            next_flip=torch.tensor(False),
            done=torch.tensor(True),
            metadata={
                "event_uuid": event["id"],
                "anchor_uuid": anchor["id"],
                "anchor_position": anchor_position,
                "target_position": position,
                "period": event.get("period"),
                "team_id": (event.get("team") or {}).get("id"),
                "match_id": self.events_path.stem,
                "history_events": len(history),
            },
        )

    @property
    def samples(self) -> List[NextActionSample]:
        return [self[index] for index in range(len(self))]

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, index: int) -> NextActionSample:
        return NextActionSample(
            **{name: tensor[index] for name, tensor in self._tensors.items()}, metadata=self._metadata[index]
        )


@lru_cache(maxsize=2)
def load_value_labels(path: Union[str, Path]) -> Dict[str, tuple]:
    """Event id -> :data:`VALUE_LABELS` from ``value_labels_360.csv`` (SPADL ``original_event_id``)."""
    labels: Dict[str, tuple] = {}
    with Path(path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            labels[row["original_event_id"]] = tuple(float(row[name]) for name in VALUE_LABELS)
    return labels


class TransitionDataset(Dataset):
    """Pairs every decision with the next one of the same match (for temporal-difference targets)
    and its dataset index (for precomputed targets such as λ-returns).

    At the end of a period the sample itself stands in as the next state; its
    ``done`` flag makes the bootstrap term vanish.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        if isinstance(dataset, ConcatDataset):
            self._offsets = [0, *dataset.cumulative_sizes[:-1]]
            self._parts = dataset.datasets
        else:
            self._offsets, self._parts = [0], [dataset]
        self._starts = torch.tensor([*self._offsets, len(dataset)])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[NextActionSample, NextActionSample, int]:
        sample = self.dataset[index]
        part = int(torch.searchsorted(self._starts, torch.tensor(index), right=True)) - 1
        following = int(sample.next_index)
        next_sample = self._parts[part][following] if following >= 0 else sample
        return sample, next_sample, index


def collate_transitions(
    items: Sequence[tuple[NextActionSample, NextActionSample, int]]
) -> tuple[NextActionBatch, NextActionBatch, torch.Tensor]:
    """(decisions, next decisions, dataset indices of the decisions)."""
    return (
        collate_next_action([item[0] for item in items]),
        collate_next_action([item[1] for item in items]),
        torch.tensor([item[2] for item in items]),
    )


def _parts(dataset: Dataset) -> List[tuple]:
    """(offset, NextActionDataset) for a dataset or a ConcatDataset of them."""
    if isinstance(dataset, ConcatDataset):
        return list(zip([0, *dataset.cumulative_sizes[:-1]], dataset.datasets))
    return [(0, dataset)]


def td_rewards_tensor(dataset: Dataset, kind: str = "xg") -> torch.Tensor:
    """Per-decision (our, their) rewards of every sample, ``(N, 2)``."""
    names = ("our_xg", "their_xg") if kind == "xg" else ("our_goals", "their_goals")
    columns = [REWARDS.index(name) for name in names]
    return torch.cat([part._tensors["reward"][:, columns] for _, part in _parts(dataset)])


def lambda_returns(
    dataset: Dataset,
    gamma: float,
    lam: float,
    bootstrap: Optional[torch.Tensor] = None,
    reward_kind: str = "xg",
) -> torch.Tensor:
    """λ-returns of every decision, ``(N, 2)`` (score, concede components).

        G_t = r_t + γ · swap_t[(1 - λ) · V_{t+1} + λ · G_{t+1}]

    where V_{t+1} = ``bootstrap[next]`` (values of the next decision in its
    actor's perspective), ``swap_t`` exchanges the components when the other
    team makes the next decision, and the bootstrap vanishes at the end of a
    period.  λ = 1 gives the discounted Monte Carlo return (no bootstrap
    needed); λ = 0 gives the one-step TD target.  Dead balls are *not*
    episode ends: the restart is the next decision and carries its own value.
    """
    if lam < 1.0 and bootstrap is None:
        raise ValueError("λ < 1 needs bootstrap values of every decision.")
    rewards = td_rewards_tensor(dataset, reward_kind)
    returns = torch.zeros_like(rewards)
    for offset, part in _parts(dataset):
        tensors, metadata = part._tensors, part._metadata
        order = sorted(range(len(part)), key=lambda i: metadata[i]["anchor_position"], reverse=True)
        for local in order:
            index = offset + local
            following = int(tensors["next_index"][local])
            if bool(tensors["done"][local]) or following < 0:
                returns[index] = rewards[index]
                continue
            nxt = offset + following
            continuation = lam * returns[nxt]
            if lam < 1.0:
                continuation = continuation + (1.0 - lam) * bootstrap[nxt]
            if bool(tensors["next_flip"][local]):
                continuation = continuation.flip(0)
            returns[index] = rewards[index] + gamma * continuation
    return returns
