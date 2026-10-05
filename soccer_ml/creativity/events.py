"""StatsBomb match and event details joined to scored decisions.

The annotations collected here (through balls, dummies, nutmegs, backheels,
miscommunications, ...) are never model inputs for the decision itself, so
they serve as independent positive and negative controls for creativity
measures (:mod:`soccer_ml.creativity.validation`).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

RESTARTS = ("Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in")
FLAIR_SHOT_TECHNIQUES = ("Backheel", "Lob", "Overhead Kick")


@lru_cache(maxsize=4)
def load_matches(matches_dir: str) -> Dict[str, Dict[str, Any]]:
    """Match id -> competition, season, date, teams and result."""
    matches: Dict[str, Dict[str, Any]] = {}
    for path in Path(matches_dir).glob("*/*.json"):
        for match in json.loads(path.read_text()):
            home, away = match["home_team"], match["away_team"]
            matches[str(match["match_id"])] = {
                "competition": match["competition"]["competition_name"],
                "season": match["season"]["season_name"],
                "match_date": match.get("match_date", ""),
                "stage": (match.get("competition_stage") or {}).get("name", ""),
                "home_team": home["home_team_name"],
                "away_team": away["away_team_name"],
                "home_score": match.get("home_score"),
                "away_score": match.get("away_score"),
                "women": home.get("home_team_gender") == "female",
            }
    return matches


def _name(value: Any) -> str:
    return value.get("name", "") if isinstance(value, dict) else ""


def _number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else float("nan")


def event_fields(event: Dict[str, Any]) -> Dict[str, Any]:
    """Player, timing and action annotations of one decision's event."""
    kind = _name(event.get("type")).lower()
    details = event.get(kind) or {}
    location = event.get("location") or [float("nan"), float("nan")]
    end = details.get("end_location") or [float("nan"), float("nan")]
    pass_type = _name(details.get("type")) if kind == "pass" else ""
    technique = _name(details.get("technique"))
    body_part = _name(details.get("body_part"))
    return {
        "player_id": (event.get("player") or {}).get("id"),
        "player": _name(event.get("player")),
        "team": _name(event.get("team")),
        "position": _name(event.get("position")),
        "minute": event.get("minute"),
        "second": event.get("second"),
        "timestamp": event.get("timestamp", ""),
        "play_pattern": _name(event.get("play_pattern")),
        "possession": event.get("possession"),
        "under_pressure": bool(event.get("under_pressure", False)),
        "start_x": float(location[0]),
        "start_y": float(location[1]),
        "end_x": float(end[0]),
        "end_y": float(end[1]),
        "outcome": _name(details.get("outcome")),
        "restart": pass_type if pass_type in RESTARTS else "",
        "height": _name(details.get("height")),
        "body_part": body_part,
        "technique": technique,
        "recipient": _name(details.get("recipient")),
        "length": _number(details.get("length")),
        # Annotation flags (independent of the model's inputs for this decision).
        "through_ball": bool(details.get("through_ball")) or technique == "Through Ball",
        "switch": bool(details.get("switch")),
        "cross": bool(details.get("cross")),
        "cut_back": bool(details.get("cut_back")),
        "shot_assist": bool(details.get("shot_assist")),
        "goal_assist": bool(details.get("goal_assist")),
        "no_touch": body_part == "No Touch",
        "miscommunication": bool(details.get("miscommunication")),
        "deflected": bool(details.get("deflected")),
        "nutmeg": bool(details.get("nutmeg")),
        "overrun": bool(details.get("overrun")),
        "first_time": bool(details.get("first_time")),
        "flair_shot": technique in FLAIR_SHOT_TECHNIQUES,
        "shot_xg": _number(details.get("statsbomb_xg")) if kind == "shot" else float("nan"),
    }


def load_nicknames(lineups_dir: Path, match_ids) -> Dict[int, str]:
    """Player id -> StatsBomb nickname (e.g. 'Pedri'), from the lineups of the given matches."""
    nicknames: Dict[int, str] = {}
    for match_id in set(match_ids):
        path = Path(lineups_dir) / f"{match_id}.json"
        if not path.exists():
            continue
        for team in json.loads(path.read_text()):
            for player in team.get("lineup", []):
                if player.get("player_nickname"):
                    nicknames[player["player_id"]] = player["player_nickname"]
    return nicknames
