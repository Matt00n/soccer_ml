"""Match context of every decision, reconstructed from StatsBomb events and match files.

For second-order analyses (score reactions, substitutions, red cards, on/off
splits, sequences) every decision needs the state *before* it happened, and
the analyses need the match's events as tables:

* ``decisions``: per decision event, the actor team's goals for and against,
  players on the pitch for both teams, time in the period, last goal;
* ``goals``, ``substitutions``, ``red_cards``, ``tactical_shifts``: one row per event;
* ``appearances``: every player's on-pitch interval(s) per match, and starters;
* ``possessions``: possession team, shots, xG and goal of every possession;
* ``matches``: teams, final score, result, competition, stage, date.

Times are ``(period, clock)`` with ``clock = minute + second / 60`` in StatsBomb's
match clock (the second half starts at 45:00), so windows never cross periods.
The penalty shoot-out (period 5) is ignored.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from soccer_ml.creativity.events import load_matches

RED_CARDS = ("Red Card", "Second Yellow")


def _clock(event: Dict[str, Any]) -> float:
    return float(event.get("minute", 0)) + float(event.get("second", 0)) / 60.0


def _card(event: Dict[str, Any]) -> str:
    for key in ("foul_committed", "bad_behaviour"):
        name = ((event.get(key) or {}).get("card") or {}).get("name", "")
        if name:
            return name
    return ""


def match_context(events_path: Path, info: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """All context tables of one match (lists of row dicts)."""
    match_id = events_path.stem
    events = sorted(json.loads(events_path.read_text()), key=lambda e: e.get("index", 0))
    events = [e for e in events if e.get("period", 1) <= 4]
    teams = list(dict.fromkeys((e.get("team") or {}).get("name") for e in events if e.get("team")))
    goals_for = defaultdict(int)
    on_pitch: Dict[str, set] = defaultdict(set)
    entered: Dict[tuple, tuple] = {}  # (team, player id) -> (period, clock) of entering
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    last_goal = None
    last_clock = {}
    for e in events:
        kind = (e.get("type") or {}).get("name", "")
        team = (e.get("team") or {}).get("name")
        period, clock = int(e.get("period", 1)), _clock(e)
        last_clock[period] = clock
        opponent = next((t for t in teams if t != team), None)
        if kind == "Starting XI":
            for entry in (e.get("tactics") or {}).get("lineup", []):
                pid = entry["player"]["id"]
                on_pitch[team].add(pid)
                entered[(team, pid)] = (1, 0.0)
                out["appearances_start"].append({"match_id": match_id, "team": team, "player_id": pid,
                                                 "player": entry["player"]["name"],
                                                 "position": (entry.get("position") or {}).get("name", "")})
            continue
        # State before this event, from the actor team's perspective.
        if e.get("id") and team:
            out["decisions"].append({
                "event_uuid": e["id"], "period_ctx": period, "clock": clock,
                "goals_for": goals_for[team], "goals_against": goals_for[opponent] if opponent else 0,
                "players_own": len(on_pitch[team]), "players_opponent": len(on_pitch[opponent]) if opponent else 11,
                "last_goal_period": last_goal[0] if last_goal else None,
                "last_goal_clock": last_goal[1] if last_goal else None,
                "last_goal_by_own": (last_goal[2] == team) if last_goal else None,
            })
        scorer = None
        if kind == "Shot" and ((e.get("shot") or {}).get("outcome") or {}).get("name") == "Goal":
            scorer = team
        elif kind == "Own Goal Against":
            scorer = opponent
        if scorer:
            goals_for[scorer] += 1
            conceding = next((t for t in teams if t != scorer), None)
            out["goals"].append({"match_id": match_id, "period": period, "clock": clock, "team": scorer,
                                 "conceding_team": conceding, "own_goal": kind == "Own Goal Against",
                                 "player": (e.get("player") or {}).get("name", ""), "possession": e.get("possession"),
                                 "play_pattern": (e.get("play_pattern") or {}).get("name", ""),
                                 "score_for_after": goals_for[scorer], "score_against_after": goals_for[conceding]})
            last_goal = (period, clock, scorer)
        if kind == "Substitution":
            off = (e.get("player") or {}).get("id")
            on = ((e.get("substitution") or {}).get("replacement") or {}).get("id")
            on_pitch[team].discard(off)
            if on is not None:
                on_pitch[team].add(on)
                entered[(team, on)] = (period, clock)
            start = entered.pop((team, off), (1, 0.0))
            out["intervals"].append({"match_id": match_id, "team": team, "player_id": off, "start_period": start[0],
                                     "start_clock": start[1], "end_period": period, "end_clock": clock, "end": "subbed off"})
            out["substitutions"].append({
                "match_id": match_id, "period": period, "clock": clock, "team": team, "player_off_id": off,
                "player_off": (e.get("player") or {}).get("name", ""), "player_on_id": on,
                "player_on": ((e.get("substitution") or {}).get("replacement") or {}).get("name", ""),
                "reason": ((e.get("substitution") or {}).get("outcome") or {}).get("name", ""),
                "goals_for": goals_for[team], "goals_against": goals_for[opponent] if opponent else 0,
            })
        if _card(e) in RED_CARDS:
            player = (e.get("player") or {}).get("id")
            if player in on_pitch[team]:
                on_pitch[team].discard(player)
                start = entered.pop((team, player), (1, 0.0))
                out["intervals"].append({"match_id": match_id, "team": team, "player_id": player,
                                         "start_period": start[0], "start_clock": start[1], "end_period": period,
                                         "end_clock": clock, "end": "sent off"})
            out["red_cards"].append({"match_id": match_id, "period": period, "clock": clock, "team": team,
                                     "player_id": player, "player": (e.get("player") or {}).get("name", ""),
                                     "card": _card(e), "goals_for": goals_for[team],
                                     "goals_against": goals_for[opponent] if opponent else 0})
        if kind == "Tactical Shift":
            out["tactical_shifts"].append({"match_id": match_id, "period": period, "clock": clock, "team": team,
                                           "formation": (e.get("tactics") or {}).get("formation"),
                                           "goals_for": goals_for[team],
                                           "goals_against": goals_for[opponent] if opponent else 0})
    end_period = max(last_clock) if last_clock else 2
    for (team, pid), start in entered.items():
        out["intervals"].append({"match_id": match_id, "team": team, "player_id": pid, "start_period": start[0],
                                 "start_clock": start[1], "end_period": end_period,
                                 "end_clock": last_clock.get(end_period, 90.0), "end": "full time"})

    # Possessions: possession team, shots, xG and goals of that team in the possession.
    possessions: Dict[int, Dict[str, Any]] = {}
    for e in events:
        pid = e.get("possession")
        if pid is None:
            continue
        row = possessions.setdefault(pid, {"match_id": match_id, "possession": pid,
                                           "possession_team": (e.get("possession_team") or {}).get("name"),
                                           "play_pattern": (e.get("play_pattern") or {}).get("name", ""),
                                           "period": e.get("period"), "start_clock": _clock(e),
                                           "shots": 0, "xg": 0.0, "goal": False})
        if (e.get("type") or {}).get("name") == "Shot" and (e.get("team") or {}).get("name") == row["possession_team"]:
            row["shots"] += 1
            row["xg"] += float((e.get("shot") or {}).get("statsbomb_xg", 0.0) or 0.0)
            row["goal"] |= ((e.get("shot") or {}).get("outcome") or {}).get("name") == "Goal"
    out["possessions"] = list(possessions.values())

    final = {team: goals_for[team] for team in teams}
    for team in teams:
        opponent = next((t for t in teams if t != team), None)
        own, other = final[team], final.get(opponent, 0)
        out["team_matches"].append({
            "match_id": match_id, "team": team, "opponent": opponent, "goals_for": own, "goals_against": other,
            "result": "win" if own > other else ("draw" if own == other else "loss"),
            "points": 3 if own > other else (1 if own == other else 0),
            "competition": info.get("competition"), "season": info.get("season"), "stage": info.get("stage"),
            "match_date": info.get("match_date"), "women": info.get("women"),
            "home": team == info.get("home_team"),
        })
    return out


def build_context(match_ids: Iterable[str], data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Context tables of all matches (see the module docstring)."""
    matches = load_matches(str(data_dir / "matches"))
    tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for match_id in sorted(set(map(str, match_ids))):
        for name, rows in match_context(data_dir / "events" / f"{match_id}.json", matches.get(match_id, {})).items():
            tables[name].extend(rows)
    frames = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    starters = frames.pop("appearances_start")
    frames["appearances"] = frames["intervals"].merge(
        starters[["match_id", "team", "player_id", "position"]].assign(starter=True),
        on=["match_id", "team", "player_id"], how="left").fillna({"starter": False})
    frames.pop("intervals")
    return frames
