"""Expected Threat (xT; Karun Singh, 2018) on a StatsBomb pitch grid.

xT(z) is the probability that a team holding the ball in zone z scores, via
a Markov chain over zones fitted from open-play events:

    xT(z) = P(shot | z) * xG(z) + P(move | z) * sum_z' T(z -> z') * xT(z')

* P(shot | z), P(move | z): share of shots and moves (passes, carries) among
  the on-ball actions started in z;
* xG(z): mean StatsBomb xG of the shots from z (smoother than the goal rate);
* T(z -> z'): successful moves z -> z' divided by *all* moves from z, so a
  failed move ends the chain with value 0.

Coordinates are StatsBomb's (120 x 80 yards, the acting team attacking
towards x = 120).  Zones follow :class:`~soccer_ml.data.pitch_grid.PitchGrid`:
index = bin_x * grid_n + bin_y.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np

from soccer_ml.data.pitch_grid import PitchGrid

SET_PIECE_PASSES = {"Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"}


class ExpectedThreat:
    def __init__(self, grid_m: int = 16, grid_n: int = 12):
        self.grid = PitchGrid(grid_m, grid_n)
        z = self.grid.num_zones
        self.shots = np.zeros(z)
        self.shot_xg = np.zeros(z)
        self.moves = np.zeros(z)
        self.transitions = np.zeros((z, z))
        self.values: Optional[np.ndarray] = None
        self.num_matches = 0

    def _zone(self, location: Sequence[float]) -> int:
        return self.grid.coords_to_zone(location[0], location[1])

    def update(self, events: Iterable[Dict[str, Any]]) -> None:
        """Add the open-play shots, passes and carries of one match."""
        for event in events:
            kind = (event.get("type") or {}).get("name")
            start = event.get("location")
            if not isinstance(start, list):
                continue
            if kind == "Shot":
                shot = event.get("shot") or {}
                if (shot.get("type") or {}).get("name") != "Open Play":
                    continue
                zone = self._zone(start)
                self.shots[zone] += 1
                self.shot_xg[zone] += float(shot.get("statsbomb_xg", 0.0) or 0.0)
            elif kind == "Pass":
                details = event.get("pass") or {}
                if (details.get("type") or {}).get("name") in SET_PIECE_PASSES:
                    continue
                outcome = (details.get("outcome") or {}).get("name")
                if outcome == "Injury Clearance":
                    continue
                zone = self._zone(start)
                self.moves[zone] += 1
                if outcome is None:
                    self.transitions[zone, self._zone(details["end_location"])] += 1
            elif kind == "Carry":
                zone = self._zone(start)
                self.moves[zone] += 1
                self.transitions[zone, self._zone(event["carry"]["end_location"])] += 1
        self.num_matches += 1

    def merge(self, other: "ExpectedThreat") -> None:
        self.shots += other.shots
        self.shot_xg += other.shot_xg
        self.moves += other.moves
        self.transitions += other.transitions
        self.num_matches += other.num_matches

    def solve(self, tolerance: float = 1e-7, max_iterations: int = 500) -> np.ndarray:
        actions = self.shots + self.moves
        with np.errstate(divide="ignore", invalid="ignore"):
            p_shot = np.where(actions > 0, self.shots / actions, 0.0)
            p_move = np.where(actions > 0, self.moves / actions, 0.0)
            xg = np.where(self.shots > 0, self.shot_xg / self.shots, 0.0)
            transition = np.where(self.moves[:, None] > 0, self.transitions / self.moves[:, None], 0.0)
        values = np.zeros(self.grid.num_zones)
        for _ in range(max_iterations):
            updated = p_shot * xg + p_move * (transition @ values)
            if np.max(np.abs(updated - values)) < tolerance:
                values = updated
                break
            values = updated
        self.values = values
        return values

    def value(self, x: float, y: float) -> float:
        if self.values is None:
            raise RuntimeError("Call solve() or load a fitted grid first.")
        return float(self.values[self._zone((x, y))])

    def save(self, path: str | Path) -> None:
        values = self.values if self.values is not None else self.solve()
        Path(path).write_text(
            json.dumps(
                {
                    "grid": [self.grid.grid_m, self.grid.grid_n],
                    "num_matches": self.num_matches,
                    "values": values.reshape(self.grid.grid_m, self.grid.grid_n).tolist(),  # [bin_x][bin_y]
                },
                indent=1,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExpectedThreat":
        data = json.loads(Path(path).read_text())
        model = cls(*data["grid"])
        model.values = np.asarray(data["values"], dtype=float).reshape(-1)
        model.num_matches = data.get("num_matches", 0)
        return model

    @classmethod
    def from_karun_grid(cls, path: str | Path) -> "ExpectedThreat":
        """Karun Singh's published grid: rows are y bins, columns x bins (attacking right)."""
        rows = np.asarray(json.loads(Path(path).read_text()), dtype=float)  # (grid_n, grid_m)
        model = cls(grid_m=rows.shape[1], grid_n=rows.shape[0])
        model.values = rows.T.reshape(-1)
        return model
