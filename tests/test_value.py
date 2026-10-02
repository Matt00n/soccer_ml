"""Tests for value baselines."""

import pytest

from soccer_ml.value.expected_threat import ExpectedThreat


def _event(kind, location, **details):
    return {"type": {"name": kind}, "location": list(location), **details}


def test_expected_threat_markov_chain():
    """Zone A only passes to B; B shoots (xG 0.3) half the time and passes back to A otherwise."""
    xt = ExpectedThreat(grid_m=4, grid_n=1)
    a, b = (10.0, 40.0), (100.0, 40.0)
    shot = {"shot": {"type": {"name": "Open Play"}, "statsbomb_xg": 0.3}}
    events = [
        _event("Pass", a, **{"pass": {"end_location": list(b)}}),
        _event("Pass", a, **{"pass": {"end_location": list(b), "outcome": {"name": "Incomplete"}}}),
        _event("Shot", b, **shot),
        _event("Pass", b, **{"pass": {"end_location": list(a)}}),
        _event("Pass", a, **{"pass": {"end_location": list(b), "type": {"name": "Corner"}}}),  # set piece: ignored
    ]
    xt.update(events)
    values = xt.solve()
    zone_a, zone_b = xt.grid.coords_to_zone(*a), xt.grid.coords_to_zone(*b)
    # xT(B) = 0.5 * 0.3 + 0.5 * xT(A); xT(A) = 1.0 * 0.5 * xT(B)  (half of A's moves fail)
    expected_b = 0.15 / (1 - 0.25)
    assert values[zone_b] == pytest.approx(expected_b, abs=1e-6)
    assert values[zone_a] == pytest.approx(0.5 * expected_b, abs=1e-6)
    assert xt.value(*b) == pytest.approx(expected_b, abs=1e-6)


def test_expected_threat_save_load_roundtrip(tmp_path):
    xt = ExpectedThreat(grid_m=3, grid_n=2)
    xt.update([_event("Shot", (110, 40), shot={"type": {"name": "Open Play"}, "statsbomb_xg": 0.2})])
    xt.solve()
    xt.save(tmp_path / "xt.json")
    loaded = ExpectedThreat.load(tmp_path / "xt.json")
    assert loaded.value(110, 40) == pytest.approx(xt.value(110, 40))
