from types import SimpleNamespace

import pandas as pd
import pytest

from f1_analysis import data


def test_get_winner_returns_abbreviation_of_position_one() -> None:
    results = pd.DataFrame(
        {"ClassifiedPosition": ["2", "1", "3"], "Abbreviation": ["HUL", "HAM", "VER"]}
    )
    assert data.get_winner(results) == "HAM"


def test_get_winner_raises_without_winner() -> None:
    results = pd.DataFrame({"ClassifiedPosition": ["R"], "Abbreviation": ["HUL"]})
    with pytest.raises(ValueError, match="no classified winner"):
        data.get_winner(results)


def test_get_winner_raises_on_missing_columns() -> None:
    with pytest.raises(ValueError, match="lack columns"):
        data.get_winner(pd.DataFrame({"Abbreviation": ["HUL"]}))


class _Laps:
    """Minimal stand-in recording the order of pick_* calls (test double, no FastF1)."""

    def __init__(self, calls: list[str], n: int = 2) -> None:
        self.calls = calls
        self.n = n

    def __len__(self) -> int:
        return self.n

    def pick_drivers(self, driver: str) -> "_Laps":
        self.calls.append(f"drivers:{driver}")
        return self

    def pick_accurate(self) -> "_Laps":
        self.calls.append("accurate")
        return self

    def pick_quicklaps(self) -> "_Laps":
        self.calls.append("quick")
        return self


def test_pick_orders() -> None:
    calls: list[str] = []
    session = SimpleNamespace(laps=_Laps(calls))
    data.pick_driver_laps(session, "HUL")
    data.pick_field_laps(session)
    assert calls == [
        "drivers:HUL",
        "accurate",
        "quick",
        "quick",
        "accurate",
    ]


def test_pick_raises_when_no_laps_left() -> None:
    session = SimpleNamespace(laps=_Laps([], n=0))
    with pytest.raises(ValueError, match="No laps left"):
        data.pick_field_laps(session)


def test_load_lap_telemetries_adds_distance_per_lap() -> None:
    def make_lap(value: float) -> SimpleNamespace:
        frame = SimpleNamespace(add_distance=lambda: pd.DataFrame({"Distance": [value]}))
        return SimpleNamespace(get_car_data=lambda: frame)

    laps = SimpleNamespace(iterrows=lambda: iter([(0, make_lap(1.0)), (1, make_lap(2.0))]))
    result = data.load_lap_telemetries(laps)
    assert [float(frame["Distance"].iloc[0]) for frame in result] == [1.0, 2.0]
