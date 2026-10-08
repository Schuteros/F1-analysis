import numpy as np
import pandas as pd
import pytest

from f1_analysis import classify


@pytest.mark.parametrize(
    ("cornering", "speed", "throttle", "brake", "drs", "expected"),
    [
        (True, 300.0, 100.0, 0.0, 0.9, "DRS Inefficiency"),  # DRS wins over everything
        (True, 300.0, 100.0, 0.0, 0.0, "Cornering"),
        (False, 300.0, 95.0, 0.5, 0.0, "Straight-Line Speed Issue"),
        (False, 200.0, 95.0, 0.1, 0.0, "Acceleration"),
        (False, 200.0, 95.0, 0.5, 0.0, "Other"),  # braking while at full throttle
        (False, 200.0, 50.0, 0.0, 0.0, "Other"),
    ],
)
def test_classify_subsection_rules(
    cornering: bool, speed: float, throttle: float, brake: float, drs: float, expected: str
) -> None:
    assert classify.classify_subsection(cornering, speed, throttle, brake, drs) == expected


def _telemetry(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    n = len(x)
    return pd.DataFrame(
        {
            "DistanceBin": np.arange(n) * 10.0,
            "X": x,
            "Y": y,
            "Speed": 200.0,
            "Throttle": 95.0,
            "Brake": 0.0,
            "DRS": 0.0,
        }
    )


def test_is_cornering_converts_decimetres_to_metres() -> None:
    angle = np.linspace(0, np.pi, 20)
    tight = 100.0 * 10.0  # radius 100 m in decimetres
    assert classify.is_cornering(_telemetry(tight * np.cos(angle), tight * np.sin(angle)))
    wide = 5000.0 * 10.0  # radius 5 km
    assert not classify.is_cornering(
        _telemetry(wide * np.cos(angle / 10), wide * np.sin(angle / 10))
    )
    assert not classify.is_cornering(_telemetry(np.arange(20) * 100.0, np.zeros(20)))


def test_find_general_problems_groups_car_subsections_only() -> None:
    telemetry = _telemetry(np.arange(30) * 100.0, np.zeros(30))  # straight, accelerating
    sections = pd.DataFrame(
        {
            "Problem": ["Car", "Driver", "None", "Car"],
            "Subsections": [[(0.0, 50.0)], [(60.0, 100.0)], [], [(100.0, 200.0), (210.0, 250.0)]],
        }
    )
    result = classify.find_general_problems(sections, telemetry)
    by_type = dict(zip(result["Problem"], result["Subsections"], strict=True))
    assert by_type["Acceleration"] == [(0.0, 50.0), (100.0, 200.0), (210.0, 250.0)]
    assert by_type["Braking"] is None  # never produced, see KNOWN_ISSUES.md
    assert by_type["Cornering"] is None


def test_find_general_problems_raises_on_subsection_without_telemetry() -> None:
    telemetry = _telemetry(np.arange(5) * 100.0, np.zeros(5))
    sections = pd.DataFrame({"Problem": ["Car"], "Subsections": [[(1000.0, 2000.0)]]})
    with pytest.raises(ValueError, match="No telemetry between"):
        classify.find_general_problems(sections, telemetry)
