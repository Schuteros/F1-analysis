import numpy as np
import pandas as pd
import pytest

from f1_analysis import telemetry
from tests import synthetic


def _circle(radius: float, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    angle = np.linspace(0.0, np.pi, n)
    return radius * np.cos(angle), radius * np.sin(angle)


@pytest.mark.parametrize("radius", [5.0, 100.0, 1000.0])
def test_curvature_of_circle_is_inverse_radius(radius: float) -> None:
    x, y = _circle(radius)
    curvature = telemetry.calculate_curvature(x, y)
    np.testing.assert_allclose(curvature[1:-1], 1.0 / radius, rtol=1e-9)


def test_curvature_padding_straight_line_and_duplicates() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    y = np.zeros(6)
    curvature = telemetry.calculate_curvature(x, y)
    assert curvature.shape == (6,)
    assert curvature[0] == 0.0 and curvature[-1] == 0.0
    assert not curvature.any()  # collinear points and zero-length sides give 0


def test_curvature_short_input_and_bad_input() -> None:
    assert telemetry.calculate_curvature(np.array([1.0]), np.array([2.0])).tolist() == [0.0]
    with pytest.raises(ValueError, match="equal length"):
        telemetry.calculate_curvature(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        telemetry.calculate_curvature(np.array([0.0, np.nan, 1.0]), np.zeros(3))


def test_smooth_curvature_needs_enough_samples() -> None:
    with pytest.raises(ValueError, match="at least"):
        telemetry.smooth_curvature(np.zeros(3))
    assert telemetry.smooth_curvature(np.ones(20)).shape == (20,)


def test_require_columns_and_finite() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan]})
    with pytest.raises(ValueError, match="missing columns"):
        telemetry.require_columns(frame, ["a", "b"], "test")
    with pytest.raises(ValueError, match="NaN"):
        telemetry.require_finite(frame, ["a"], "test")


def test_bin_lap_telemetry_converts_brake_and_drs() -> None:
    lap = synthetic.make_lap_telemetry()
    binned = telemetry.bin_lap_telemetry(lap)
    assert set(binned["DRS"].unique()) <= {0.0, 1.0}
    assert binned["Brake"].max() == 1.0
    assert (binned["DistanceBin"] == binned["Distance"]).all()


def test_get_average_telemetry_on_synthetic_laps() -> None:
    laps = [synthetic.make_lap_telemetry(offset_m=k * 0.5) for k in range(3)]
    result = telemetry.get_average_telemetry(laps, synthetic.make_fastest_lap(laps[0]))
    assert {"X", "Y", "Curvature", "Speed", "DistanceBin"} <= set(result.columns)
    assert result["Curvature"].max() > 0
    assert np.isfinite(result.select_dtypes("number").to_numpy()).all()


def test_get_average_telemetry_raises_without_laps() -> None:
    lap = synthetic.make_lap_telemetry()
    with pytest.raises(ValueError, match="no non-empty"):
        telemetry.get_average_telemetry([lap.iloc[0:0]], synthetic.make_fastest_lap(lap))


def test_get_average_speeds_is_indexed_by_distance_bin() -> None:
    speeds = telemetry.get_average_speeds([synthetic.make_lap_telemetry()])
    assert speeds.index.name == "DistanceBin"
    assert list(speeds.columns) == ["Speed"]
    with pytest.raises(ValueError, match="no non-empty"):
        telemetry.get_average_speeds([])


def test_synchronize_speed_data_uses_overlap() -> None:
    distance = np.array([0.0, 10.0, 20.0, 30.0])
    other = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    grid, speed, avg = telemetry.synchronize_speed_data(
        distance, distance * 2.0, other, other * 3.0
    )
    assert grid[0] == 10.0 and grid[-1] == 30.0 and len(grid) == 4
    np.testing.assert_allclose(speed, grid * 2.0)
    np.testing.assert_allclose(avg, grid * 3.0)
    with pytest.raises(ValueError, match="do not overlap"):
        telemetry.synchronize_speed_data(distance, distance, other + 100.0, other)


def test_slice_and_section_paths() -> None:
    frame = pd.DataFrame(
        {"DistanceBin": [0.0, 10.0, 20.0, 30.0], "X": [1.0, 2.0, 3.0, 4.0], "Y": [0.0] * 4}
    )
    assert telemetry.slice_by_distance_bin(frame, 10.0, 20.0)["X"].tolist() == [2.0, 3.0]
    paths = telemetry.section_paths(frame, [(100.0, 200.0), (10.0, 20.0)])
    assert [p.number for p in paths] == [2]  # empty first section skipped, numbering kept
    assert paths[0].x.tolist() == [2.0, 3.0]
