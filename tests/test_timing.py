import numpy as np
import pandas as pd
import pytest

from f1_analysis import timing
from tests import synthetic


def _lap(speed_mps: float, length_m: float = 1000.0) -> pd.DataFrame:
    distance = np.arange(0.0, length_m + 1, 10.0)
    return pd.DataFrame(
        {"Distance": distance, "Time": pd.to_timedelta(distance / speed_mps, unit="s")}
    )


def test_lap_section_times_constant_speed() -> None:
    # 50 m/s: 1000 m takes 20 s; midpoints at 250 and 600 m
    durations = timing.lap_section_times(_lap(50.0), np.array([250.0, 600.0]))
    assert durations == pytest.approx([5.0, 7.0, 8.0])
    assert sum(durations) == pytest.approx(20.0)


def test_section_times_snap_midpoint_to_nearest_sample() -> None:
    durations = timing.lap_section_times(_lap(50.0), np.array([254.0]))  # nearest sample: 250 m
    assert durations == pytest.approx([5.0, 15.0])


def test_get_and_mean_section_times() -> None:
    laps = [_lap(50.0), _lap(100.0)]
    midpoints = np.array([500.0])
    lists = timing.get_section_times(laps, midpoints)
    assert lists[0] == pytest.approx([10.0, 5.0])
    assert lists[1] == pytest.approx([10.0, 5.0])
    assert timing.mean_section_times(lists) == pytest.approx({0: 7.5, 1: 7.5})


def test_section_times_validate_input() -> None:
    with pytest.raises(ValueError, match="no laps"):
        timing.get_section_times([], np.array([1.0]))
    with pytest.raises(ValueError, match="at least one midpoint"):
        timing.lap_section_times(_lap(50.0), np.array([]))
    with pytest.raises(ValueError, match="missing columns"):
        timing.lap_section_times(pd.DataFrame({"Distance": [1.0]}), np.array([1.0]))
    with pytest.raises(ValueError, match="empty"):
        timing.lap_section_times(_lap(50.0).iloc[0:0], np.array([1.0]))


def test_synthetic_lap_total_time() -> None:
    lap = synthetic.make_lap_telemetry()
    durations = timing.lap_section_times(lap, np.array([300.0, 900.0, 1500.0]))
    assert sum(durations) == pytest.approx(lap["Time"].dt.total_seconds().iloc[-1])


def test_z_scores_known_values() -> None:
    reference = {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0}
    car = {0: 10.0, 1: 11.0, 2: 10.0, 3: 11.0}  # differences 0, 1, 0, 1 -> mean 0.5, std 0.5
    z = timing.find_and_normalize_section_times(reference, car)
    assert z == pytest.approx({0: -1.0, 1: 1.0, 2: -1.0, 3: 1.0})


def test_z_scores_have_zero_mean_and_unit_std() -> None:
    offsets = [0.1, 0.5, -0.2, 0.9, 0.0, 0.3]
    reference = {i: 5.0 + i for i in range(6)}
    car = {i: reference[i] + offsets[i] for i in range(6)}
    z = np.array(list(timing.find_and_normalize_section_times(reference, car).values()))
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std() == pytest.approx(1.0)


def test_z_scores_identical_cars_give_zeros_not_nan() -> None:
    times = {0: 1.0, 1: 2.0}
    assert timing.find_and_normalize_section_times(times, times) == {0: 0.0, 1: 0.0}


def test_z_scores_reject_mismatch_and_nan() -> None:
    with pytest.raises(ValueError, match="section numbers differ"):
        timing.find_and_normalize_section_times({0: 1.0}, {1: 1.0})
    with pytest.raises(ValueError, match="non-finite"):
        timing.find_and_normalize_section_times({0: 1.0, 1: 2.0}, {0: np.nan, 1: 2.0})


def test_driver_consistency_relative_to_reference_std() -> None:
    reference_std = {0: 1.0, 1: 0.0}
    car = {0: [10.0, 13.0], 1: [4.0, 6.0]}  # std 1.5 and 1.0
    result = timing.driver_consistency(reference_std, car)
    assert result[0] == pytest.approx(0.5)  # 50 % more spread
    assert result[1] == pytest.approx(1.0)  # zero reference std -> denominator 1
    # a car steadier than the reference is negative
    assert timing.driver_consistency({0: 1.0}, {0: [11.0, 11.5]})[0] < 0


def test_driver_consistency_rejects_nan_and_mismatch() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        timing.driver_consistency({0: 1.0}, {0: [np.nan, 2.0]})
    with pytest.raises(ValueError, match="section numbers differ"):
        timing.driver_consistency({0: 1.0}, {1: [1.0, 2.0]})


def test_field_reference_std_is_median_of_each_drivers_own_std() -> None:
    # Pace gaps between drivers must not count: B is 10 s slower than A but equally steady.
    steady = [1.0, 2.0, 1.0, 2.0]  # std 0.5
    per_driver = {
        "A": {0: steady},
        "B": {0: [x + 10.0 for x in steady]},
        "C": {0: [1.0, 5.0, 1.0, 5.0]},  # std 2.0
    }
    assert timing.field_reference_std(per_driver, min_laps=4) == {0: pytest.approx(0.5)}


def test_field_reference_std_skips_drivers_with_too_few_laps() -> None:
    per_driver = {
        "A": {0: [1.0, 2.0, 1.0, 2.0]},  # std 0.5
        "B": {0: [1.0, 9.0]},  # too few laps, would be std 4.0
    }
    assert timing.field_reference_std(per_driver, min_laps=4) == {0: pytest.approx(0.5)}
    with pytest.raises(ValueError, match="at least 9 laps"):
        timing.field_reference_std(per_driver, min_laps=9)


def test_build_performance_table() -> None:
    table = timing.build_performance_table({0: 1.0, 1: -1.0}, {0: 0.1, 1: 0.3})
    assert list(table.columns) == ["Time Diff Z-Score", "Consistency"]
    assert table["Consistency"].tolist() == [0.1, 0.3]
    with pytest.raises(ValueError, match="section numbers differ"):
        timing.build_performance_table({0: 1.0}, {1: 1.0})


def test_mean_section_times_is_mean_of_each_sections_laps() -> None:
    rng = np.random.default_rng(0)
    laps = [_lap(float(speed)) for speed in rng.uniform(40.0, 90.0, size=7)]
    lists = timing.get_section_times(laps, np.array([120.0, 310.0, 480.0, 655.0, 800.0, 930.0]))
    means = timing.mean_section_times(lists)
    assert means.keys() == lists.keys()
    for section, values in lists.items():
        assert means[section] == float(np.mean(values))
