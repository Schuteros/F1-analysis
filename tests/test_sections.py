import numpy as np
import pandas as pd
import pytest

from f1_analysis import sections, telemetry
from tests import synthetic


def test_find_runs_min_length_and_trailing_run() -> None:
    distance = np.arange(0.0, 100.0, 10.0)
    mask = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1], dtype=bool)
    assert sections.find_runs(mask, distance, 0.0) == [(10.0, 20.0), (40.0, 40.0), (70.0, 90.0)]
    assert sections.find_runs(mask, distance, 15.0) == [(70.0, 90.0)]
    with pytest.raises(ValueError, match="differ in shape"):
        sections.find_runs(mask, distance[:5], 0.0)


def test_merge_runs_merges_gaps_up_to_threshold() -> None:
    runs = [(0.0, 10.0), (15.0, 20.0), (40.0, 50.0)]
    assert sections.merge_runs(runs, 5.0) == [(0.0, 20.0), (40.0, 50.0)]  # gap 5 merges (<=)
    assert sections.merge_runs(runs, 4.9) == runs
    assert sections.merge_runs(runs, 0.0) == runs
    assert sections.merge_runs([], 5.0) == []


def test_detect_sections_on_sine_speed_trace() -> None:
    distance = np.arange(0.0, 1500.0)
    speed = 200.0 + 100.0 * np.sin(2 * np.pi * distance / 500.0)
    result = sections.detect_sections(distance, speed)
    assert np.allclose(result.max_idxs, [125, 625, 1125], atol=2)
    assert np.allclose(result.min_idxs, [375, 875, 1375], atol=2)
    # 6 extremes + lap start + lap end -> 7 midpoints between them
    assert len(result.midpoints) == 7
    assert np.allclose(result.midpoints[1:4], [250, 500, 750], atol=2)
    assert (np.diff(result.midpoints) > 0).all()


def test_detect_sections_rejects_short_trace() -> None:
    with pytest.raises(ValueError, match="at least"):
        sections.detect_sections(np.arange(10.0), np.arange(10.0))


def test_calculate_sections_on_synthetic_track() -> None:
    laps = [synthetic.make_lap_telemetry(offset_m=k * 0.5) for k in range(3)]
    average = telemetry.get_average_telemetry(laps, synthetic.make_fastest_lap(laps[0]))
    before = average.copy()
    result = sections.calculate_sections(average)
    pd.testing.assert_frame_equal(average, before)  # input not mutated
    assert len(result["corners"]) == 2
    assert len(result["braking_zones"]) == 2
    assert len(result["straights"]) == 2
    turn1 = (synthetic.STRAIGHT_M, synthetic.STRAIGHT_M + synthetic.TURN_M)
    corner = result["corners"][0]
    assert turn1[0] - 20 <= corner[0] and corner[1] <= turn1[1] + 20
    # straight ends where braking starts, braking starts before the corner
    brake = result["braking_zones"][0]
    assert result["straights"][0][1] == brake[0] < corner[0]


def test_find_braking_zones_filters_and_keeps_default_gap_of_zero() -> None:
    frame = pd.DataFrame(
        {
            "Distance": np.arange(0.0, 200.0, 10.0),
            "Brake": [0.0] * 3 + [1.0] * 4 + [0.0] * 2 + [1.0] * 5 + [0.0] * 6,
        }
    )
    # runs: 30-60 (30 m) and 90-130 (40 m); the 30 m gap is not merged at gap 0
    assert sections.find_braking_zones(frame) == [(30.0, 60.0), (90.0, 130.0)]
    assert sections.find_braking_zones(frame, gap_threshold=30.0) == [(30.0, 130.0)]
    assert sections.find_braking_zones(frame, min_braking_length=35.0) == [(90.0, 130.0)]


def test_find_exits_detects_sustained_acceleration() -> None:
    distance = np.arange(0.0, 400.0, 10.0)
    speed = np.concatenate([np.full(10, 100.0), np.linspace(100.0, 300.0, 15), np.full(15, 300.0)])
    frame = pd.DataFrame({"Distance": distance, "Speed": speed})
    exits = sections.find_exits(frame)
    assert len(exits) == 1
    assert 90.0 <= exits[0][0] <= 110.0 and 230.0 <= exits[0][1] <= 250.0


def test_calculate_acceleration_first_sample_is_zero() -> None:
    frame = pd.DataFrame({"Distance": [0.0, 10.0, 20.0], "Speed": [100.0, 110.0, 120.0]})
    acceleration = sections.calculate_acceleration(frame["Distance"], frame["Speed"])
    assert acceleration.iloc[0] == 0.0
    assert (acceleration.iloc[1:] > 0).all()


def test_straight_is_cut_at_first_braking_start_inside_it() -> None:
    result = sections.cut_straights_at_braking([(0.0, 500.0), (600.0, 700.0)], [(450.0, 480.0)])
    assert result == [(0.0, 450.0), (600.0, 700.0)]


def test_remove_overlaps_splits_at_overlap_midpoint() -> None:
    result = sections.remove_overlaps([(0.0, 100.0)], [(80.0, 200.0)], min_corner_length=50.0)
    assert result == [(0.0, 90.0), (90.0, 100.0)]
    # no overlap: untouched
    assert sections.remove_overlaps([(0.0, 50.0)], [(80.0, 200.0)]) == [(0.0, 50.0)]


def test_remove_overlaps_drops_target_overlapping_short_reference() -> None:
    # Documents current behaviour (KNOWN_ISSUES.md): the overlapped piece disappears.
    assert sections.remove_overlaps([(0.0, 100.0)], [(80.0, 90.0)], min_corner_length=50.0) == []


def test_remove_zones_from_sections_keeps_outer_parts() -> None:
    result = sections.remove_zones_from_sections([(0.0, 100.0), (200.0, 300.0)], [(40.0, 60.0)])
    assert result == [(0.0, 40.0), (60.0, 100.0), (200.0, 300.0)]


def test_find_and_override_drs_zones() -> None:
    frame = pd.DataFrame(
        {
            "Distance": np.arange(0.0, 100.0, 10.0),
            "Throttle": [100.0] * 4 + [50.0] + [100.0] * 5,
            "DRS": [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        }
    )
    result = sections.find_and_override_drs_zones(frame, [(0.0, 90.0)], [], [], [])
    assert result.drs_zones == [(10.0, 30.0)]
    assert result.straights == [(0.0, 10.0), (30.0, 90.0)]


def test_detect_slow_subsections_finds_long_slow_runs_in_flagged_sections() -> None:
    distance = np.arange(0.0, 1000.0, 5.0)
    field = pd.DataFrame({"Speed": np.full(len(distance), 200.0)}, index=distance)
    car_speed = np.full(len(distance), 200.0)
    car_speed[(distance >= 100) & (distance < 220)] = 180.0  # 120 m slow, inside section 1
    car_speed[(distance >= 600) & (distance < 620)] = 180.0  # 20 m slow, too short
    car = pd.DataFrame({"Speed": car_speed}, index=distance)
    labelled = pd.DataFrame(
        {
            "Problem": ["None", "Car", "Driver"],
            "Time Diff Z-Score": [0.0, 1.0, 1.0],
            "Consistency": [0.0, 0.5, 0.5],
        }
    )
    result = sections.detect_slow_subsections(car, field, labelled, np.array([50.0, 500.0]))
    assert result["Subsections"].iloc[0] == []
    assert result["Subsections"].iloc[1] == [(100.0, 215.0)]
    assert result["Subsections"].iloc[2] == []


def test_detect_slow_subsections_checks_section_count() -> None:
    labelled = pd.DataFrame({"Problem": ["Car"], "Time Diff Z-Score": [0.0], "Consistency": [0.0]})
    speeds = pd.DataFrame({"Speed": [1.0, 2.0]}, index=[0.0, 1.0])
    with pytest.raises(ValueError, match="Expected 3 sections"):
        sections.detect_slow_subsections(speeds, speeds, labelled, np.array([1.0, 2.0]))
