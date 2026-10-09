"""Section times, time-difference z-scores and driver consistency. Pure maths on telemetry."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from f1_analysis import config
from f1_analysis.telemetry import FloatArray, require_columns

SectionTimeLists = dict[int, list[float]]  # section number -> one duration (s) per lap


def _nearest_sample_index(distance: FloatArray, position: float) -> int:
    return int(np.abs(distance - position).argmin())


def lap_section_times(telemetry: pd.DataFrame, midpoints: FloatArray) -> list[float]:
    """Duration (s) of each section of one lap; there are ``len(midpoints) + 1`` sections.

    Section 0 is the time from lap start to the first midpoint, the last section runs from the
    final midpoint to the end of the telemetry. Each midpoint is snapped to the nearest sample.
    """
    require_columns(telemetry, ["Distance", "Time"], "lap_section_times")
    if telemetry.empty:
        raise ValueError("lap_section_times: lap telemetry is empty")
    if len(midpoints) == 0:
        raise ValueError("lap_section_times: need at least one midpoint")
    seconds = telemetry["Time"].dt.total_seconds().to_numpy(dtype=float)
    distance = telemetry["Distance"].to_numpy(dtype=float)
    at_midpoints = [seconds[_nearest_sample_index(distance, float(mp))] for mp in midpoints]
    durations = [at_midpoints[0]]
    durations += [at_midpoints[i] - at_midpoints[i - 1] for i in range(1, len(midpoints))]
    durations.append(float(seconds[-1] - at_midpoints[-1]))
    return [float(d) for d in durations]


def get_section_times(
    lap_telemetries: Sequence[pd.DataFrame], midpoints: FloatArray
) -> SectionTimeLists:
    """Section durations (s) for every lap: ``{section: [lap1, lap2, ...]}``."""
    if not lap_telemetries:
        raise ValueError("get_section_times: no laps given")
    per_lap = [lap_section_times(lap, midpoints) for lap in lap_telemetries]
    return {section: [lap[section] for lap in per_lap] for section in range(len(midpoints) + 1)}


def calculate_section_times(
    lap_telemetries: Sequence[pd.DataFrame], midpoints: FloatArray
) -> dict[int, float]:
    """Mean section duration (s) over laps: ``{section: mean}``.

    Same numbers as the mean of get_section_times. The original two functions were copies
    differing only in this final averaging (and a silent 0 for an empty list); both are kept,
    see KNOWN_ISSUES.md.
    """
    times = get_section_times(lap_telemetries, midpoints)
    return {section: float(np.mean(values)) for section, values in times.items()}


def _require_same_sections(a: Mapping[int, object], b: Mapping[int, object], where: str) -> None:
    if set(a) != set(b):
        raise ValueError(f"{where}: section numbers differ: {sorted(a)} vs {sorted(b)}")


def find_and_normalize_section_times(
    average_section_times: Mapping[int, float], car_average_section_times: Mapping[int, float]
) -> dict[int, float]:
    """Z-score of (car - reference) section time differences.

    The differences are centred on their mean over sections and divided by their (population)
    standard deviation, so the score says which sections lose more time than the car's typical
    deficit. A standard deviation of exactly 0 is replaced by ZERO_STD_REPLACEMENT.
    """
    _require_same_sections(average_section_times, car_average_section_times, "z-scores")
    differences = {
        section: car_average_section_times[section] - average
        for section, average in average_section_times.items()
    }
    values = list(differences.values())
    if not np.isfinite(values).all():
        raise ValueError("find_and_normalize_section_times: non-finite section time difference")
    std = float(np.std(values)) or config.ZERO_STD_REPLACEMENT
    mean = float(np.mean(values))
    return {section: (diff - mean) / std for section, diff in differences.items()}


def section_std(section_times: Mapping[int, Sequence[float]]) -> dict[int, float]:
    """Population standard deviation (s) of each section's lap times."""
    return {section: float(np.std(times)) for section, times in section_times.items()}


def field_reference_std(
    per_driver_times: Mapping[str, Mapping[int, Sequence[float]]],
    min_laps: int = config.MIN_LAPS_FOR_STD,
) -> dict[int, float]:
    """Typical lap-to-lap spread (s) per section: median over drivers of each driver's own std.

    Each driver's std is computed on their own laps, so pace gaps between drivers do not count
    as inconsistency. Drivers with fewer than ``min_laps`` laps are left out.
    """
    stds = {
        driver: section_std(times)
        for driver, times in per_driver_times.items()
        if all(len(laps) >= min_laps for laps in times.values())
    }
    if not stds:
        raise ValueError(f"field_reference_std: no driver has at least {min_laps} laps")
    sections = set.intersection(*(set(driver_std) for driver_std in stds.values()))
    return {
        section: float(np.median([driver_std[section] for driver_std in stds.values()]))
        for section in sorted(sections)
    }


def driver_consistency(
    reference_std: Mapping[int, float], car_times: Mapping[int, Sequence[float]]
) -> dict[int, float]:
    """Relative excess lap-to-lap spread: (std(car) - reference_std) / reference_std.

    0 means as consistent as the reference (the median driver in the pipeline); 0.2 means 20 %
    more spread. A reference of exactly 0 is replaced by ZERO_STD_REPLACEMENT in the denominator.
    """
    _require_same_sections(reference_std, car_times, "consistency")
    car_std = section_std(car_times)
    result = {}
    for section, std in car_std.items():
        denominator = reference_std[section] or config.ZERO_STD_REPLACEMENT
        result[section] = (std - reference_std[section]) / denominator
    if not np.isfinite(list(result.values())).all():
        raise ValueError("driver_consistency: non-finite result (NaN lap times?)")
    return result


def build_performance_table(
    time_diff_z_scores: Mapping[int, float], consistency: Mapping[int, float]
) -> pd.DataFrame:
    """One row per section with 'Time Diff Z-Score' and 'Consistency'."""
    _require_same_sections(time_diff_z_scores, consistency, "performance table")
    return pd.DataFrame({"Time Diff Z-Score": time_diff_z_scores, "Consistency": consistency})
