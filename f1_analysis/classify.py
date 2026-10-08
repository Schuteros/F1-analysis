"""Classify the type of problem in sections labelled as car problems."""

import pandas as pd

from f1_analysis import config
from f1_analysis.attribution import CAR
from f1_analysis.telemetry import (
    Section,
    calculate_curvature,
    require_columns,
    require_finite,
    slice_by_distance_bin,
)


def classify_subsection(
    is_cornering: bool,
    mean_speed: float,
    mean_throttle: float,
    mean_brake: float,
    mean_drs: float,
) -> str:
    """Problem type from a subsection's averages. The first matching rule wins.

    DRS open for most of it, then corner, then fast + full throttle (straight-line speed),
    then full throttle without braking (acceleration), else "Other".
    """
    if mean_drs > config.DRS_THRESHOLD:
        return "DRS Inefficiency"
    if is_cornering:
        return "Cornering"
    if mean_speed > config.HIGH_SPEED_THRESHOLD and mean_throttle > config.HIGH_THROTTLE_THRESHOLD:
        return "Straight-Line Speed Issue"
    if mean_throttle > config.HIGH_THROTTLE_THRESHOLD and mean_brake < config.LOW_BRAKE_THRESHOLD:
        return "Acceleration"
    return "Other"


def is_cornering(subsection_telemetry: pd.DataFrame) -> bool:
    """True if any point of the path has curvature above CLASSIFY_CURVATURE_THRESHOLD_PER_M.

    X/Y are decimetres, so they are converted to metres first.
    """
    x = subsection_telemetry["X"].to_numpy(dtype=float) / config.POSITION_UNITS_PER_M
    y = subsection_telemetry["Y"].to_numpy(dtype=float) / config.POSITION_UNITS_PER_M
    curvature = calculate_curvature(x, y)
    return bool((curvature > config.CLASSIFY_CURVATURE_THRESHOLD_PER_M).any())


def find_general_problems(
    problematic_sections: pd.DataFrame, car_average_telemetry: pd.DataFrame
) -> pd.DataFrame:
    """Group the subsections of "Car" sections by problem type.

    ``problematic_sections`` needs 'Problem' and 'Subsections' (from detect_slow_subsections);
    the telemetry needs DistanceBin, X, Y, Speed, Throttle, Brake and DRS.
    Returns one row per type in PROBLEM_TYPES with a list of (start, end), or None if empty.
    """
    require_columns(problematic_sections, ["Problem", "Subsections"], "find_general_problems")
    require_finite(
        car_average_telemetry,
        ["DistanceBin", "X", "Y", "Speed", "Throttle", "Brake", "DRS"],
        "find_general_problems",
    )
    found: dict[str, list[Section]] = {name: [] for name in config.PROBLEM_TYPES}
    for _, section in problematic_sections.iterrows():
        if section["Problem"] != CAR:
            continue
        for start, end in section["Subsections"]:
            rows = slice_by_distance_bin(car_average_telemetry, start, end)
            if rows.empty:
                raise ValueError(f"No telemetry between {start} m and {end} m to classify")
            problem = classify_subsection(
                is_cornering(rows),
                float(rows["Speed"].mean()),
                float(rows["Throttle"].mean()),
                float(rows["Brake"].mean()),
                float(rows["DRS"].mean()),
            )
            found[problem].append((start, end))
    return pd.DataFrame(
        {
            "Problem": list(found),
            "Subsections": [subsections or None for subsections in found.values()],
        }
    )
