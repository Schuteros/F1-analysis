"""Find corners, braking zones, exits, straights and DRS zones; remove overlaps.

Sections are (start, end) distances in metres along the lap. Nothing here mutates its inputs.
"""

from collections.abc import Sequence
from typing import NamedTuple, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from f1_analysis import config
from f1_analysis.telemetry import (
    FloatArray,
    Section,
    require_columns,
    require_finite,
    smooth_curvature,
    synchronize_speed_data,
)

BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]


class Sections(TypedDict):
    """All section types found on one lap."""

    corners: list[Section]
    braking_zones: list[Section]
    exits: list[Section]
    straights: list[Section]


class DrsOverride(NamedTuple):
    """DRS zones and the other section lists with the DRS zones cut out."""

    drs_zones: list[Section]
    straights: list[Section]
    braking_zones: list[Section]
    corners: list[Section]
    exits: list[Section]


class SpeedExtremes(NamedTuple):
    """Section midpoints (m) and the sample indices of speed maxima and minima."""

    midpoints: FloatArray
    max_idxs: IntArray
    min_idxs: IntArray


def find_runs(mask: BoolArray, distance: FloatArray, min_length: float) -> list[Section]:
    """(start, end) distances of each run of True in ``mask`` at least ``min_length`` long."""
    if mask.shape != distance.shape:
        raise ValueError(f"mask {mask.shape} and distance {distance.shape} differ in shape")
    runs: list[Section] = []
    start: float | None = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = float(distance[i])
        elif not active and start is not None:
            end = float(distance[i - 1])
            if end - start >= min_length:
                runs.append((start, end))
            start = None
    if start is not None:
        end = float(distance[-1])
        if end - start >= min_length:
            runs.append((start, end))
    return runs


def merge_runs(runs: Sequence[Section], gap: float) -> list[Section]:
    """Merge consecutive runs whose gap is at most ``gap`` metres."""
    merged: list[Section] = []
    for run in runs:
        if not merged or run[0] - merged[-1][1] > gap:
            merged.append(run)
        else:
            merged[-1] = (merged[-1][0], run[1])
    return merged


def detect_sections(distance: FloatArray, speed: FloatArray) -> SpeedExtremes:
    """Midpoints between consecutive speed maxima/minima (plus lap start and end).

    Speed is Savitzky-Golay smoothed first. Maxima are acceleration points, minima braking
    points; peaks must be at least SPEED_PEAK_MIN_SEPARATION_SAMPLES apart.
    """
    if distance.shape != speed.shape:
        raise ValueError("distance and speed must have the same shape")
    if len(speed) < config.SPEED_SMOOTHING_WINDOW_SAMPLES:
        raise ValueError(
            f"Need at least {config.SPEED_SMOOTHING_WINDOW_SAMPLES} samples, got {len(speed)}"
        )
    smooth: FloatArray = savgol_filter(
        speed,
        window_length=config.SPEED_SMOOTHING_WINDOW_SAMPLES,
        polyorder=config.SPEED_SMOOTHING_POLYORDER,
    )
    separation = config.SPEED_PEAK_MIN_SEPARATION_SAMPLES
    max_idxs, _ = find_peaks(smooth, distance=separation)
    min_idxs, _ = find_peaks(-smooth, distance=separation)
    extremes = np.sort(np.concatenate([max_idxs, min_idxs]))
    extremes = np.insert(extremes, 0, 0)
    extremes = np.append(extremes, len(smooth) - 1)
    midpoint_idxs = (extremes[1:] + extremes[:-1]) / 2
    return SpeedExtremes(distance[midpoint_idxs.astype(int)], max_idxs, min_idxs)


def with_smoothed_curvature(average_telemetry: pd.DataFrame) -> pd.DataFrame:
    """Copy of the telemetry whose Curvature column is smoothed.

    The original find_corners smoothed Curvature in place, so find_straights (called later)
    saw smoothed values too. calculate_sections calls this once and feeds both finders.
    """
    require_finite(average_telemetry, ["Curvature"], "with_smoothed_curvature")
    smoothed = smooth_curvature(average_telemetry["Curvature"].to_numpy(dtype=float))
    return average_telemetry.assign(Curvature=smoothed)


def find_corners(
    average_telemetry: pd.DataFrame,
    curvature_threshold: float = config.CORNER_CURVATURE_THRESHOLD_PER_DM,
    gap_threshold: float = config.CORNER_GAP_M,
    min_corner_length: float = config.CORNER_MIN_LENGTH_M,
) -> list[Section]:
    """Corners: runs where (already smoothed) Curvature exceeds the threshold.

    Needs 'Distance' (m) and 'Curvature' (1/dm; smooth it first with with_smoothed_curvature).
    """
    require_finite(average_telemetry, ["Distance", "Curvature"], "find_corners")
    mask = average_telemetry["Curvature"].to_numpy(dtype=float) > curvature_threshold
    distance = average_telemetry["Distance"].to_numpy(dtype=float)
    return merge_runs(find_runs(mask, distance, min_corner_length), gap_threshold)


def find_braking_zones(
    average_telemetry: pd.DataFrame,
    brake_threshold: float = config.BRAKE_THRESHOLD,
    gap_threshold: float = config.BRAKING_GAP_M,
    min_braking_length: float = config.BRAKING_MIN_LENGTH_M,
) -> list[Section]:
    """Braking zones: runs where the averaged Brake value exceeds the threshold."""
    require_finite(average_telemetry, ["Distance", "Brake"], "find_braking_zones")
    mask = average_telemetry["Brake"].to_numpy(dtype=float) > brake_threshold
    distance = average_telemetry["Distance"].to_numpy(dtype=float)
    return merge_runs(find_runs(mask, distance, min_braking_length), gap_threshold)


def cut_straights_at_braking(
    straights: Sequence[Section], braking_zones: Sequence[Section]
) -> list[Section]:
    """End each straight at the first braking zone (in list order) that starts inside it."""
    adjusted = list(straights)
    for i, (start, end) in enumerate(adjusted):
        for braking_start, _ in braking_zones:
            if start < braking_start <= end:
                adjusted[i] = (start, braking_start)
                break
    return adjusted


def find_straights(
    average_telemetry: pd.DataFrame,
    curvature_threshold: float = config.STRAIGHT_CURVATURE_THRESHOLD_PER_DM,
    gap_threshold: float = config.STRAIGHT_GAP_M,
    min_straight_length: float = config.STRAIGHT_MIN_LENGTH_M,
    braking_zones: Sequence[Section] | None = None,
) -> list[Section]:
    """Straights: runs where Curvature is below the threshold, ended at braking zone starts."""
    require_finite(average_telemetry, ["Distance", "Curvature"], "find_straights")
    mask = average_telemetry["Curvature"].to_numpy(dtype=float) < curvature_threshold
    distance = average_telemetry["Distance"].to_numpy(dtype=float)
    straights = merge_runs(find_runs(mask, distance, min_straight_length), gap_threshold)
    if braking_zones:
        straights = cut_straights_at_braking(straights, braking_zones)
    return straights


def calculate_acceleration(distance: pd.Series, speed: pd.Series) -> pd.Series:
    """Longitudinal acceleration (m/s^2) along the lap from a speed-vs-distance trace.

    a = dv/dt with dt = ds/v. Speed above EXIT_SPEED_IS_KMH_ABOVE anywhere is taken to be km/h.
    Undefined values (first sample, zero speed) become 0, as in the original.
    """
    is_kmh = speed.max() > config.EXIT_SPEED_IS_KMH_ABOVE
    speed_mps = speed / config.KMH_PER_MPS if is_kmh else speed
    dt = distance.diff() / speed_mps.replace(0, np.nan)
    acceleration = speed_mps.diff() / dt
    return acceleration.replace([np.inf, -np.inf], np.nan).fillna(0)


def find_exits(
    average_telemetry: pd.DataFrame,
    acceleration_threshold: float = config.EXIT_ACCELERATION_THRESHOLD_FIND_EXITS,
    min_exit_length: float = config.EXIT_MIN_LENGTH_M,
) -> list[Section]:
    """Exit phases: runs where acceleration exceeds the threshold."""
    require_finite(average_telemetry, ["Distance", "Speed"], "find_exits")
    acceleration = calculate_acceleration(average_telemetry["Distance"], average_telemetry["Speed"])
    mask = acceleration.to_numpy(dtype=float) > acceleration_threshold
    distance = average_telemetry["Distance"].to_numpy(dtype=float)
    return find_runs(mask, distance, min_exit_length)


def calculate_sections(
    average_telemetry: pd.DataFrame,
    curvature_threshold: float = config.SECTIONS_CURVATURE_THRESHOLD_PER_DM,
    brake_threshold: float = config.BRAKE_THRESHOLD,
    acceleration_threshold: float = config.EXIT_ACCELERATION_THRESHOLD_SECTIONS,
    min_corner_length: float = config.CORNER_MIN_LENGTH_M,
) -> Sections:
    """All sections of a lap. Corners and straights see smoothed curvature, as in the original."""
    smoothed = with_smoothed_curvature(average_telemetry)
    corners = find_corners(smoothed, curvature_threshold, min_corner_length=min_corner_length)
    braking_zones = find_braking_zones(average_telemetry, brake_threshold)
    straights = find_straights(smoothed, curvature_threshold, braking_zones=braking_zones)
    exits = find_exits(average_telemetry, acceleration_threshold=acceleration_threshold)
    return Sections(corners=corners, braking_zones=braking_zones, exits=exits, straights=straights)


def remove_zones_from_sections(
    sections: Sequence[Section], zones: Sequence[Section]
) -> list[Section]:
    """Cut every zone out of every section, keeping the parts before and after."""
    updated: list[Section] = []
    for section_start, section_end in sections:
        pieces = [(section_start, section_end)]
        for zone_start, zone_end in zones:
            next_pieces: list[Section] = []
            for start, end in pieces:
                if zone_start <= end and zone_end >= start:
                    if start < zone_start:
                        next_pieces.append((start, zone_start))
                    if end > zone_end:
                        next_pieces.append((zone_end, end))
                else:
                    next_pieces.append((start, end))
            pieces = next_pieces
        updated.extend(pieces)
    return updated


def find_and_override_drs_zones(
    average_telemetry: pd.DataFrame,
    straights: Sequence[Section],
    braking_zones: Sequence[Section],
    corners: Sequence[Section],
    exits: Sequence[Section],
    throttle_threshold: float = config.DRS_ZONE_THROTTLE_THRESHOLD,
) -> DrsOverride:
    """DRS zones (DRS open and throttle high) and the other sections with them cut out."""
    columns = ["Distance", "Throttle", "DRS"]
    require_finite(average_telemetry, columns, "find_and_override_drs_zones")
    mask = (average_telemetry["DRS"].to_numpy(dtype=float) > 0) & (
        average_telemetry["Throttle"].to_numpy(dtype=float) >= throttle_threshold
    )
    distance = average_telemetry["Distance"].to_numpy(dtype=float)
    drs_zones = find_runs(mask, distance, 0.0)
    return DrsOverride(
        drs_zones,
        remove_zones_from_sections(straights, drs_zones),
        remove_zones_from_sections(braking_zones, drs_zones),
        remove_zones_from_sections(corners, drs_zones),
        remove_zones_from_sections(exits, drs_zones),
    )


def remove_overlaps(
    target_sections: Sequence[Section],
    reference_sections: Sequence[Section],
    min_corner_length: float = config.OVERLAP_MIN_REFERENCE_LENGTH_M,
) -> list[Section]:
    """Split target sections at the midpoint of their overlap with each reference section.

    A target piece overlapping a reference shorter than ``min_corner_length`` is DROPPED
    (the original docstring says "not split", the code discards it; see KNOWN_ISSUES.md).
    """
    updated: list[Section] = []
    for target_start, target_end in target_sections:
        pieces = [(target_start, target_end)]
        for ref_start, ref_end in reference_sections:
            next_pieces: list[Section] = []
            for start, end in pieces:
                if ref_start <= end and ref_end >= start:
                    if ref_end - ref_start < min_corner_length:
                        continue
                    midpoint = (max(start, ref_start) + min(end, ref_end)) / 2
                    if start < midpoint:
                        next_pieces.append((start, midpoint))
                    if end > midpoint:
                        next_pieces.append((midpoint, end))
                else:
                    next_pieces.append((start, end))
            pieces = next_pieces
        updated.extend(pieces)
    return updated


def find_slow_runs(
    distance: FloatArray, speed: FloatArray, avg_speed: FloatArray, start_idx: int, end_idx: int
) -> list[Section]:
    """Runs of samples in [start_idx, end_idx) where the car is slower than the average."""
    runs: list[Section] = []
    slow_start: float | None = None
    for idx in range(start_idx, end_idx):
        if speed[idx] < avg_speed[idx]:
            if slow_start is None:
                slow_start = float(distance[idx])
        elif slow_start is not None:
            runs.append((slow_start, float(distance[idx - 1])))
            slow_start = None
    if slow_start is not None:
        runs.append((slow_start, float(distance[end_idx - 1])))
    return runs


def drop_short_runs(runs: Sequence[Section], min_length: float) -> list[Section]:
    """Keep runs at least ``min_length`` metres long."""
    return [(start, end) for start, end in runs if end - start >= min_length]


def detect_slow_subsections(
    car_speeds: pd.DataFrame,
    field_speeds: pd.DataFrame,
    labelled_sections: pd.DataFrame,
    midpoints: FloatArray,
) -> pd.DataFrame:
    """Add a 'Subsections' column: where a Car/Driver-problem section is slower than average.

    ``car_speeds`` and ``field_speeds`` come from telemetry.get_average_speeds. Section i runs
    from midpoint i-1 to midpoint i; rows are matched to sections by position.
    """
    require_columns(
        labelled_sections,
        ["Problem", "Time Diff Z-Score", "Consistency"],
        "detect_slow_subsections",
    )
    if len(labelled_sections) != len(midpoints) + 1:
        raise ValueError(
            f"Expected {len(midpoints) + 1} sections for {len(midpoints)} midpoints, "
            f"got {len(labelled_sections)}"
        )
    grid, speed, avg_speed = synchronize_speed_data(
        car_speeds.index.to_numpy(dtype=float),
        car_speeds["Speed"].to_numpy(dtype=float),
        field_speeds.index.to_numpy(dtype=float),
        field_speeds["Speed"].to_numpy(dtype=float),
    )
    rows = []
    for position, (_, section) in enumerate(labelled_sections.iterrows()):
        subsections: list[Section] = []
        if section["Problem"] in ("Car", "Driver"):
            start_idx = int(np.searchsorted(grid, midpoints[position - 1])) if position > 0 else 0
            end_idx = (
                int(np.searchsorted(grid, midpoints[position]))
                if position < len(midpoints)
                else len(grid)
            )
            slow = find_slow_runs(grid, speed, avg_speed, start_idx, end_idx)
            subsections = drop_short_runs(
                merge_runs(slow, config.SLOW_MERGE_GAP_M), config.SLOW_MIN_LENGTH_M
            )
        rows.append(
            {
                "Problem": section["Problem"],
                "Time Diff Z-Score": section["Time Diff Z-Score"],
                "Consistency": section["Consistency"],
                "Subsections": subsections,
            }
        )
    return pd.DataFrame(rows)
