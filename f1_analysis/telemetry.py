"""Average telemetry, speed-trace synchronisation and curvature. Pure data transforms."""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import savgol_filter

from f1_analysis import config
from f1_analysis.data import FastestLapData

FloatArray = npt.NDArray[np.float64]
Section = tuple[float, float]  # (start, end) distance along the lap, metres


class SectionPath(NamedTuple):
    """X/Y path of one section. ``number`` is its 1-based position in the section list."""

    number: int
    x: FloatArray
    y: FloatArray


def require_columns(frame: pd.DataFrame, columns: Sequence[str], where: str) -> None:
    """Raise ValueError if any of ``columns`` is missing from ``frame``."""
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{where}: missing columns {missing}; have {list(frame.columns)}")


def require_finite(frame: pd.DataFrame, columns: Sequence[str], where: str) -> None:
    """Raise ValueError if any of ``columns`` contains NaN or infinity."""
    require_columns(frame, columns, where)
    for name in columns:
        if not np.isfinite(frame[name].to_numpy(dtype=float)).all():
            raise ValueError(f"{where}: column '{name}' contains NaN or infinite values")


def calculate_curvature(x: FloatArray, y: FloatArray) -> FloatArray:
    """Curvature at each point from it and its two neighbours (circle through 3 points).

    A circle through a triangle with sides a, b, c and area A has radius R = abc / (4A).
    The original code uses ``config.CURVATURE_AREA_FACTOR`` (2) instead of 4, so it returns
    1 / (2R): half the true curvature. Kept as is, see KNOWN_ISSUES.md.
    Units are 1/(unit of x, y). Ends are padded with 0; coincident points give 0.
    """
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x and y must be 1-D arrays of equal length, got {x.shape} and {y.shape}")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("calculate_curvature: x and y must be finite")
    curvature = np.zeros(len(x), dtype=np.float64)
    if len(x) < 3:
        return curvature
    x1, x2, x3 = x[:-2], x[1:-1], x[2:]
    y1, y2, y3 = y[:-2], y[1:-1], y[2:]
    side1 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    side2 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    side3 = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    area = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    product = side1 * side2 * side3
    curvature[1:-1] = np.divide(
        config.CURVATURE_AREA_FACTOR * area, product, out=np.zeros_like(area), where=product != 0
    )
    return curvature


def smooth_curvature(curvature: FloatArray) -> FloatArray:
    """Savitzky-Golay smoothing of a curvature trace (reduces point-to-point noise)."""
    window = config.CURVATURE_SMOOTHING_WINDOW_SAMPLES
    if len(curvature) < window:
        raise ValueError(f"Need at least {window} samples to smooth, got {len(curvature)}")
    smoothed: FloatArray = savgol_filter(
        curvature, window_length=window, polyorder=config.CURVATURE_SMOOTHING_POLYORDER
    )
    return smoothed


def _distance_bin(distance: pd.Series) -> pd.Series:
    return (distance / config.DISTANCE_BIN_M).round() * config.DISTANCE_BIN_M


def bin_lap_telemetry(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Average one lap's numeric channels in DISTANCE_BIN_M-metre distance bins."""
    require_columns(telemetry, ["Distance", "Brake", "DRS"], "bin_lap_telemetry")
    lap = telemetry.copy()
    lap["Brake"] = lap["Brake"].astype(int)
    lap["DRS"] = lap["DRS"].isin(config.DRS_OPEN_CODES).astype(int)
    lap["DistanceBin"] = _distance_bin(lap["Distance"])
    numeric_cols = lap.select_dtypes(include=["number"]).columns
    return lap.groupby("DistanceBin", as_index=False)[numeric_cols].mean()


def average_binned_laps(binned_laps: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Mean over laps per distance bin; time columns are dropped."""
    combined = pd.concat(binned_laps).groupby("DistanceBin", as_index=False).mean()
    return combined.drop(columns=["SessionTime", "Time"], errors="ignore")


def attach_position(combined: pd.DataFrame, fastest: FastestLapData) -> pd.DataFrame:
    """Add X, Y (decimetres) from the fastest lap; bins with no position are dropped."""
    require_columns(fastest.position, ["X", "Y"], "attach_position (position)")
    require_columns(fastest.car, ["Distance"], "attach_position (car)")
    merged = pd.merge_asof(
        fastest.car,
        fastest.position[["X", "Y"]],
        left_index=True,
        right_index=True,
        direction="nearest",
    )
    merged["DistanceBin"] = _distance_bin(merged["Distance"])
    joined = pd.merge(combined, merged[["DistanceBin", "X", "Y"]], on="DistanceBin", how="left")
    return joined.dropna(subset=["X", "Y"])


def get_average_telemetry(
    lap_telemetries: Sequence[pd.DataFrame], fastest: FastestLapData
) -> pd.DataFrame:
    """Lap-averaged telemetry with fastest-lap X/Y and curvature (1/dm).

    Empty lap telemetries are skipped; raises ValueError if none remain.
    """
    binned = [bin_lap_telemetry(lap) for lap in lap_telemetries if not lap.empty]
    if not binned:
        raise ValueError("get_average_telemetry: no non-empty lap telemetry to average")
    result = attach_position(average_binned_laps(binned), fastest)
    curvature = calculate_curvature(
        result["X"].to_numpy(dtype=float), result["Y"].to_numpy(dtype=float)
    )
    result = result.assign(Curvature=curvature)
    require_finite(
        result,
        ["Distance", "DistanceBin", "Speed", "Throttle", "Brake", "DRS", "X", "Y", "Curvature"],
        "get_average_telemetry",
    )
    return result


def get_average_speeds(lap_telemetries: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Mean speed per distance bin over laps, indexed by DistanceBin."""
    per_lap = []
    for telemetry in lap_telemetries:
        if telemetry.empty:
            continue
        require_columns(telemetry, ["Distance", "Speed"], "get_average_speeds")
        binned = telemetry.assign(DistanceBin=_distance_bin(telemetry["Distance"]))
        per_lap.append(binned.groupby("DistanceBin", as_index=False)["Speed"].mean())
    if not per_lap:
        raise ValueError("get_average_speeds: no non-empty lap telemetry to average")
    combined: pd.DataFrame = pd.concat(per_lap).groupby("DistanceBin").mean()
    return combined.dropna(how="all")


def synchronize_speed_data(
    distance: FloatArray,
    speed: FloatArray,
    avg_speed_distance: FloatArray,
    avg_speed: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Resample two speed traces onto one common distance grid.

    The grid spans the overlap of both traces and has the mean of the two sample counts.
    Returns (grid distance, speed on grid, avg_speed on grid).
    """
    num_points = (len(distance) + len(avg_speed_distance)) // 2
    start = max(distance.min(), avg_speed_distance.min())
    stop = min(distance.max(), avg_speed_distance.max())
    if start >= stop:
        raise ValueError(f"Speed traces do not overlap in distance ({start} >= {stop})")
    new_distance: FloatArray = np.linspace(start, stop, num_points)
    speed_interp: FloatArray = np.interp(new_distance, distance, speed)
    avg_speed_interp: FloatArray = np.interp(new_distance, avg_speed_distance, avg_speed)
    return new_distance, speed_interp, avg_speed_interp


def speed_at(distance: FloatArray, speed: FloatArray, positions: FloatArray) -> FloatArray:
    """Speed linearly interpolated at ``positions`` (m) along the lap."""
    interpolated: FloatArray = np.interp(positions, distance, speed)
    return interpolated


def slice_by_distance_bin(telemetry: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    """Rows whose DistanceBin lies in [start, end] (inclusive)."""
    require_columns(telemetry, ["DistanceBin"], "slice_by_distance_bin")
    in_range = (telemetry["DistanceBin"] >= start) & (telemetry["DistanceBin"] <= end)
    return telemetry[in_range]


def section_paths(
    telemetry: pd.DataFrame, sections: Sequence[Section], skip_empty: bool = True
) -> list[SectionPath]:
    """X/Y path for each section; ``number`` counts all sections, including skipped ones."""
    paths = []
    for number, (start, end) in enumerate(sections, start=1):
        rows = slice_by_distance_bin(telemetry, start, end)
        if skip_empty and rows.empty:
            continue
        paths.append(
            SectionPath(number, rows["X"].to_numpy(dtype=float), rows["Y"].to_numpy(dtype=float))
        )
    return paths
