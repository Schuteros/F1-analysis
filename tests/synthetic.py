"""Synthetic telemetry for tests: a stadium-shaped track, no FastF1 involved."""

import numpy as np
import numpy.typing as npt
import pandas as pd

from f1_analysis.data import FastestLapData

FloatArray = npt.NDArray[np.float64]

STRAIGHT_M = 600.0  # length of each straight
RADIUS_M = 100.0  # radius of each 180-degree turn
SPACING_M = 10.0  # telemetry sample spacing
FAST_KMH = 300.0
SLOW_KMH = 100.0
SESSION_START_S = 3600.0
TURN_M = np.pi * RADIUS_M
LAP_M = 2 * STRAIGHT_M + 2 * TURN_M


def track_xy_m(distance: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Position in metres at ``distance`` along a stadium: straight, turn, straight, turn."""
    d = np.mod(distance, LAP_M)
    x = np.zeros_like(d)
    y = np.zeros_like(d)
    for i, s in enumerate(d):
        if s < STRAIGHT_M:
            x[i], y[i] = s, 0.0
        elif s < STRAIGHT_M + TURN_M:
            angle = (s - STRAIGHT_M) / RADIUS_M
            x[i] = STRAIGHT_M + RADIUS_M * np.sin(angle)
            y[i] = RADIUS_M - RADIUS_M * np.cos(angle)
        elif s < 2 * STRAIGHT_M + TURN_M:
            x[i], y[i] = STRAIGHT_M - (s - STRAIGHT_M - TURN_M), 2 * RADIUS_M
        else:
            angle = (s - 2 * STRAIGHT_M - TURN_M) / RADIUS_M
            x[i] = -RADIUS_M * np.sin(angle)
            y[i] = RADIUS_M + RADIUS_M * np.cos(angle)
    return x, y


def in_turn(distance: FloatArray) -> npt.NDArray[np.bool_]:
    d = np.mod(distance, LAP_M)
    first = (d >= STRAIGHT_M) & (d < STRAIGHT_M + TURN_M)
    second = d >= 2 * STRAIGHT_M + TURN_M
    return np.asarray(first | second)


def make_lap_telemetry(offset_m: float = 0.0, speed_scale: float = 1.0) -> pd.DataFrame:
    """One lap of car telemetry (columns as FastF1 car data after add_distance)."""
    distance = np.arange(0.0, LAP_M, SPACING_M) + offset_m
    turning = in_turn(distance)
    speed = np.where(turning, SLOW_KMH, FAST_KMH) * speed_scale
    ahead = in_turn(distance + 3 * SPACING_M)  # braking starts 30 m before a turn
    braking = ahead & ~turning
    dt = np.diff(distance, prepend=distance[0]) / (speed / 3.6)
    seconds = np.cumsum(dt)
    drs = np.where(distance < STRAIGHT_M - 100, 12, 0)
    return pd.DataFrame(
        {
            "Time": pd.to_timedelta(seconds, unit="s"),
            "SessionTime": pd.to_timedelta(seconds + SESSION_START_S, unit="s"),
            "RPM": np.where(turning, 9000.0, 12000.0),
            "Speed": speed,
            "nGear": np.where(turning, 3, 8),
            "Throttle": np.where(turning | braking, 20.0, 100.0),
            "Brake": braking,
            "DRS": drs,
            "Distance": distance,
        }
    )


def make_fastest_lap(lap: pd.DataFrame) -> FastestLapData:
    """Fastest-lap data (car + position) for a lap built by make_lap_telemetry."""
    x_m, y_m = track_xy_m(lap["Distance"].to_numpy(dtype=float))
    car = lap.set_index("SessionTime")
    position = pd.DataFrame({"X": x_m * 10.0, "Y": y_m * 10.0}, index=car.index)
    return FastestLapData(car=car, position=position)
