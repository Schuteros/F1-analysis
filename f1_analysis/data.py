"""Load a FastF1 session and pick laps. Loading only: nothing here analyses."""

from pathlib import Path
from typing import NamedTuple

import fastf1
import pandas as pd
from fastf1.core import Laps, Session


class FastestLapData(NamedTuple):
    """Car and position telemetry of one lap, both indexed by SessionTime."""

    car: pd.DataFrame  # includes a Distance column (m)
    position: pd.DataFrame  # X, Y in decimetres


def load_session(year: int, event: str, session_name: str, cache_dir: Path) -> Session:
    """Load a session with the FastF1 on-disk cache enabled."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    session = fastf1.get_session(year, event, session_name)
    session.load()
    return session


def get_winner(results: pd.DataFrame) -> str:
    """Abbreviation of the driver with ClassifiedPosition "1"."""
    missing = {"ClassifiedPosition", "Abbreviation"} - set(results.columns)
    if missing:
        raise ValueError(f"Session results lack columns: {sorted(missing)}")
    winners = results[results["ClassifiedPosition"] == "1"]["Abbreviation"]
    if winners.empty:
        raise ValueError("Session results contain no classified winner (ClassifiedPosition '1').")
    return str(winners.iloc[0])


def _require_laps(laps: Laps, description: str) -> Laps:
    if len(laps) == 0:
        raise ValueError(f"No laps left after picking {description}.")
    return laps


def pick_driver_laps(session: Session, driver: str) -> Laps:
    """Driver's accurate laps, then quick laps (the order the original main() used)."""
    laps = session.laps.pick_drivers(driver).pick_accurate().pick_quicklaps()
    return _require_laps(laps, f"accurate+quick laps of {driver}")


def pick_field_laps(session: Session) -> Laps:
    """All drivers' quick laps, then accurate laps (the order analyze_performance used)."""
    laps = session.laps.pick_quicklaps().pick_accurate()
    return _require_laps(laps, "quick+accurate laps of the field")


def load_lap_telemetries(laps: Laps) -> list[pd.DataFrame]:
    """Car telemetry with a Distance column for each lap, in lap order."""
    return [lap.get_car_data().add_distance() for _, lap in laps.iterrows()]


def _index_by_session_time(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if not pd.api.types.is_timedelta64_dtype(frame["SessionTime"]):
        frame["SessionTime"] = pd.to_timedelta(frame["SessionTime"])
    return frame.set_index("SessionTime")


def load_fastest_lap(laps: Laps) -> FastestLapData:
    """Car and position telemetry of the fastest lap in ``laps``."""
    fastest = laps.pick_fastest()
    position = _index_by_session_time(fastest.get_pos_data())
    car = _index_by_session_time(fastest.get_car_data().add_distance())
    return FastestLapData(car=car, position=position)
