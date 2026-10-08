"""Save reference outputs from the ORIGINAL code (scripts/legacy_main.py).

Run from the repo root:  python scripts/make_reference.py [--out reference/old]

Writes numbers (JSON/CSV) and PNGs. Steps that crash in the original code are not
fixed: the traceback goes to errors.json and only what runs is referenced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import traceback
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import fastf1
import fastf1.plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".fastf1-cache"
YEAR, EVENT, SESSION, DRIVER = 2025, "CHINESE", "Sprint", "HUL"


def load_legacy() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "legacy_main", ROOT / "scripts" / "legacy_main.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jsonable(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return {
            "columns": [str(c) for c in obj.columns],
            "index": [jsonable(i) for i in obj.index],
            "data": [[jsonable(v) for v in row] for row in obj.itertuples(index=False)],
        }
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def dump(out: Path, name: str, obj: Any) -> None:
    (out / f"{name}.json").write_text(json.dumps(jsonable(obj), indent=1), encoding="utf-8")


def attempt(errors: dict[str, str], name: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # recorded, never swallowed: see errors.json
        errors[name] = traceback.format_exc()
        print(f"[{name}] CRASHED (recorded)")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "reference" / "old")
    out: Path = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    old = load_legacy()
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=True, misc_mpl_mods=False, color_scheme="fastf1"
    )

    saved_figs: list[str] = []

    def save_instead_of_show() -> None:
        name = f"plot_{len(saved_figs):02d}.png"
        plt.gcf().savefig(out / name, dpi=100)
        saved_figs.append(name)
        plt.close("all")

    plt.show = save_instead_of_show  # type: ignore[assignment]

    errors: dict[str, str] = {}
    gp = fastf1.get_session(YEAR, EVENT, SESSION)
    gp.load()
    winner = gp.results[gp.results["ClassifiedPosition"] == "1"]["Abbreviation"].iloc[0]
    dump(out, "meta", {"driver": DRIVER, "winner": winner})

    driver_laps = gp.laps.pick_drivers(DRIVER).pick_accurate().pick_quicklaps()
    telemetry = old.get_average_telemetry(driver_laps)

    sections = old.calculate_sections(telemetry)  # NB: mutates telemetry (as the original does)
    dump(out, "sections", sections)
    telemetry.to_csv(out / "average_telemetry.csv", index=False)

    old.plot_sections(telemetry, sections["corners"], title="Corners", section_label="Corner")
    old.plot_sections(
        telemetry, sections["braking_zones"], title="Braking Zones", section_label="Braking Zone"
    )
    old.plot_sections(telemetry, sections["exits"], title="Exits", section_label="Exit")
    old.plot_sections(telemetry, sections["straights"], title="Straights", section_label="Straight")

    # Section times: never wired in main(); midpoints from the speed trace of the averaged lap.
    midpoints, max_idxs, min_idxs = old.detect_sections(
        telemetry["Distance"].values, telemetry["Speed"].values
    )
    dump(out, "midpoints", {"midpoints": midpoints, "max_idxs": max_idxs, "min_idxs": min_idxs})
    driver_times = old.calculate_section_times(driver_laps, midpoints)
    dump(out, "section_times_avg", driver_times)
    dump(out, "section_times_raw", old.get_section_times(driver_laps, midpoints))

    # Car-vs-driver: winner is the compare car.
    all_laps = gp.laps.pick_quicklaps().pick_accurate()
    winner_laps = gp.laps.pick_drivers(winner).pick_accurate().pick_quicklaps()
    winner_times = old.calculate_section_times(winner_laps, midpoints)

    driver_analysis = attempt(
        errors,
        "analyze_performance_driver",
        lambda: old.analyze_performance(gp, midpoints, winner, driver_times),
    )
    winner_analysis = attempt(
        errors,
        "analyze_performance_winner",
        lambda: old.analyze_performance(gp, midpoints, winner, winner_times),
    )
    if driver_analysis is not None and winner_analysis is not None:
        dump(out, "analysis_driver", driver_analysis)
        dump(out, "analysis_winner", winner_analysis)
        problems = attempt(
            errors,
            "problem_analysis",
            lambda: old.problem_analysis(driver_analysis, winner_analysis),
        )
        if problems is not None:
            dump(out, "problem_analysis", problems)
            with_sub = attempt(
                errors,
                "detect_slow_subsections",
                lambda: old.detect_slow_subsections(driver_laps, problems, midpoints, all_laps),
            )
            if with_sub is not None:
                dump(out, "problem_subsections", with_sub)
                general = attempt(
                    errors,
                    "find_general_problems",
                    lambda: old.find_general_problems(with_sub, telemetry),
                )
                if general is not None:
                    dump(out, "general_problems", general)
                    attempt(
                        errors,
                        "plot_track_with_problems",
                        lambda: old.plot_track_with_problems(telemetry, general, with_sub),
                    )
    dump(out, "errors", errors)
    dump(out, "plots", saved_figs)
    print(f"Reference written to {out}; crashes: {list(errors)}")


if __name__ == "__main__":
    main()
