"""Run the refactored package on the reference session and compare with reference/old.

Run from the repo root:  python scripts/compare_reference.py

Needs real FastF1 data (the cache from make_reference.py), so it is a script, not a pytest test.
Lists and indices must match exactly; floats must match with np.allclose. PNG comparison is
informational only (look at the images side by side).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import fastf1.plotting
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_reference import DRIVER, EVENT, SESSION, YEAR, dump  # noqa: E402

from f1_analysis import cli, data, sections, telemetry, timing  # noqa: E402

RTOL = 1e-12
PLOT_NAMES = ["corners", "braking_zones", "exits", "straights", "problems"]  # old plot_00..04


def differences(old: Any, new: Any, path: str, exact: bool) -> list[str]:
    """Human-readable list of differences between two JSON-like values."""
    if isinstance(old, dict) and isinstance(new, dict):
        if old.keys() != new.keys():
            return [f"{path}: keys {sorted(old)} != {sorted(new)}"]
        return [d for k in old for d in differences(old[k], new[k], f"{path}.{k}", exact)]
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{path}: length {len(old)} != {len(new)}"]
        return [
            d
            for i, (a, b) in enumerate(zip(old, new, strict=True))
            for d in differences(a, b, f"{path}[{i}]", exact)
        ]
    if isinstance(old, float) or isinstance(new, float):
        if old is None or new is None:
            return [f"{path}: {old!r} != {new!r}"]
        if exact:
            return [] if old == new else [f"{path}: {old!r} != {new!r}"]
        same = np.isclose(new, old, rtol=RTOL, atol=0.0, equal_nan=True)
        return [] if same else [f"{path}: {old!r} != {new!r}"]
    return [] if old == new else [f"{path}: {old!r} != {new!r}"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=ROOT / "reference" / "old")
    parser.add_argument("--new", type=Path, default=ROOT / "reference" / "new")
    parser.add_argument(
        "--curvature-scale",
        type=float,
        default=1.0,
        help="factor between new and old Curvature column (2 after the curvature unit fix)",
    )
    args = parser.parse_args()
    old_dir: Path = args.old
    new_dir: Path = args.new
    new_dir.mkdir(parents=True, exist_ok=True)

    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=True, misc_mpl_mods=False, color_scheme="fastf1"
    )
    session = data.load_session(YEAR, EVENT, SESSION, ROOT / ".fastf1-cache")
    winner = data.get_winner(session.results)
    inputs = cli.load_inputs(session, DRIVER, winner)
    result = cli.run_pipeline(inputs)
    midpoints = result.midpoints

    new: dict[str, tuple[Any, bool]] = {  # name -> (value, must match exactly)
        "meta": ({"driver": DRIVER, "winner": winner}, True),
        "sections": (dict(result.sections), True),
        "midpoints": (
            {
                "midpoints": midpoints,
                "max_idxs": sections.detect_sections(
                    result.average_telemetry["Distance"].to_numpy(dtype=float),
                    result.average_telemetry["Speed"].to_numpy(dtype=float),
                ).max_idxs,
                "min_idxs": sections.detect_sections(
                    result.average_telemetry["Distance"].to_numpy(dtype=float),
                    result.average_telemetry["Speed"].to_numpy(dtype=float),
                ).min_idxs,
            },
            True,
        ),
        "section_times_avg": (
            timing.calculate_section_times(inputs.subject_laps, midpoints),
            False,
        ),
        "section_times_raw": (timing.get_section_times(inputs.subject_laps, midpoints), False),
        "analysis_driver": (result.subject_analysis, False),
        "analysis_winner": (result.compare_analysis, False),
        "problem_analysis": (result.labelled, False),
        "problem_subsections": (result.with_subsections, False),
        "general_problems": (result.general_problems, False),
    }
    problems: list[str] = []
    for name, (value, exact) in new.items():
        dump(new_dir, name, value)
        old_json = json.loads((old_dir / f"{name}.json").read_text(encoding="utf-8"))
        new_json = json.loads((new_dir / f"{name}.json").read_text(encoding="utf-8"))
        found = differences(old_json, new_json, name, exact)
        print(f"{'OK  ' if not found else 'DIFF'} {name}")
        problems += found

    # Average telemetry: the old code smoothed Curvature and added Acceleration in place.
    average = result.average_telemetry
    expected = average.assign(
        Curvature=telemetry.smooth_curvature(average["Curvature"].to_numpy(dtype=float)),
        Acceleration=sections.calculate_acceleration(average["Distance"], average["Speed"]),
    )
    old_csv = pd.read_csv(old_dir / "average_telemetry.csv")
    old_csv["Curvature"] *= args.curvature_scale  # intended unit change only; all else unscaled
    if list(old_csv.columns) != list(expected.columns):
        problems.append(
            f"average_telemetry columns {list(old_csv.columns)} != {list(expected.columns)}"
        )
    else:
        for column in old_csv.columns:
            if not np.allclose(old_csv[column], expected[column], rtol=RTOL, atol=0.0):
                problems.append(f"average_telemetry.{column} differs")
    print(
        f"{'OK  ' if not any(p.startswith('average_telemetry') for p in problems) else 'DIFF'} average_telemetry"
    )

    # Plots: the new CLI code path, then a byte comparison (informational).
    plot_dir = new_dir
    cli.make_plots(result, plot_dir)
    for i, name in enumerate(PLOT_NAMES):
        old_png = old_dir / f"plot_{i:02d}.png"
        new_png = plot_dir / f"{name}.png"
        same = old_png.read_bytes() == new_png.read_bytes()
        print(
            f"{'identical' if same else 'differs  '} PNG {name}  ({old_png.name} vs {new_png.name})"
        )

    print()
    if problems:
        print(f"{len(problems)} DIFFERENCE(S):")
        for line in problems[:50]:
            print("  ", line)
        raise SystemExit(1)
    print("All numbers match the reference.")


if __name__ == "__main__":
    main()
