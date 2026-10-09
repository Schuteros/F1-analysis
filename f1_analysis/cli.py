"""Command line entry point and the pipeline that wires the modules together."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import fastf1.plotting
import matplotlib
import pandas as pd

from f1_analysis import (
    attribution,
    classify,
    config,
    data,
    plotting,
    sections,
    telemetry,
    timing,
)
from f1_analysis.telemetry import FloatArray


class AnalysisInputs(NamedTuple):
    """Per-lap telemetry (with Distance and Time) for every group the analysis needs.

    The lap-picking order differs between groups on purpose, to match the original code:
    see data.pick_driver_laps / pick_field_laps.
    """

    subject_laps: list[pd.DataFrame]  # the analysed driver
    subject_fastest: data.FastestLapData  # fastest lap of the analysed driver (for X/Y)
    compare_laps: list[pd.DataFrame]  # the comparison car, picked like subject_laps
    field_laps: list[pd.DataFrame]  # all drivers


class AnalysisResult(NamedTuple):
    """Everything the pipeline computes."""

    average_telemetry: pd.DataFrame
    sections: sections.Sections
    midpoints: FloatArray  # m; boundaries between the timing sections
    subject_analysis: pd.DataFrame  # z-score and consistency per timing section
    compare_analysis: pd.DataFrame
    labelled: pd.DataFrame  # + Problem (Car/Driver/None)
    with_subsections: pd.DataFrame  # + Subsections
    general_problems: pd.DataFrame  # problem type -> subsections


def run_pipeline(inputs: AnalysisInputs) -> AnalysisResult:
    """Run the whole analysis on already-loaded telemetry. No FastF1, no plotting."""
    average = telemetry.get_average_telemetry(inputs.subject_laps, inputs.subject_fastest)
    found = sections.calculate_sections(average)
    midpoints = sections.detect_sections(
        average["Distance"].to_numpy(dtype=float), average["Speed"].to_numpy(dtype=float)
    ).midpoints

    field_times = timing.get_section_times(inputs.field_laps, midpoints)
    field_average = timing.calculate_section_times(inputs.field_laps, midpoints)

    def analyse(laps: list[pd.DataFrame]) -> pd.DataFrame:
        car_times = timing.get_section_times(laps, midpoints)
        car_average = timing.calculate_section_times(laps, midpoints)
        z_scores = timing.find_and_normalize_section_times(field_average, car_average)
        consistency = timing.driver_consistency(field_times, car_times)
        return timing.build_performance_table(z_scores, consistency)

    subject_analysis = analyse(inputs.subject_laps)
    compare_analysis = analyse(inputs.compare_laps)
    labelled = attribution.problem_analysis(subject_analysis, compare_analysis)
    with_subsections = sections.detect_slow_subsections(
        telemetry.get_average_speeds(inputs.subject_laps),
        telemetry.get_average_speeds(inputs.field_laps),
        labelled,
        midpoints,
    )
    general = classify.find_general_problems(with_subsections, average)
    return AnalysisResult(
        average,
        found,
        midpoints,
        subject_analysis,
        compare_analysis,
        labelled,
        with_subsections,
        general,
    )


def load_inputs(session: fastf1.core.Session, driver: str, compare_to: str) -> AnalysisInputs:
    """Pick laps from a loaded session and read their telemetry."""
    subject = data.pick_driver_laps(session, driver)
    compare = data.pick_driver_laps(session, compare_to)
    field = data.pick_field_laps(session)
    return AnalysisInputs(
        subject_laps=data.load_lap_telemetries(subject),
        subject_fastest=data.load_fastest_lap(subject),
        compare_laps=data.load_lap_telemetries(compare),
        field_laps=data.load_lap_telemetries(field),
    )


def make_plots(result: AnalysisResult, save_dir: Path | None) -> None:
    """Draw all plots; save PNGs into ``save_dir`` or show them when it is None."""
    track = result.average_telemetry
    track_x = track["X"].to_numpy(dtype=float)
    track_y = track["Y"].to_numpy(dtype=float)

    def target(name: str) -> Path | None:
        return None if save_dir is None else save_dir / f"{name}.png"

    found = result.sections
    section_plots = [
        ("corners", "Corners", "Corner", found["corners"]),
        ("braking_zones", "Braking Zones", "Braking Zone", found["braking_zones"]),
        ("exits", "Exits", "Exit", found["exits"]),
        ("straights", "Straights", "Straight", found["straights"]),
    ]
    for name, title, label, found_sections in section_plots:
        paths = telemetry.section_paths(track, found_sections)
        plotting.plot_sections(
            track_x, track_y, paths, title=title, section_label=label, save_to=target(name)
        )

    problem_paths = {
        name: telemetry.section_paths(track, subsections, skip_empty=False)
        for name, subsections in classify.problem_sections(result.general_problems).items()
    }
    driver_paths = telemetry.section_paths(
        track, classify.driver_subsections(result.with_subsections)
    )
    plotting.plot_track_with_problems(
        track_x, track_y, problem_paths, driver_paths, save_to=target("problems")
    )


def format_report(result: AnalysisResult, driver: str, compare_to: str) -> str:
    """Plain-text summary of the analysis."""
    lines = [f"Driver {driver} compared with {compare_to}", ""]
    found = result.sections
    for name, count in [
        ("corners", len(found["corners"])),
        ("braking_zones", len(found["braking_zones"])),
        ("exits", len(found["exits"])),
        ("straights", len(found["straights"])),
    ]:
        lines.append(f"{name}: {count}")
    lines += ["", "Per timing section (z-score of time lost, consistency, label):"]
    lines.append(result.labelled.to_string())
    lines += ["", "Car problems by type:", result.general_problems.to_string()]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="f1-analysis", description="Car vs driver section analysis from F1 telemetry."
    )
    parser.add_argument("--year", type=int, default=config.DEFAULT_YEAR)
    parser.add_argument("--event", default=config.DEFAULT_EVENT, help="event name, e.g. CHINESE")
    parser.add_argument("--session", default=config.DEFAULT_SESSION, help="e.g. Race, Sprint, Q")
    parser.add_argument("--driver", default=config.DEFAULT_DRIVER, help="three-letter code")
    parser.add_argument(
        "--compare-to",
        default=None,
        help="three-letter code of the comparison car (default: the session winner)",
    )
    parser.add_argument(
        "--save-plots",
        type=Path,
        default=None,
        metavar="DIR",
        help="save PNGs into DIR instead of opening windows",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(config.DEFAULT_CACHE_DIR))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.save_plots is not None:
        matplotlib.use("Agg")
    fastf1.plotting.setup_mpl(
        mpl_timedelta_support=True, misc_mpl_mods=False, color_scheme="fastf1"
    )
    session = data.load_session(args.year, args.event, args.session, args.cache_dir)
    compare_to = args.compare_to or data.get_winner(session.results)
    result = run_pipeline(load_inputs(session, args.driver, compare_to))
    print(format_report(result, args.driver, compare_to))
    make_plots(result, args.save_plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
