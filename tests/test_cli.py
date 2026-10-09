from pathlib import Path

import numpy as np
import pytest

from f1_analysis import cli, config
from tests import synthetic


def _inputs() -> cli.AnalysisInputs:
    n_laps = config.MIN_LAPS_FOR_STD
    scales = [0.97, 1.0, 1.03, 0.99]
    subject = [
        synthetic.make_lap_telemetry(offset_m=k * 0.5, speed_scale=scales[k % len(scales)])
        for k in range(n_laps)
    ]
    compare = [
        synthetic.make_lap_telemetry(offset_m=k * 0.5, speed_scale=1.05) for k in range(n_laps)
    ]
    steady = [
        synthetic.make_lap_telemetry(offset_m=k * 0.5, speed_scale=1.0) for k in range(n_laps)
    ]
    return cli.AnalysisInputs(
        subject_laps=subject,
        subject_fastest=synthetic.make_fastest_lap(subject[1]),
        compare_laps=compare,
        field_laps=subject + compare + steady,
        field_laps_by_driver={"AAA": subject, "BBB": compare, "CCC": steady},
    )


def test_pipeline_end_to_end_on_synthetic_telemetry() -> None:
    result = cli.run_pipeline(_inputs())

    assert len(result.sections["corners"]) == 2
    assert len(result.sections["straights"]) == 2
    n_sections = len(result.midpoints) + 1
    assert len(result.subject_analysis) == len(result.compare_analysis) == n_sections
    assert len(result.labelled) == len(result.with_subsections) == n_sections
    assert set(result.labelled["Problem"]) <= {"Car", "Driver", "None"}
    assert np.isfinite(result.subject_analysis.to_numpy(dtype=float)).all()
    # z-scores are centred over sections
    assert result.subject_analysis["Time Diff Z-Score"].mean() == pytest.approx(0.0, abs=1e-9)
    assert list(result.general_problems["Problem"]) == list(config.PROBLEM_TYPES)


def test_pipeline_is_deterministic_and_does_not_mutate_inputs() -> None:
    inputs = _inputs()
    before = [lap.copy() for lap in inputs.subject_laps]
    first = cli.run_pipeline(inputs)
    second = cli.run_pipeline(inputs)
    assert first.sections == second.sections
    for lap, original in zip(inputs.subject_laps, before, strict=True):
        assert lap.equals(original)


def test_make_plots_saves_all_pngs(tmp_path: Path) -> None:
    cli.make_plots(cli.run_pipeline(_inputs()), tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "braking_zones.png",
        "corners.png",
        "exits.png",
        "problems.png",
        "straights.png",
    ]


def test_format_report_mentions_driver_and_sections() -> None:
    result = cli.run_pipeline(_inputs())
    report = cli.format_report(result, "HUL", "HAM")
    assert "HUL compared with HAM" in report and "corners: 2" in report


def test_parse_args_defaults_match_original_hard_coded_values() -> None:
    args = cli.parse_args([])
    assert (args.year, args.event, args.session, args.driver) == (2025, "CHINESE", "Sprint", "HUL")
    assert args.compare_to is None and args.save_plots is None


def test_parse_args_reads_options() -> None:
    args = cli.parse_args(
        [
            "--year",
            "2024",
            "--event",
            "Monza",
            "--session",
            "R",
            "--driver",
            "VER",
            "--compare-to",
            "NOR",
            "--save-plots",
            "out",
        ]
    )
    assert (args.year, args.compare_to, args.save_plots) == (2024, "NOR", Path("out"))


def test_each_car_gets_its_own_consistency() -> None:
    result = cli.run_pipeline(_inputs())
    subject = result.subject_analysis["Consistency"]
    compare = result.compare_analysis["Consistency"]
    assert not subject.equals(compare)
    # subject laps vary in speed from lap to lap, the comparison car's do not
    assert subject.mean() > compare.mean()
