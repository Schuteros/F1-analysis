from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from f1_analysis import classify, plotting, telemetry
from f1_analysis.telemetry import SectionPath


def _path(number: int) -> SectionPath:
    return SectionPath(number, np.array([0.0, 1.0]), np.array([0.0, number * 1.0]))


def _legend_labels(fig: Figure) -> list[str]:
    legend = fig.axes[0].get_legend()
    assert legend is not None
    return [text.get_text() for text in legend.get_texts()]


def test_sections_figure_labels_use_section_numbers() -> None:
    track = np.arange(3.0)
    fig = plotting.build_sections_figure(track, track, [_path(1), _path(3)], "T", "Corner")
    assert _legend_labels(fig) == ["Full Track", "Corner 1", "Corner 3"]


def test_problem_figure_legend_has_each_type_once() -> None:
    track = np.arange(3.0)
    fig = plotting.build_track_with_problems(
        track,
        track,
        {"Cornering": [_path(1), _path(2)], "Acceleration": [], "Other": [_path(3)]},
        [_path(4), _path(5)],
    )
    assert _legend_labels(fig) == ["Track", "Cornering", "Other", "Driver Problem"]


def test_plot_functions_save_to_file(tmp_path: Path) -> None:
    track = np.arange(3.0)
    target = tmp_path / "nested" / "sections.png"
    plotting.plot_sections(track, track, [_path(1)], save_to=target)
    assert target.stat().st_size > 0
    target2 = tmp_path / "problems.png"
    plotting.plot_track_with_problems(track, track, {"Other": [_path(1)]}, [], save_to=target2)
    assert target2.stat().st_size > 0
    target3 = tmp_path / "speed.png"
    distance = np.arange(10.0)
    plotting.plot_section_graph(
        distance, distance, [3], [6], np.array([4.5]), np.array([4.5]), save_to=target3
    )
    assert target3.stat().st_size > 0
    target4 = tmp_path / "path.png"
    plotting.visualize_subsection(track, track, save_to=target4)
    assert target4.stat().st_size > 0


def test_speed_at_interpolates() -> None:
    result = telemetry.speed_at(np.array([0.0, 10.0]), np.array([0.0, 100.0]), np.array([5.0]))
    assert result.tolist() == [50.0]


def test_problem_sections_and_driver_subsections() -> None:
    general = pd.DataFrame({"Problem": ["Braking", "Other"], "Subsections": [None, [(1.0, 2.0)]]})
    assert classify.problem_sections(general) == {"Braking": [], "Other": [(1.0, 2.0)]}
    labelled = pd.DataFrame(
        {"Problem": ["Car", "Driver", "Driver"], "Subsections": [[(0.0, 1.0)], [(5.0, 6.0)], []]}
    )
    assert classify.driver_subsections(labelled) == [(5.0, 6.0)]
    with pytest.raises(ValueError, match="missing columns"):
        classify.driver_subsections(pd.DataFrame({"Problem": []}))
