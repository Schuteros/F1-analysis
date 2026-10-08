"""All plots. Plotting only draws what it is given; nothing is computed here.

Every ``plot_*`` function shows the figure, or saves it when ``save_to`` is given.
The ``build_*`` functions return the figure without showing or saving it.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from f1_analysis import config
from f1_analysis.telemetry import FloatArray, SectionPath

NO_LEGEND_LABEL = "_nolegend_"  # matplotlib: line is not listed in the legend


def output_figure(fig: Figure, save_to: Path | None) -> None:
    """Save the figure to ``save_to`` (and close it), or show it when ``save_to`` is None."""
    if save_to is None:
        plt.show()
        return
    save_to.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, dpi=config.PLOT_DPI)
    plt.close(fig)


def _legend_label(label: str, seen: set[str]) -> str:
    """``label`` the first time it is used, then NO_LEGEND_LABEL, so the legend has no repeats."""
    if label in seen:
        return NO_LEGEND_LABEL
    seen.add(label)
    return label


def _draw_track(ax: Axes, track_x: FloatArray, track_y: FloatArray, label: str) -> None:
    ax.plot(
        track_x, track_y, color=config.TRACK_COLOR, label=label, linewidth=config.TRACK_LINEWIDTH
    )


def _finish_track_axes(ax: Axes, title: str) -> None:
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)


def build_section_graph(
    distance: FloatArray,
    speed: FloatArray,
    max_idxs: Sequence[int],
    min_idxs: Sequence[int],
    section_midpoints: FloatArray,
    midpoint_speeds: FloatArray,
) -> Figure:
    """Speed trace with its maxima, minima and section midpoints marked."""
    fig, ax = plt.subplots()
    ax.plot(distance, speed, label="Speed", color="blue")
    ax.scatter(
        distance[max_idxs],
        speed[max_idxs],
        color="green",
        label="Refined Maxima (Acceleration)",
        zorder=3,
    )
    ax.scatter(
        distance[min_idxs],
        speed[min_idxs],
        color="red",
        label="Refined Minima (Braking)",
        zorder=3,
    )
    ax.scatter(
        section_midpoints, midpoint_speeds, color="black", label="Section Midpoints", zorder=3
    )
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.legend()
    return fig


def plot_section_graph(
    distance: FloatArray,
    speed: FloatArray,
    max_idxs: Sequence[int],
    min_idxs: Sequence[int],
    section_midpoints: FloatArray,
    midpoint_speeds: FloatArray,
    save_to: Path | None = None,
) -> None:
    fig = build_section_graph(
        distance, speed, max_idxs, min_idxs, section_midpoints, midpoint_speeds
    )
    output_figure(fig, save_to)


def build_subsection_path(x: FloatArray, y: FloatArray) -> Figure:
    """The X/Y path of one subsection."""
    fig, ax = plt.subplots(figsize=config.SPEED_FIGSIZE)
    ax.plot(x, y, label="Path Subsection", color="blue", linestyle="-", marker="o")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Visualization of Path Subsection")
    ax.legend()
    ax.grid(True)
    return fig


def visualize_subsection(x: FloatArray, y: FloatArray, save_to: Path | None = None) -> None:
    output_figure(build_subsection_path(x, y), save_to)


def build_sections_figure(
    track_x: FloatArray,
    track_y: FloatArray,
    paths: Sequence[SectionPath],
    title: str,
    section_label: str,
) -> Figure:
    """The full track in grey with each section drawn on top in its own colour."""
    fig, ax = plt.subplots(figsize=config.TRACK_FIGSIZE)
    _draw_track(ax, track_x, track_y, "Full Track")
    for path in paths:
        ax.plot(
            path.x,
            path.y,
            label=f"{section_label} {path.number}",
            linewidth=config.SECTION_LINEWIDTH,
        )
    _finish_track_axes(ax, title)
    return fig


def plot_sections(
    track_x: FloatArray,
    track_y: FloatArray,
    paths: Sequence[SectionPath],
    title: str = "Track Sections",
    section_label: str = "Section",
    save_to: Path | None = None,
) -> None:
    output_figure(build_sections_figure(track_x, track_y, paths, title, section_label), save_to)


def build_track_with_problems(
    track_x: FloatArray,
    track_y: FloatArray,
    problem_paths: Mapping[str, Sequence[SectionPath]],
    driver_paths: Sequence[SectionPath],
) -> Figure:
    """The track with car-problem sections coloured by type and driver problems in cyan."""
    fig, ax = plt.subplots(figsize=config.TRACK_FIGSIZE)
    _draw_track(ax, track_x, track_y, "Track")
    seen: set[str] = {"Track"}
    for problem_type, paths in problem_paths.items():
        for path in paths:
            ax.plot(
                path.x,
                path.y,
                color=config.PROBLEM_COLORS.get(problem_type, "black"),
                linewidth=config.SECTION_LINEWIDTH,
                label=_legend_label(problem_type, seen),
            )
    for path in driver_paths:
        ax.plot(
            path.x,
            path.y,
            color=config.PROBLEM_COLORS["Driver"],
            linewidth=config.SECTION_LINEWIDTH,
            label=_legend_label("Driver Problem", seen),
        )
    _finish_track_axes(ax, "Race Track with Problem Sections")
    return fig


def plot_track_with_problems(
    track_x: FloatArray,
    track_y: FloatArray,
    problem_paths: Mapping[str, Sequence[SectionPath]],
    driver_paths: Sequence[SectionPath],
    save_to: Path | None = None,
) -> None:
    output_figure(build_track_with_problems(track_x, track_y, problem_paths, driver_paths), save_to)
