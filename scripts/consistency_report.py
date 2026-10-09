"""Table of every section whose Car/Driver label changed with the consistency fix.

Run from the repo root after compare_reference.py has written reference/fix-consistency/:
    python scripts/consistency_report.py

Old values come from reference/old (original code); new values from the current package.
Spreads are population standard deviations of lap times per timing section, in seconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_reference import DRIVER, EVENT, SESSION, YEAR  # noqa: E402

from f1_analysis import cli, data, timing  # noqa: E402

OLD = ROOT / "reference" / "old"
NEW = ROOT / "reference" / "fix-consistency"


def column(directory: Path, name: str, field: str) -> list[object]:
    table = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    return [row[table["columns"].index(field)] for row in table["data"]]


def main() -> None:
    session = data.load_session(YEAR, EVENT, SESSION, ROOT / ".fastf1-cache")
    winner = data.get_winner(session.results)
    inputs = cli.load_inputs(session, DRIVER, winner)
    result = cli.run_pipeline(inputs)
    midpoints = result.midpoints

    # Spreads (s) per section, exactly as the pipeline uses them.
    std_subject = timing.section_std(timing.get_section_times(inputs.subject_laps, midpoints))
    std_compare = timing.section_std(timing.get_section_times(inputs.compare_laps, midpoints))
    std_field = timing.section_std(timing.get_section_times(inputs.field_laps, midpoints))
    # The original reference used the comparison car's laps picked quick-then-accurate.
    old_compare_laps = session.laps.pick_drivers(winner).pick_quicklaps().pick_accurate()
    std_old_ref = timing.section_std(
        timing.get_section_times(data.load_lap_telemetries(old_compare_laps), midpoints)
    )

    old_label = column(OLD, "problem_analysis", "Problem")
    old_cons = column(OLD, "problem_analysis", "Consistency")
    old_z = column(OLD, "problem_analysis", "Time Diff Z-Score")
    old_z_compare = column(OLD, "analysis_winner", "Time Diff Z-Score")
    new_label = list(result.labelled["Problem"])
    new_cons = list(result.subject_analysis["Consistency"])
    new_cons_compare = list(result.compare_analysis["Consistency"])
    new_z = list(result.subject_analysis["Time Diff Z-Score"])

    edges = [0.0, *[float(m) for m in midpoints], float(result.average_telemetry["Distance"].max())]
    assert all(np.isclose(a, b, rtol=1e-12) for a, b in zip(old_z, new_z, strict=True))
    assert all(
        np.isclose(a, b, rtol=1e-12)
        for a, b in zip(old_z_compare, result.compare_analysis["Time Diff Z-Score"], strict=True)
    )  # z-scores are untouched by the fix

    lines = [
        f"# Label changes, {DRIVER} vs {winner} (2025 Chinese Sprint)",
        "",
        "Spreads = std of lap times per section (s).",
        f"Old consistency (identical for both cars) = (std field - std {winner}) / std {winner}.",
        "New consistency, per car = (std car - std field) / std field.",
        "",
        "| Section | Distance (m) | z-score | Old label | New label | std "
        f"{DRIVER} | std {winner} | std field | Old consistency (both cars) | "
        f"New consistency {DRIVER} | New consistency {winner} |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    changed = 0
    for i, (old, new) in enumerate(zip(old_label, new_label, strict=True)):
        if old == new:
            continue
        changed += 1
        lines.append(
            f"| {i} | {edges[i]:.0f}-{edges[i + 1]:.0f} | {new_z[i]:+.2f} | {old} | {new} "
            f"| {std_subject[i]:.3f} | {std_compare[i]:.3f} | {std_field[i]:.3f} "
            f"| {float(old_cons[i]):+.2f} (std {winner} {std_old_ref[i]:.3f}) "  # type: ignore[arg-type]
            f"| {new_cons[i]:+.2f} | {new_cons_compare[i]:+.2f} |"
        )
    lines += ["", f"{changed} of {len(new_label)} sections changed label."]
    lines += [
        "",
        "## All sections (old -> new)",
        "",
        "| Section | z-score | Old label | New label |",
        "|---|---|---|---|",
    ]
    for i, (old, new) in enumerate(zip(old_label, new_label, strict=True)):
        lines.append(f"| {i} | {new_z[i]:+.2f} | {old} | {new} |")
    text = "\n".join(lines) + "\n"
    (NEW / "label_changes.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
