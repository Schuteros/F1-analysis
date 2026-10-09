# F1 section analysis

Splits a lap into corners, braking zones, exits and straights from public F1 telemetry, then asks in
which sections a driver loses time and whether the cause looks like the car or the driver.

## Example output

![Problem sections on track](docs/example.png)

**Finding:** _(to be filled in)_

## How it works

- **Sections:** average the driver's quick laps per metre of track; corners = high curvature,
  braking = brake on, exits = strong acceleration, straights = low curvature.
- **Section times:** cut the lap at the midpoints between speed peaks and troughs and time each piece.
- **Consistency:** how much more the lap-to-lap spread of a section is than a reference car's.
- **Car or driver:** slow or inconsistent sections count as the car's if the comparison car shows the
  same symptom, otherwise the driver's.
- **Problem type:** car sections are split into slow stretches and named by their averages
  (DRS, cornering, straight-line speed, acceleration, other).

## Install and run

```
python -m venv .venv && .venv\Scripts\activate
pip install -e .
f1-analysis --year 2025 --event CHINESE --session Sprint --driver HUL --save-plots plots
```

Options: `--compare-to CODE` (default: the winner), `--cache-dir DIR`. Without `--save-plots`
the figures open in windows. Developers: `pip install -e ".[dev]"`, then `mypy`, `ruff check`, `pytest`.
Regression check against the saved reference (needs the FastF1 cache): `python scripts/compare_reference.py --old reference/fix-consistency`.

## Limitations

- Public telemetry only (about 4 Hz, positions from one lap).
- Thresholds are hand-tuned (`f1_analysis/config.py`), not fitted.
- Not validated against team data.
- Some behaviour is odd but intentionally kept: see `KNOWN_ISSUES.md`.
