# Known issues

Things in the original `main.py` that look wrong or odd. **None of them were fixed**: the refactor
keeps the numbers identical (see `scripts/compare_reference.py`). Each item says what the code does
now and where it lives. Decide what to change later; every item is a one-line change in `config.py`
or a small edit in the named function.

## A. Results are affected (found while refactoring)

1. **Curvature is half the true value.** `telemetry.calculate_curvature` computes `2*area/(abc)`.
   For a circle of radius R this gives `1/(2R)`, not `1/R` (true Menger curvature is `4*area/(abc)`).
   Effect: every curvature threshold selects radii half as large as its comment says
   (`0.0002 1/dm` really means R < 250 m, not 500 m). Constant: `config.CURVATURE_AREA_FACTOR` (2.0;
   4.0 would be correct). The unit test checks `1/(2r)` and the factor-4 relation.
2. **"Consistency" is the same number for both cars.** `calculate_driver_consistency` compares the
   *comparison car's* lap-to-lap spread with the *whole field's* spread. It does not use the analysed
   driver at all, so `analysis_driver` and `analysis_winner` have identical Consistency columns
   (verified in the reference output). Consequence in `attribution.label_section`: if a section is
   inconsistent, the other car is inconsistent by definition, so the label is always "Car". "Driver"
   can only come from the slow-but-consistent branch.
3. **Lap-picking order differs between call sites.** `pick_accurate().pick_quicklaps()` (main),
   `pick_quicklaps().pick_accurate()` (analyze_performance, consistency). The quick-lap cut-off is
   relative to the fastest lap in the set, so the order can select different laps. Kept as three
   functions in `data.py`.
4. **Averaged telemetry is sparse.** X/Y come only from the fastest lap's samples and bins without a
   position are dropped, leaving about one row per 15 m (357 rows for the 5.4 km reference lap).
   Curvature, the 7-sample smoothing and all 10-50 m length thresholds work on that grid.
   Duplicate `DistanceBin` rows in the fastest lap would duplicate rows in the merge (not seen so far).
5. **`find_general_problems` never produces "Braking"** and only looks at "Car" sections.
   `HIGH_BRAKE_THRESHOLD` and `LOW_THROTTLE_THRESHOLD` are unused (kept in `config.py`, marked unused).
6. **`remove_overlaps` deletes instead of "not splitting".** When the reference section is shorter than
   the minimum, the docstring says not to split; the code `continue`s and drops the overlapped piece.
   Not called anywhere. Behaviour kept and pinned by a test.
7. **Exit acceleration threshold differs by entry point.** `find_exits` default 3 m/s^2,
   `calculate_sections` passes 3.75. Constants: `EXIT_ACCELERATION_THRESHOLD_FIND_EXITS` / `_SECTIONS`.

## B. Mismatches the brief already listed

8. **Curvature thresholds.** 0.002 (`CLASSIFY_CURVATURE_THRESHOLD_PER_M`, metres) vs 0.0002
   (`CORNER_/STRAIGHT_/SECTIONS_CURVATURE_THRESHOLD_PER_DM`, decimetres). Same physical value; FastF1
   X/Y are decimetres and the classifier divides by 10. Three per-dm constants exist because three
   functions each had their own default.
9. ~~**`calculate_section_times` and `get_section_times`** were copies.~~ **Resolved in
   `dedupe-section-times`.** A test showed the first was exactly the mean of the second (identical
   floats on 7 synthetic laps, 7 sections). `calculate_section_times` is deleted; `get_section_times`
   stays and `mean_section_times` averages its output. The pipeline used both (means for z-scores,
   lists for consistency), so it now computes the lists once per group.
10. **`find_braking_zones` default `gap_threshold=00`** (0). Zones are merged only if they touch or
    overlap. `config.BRAKING_GAP_M`.
11. **`main()` only plotted sections.** The car-vs-driver pipeline was never called. The CLI now runs
    all of it (the reference script did the same against the old functions).
12. **Hard-coded session and driver** (2025 Chinese Sprint, HUL): now the CLI defaults.

## C. Fragile spots (behaviour kept, now fail loudly or are documented)

13. **Hidden in-place mutation.** `find_corners` smoothed `Curvature` in the caller's DataFrame and
    `find_exits` added an `Acceleration` column, so `find_straights` saw smoothed curvature only
    because it ran after `find_corners`. Now explicit: `sections.with_smoothed_curvature` once, then
    both finders. **API change:** `find_corners` expects already-smoothed curvature; call
    `calculate_sections` for the original behaviour.
14. **Silent fallbacks kept on purpose, with a named constant or comment:** a zero standard deviation
    is replaced by 1 (`ZERO_STD_REPLACEMENT`, z-scores and consistency); undefined acceleration
    (first sample, zero speed) becomes 0; speed is assumed km/h if its maximum exceeds 100.
15. **Made stricter (raise `ValueError` instead of NaN/None/crash):** missing columns, NaN or
    infinite values in used columns, empty lap lists, `get_average_speeds` returning `None`, traces
    shorter than the smoothing window, mismatching section numbers, subsections with no telemetry.
    None of these triggered on the reference session.
16. **Row matching by position.** `problem_analysis` and `detect_slow_subsections` used DataFrame
    labels as positions (`.iloc[i]`, `midpoints[i - 1]`). Identical for a 0..n-1 index; now positional
    and length-checked.
17. **`calculate_curvature` length quirk.** For fewer than 2 points the original returned a list of
    length 2 (`[0] + [] + [0]`); now it returns zeros of the input length. Same classification result.
18. **Unused in any pipeline:** `plot_section_graph`, `visualize_subsection`, `remove_overlaps`,
    `find_and_override_drs_zones`. Moved and tested, not wired in.
19. **Debug `print`s removed** (curvature statistics, detected straights, acceleration statistics,
    column list). Output text only; no effect on results.
