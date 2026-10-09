"""All thresholds and constants. Every value states its unit and meaning.

Values are copied from the original ``main.py``; see KNOWN_ISSUES.md for the oddities.
"""

from typing import Final

# --- CLI defaults (were hard-coded in main()) -------------------------------------------------
DEFAULT_YEAR: Final = 2025  # season
DEFAULT_EVENT: Final = "CHINESE"  # event name as accepted by fastf1.get_session
DEFAULT_SESSION: Final = "Sprint"  # session name as accepted by fastf1.get_session
DEFAULT_DRIVER: Final = "HUL"  # three-letter driver code analysed
DEFAULT_CACHE_DIR: Final = ".fastf1-cache"  # FastF1 on-disk cache directory

# --- Units ------------------------------------------------------------------------------------
POSITION_UNITS_PER_M: Final = 10.0  # FastF1 X/Y are in decimetres: 10 dm = 1 m
KMH_PER_MPS: Final = 3.6  # km/h per m/s

# --- Average telemetry ------------------------------------------------------------------------
DISTANCE_BIN_M: Final = 0.5  # m; laps are binned to this distance resolution before averaging
DRS_OPEN_CODES: Final = (10, 12, 14)  # FastF1 DRS channel values meaning "flap open"

# --- Speed-trace section detection (detect_sections) ------------------------------------------
SPEED_PEAK_MIN_SEPARATION_SAMPLES: Final = 25  # samples; min gap between speed peaks/troughs
SPEED_SMOOTHING_WINDOW_SAMPLES: Final = 25  # samples; Savitzky-Golay window for speed (odd)
SPEED_SMOOTHING_POLYORDER: Final = 2  # polynomial order of the speed Savitzky-Golay filter

# --- Curvature smoothing ----------------------------------------------------------------------
CURVATURE_SMOOTHING_WINDOW_SAMPLES: Final = 7  # samples; Savitzky-Golay window (odd)
CURVATURE_SMOOTHING_POLYORDER: Final = 2  # polynomial order of the curvature filter

# --- Curvature ----------------------------------------------------------------------------------
CURVATURE_AREA_FACTOR: Final = 4.0  # curvature = factor * triangle area / (abc) = 1 / radius

# Thresholds are curvature (1/radius) above which a point counts as "corner", all radius < 250 m.
# Averaged telemetry has X/Y in decimetres (1/dm), the classifier uses metres (1/m):
# 0.0004 1/dm == 0.004 1/m == radius 250 m. Kept as separate names so the (apparent) mismatch
# stays visible. (These are twice the pre-fix values, because the curvature function now
# returns twice what it used to; the radius cut-off is unchanged.)
CORNER_CURVATURE_THRESHOLD_PER_DM: Final = 0.0004  # 1/dm; find_corners default (radius 2500 dm)
STRAIGHT_CURVATURE_THRESHOLD_PER_DM: Final = 0.0004  # 1/dm; find_straights default (max curvature)
SECTIONS_CURVATURE_THRESHOLD_PER_DM: Final = 0.0004  # 1/dm; calculate_sections default
CLASSIFY_CURVATURE_THRESHOLD_PER_M: Final = 0.004  # 1/m; classifier "is a corner" (radius 250 m)

# --- Corners ----------------------------------------------------------------------------------
CORNER_GAP_M: Final = 50.0  # m; corners closer than this are merged
CORNER_MIN_LENGTH_M: Final = 10.0  # m; shorter corners are discarded

# --- Braking zones ----------------------------------------------------------------------------
BRAKE_THRESHOLD: Final = 0.2  # 0..1; averaged brake value above this counts as braking
BRAKING_GAP_M: Final = 0.0  # m; braking zones closer than this are merged (0 = effectively never)
BRAKING_MIN_LENGTH_M: Final = 20.0  # m; shorter braking zones are discarded

# --- Straights --------------------------------------------------------------------------------
STRAIGHT_GAP_M: Final = 50.0  # m; straights closer than this are merged
STRAIGHT_MIN_LENGTH_M: Final = 20.0  # m; shorter straights are discarded

# --- Exits ------------------------------------------------------------------------------------
EXIT_ACCELERATION_THRESHOLD_FIND_EXITS: Final = 3.0  # m/s^2; find_exits default
EXIT_ACCELERATION_THRESHOLD_SECTIONS: Final = 3.75  # m/s^2; value calculate_sections passes
EXIT_MIN_LENGTH_M: Final = 20.0  # m; shorter exits are discarded
EXIT_SPEED_IS_KMH_ABOVE: Final = 100.0  # speed units; if max speed exceeds this, assume km/h

# --- DRS zones --------------------------------------------------------------------------------
DRS_ZONE_THROTTLE_THRESHOLD: Final = 80.0  # %; throttle at or above this counts toward a DRS zone

# --- Overlap removal --------------------------------------------------------------------------
OVERLAP_MIN_REFERENCE_LENGTH_M: Final = 50.0  # m; reference sections shorter than this are special

# --- Slow subsections -------------------------------------------------------------------------
SLOW_MERGE_GAP_M: Final = 6.0  # m; slow runs closer than this are merged
SLOW_MIN_LENGTH_M: Final = 50.0  # m; shorter slow runs are discarded

# --- Car vs driver labelling ------------------------------------------------------------------
Z_SCORE_PROBLEM_THRESHOLD: Final = 0.75  # z-score; section time is "bad" above this
CONSISTENCY_PROBLEM_THRESHOLD: Final = 0.2  # relative std excess; "inconsistent" above this
ZERO_STD_REPLACEMENT: Final = 1.0  # divisor used when a standard deviation is exactly 0
# Drivers with fewer laps than this are left out of the field's consistency reference. The
# relative error of a standard deviation from n laps is about 1/sqrt(2(n-1)): 27 % at 8 laps,
# 50 % at 3. Below 8 it is too noisy to be a reference (and it drops early retirements).
MIN_LAPS_FOR_STD: Final = 8  # laps

# --- Problem classification (find_general_problems) -------------------------------------------
HIGH_THROTTLE_THRESHOLD: Final = 90.0  # %; mean throttle above this is "high"
HIGH_SPEED_THRESHOLD: Final = 290.0  # km/h; mean speed above this is "high"
LOW_BRAKE_THRESHOLD: Final = 0.2  # 0..1; mean brake below this is "low"
DRS_THRESHOLD: Final = 0.4  # 0..1; mean DRS-open fraction above this means DRS inefficiency
# Unused in the original code (kept for reference, see KNOWN_ISSUES.md):
HIGH_BRAKE_THRESHOLD: Final = 0.6  # 0..1; unused
LOW_THROTTLE_THRESHOLD: Final = 20.0  # %; unused

PROBLEM_TYPES: Final = (
    "Braking",
    "Acceleration",
    "Cornering",
    "DRS Inefficiency",
    "Straight-Line Speed Issue",
    "Other",
)  # problem categories, in output order

# --- Plotting ---------------------------------------------------------------------------------
PROBLEM_COLORS: Final = {
    "Braking": "red",
    "Acceleration": "blue",
    "Cornering": "green",
    "DRS Inefficiency": "purple",
    "Straight-Line Speed Issue": "orange",
    "Other": "black",
    "Driver": "cyan",
}  # matplotlib colour per problem type
TRACK_COLOR: Final = "gray"  # matplotlib colour of the full track outline
TRACK_LINEWIDTH: Final = 1.0  # points; full track outline
SECTION_LINEWIDTH: Final = 2.0  # points; highlighted sections
TRACK_FIGSIZE: Final = (10, 8)  # inches (width, height) of track plots
SPEED_FIGSIZE: Final = (8, 6)  # inches (width, height) of the path-subsection plot
PLOT_DPI: Final = 100  # dots per inch for saved plots
