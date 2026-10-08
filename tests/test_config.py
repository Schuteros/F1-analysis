import math

from f1_analysis import config


def test_curvature_thresholds_are_the_same_physical_value() -> None:
    # 0.0002 1/dm and 0.002 1/m are both a radius of 500 m (1 m = 10 dm).
    per_dm_as_per_m = config.CORNER_CURVATURE_THRESHOLD_PER_DM * config.POSITION_UNITS_PER_M
    assert math.isclose(per_dm_as_per_m, config.CLASSIFY_CURVATURE_THRESHOLD_PER_M)


def test_problem_colors_cover_all_problem_types() -> None:
    assert set(config.PROBLEM_TYPES) <= set(config.PROBLEM_COLORS)
