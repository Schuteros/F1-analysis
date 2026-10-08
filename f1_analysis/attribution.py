"""Car-vs-driver labelling of sections.

A section where the subject is slow or inconsistent is compared with a second car:
if the second car shows the same symptom the cause is probably the car, else the driver.
"""

import pandas as pd

from f1_analysis import config
from f1_analysis.telemetry import require_columns

NO_PROBLEM = "None"  # label written as the string "None", as in the original
CAR = "Car"
DRIVER = "Driver"


def label_section(
    z_score: float, consistency: float, other_z_score: float, other_consistency: float
) -> str:
    """Label one section "Car", "Driver" or "None".

    Slow (z above Z_SCORE_PROBLEM_THRESHOLD) or inconsistent (above
    CONSISTENCY_PROBLEM_THRESHOLD) sections are problems. Inconsistent subject: Car if the
    other car is inconsistent too. Slow but consistent subject: Car if the other car is also slow.
    """
    slow = z_score > config.Z_SCORE_PROBLEM_THRESHOLD
    inconsistent = consistency > config.CONSISTENCY_PROBLEM_THRESHOLD
    if inconsistent:
        other_inconsistent = other_consistency > config.CONSISTENCY_PROBLEM_THRESHOLD
        return CAR if other_inconsistent else DRIVER
    if slow:
        other_slow = other_z_score > config.Z_SCORE_PROBLEM_THRESHOLD
        return CAR if other_slow else DRIVER
    return NO_PROBLEM


def problem_analysis(
    first_car_analysis: pd.DataFrame, second_car_analysis: pd.DataFrame
) -> pd.DataFrame:
    """Label every section of the first car using the second car as the comparison.

    Both tables need 'Time Diff Z-Score' and 'Consistency', one row per section, in the same
    order (rows are matched by position). Returns the first car's columns plus 'Problem'.
    """
    columns = ["Time Diff Z-Score", "Consistency"]
    require_columns(first_car_analysis, columns, "problem_analysis (first car)")
    require_columns(second_car_analysis, columns, "problem_analysis (second car)")
    if len(first_car_analysis) != len(second_car_analysis):
        raise ValueError(
            f"problem_analysis: {len(first_car_analysis)} sections vs {len(second_car_analysis)}"
        )
    labels = [
        label_section(z, consistency, other_z, other_consistency)
        for z, consistency, other_z, other_consistency in zip(
            first_car_analysis["Time Diff Z-Score"],
            first_car_analysis["Consistency"],
            second_car_analysis["Time Diff Z-Score"],
            second_car_analysis["Consistency"],
            strict=True,
        )
    ]
    return first_car_analysis[columns].assign(Problem=labels)
