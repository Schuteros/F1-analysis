import pandas as pd
import pytest

from f1_analysis import attribution

SLOW = 1.0  # above the 0.75 z-score threshold
OK = 0.0
INCONSISTENT = 0.5  # above the 0.2 consistency threshold
STEADY = 0.0


@pytest.mark.parametrize(
    ("z", "consistency", "other_z", "other_consistency", "expected"),
    [
        # nothing wrong
        (OK, STEADY, SLOW, INCONSISTENT, "None"),
        # slow + consistent: car only if the other car is slow too
        (SLOW, STEADY, SLOW, STEADY, "Car"),
        (SLOW, STEADY, OK, STEADY, "Driver"),
        (SLOW, STEADY, OK, INCONSISTENT, "Driver"),  # other's consistency is not consulted
        # slow + inconsistent: car only if the other car is inconsistent too
        (SLOW, INCONSISTENT, OK, INCONSISTENT, "Car"),
        (SLOW, INCONSISTENT, SLOW, STEADY, "Driver"),  # other's slowness is not consulted
        # not slow but inconsistent
        (OK, INCONSISTENT, OK, INCONSISTENT, "Car"),
        (OK, INCONSISTENT, OK, STEADY, "Driver"),
    ],
)
def test_label_section_rules(
    z: float, consistency: float, other_z: float, other_consistency: float, expected: str
) -> None:
    assert attribution.label_section(z, consistency, other_z, other_consistency) == expected


def test_label_section_thresholds_are_strict() -> None:
    assert attribution.label_section(0.75, 0.2, 5.0, 5.0) == "None"
    assert attribution.label_section(0.7501, 0.0, 0.7501, 0.0) == "Car"


def test_problem_analysis_labels_each_row_by_position() -> None:
    first = pd.DataFrame(
        {"Time Diff Z-Score": [OK, SLOW, OK], "Consistency": [STEADY, STEADY, INCONSISTENT]}
    )
    second = pd.DataFrame(
        {"Time Diff Z-Score": [OK, OK, OK], "Consistency": [STEADY, STEADY, INCONSISTENT]}
    )
    result = attribution.problem_analysis(first, second)
    assert result["Problem"].tolist() == ["None", "Driver", "Car"]
    assert list(result.columns) == ["Time Diff Z-Score", "Consistency", "Problem"]


def test_problem_analysis_validates_inputs() -> None:
    table = pd.DataFrame({"Time Diff Z-Score": [0.0], "Consistency": [0.0]})
    with pytest.raises(ValueError, match="1 sections vs 2"):
        attribution.problem_analysis(table, pd.concat([table, table]))
    with pytest.raises(ValueError, match="missing columns"):
        attribution.problem_analysis(table, pd.DataFrame({"Consistency": [0.0]}))
