"""Edge-case tests for DataQualityChecker (card Q1). Author: Shepard.

Covers the completeness and uniqueness checks from card C1.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.quality import DataQualityChecker

DATA = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"
KEYS = {"dimension", "check", "column", "issues", "of", "percent", "note"}


def _find(findings, check, column=None):
    for f in findings:
        if f["check"] == check and (column is None or f["column"] == column):
            return f
    raise AssertionError(f"no finding for {check!r} / {column!r}")


def test_checks_do_not_change_the_input():
    df = pd.DataFrame({"Glucose": [0, 100], "Outcome": [0, 1]})
    before = df.copy()
    checker = DataQualityChecker(df)
    checker.check_completeness()
    checker.check_uniqueness()
    pd.testing.assert_frame_equal(df, before)


def test_non_dataframe_is_rejected():
    with pytest.raises(TypeError):
        DataQualityChecker([1, 2, 3])


def test_every_finding_has_the_seven_keys():
    checker = DataQualityChecker(DataLoader(DATA).load())
    findings = checker.check_completeness() + checker.check_uniqueness()
    assert findings
    assert all(set(f) == KEYS for f in findings)


def test_percent_is_calculated_correctly():
    df = pd.DataFrame({"Glucose": [0, 100, 110, 120], "Outcome": [0, 1, 0, 1]})
    findings = DataQualityChecker(df).check_completeness()
    zeros = _find(findings, "Zero recorded where zero is impossible", "Glucose")
    assert (zeros["issues"], zeros["of"], zeros["percent"]) == (1, 4, 25.0)


def test_pregnancies_zero_is_not_a_hidden_zero():
    df = pd.DataFrame({"Pregnancies": [0, 0, 1], "Glucose": [90, 95, 100]})
    columns = [
        f["column"]
        for f in DataQualityChecker(df).check_completeness()
        if f["check"] == "Zero recorded where zero is impossible"
    ]
    assert columns == ["Glucose"]


def test_missing_hidden_zero_columns_do_not_crash():
    df = pd.DataFrame({"Age": [30, 40], "Outcome": [0, 1]})
    findings = DataQualityChecker(df).check_completeness()
    assert all("Zero" not in f["check"] for f in findings)


def test_rows_with_a_hidden_zero_on_real_data():
    checker = DataQualityChecker(DataLoader(DATA).load())
    row = _find(
        checker.check_completeness(), "Rows with at least one hidden zero"
    )
    assert (row["issues"], row["of"], row["percent"]) == (376, 768, 49.0)


def test_patient_id_is_ignored_when_looking_for_duplicates():
    df = pd.DataFrame(
        {"PatientID": [1, 2], "Glucose": [100, 100], "Outcome": [0, 0]}
    )
    full = _find(DataQualityChecker(df).check_uniqueness(), "Fully duplicated rows")
    assert full["issues"] == 1


def test_same_features_different_outcome_is_only_a_feature_duplicate():
    df = pd.DataFrame({"Glucose": [100, 100], "Outcome": [0, 1]})
    findings = DataQualityChecker(df).check_uniqueness()
    full = _find(findings, "Fully duplicated rows")
    features = _find(findings, "Duplicate rows ignoring Outcome")
    assert (full["issues"], features["issues"]) == (0, 1)


def test_blank_cells_are_reported_for_every_column():
    df = pd.DataFrame({"Glucose": [100, None], "Age": [30, 40]})
    blanks = {
        f["column"]: f["issues"]
        for f in DataQualityChecker(df).check_completeness()
        if f["check"] == "Blank (NaN) cells"
    }
    assert blanks == {"Glucose": 1, "Age": 0}
