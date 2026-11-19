"""Tests for DataQualityChecker, completeness and uniqueness (card C1).

Author: Shepard. Card Q1 adds more cases on Monday.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import DataLoader
from src.quality import DataQualityChecker

DATA = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"


def _by_column(findings, check):
    return {f["column"]: f["issues"] for f in findings if f["check"] == check}


def test_hidden_zero_counts_match_the_real_data():
    checker = DataQualityChecker(DataLoader(DATA).load())
    zeros = _by_column(
        checker.check_completeness(), "Zero recorded where zero is impossible"
    )
    assert zeros == {
        "Glucose": 5,
        "BloodPressure": 35,
        "SkinThickness": 227,
        "Insulin": 374,
        "BMI": 11,
    }


def test_blank_cells_are_counted():
    df = pd.DataFrame({"Glucose": [100, None, 90], "Outcome": [0, 1, 0]})
    findings = DataQualityChecker(df).check_completeness()
    blanks = _by_column(findings, "Blank (NaN) cells")
    assert blanks["Glucose"] == 1 and blanks["Outcome"] == 0


def test_duplicate_rows_are_found():
    df = pd.DataFrame({"Glucose": [100, 100, 90], "Outcome": [0, 0, 1]})
    findings = DataQualityChecker(df).check_uniqueness()
    full = [f for f in findings if f["check"] == "Fully duplicated rows"][0]
    assert full["issues"] == 1


def test_real_data_has_no_duplicates():
    checker = DataQualityChecker(DataLoader(DATA).load())
    assert all(f["issues"] == 0 for f in checker.check_uniqueness())


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        DataQualityChecker(pd.DataFrame())
