"""Tests for CleaningLog (card CL1). Author: Shepard."""
import pandas as pd
import pytest

from src.cleaning_log import CleaningLog


def _add_example(log):
    return log.add(
        "Completeness",
        "Insulin",
        "374 zeros recorded",
        "Converted 0 to missing",
        374,
        "Zero insulin is impossible",
    )


def test_steps_are_numbered_automatically():
    log = CleaningLog()
    _add_example(log)
    _add_example(log)
    assert [e.step for e in log.entries] == [1, 2]
    assert len(log) == 2


def test_unknown_dimension_is_rejected():
    with pytest.raises(ValueError):
        CleaningLog().add("Beauty", "Age", "x", "y", 1, "z")


def test_negative_rows_are_rejected():
    with pytest.raises(ValueError):
        CleaningLog().add("Validity", "Age", "x", "y", -1, "z")


def test_empty_log_still_has_columns():
    assert list(CleaningLog().to_frame().columns) == CleaningLog.COLUMNS


def test_save_writes_a_readable_csv(tmp_path):
    log = CleaningLog()
    _add_example(log)
    path = log.save(tmp_path / "out" / "cleaning_log.csv")
    back = pd.read_csv(path)
    assert back.loc[0, "column"] == "Insulin"
    assert back.loc[0, "rows_affected"] == 374
