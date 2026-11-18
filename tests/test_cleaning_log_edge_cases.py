"""Edge-case tests for CleaningLog (card Q1). Author: Shepard."""
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from src.cleaning_log import CleaningLog, LogEntry


def _add(log, rows=5, column="Insulin"):
    return log.add("Completeness", column, "issue", "action", rows, "reason")


def test_add_returns_the_new_entry():
    entry = _add(CleaningLog())
    assert isinstance(entry, LogEntry)
    assert (entry.step, entry.column, entry.rows_affected) == (1, "Insulin", 5)


def test_an_entry_cannot_be_changed():
    entry = _add(CleaningLog())
    with pytest.raises(FrozenInstanceError):
        entry.rows_affected = 99


def test_empty_text_is_rejected():
    with pytest.raises(ValueError):
        CleaningLog().add("Completeness", "Insulin", "  ", "action", 1, "reason")


def test_float_row_count_is_rejected():
    with pytest.raises(ValueError):
        _add(CleaningLog(), rows=2.5)


def test_zero_rows_is_allowed():
    assert _add(CleaningLog(), rows=0).rows_affected == 0


def test_entries_cannot_be_edited_from_outside():
    log = CleaningLog()
    _add(log)
    with pytest.raises(AttributeError):
        log.entries.append("tamper")
    assert len(log) == 1


def test_all_six_dimensions_are_accepted():
    log = CleaningLog()
    for dimension in CleaningLog.DIMENSIONS:
        log.add(dimension, "col", "issue", "action", 1, "reason")
    assert len(log) == 6


def test_frame_has_one_row_per_entry_in_order():
    log = CleaningLog()
    _add(log, column="Glucose")
    _add(log, column="BMI")
    frame = log.to_frame()
    assert frame["step"].tolist() == [1, 2]
    assert frame["column"].tolist() == ["Glucose", "BMI"]


def test_save_creates_missing_folders_and_returns_the_path(tmp_path):
    log = CleaningLog()
    _add(log)
    target = tmp_path / "a" / "b" / "log.csv"
    assert log.save(target) == target
    assert pd.read_csv(target).shape == (1, 7)
