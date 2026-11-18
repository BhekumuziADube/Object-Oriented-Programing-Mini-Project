"""Failure-case tests for DataLoader (card Q1). Author: Shepard.

The loader may report problems with its own DataLoadError or with Python's
FileNotFoundError / ValueError, so the tests accept any of them.
"""
from pathlib import Path

import pytest

from src import data_loader
from src.data_loader import DataLoader

DATA = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"
HEADER = (
    "Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,"
    "DiabetesPedigreeFunction,Age,Outcome"
)
LOAD_ERRORS = tuple(
    e
    for e in (
        getattr(data_loader, "DataLoadError", None),
        FileNotFoundError,
        ValueError,
    )
    if e is not None
)
NOT_LOADED_ERRORS = tuple(
    e
    for e in (getattr(data_loader, "DataLoadError", None), RuntimeError)
    if e is not None
)


def _write(tmp_path, text, name="data.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_missing_file_raises(tmp_path):
    with pytest.raises(LOAD_ERRORS):
        DataLoader(tmp_path / "nope.csv").load()


def test_empty_file_raises(tmp_path):
    with pytest.raises(LOAD_ERRORS):
        DataLoader(_write(tmp_path, "")).load()


def test_header_only_file_raises(tmp_path):
    with pytest.raises(LOAD_ERRORS):
        DataLoader(_write(tmp_path, HEADER + "\n")).load()


def test_missing_column_raises(tmp_path):
    with pytest.raises(LOAD_ERRORS):
        DataLoader(_write(tmp_path, "Pregnancies,Outcome\n1,0\n")).load()


def test_non_binary_target_raises(tmp_path):
    row = "1,100,70,20,80,30.0,0.5,40,2"
    with pytest.raises(LOAD_ERRORS):
        DataLoader(_write(tmp_path, HEADER + "\n" + row + "\n")).load()


def test_summary_before_load_raises():
    with pytest.raises(NOT_LOADED_ERRORS):
        DataLoader(DATA).summary()


def test_path_property_returns_a_path():
    assert DataLoader(DATA).path == DATA


def test_returned_data_is_a_copy():
    loader = DataLoader(DATA)
    df = loader.load()
    expected = loader.summary()["positive_rate"]
    df["Outcome"] = 0  # changing our copy must not change the loader's data
    assert loader.summary()["positive_rate"] == expected


def test_summary_after_load_reports_rows():
    loader = DataLoader(DATA)
    loader.load()
    assert loader.summary()["rows"] == 768
