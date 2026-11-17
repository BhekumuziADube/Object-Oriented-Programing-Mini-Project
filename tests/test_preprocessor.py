"""Tests for ZeroToNaN and Preprocessor (card C6). Author: Shepard.

Tests use scikit-learn's SimpleImputer, so they do not depend on the
team's own Imputer classes (card C4).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from src.preprocessor import Preprocessor, ZeroToNaN

DATA = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"


def _frame():
    return pd.DataFrame(
        {
            "Pregnancies": [0, 2, 1, 4],
            "Glucose": [100, 0, 120, 90],
            "BloodPressure": [70, 80, 0, 72],
            "SkinThickness": [20, 0, 25, 30],
            "Insulin": [0, 100, 0, 85],
            "BMI": [30.0, 28.0, 0.0, 33.0],
            "DiabetesPedigreeFunction": [0.5, 0.3, 0.4, 0.6],
            "Age": [30, 40, 50, 25],
        }
    )


def _preprocessor(log=None):
    return Preprocessor(SimpleImputer(strategy="median"), log=log)


class _FakeLog:
    """Stands in for CleaningLog: only needs an add() method."""

    def __init__(self):
        self.entries = []

    def add(self, *args):
        self.entries.append(args)


def test_zero_to_nan_only_touches_impossible_zero_columns():
    out = ZeroToNaN().transform(_frame())
    assert out["Glucose"].isna().sum() == 1
    assert out["Pregnancies"].tolist() == [0, 2, 1, 4]  # zero is valid here


def test_zero_to_nan_does_not_change_the_input():
    df = _frame()
    ZeroToNaN().transform(df)
    assert (df["Glucose"] == 0).sum() == 1


def test_fit_transform_leaves_no_missing_values():
    X = _preprocessor().fit_transform(_frame())
    assert X.shape == (4, 8)
    assert not np.isnan(X).any()


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        _preprocessor().transform(_frame())


def test_scaler_is_fitted_on_training_rows_only():
    train, test = _frame().iloc[:3], _frame().iloc[3:]
    prep = _preprocessor()
    X_train = prep.fit_transform(train)
    X_test = prep.transform(test)
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-9)
    assert not np.allclose(X_test.mean(axis=0), 0)  # test was not re-fitted


def test_feature_names_keep_the_column_order():
    prep = _preprocessor().fit(_frame())
    assert prep.feature_names == list(_frame().columns)


def test_hidden_zeros_are_logged_per_column():
    log = _FakeLog()
    _preprocessor(log).fit(_frame())
    counts = {entry[1]: entry[4] for entry in log.entries}
    assert counts == {
        "Glucose": 1,
        "BloodPressure": 1,
        "SkinThickness": 1,
        "Insulin": 2,
        "BMI": 1,
    }


def test_real_data_comes_out_complete():
    X = pd.read_csv(DATA).drop(columns="Outcome")
    out = _preprocessor().fit_transform(X)
    assert out.shape == (768, 8)
    assert not np.isnan(out).any()
