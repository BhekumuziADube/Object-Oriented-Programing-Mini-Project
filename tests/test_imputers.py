"""Tests for the imputers (card Q2)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from src.imputers import IMPUTERS, Imputer, MedianImputer


@pytest.fixture
def train():
    """Small training table: the median of Glucose is 120 and of BMI is 30."""
    return pd.DataFrame(
        {
            "Glucose": [100.0, 120.0, np.nan, 140.0],
            "BMI": [20.0, np.nan, 30.0, 40.0],
        }
    )


def test_base_class_cannot_be_created():
    with pytest.raises(TypeError):
        Imputer()


def test_median_imputer_is_registered():
    assert IMPUTERS["median"] is MedianImputer


def test_fit_returns_the_imputer_itself(train):
    imputer = MedianImputer()
    assert imputer.fit(train) is imputer


def test_no_nan_left_after_imputing(train):
    filled = MedianImputer().fit(train).transform(train)
    assert filled.isna().sum().sum() == 0


def test_missing_values_get_the_training_median(train):
    filled = MedianImputer().fit(train).transform(train)
    assert filled.loc[2, "Glucose"] == 120.0
    assert filled.loc[1, "BMI"] == 30.0


def test_known_values_are_not_changed(train):
    filled = MedianImputer().fit(train).transform(train)
    assert filled.loc[0, "Glucose"] == 100.0
    assert filled.loc[3, "BMI"] == 40.0


def test_new_data_gets_the_training_median_not_its_own(train):
    imputer = MedianImputer().fit(train)
    new_data = pd.DataFrame({"Glucose": [np.nan, 1000.0], "BMI": [np.nan, 5.0]})
    filled = imputer.transform(new_data)
    assert filled.loc[0, "Glucose"] == 120.0
    assert filled.loc[0, "BMI"] == 30.0


def test_result_keeps_columns_and_row_numbers(train):
    filled = MedianImputer().fit(train).transform(train)
    assert isinstance(filled, pd.DataFrame)
    assert list(filled.columns) == list(train.columns)
    assert filled.index.equals(train.index)


def test_input_table_is_not_changed(train):
    MedianImputer().fit(train).transform(train)
    assert train.isna().sum().sum() == 2


def test_transform_before_fit_raises(train):
    with pytest.raises(NotFittedError):
        MedianImputer().transform(train)


def test_fit_transform_matches_fit_then_transform(train):
    combined = MedianImputer().fit_transform(train)
    separate = MedianImputer().fit(train).transform(train)
    pd.testing.assert_frame_equal(combined, separate)


def test_imputer_can_be_cloned():
    assert isinstance(clone(MedianImputer()), MedianImputer)
