"""Tests for the zeros-kept control imputer (card X1)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.imputers import IMPUTERS, ZerosKeptImputer


@pytest.fixture
def data():
    return pd.DataFrame(
        {"Glucose": [0.0, 120.0, 140.0], "BMI": [30.0, 0.0, 25.0]}
    )


def test_zeros_kept_imputer_is_registered():
    assert IMPUTERS["zeros_kept"] is ZerosKeptImputer


def test_zeros_kept_imputer_says_it_keeps_zeros():
    assert ZerosKeptImputer.keeps_zeros is True


def test_fit_returns_the_imputer_itself(data):
    imputer = ZerosKeptImputer()
    assert imputer.fit(data) is imputer


def test_data_comes_back_unchanged(data):
    result = ZerosKeptImputer().fit(data).transform(data)
    pd.testing.assert_frame_equal(result, data)


def test_zeros_are_kept(data):
    result = ZerosKeptImputer().fit(data).transform(data)
    assert result.loc[0, "Glucose"] == 0.0
    assert result.loc[1, "BMI"] == 0.0


def test_result_is_a_copy(data):
    result = ZerosKeptImputer().fit(data).transform(data)
    result.loc[0, "Glucose"] = 999.0
    assert data.loc[0, "Glucose"] == 0.0


def test_missing_values_raise_a_clear_error(data):
    broken = data.copy()
    broken.loc[0, "Glucose"] = np.nan
    with pytest.raises(ValueError, match="ZeroToNaN"):
        ZerosKeptImputer().fit(broken).transform(broken)


def test_can_be_cloned():
    assert isinstance(clone(ZerosKeptImputer()), ZerosKeptImputer)
