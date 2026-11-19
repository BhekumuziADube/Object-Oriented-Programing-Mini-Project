"""Tests for the stratified train and test split (card Q2)."""

import pandas as pd
import pytest

from src.data_split import split_data


def make_data(rows=100, positives=30):
    """Build a small table with an Outcome column."""
    return pd.DataFrame(
        {
            "Glucose": range(rows),
            "Outcome": [1] * positives + [0] * (rows - positives),
        }
    )


def test_split_is_80_20():
    X_train, X_test, y_train, y_test = split_data(make_data())
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_split_sizes_for_the_real_data_size():
    data = make_data(rows=768, positives=268)
    X_train, X_test, _, _ = split_data(data)
    assert len(X_train) == 614
    assert len(X_test) == 154


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_split_is_stratified_for_any_seed(seed):
    _, _, y_train, y_test = split_data(make_data(), random_state=seed)
    assert y_train.sum() == 24
    assert y_test.sum() == 6


def test_train_and_test_do_not_overlap_and_use_every_row():
    data = make_data()
    X_train, X_test, _, _ = split_data(data)
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert len(X_train) + len(X_test) == len(data)


def test_split_is_the_same_every_time():
    first = split_data(make_data())
    second = split_data(make_data())
    for a, b in zip(first, second):
        assert a.equals(b)


def test_outcome_is_not_in_the_features():
    X_train, X_test, _, _ = split_data(make_data())
    assert "Outcome" not in X_train.columns
    assert "Outcome" not in X_test.columns


def test_missing_outcome_column_raises():
    with pytest.raises(ValueError, match="not found"):
        split_data(make_data().drop(columns="Outcome"))


@pytest.mark.parametrize("bad_size", [0, 1, 1.5, -0.2])
def test_bad_test_size_raises(bad_size):
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_data(make_data(), test_size=bad_size)
