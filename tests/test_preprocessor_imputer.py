"""PP1: the Preprocessor must work with every registered imputer.

Pair session PP1. Driver: Shepard. Navigator: Bhekumuzi.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.imputers import IMPUTERS
from src.preprocessor import Preprocessor

DATA = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"


def _make(entry):
    """IMPUTERS may hold classes or ready-made objects; accept both."""
    return entry() if isinstance(entry, type) else entry


@pytest.fixture(scope="module")
def split():
    df = pd.read_csv(DATA)
    X = df.drop(columns=["Outcome"])
    return train_test_split(
        X, df["Outcome"], test_size=0.2, stratify=df["Outcome"], random_state=42
    )


@pytest.mark.parametrize("name", sorted(IMPUTERS))
def test_imputer_fits_inside_preprocessor(name, split):
    X_train, X_test, _, _ = split
    prep = Preprocessor(_make(IMPUTERS[name]))
    train_out = prep.fit_transform(X_train)
    test_out = prep.transform(X_test)
    assert train_out.shape == X_train.shape
    assert test_out.shape == X_test.shape
    assert not np.isnan(train_out).any()
    assert not np.isnan(test_out).any()


@pytest.mark.parametrize("name", sorted(IMPUTERS))
def test_test_rows_do_not_change_what_was_learned(name, split):
    X_train, X_test, _, _ = split
    prep = Preprocessor(_make(IMPUTERS[name])).fit(X_train)
    before = prep.transform(X_train.head(10))
    prep.transform(X_test * 100)  # extreme test rows: transform only
    after = prep.transform(X_train.head(10))
    np.testing.assert_allclose(before, after)


@pytest.mark.parametrize("name", sorted(IMPUTERS))
def test_preprocessor_leaves_the_imputer_object_unfitted(name, split):
    X_train, _, _, _ = split
    imputer = _make(IMPUTERS[name])
    Preprocessor(imputer).fit(X_train)
    assert not hasattr(imputer, "statistics_")  # a clone was fitted, not this
