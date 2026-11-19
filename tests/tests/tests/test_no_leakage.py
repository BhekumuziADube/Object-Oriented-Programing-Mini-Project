"""Test that imputers learn from training folds only (card Q2)."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.imputers import MedianImputer


class RecordingImputer(MedianImputer):
    """A MedianImputer that remembers how many rows each fit() call saw."""

    rows_seen = []

    def fit(self, X, y=None):
        RecordingImputer.rows_seen.append(len(X))
        return super().fit(X, y)


def make_data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    X.iloc[::10, 0] = np.nan
    y = pd.Series(np.tile([0, 1], 50))
    return X, y


def test_imputer_is_fitted_on_each_training_fold_only():
    RecordingImputer.rows_seen = []
    X, y = make_data()
    pipeline = Pipeline(
        [("impute", RecordingImputer()), ("model", LogisticRegression())]
    )
    cross_val_score(pipeline, X, y, cv=StratifiedKFold(5))
    assert RecordingImputer.rows_seen == [80, 80, 80, 80, 80]


def test_imputer_never_sees_all_rows_during_cross_validation():
    RecordingImputer.rows_seen = []
    X, y = make_data()
    pipeline = Pipeline(
        [("impute", RecordingImputer()), ("model", LogisticRegression())]
    )
    cross_val_score(pipeline, X, y, cv=StratifiedKFold(5))
    assert max(RecordingImputer.rows_seen) < len(X)
