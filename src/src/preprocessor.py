"""Preprocessor: turns raw patient data into model-ready numbers.

Card C6. Author: Shepard.

The order of the steps is fixed by the team (see docs/INTERFACES.md):
    1. ZeroToNaN   impossible zeros become missing values
    2. Scale       StandardScaler (KNN imputation needs scaled features)
    3. Impute      any imputer: median, KNN, or zeros kept
Everything is fitted on TRAINING rows only. Test rows are only transformed,
so no information from the test set leaks into the model.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace zeros with missing values in columns where zero is impossible."""

    COLUMNS: List[str] = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    def fit(self, X, y=None):
        return self  # nothing to learn

    def transform(self, X):
        X = X.copy()  # never change the caller's data
        cols = [c for c in self.COLUMNS if c in X.columns]
        X[cols] = X[cols].replace(0, np.nan)
        return X
