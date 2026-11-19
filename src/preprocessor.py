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

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


class Preprocessor:
    """Zeros to missing, scale, impute, all in one fitted pipeline."""

    def __init__(self, imputer, log=None) -> None:
        self._imputer = imputer
        self._log = log  # optional CleaningLog (card CL1)
        self._pipe: Optional[Pipeline] = None
        self._feature_names: List[str] = []

    @property
    def feature_names(self) -> List[str]:
        """Column names seen when fitting (the output keeps this order)."""
        return list(self._feature_names)

    def build_pipeline(self) -> Pipeline:
        """A new, unfitted pipeline (clone gives it its own imputer copy)."""
        return Pipeline(
            [
                ("zero_to_nan", ZeroToNaN()),
                ("scale", StandardScaler()),
                ("impute", clone(self._imputer)),
            ]
        )

    def fit(self, X_train: pd.DataFrame) -> "Preprocessor":
        """Learn scaling and imputation values from the TRAINING rows only."""
        self._pipe = self.build_pipeline()
        self._pipe.fit(X_train)
        self._feature_names = list(X_train.columns)
        self._log_hidden_zeros(X_train)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply what was learned in fit() to any rows (train or test)."""
        if self._pipe is None:
            raise RuntimeError("Call fit() before transform().")
        return self._pipe.transform(X)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit on the training rows, then transform them."""
        return self.fit(X_train).transform(X_train)

    def _log_hidden_zeros(self, X_train: pd.DataFrame) -> None:
        """Write one cleaning-log entry per affected column (if a log exists)."""
        if self._log is None:
            return
        for col in ZeroToNaN.COLUMNS:
            if col not in X_train.columns:
                continue
            n = int((X_train[col] == 0).sum())
            self._log.add(
                "Completeness",
                col,
                f"{n} hidden zeros in the training rows",
                "Converted 0 to missing inside the pipeline",
                n,
                "Zero is impossible for this measurement",
            )
