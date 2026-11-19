"""Preprocessing for model-ready Pima diabetes features."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace impossible zero placeholders with missing values."""

    COLUMNS: List[str] = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        cols = [c for c in self.COLUMNS if c in X.columns]
        if cols:
            X[cols] = X[cols].replace(0, np.nan)
        return X


class Preprocessor:
    """Apply hidden-zero conversion, scaling, and imputation."""

    def __init__(self, imputer, log=None) -> None:
        self._imputer = imputer
        self._log = log
        self._pipe: Optional[Pipeline] = None
        self._feature_names: List[str] = []

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("zero_to_nan", ZeroToNaN()),
                ("scale", StandardScaler()),
                ("impute", clone(self._imputer)),
            ]
        )

    def fit(self, X_train: pd.DataFrame) -> "Preprocessor":
        self._pipe = self.build_pipeline()
        self._pipe.fit(X_train)
        self._feature_names = list(X_train.columns)
        self._log_hidden_zeros(X_train)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("Call fit() before transform().")
        return self._pipe.transform(X)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def _log_hidden_zeros(self, X_train: pd.DataFrame) -> None:
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
