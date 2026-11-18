"""Models: one common interface (BaseModel) and concrete model classes.

Card M1. Author: Shepard.

Teammates add their models in this same file:
    M2 (Bhekumuzi): RandomForestModel
    M3 (Nicole):    KNNModel
    X1 (Bhekumuzi): BaselineModel
Every model follows the same interface, so the tuner, evaluator and
experiment runner can use any of them without knowing which one it is.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted


class BaseModel(ABC):
    """Abstract base class: every model must follow this interface."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._model = self.build_estimator()
        self._is_trained = False

    @abstractmethod
    def build_estimator(self):
        """Return a fresh, unfitted scikit-learn estimator."""

    def train(self, X, y) -> None:
        """Fit the model on training data."""
        self._model.fit(X, y)
        self._is_trained = True

    def predict(self, X) -> np.ndarray:
        """Predict 0/1 labels."""
        self._check_trained()
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Probability of diabetes (class 1) for each row."""
        self._check_trained()
        return self._model.predict_proba(X)[:, 1]

    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError(f"{self.name} has not been trained yet")


class LogisticRegressionModel(BaseModel):
    """Logistic regression: a simple, transparent baseline for prediction."""

    def __init__(self, C: float = 1.0) -> None:
        self._C = C
        super().__init__("Logistic Regression")

    def build_estimator(self):
        return LogisticRegression(C=self._C, max_iter=1000)
