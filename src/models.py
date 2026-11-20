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
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


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


class BaselineModel(BaseModel):
    """Majority-class baseline: always guesses the most common outcome (card X1).

    Any real model has to beat this. Its probabilities are the class shares in
    the training data, the same for every patient, so its ROC-AUC is 0.5.
    """

    def __init__(self) -> None:
        super().__init__("Majority-class Baseline")

    def build_estimator(self):
        return DummyClassifier(strategy="prior")
      
      
class KNNModel(BaseModel):
    """K-nearest neighbours classifier for card M3."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        p: int = 2,
    ) -> None:
        self._n_neighbors = n_neighbors
        self._weights = weights
        self._p = p
        super().__init__("KNN")

    def build_estimator(self):
        return KNeighborsClassifier(
            n_neighbors=self._n_neighbors,
            weights=self._weights,
            p=self._p,
        )
      
class RandomForestModel(BaseModel):
    """Random forest: many decision trees that vote together (card M2)."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_leaf = min_samples_leaf
        self._random_state = random_state
        super().__init__("Random Forest")

    def build_estimator(self):
        return RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            min_samples_leaf=self._min_samples_leaf,
            random_state=self._random_state,
        )

    def feature_importances(self) -> pd.Series:
        """How much each feature matters, largest first (after training)."""
        self._check_trained()
        scores = self._model.feature_importances_
        names = getattr(self._model, "feature_names_in_", None)
        if names is None:
            names = [f"feature_{i}" for i in range(len(scores))]
        return pd.Series(scores, index=names).sort_values(ascending=False)


class BaselineModel(BaseModel):
    """Always predict the most frequent training outcome."""

    def __init__(self) -> None:
        super().__init__("Baseline")

    def build_estimator(self):
        return DummyClassifier(strategy="most_frequent")


MODELS = {
    "Logistic Regression": LogisticRegressionModel,
    "Random Forest": RandomForestModel,
    "KNN": KNNModel,
    "Baseline": BaselineModel,
}
