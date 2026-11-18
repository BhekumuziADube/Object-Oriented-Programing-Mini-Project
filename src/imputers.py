"""Imputers for the hidden missing values in the Pima diabetes data (card C4).

An imputer fills in missing values (NaN). All imputers follow the
scikit-learn interface (fit and transform) so they can sit inside a Pipeline
and be swapped for one another without changing the rest of the code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.utils.validation import check_is_fitted


class Imputer(BaseEstimator, TransformerMixin, ABC):
    """Abstract base class for every imputer.

    fit() learns what it needs from the training data only. transform()
    returns a copy of the data with the missing values filled in.
    fit_transform() comes from TransformerMixin.
    """

    @abstractmethod
    def fit(self, X, y=None) -> "Imputer":
        """Learn from the training data and return self."""

    @abstractmethod
    def transform(self, X):
        """Return a copy of X with every missing value filled in."""


class MedianImputer(Imputer):
    """Fills missing values with the median of each column.

    The medians are learned from the training data in fit() and reused in
    transform(), so test data never influences them (no data leakage).
    """

    def fit(self, X, y=None) -> "MedianImputer":
        """Learn the median of every column from the training data."""
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.imputer_.fit(X)
        self.medians_ = self.imputer_.statistics_
        return self

    def transform(self, X):
        """Fill missing values with the medians learned in fit()."""
        check_is_fitted(self, "imputer_")
        filled = self.imputer_.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(filled, columns=X.columns, index=X.index)
        return filled

class KNNImputerStrategy(Imputer):
    """Fill missing values using nearby rows from the training data."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None) -> "KNNImputerStrategy":
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "imputer_")
        filled = self.imputer_.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(filled, index=X.index, columns=X.columns)
        return filled
# Registry of the available imputers: name -> class.
# Each imputer card adds one line inside the brace

 IMPUTERS: dict[str, type[Imputer]] = {
    "knn": KNNImputerStrategy,
    "median": MedianImputer,
}