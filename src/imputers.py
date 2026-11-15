"""Imputers for the hidden missing values in the Pima diabetes data (card C4).

An imputer fills in missing values (NaN). All imputers follow the
scikit-learn interface (fit and transform) so they can sit inside a Pipeline
and be swapped for one another without changing the rest of the code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
    """Fills missing values with the median of each column (to be written)."""

    def fit(self, X, y=None) -> "MedianImputer":
        raise NotImplementedError("MedianImputer.fit is not written yet.")

    def transform(self, X):
        raise NotImplementedError("MedianImputer.transform is not written yet.")


# Registry of the available imputers: name -> class.
# Each imputer card adds one line inside the braces.
IMPUTERS: dict[str, type[Imputer]] = {
}
