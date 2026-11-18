"""Hyperparameter tuning for the diabetes models (card T1).

The tuner wraps scikit-learn GridSearchCV around a whole Pipeline (ZeroToNaN,
StandardScaler, Imputer, model). In every cross-validation fold the scaler and
the imputer are fitted on that fold's training part only, so no information
leaks from the validation part. Tuning uses the training data only; the test
set is used once, at the very end.
"""

from __future__ import annotations

from math import prod

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
N_SPLITS = 5
SCORING = "roc_auc"
MAX_COMBINATIONS = 20

# Grid for logistic regression (card T1). The names are the model's own
# settings; the tuner adds the "model__" prefix when the estimator is a Pipeline.
LOGISTIC_GRID = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"],
}


class HyperparameterTuner:
    """Searches a grid of settings with stratified cross-validation.

    estimator: a Pipeline (or a single model) to tune.
    param_grid: dictionary of setting name -> list of values to try. When the
        estimator is a Pipeline, names without "__" are given to the pipeline
        step called model_step (for example "C" becomes "model__C").
    The search is limited to max_combinations (20 in the Kanban plan).
    """

    def __init__(
        self,
        estimator,
        param_grid: dict,
        model_step: str = "model",
        n_splits: int = N_SPLITS,
        scoring: str = SCORING,
        max_combinations: int = MAX_COMBINATIONS,
        random_state: int = RANDOM_STATE,
        n_jobs: int = -1,
    ):
        self._estimator = estimator
        self._model_step = model_step
        self._n_splits = n_splits
        self._scoring = scoring
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._param_grid = self._prepare_grid(param_grid)
        self._combinations = prod(len(v) for v in self._param_grid.values())
        if self._combinations > max_combinations:
            raise ValueError(
                f"The grid has {self._combinations} combinations; "
                f"the limit is {max_combinations}."
            )
        self._search: GridSearchCV | None = None

    def _prepare_grid(self, param_grid: dict) -> dict:
        """Check the grid and add the pipeline step prefix where needed."""
        if not param_grid:
            raise ValueError("param_grid must not be empty.")
        is_pipeline = isinstance(self._estimator, Pipeline)
        grid = {}
        for name, values in param_grid.items():
            if is_pipeline and "__" not in name:
                name = f"{self._model_step}__{name}"
            grid[name] = list(values)
        return grid

    @property
    def n_combinations(self) -> int:
        """Number of setting combinations that will be tried."""
        return self._combinations

    def fit(self, X, y) -> "HyperparameterTuner":
        """Run the search on the training data and keep the best model."""
        cv = StratifiedKFold(
            n_splits=self._n_splits, shuffle=True, random_state=self._random_state
        )
        self._search = GridSearchCV(
            self._estimator,
            self._param_grid,
            scoring=self._scoring,
            cv=cv,
            n_jobs=self._n_jobs,
            refit=True,
        )
        self._search.fit(X, y)
        return self
