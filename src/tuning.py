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

# Random Forest grid (card T2). Author: Nicole.
# 2 tree counts × 3 depth limits × 2 leaf sizes = 12 combinations.
# The tuner adds the pipeline's model-step prefix automatically.
RANDOM_FOREST_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 3],
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

    def _check_fitted(self) -> None:
        """Stop with a clear message if fit() has not been called."""
        if self._search is None:
            raise RuntimeError("Call fit() before using the results.")

    def best_params(self) -> dict:
        """Return the best settings found."""
        self._check_fitted()
        return dict(self._search.best_params_)

    def best_score(self) -> float:
        """Return the best mean cross-validation score."""
        self._check_fitted()
        return float(self._search.best_score_)

    def best_estimator(self):
        """Return the best pipeline, refitted on all the training data."""
        self._check_fitted()
        return self._search.best_estimator_

    def cv_results(self) -> pd.DataFrame:
        """Return every combination with its mean score, best first."""
        self._check_fitted()
        results = pd.DataFrame(self._search.cv_results_)
        columns = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        return results[columns].sort_values("rank_test_score").reset_index(drop=True)


if __name__ == "__main__":
    # Quick manual check: python src/tuning.py (uses the training part only)
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = pd.read_csv("data/diabetes.csv")
    for column in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        data[column] = data[column].replace(0, np.nan)
    X = data.drop(columns="Outcome")
    y = data["Outcome"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("impute", SimpleImputer(strategy="median")),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    tuner = HyperparameterTuner(pipeline, LOGISTIC_GRID).fit(X_train, y_train)
    print("combinations tried:", tuner.n_combinations)
    print("best settings:", tuner.best_params())
    print("best CV roc_auc:", round(tuner.best_score(), 3))
