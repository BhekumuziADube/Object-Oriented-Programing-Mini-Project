"""KNN tuning: grid search and comparison with default settings.

Card T3. Author: Shepard.

Question answered: does tuning the KNN model help, compared with the
default settings? Both are scored with the SAME cross-validation folds on
the TRAINING rows only. The test set is never touched here.

Run from the repository root:
    python -m src.knn_tuning
and the result is saved to results/knn_tuning.csv.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.imputers import IMPUTERS
from src.preprocessor import Preprocessor

KNN_GRID: Dict[str, List] = {
    "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],  # 1 = Manhattan distance, 2 = Euclidean distance
}
SCORING = "roc_auc"
SEED = 42


class KNNTuning:
    """Compare default KNN settings with the best settings from a grid."""

    def __init__(
        self,
        imputer,
        grid: Optional[Dict[str, List]] = None,
        folds: int = 5,
        seed: int = SEED,
    ) -> None:
        self._imputer = imputer
        self._grid = dict(grid) if grid is not None else dict(KNN_GRID)
        self._cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        self._search: Optional[GridSearchCV] = None

    @property
    def grid_size(self) -> int:
        """How many setting combinations the grid contains."""
        size = 1
        for values in self._grid.values():
            size *= len(values)
        return size

    def build_pipeline(self) -> Pipeline:
        """Preprocessor steps (zeros to missing, scale, impute) plus KNN."""
        steps = Preprocessor(self._imputer).build_pipeline().steps
        return Pipeline(steps + [("model", KNeighborsClassifier())])

    def default_score(self, X, y) -> np.ndarray:
        """Cross-validated ROC-AUC of KNN with its default settings."""
        return cross_val_score(
            self.build_pipeline(), X, y, cv=self._cv, scoring=SCORING
        )

    def tune(self, X, y) -> GridSearchCV:
        """Grid search over the whole pipeline (imputer fitted inside each fold)."""
        self._search = GridSearchCV(
            self.build_pipeline(),
            self._grid,
            cv=self._cv,
            scoring=SCORING,
            refit=True,
        )
        self._search.fit(X, y)
        return self._search

    @property
    def best_params(self) -> Dict:
        if self._search is None:
            raise RuntimeError("Call tune() or compare() first.")
        return dict(self._search.best_params_)

    def compare(self, X, y) -> pd.DataFrame:
        """Table with one row for default settings and one for tuned settings."""
        default = self.default_score(X, y)
        search = self.tune(X, y)
        best = search.best_index_
        rows = [
            {
                "version": "default",
                "settings": "n_neighbors=5, weights=uniform, p=2",
                "cv_auc_mean": round(float(default.mean()), 4),
                "cv_auc_sd": round(float(default.std()), 4),
            },
            {
                "version": "tuned",
                "settings": ", ".join(
                    f"{k.replace('model__', '')}={v}"
                    for k, v in search.best_params_.items()
                ),
                "cv_auc_mean": round(float(search.best_score_), 4),
                "cv_auc_sd": round(
                    float(search.cv_results_["std_test_score"][best]), 4
                ),
            },
        ]
        return pd.DataFrame(rows)

    def save(self, table: pd.DataFrame, path="results/knn_tuning.csv") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        return path


def _make(entry):
    """IMPUTERS may hold classes or ready-made objects; accept both."""
    return entry() if isinstance(entry, type) else entry


def main() -> None:
    df = pd.read_csv("data/diabetes.csv")
    X = df.drop(columns=["Outcome"])
    X_train, _, y_train, _ = train_test_split(
        X, df["Outcome"], test_size=0.2, stratify=df["Outcome"], random_state=SEED
    )
    tuning = KNNTuning(_make(IMPUTERS["median"]))
    table = tuning.compare(X_train, y_train)
    print(table.to_string(index=False))
    gain = table.loc[1, "cv_auc_mean"] - table.loc[0, "cv_auc_mean"]
    print(f"Tuning changed CV ROC-AUC by {gain:+.4f}")
    print("saved", tuning.save(table))


if __name__ == "__main__":
    main()
