"""ExperimentRunner: every model with every imputer, tuned, tested once.

Card E2. Author: Shepard.

What one run does
    1. Check the data file has not changed (if the checksum helper exists).
    2. Split once: stratified 80/20, fixed seed.
    3. For every model and every imputer: tune the settings with grid search
       and cross-validation on the TRAINING rows only (the imputer is fitted
       inside each fold, so nothing leaks).
    4. Score each tuned pipeline ONCE on the test rows.
    5. Mark the combination with the best cross-validation score. The choice
       never looks at the test score.
    6. Save results/final_test.csv and a metadata file next to it.

Run from the repository root:
    python -m src.experiment_runner            full run
    python -m src.experiment_runner --quick    small grids, for a fast check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.imputers import IMPUTERS
from src.models import MODELS
from src.preprocessor import Preprocessor

TARGET = "Outcome"
SEEDS = {"split": 42, "cross_validation": 42, "random_forest": 42}

# Settings to try, keyed by the scikit-learn estimator behind each model.
FULL_GRIDS: Dict[str, Dict[str, list]] = {
    "LogisticRegression": {"model__C": [0.01, 0.1, 1, 10, 100]},
    "RandomForestClassifier": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 3, 5],
    },
    "KNeighborsClassifier": {
        "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
}
QUICK_GRIDS: Dict[str, Dict[str, list]] = {
    "LogisticRegression": {"model__C": [0.1, 1]},
    "RandomForestClassifier": {"model__n_estimators": [50], "model__max_depth": [5]},
    "KNeighborsClassifier": {"model__n_neighbors": [5, 11]},
}
RESULT_COLUMNS = [
    "model", "imputer", "is_baseline", "best_params", "cv_auc",
    "accuracy", "precision", "recall", "f1", "roc_auc", "chosen_by_cv",
]


def _make(entry):
    """The registries may hold classes or ready-made objects; accept both."""
    return entry() if isinstance(entry, type) else entry


class ExperimentRunner:
    """Runs the whole comparison and saves the final test results."""

    def __init__(
        self,
        data_path="data/diabetes.csv",
        out_dir="results",
        quick: bool = False,
        folds: int = 5,
        models: Optional[List[str]] = None,
        imputers: Optional[List[str]] = None,
    ) -> None:
        self._data_path = Path(data_path)
        self._out_dir = Path(out_dir)
        self._quick = quick
        self._grids = QUICK_GRIDS if quick else FULL_GRIDS
        self._cv = StratifiedKFold(
            n_splits=2 if quick else folds,
            shuffle=True,
            random_state=SEEDS["cross_validation"],
        )
        self._model_names = list(models) if models else list(MODELS)
        self._imputer_names = list(imputers) if imputers else list(IMPUTERS)
        self._results: Optional[pd.DataFrame] = None

    # ----- step 1 and 2: data -----------------------------------------
    def check_data(self) -> None:
        """Stop early if the data file does not match its recorded checksum."""
        try:
            from src.reproducibility import verify_data
        except ImportError:
            print("note: src/reproducibility.py not found, checksum skipped")
            return
        checksum = self._data_path.with_name(self._data_path.name + ".sha256")
        if checksum.exists():
            verify_data(self._data_path)
        else:
            print("note: no .sha256 file next to the data, checksum skipped")

    def load_split(self):
        """Stratified 80/20 split with the project seed."""
        df = pd.read_csv(self._data_path)
        X = df.drop(columns=[TARGET])
        return train_test_split(
            X,
            df[TARGET],
            test_size=0.2,
            stratify=df[TARGET],
            random_state=SEEDS["split"],
        )

    # ----- step 3 and 4: one combination ------------------------------
    def build_pipeline(self, estimator, imputer) -> Pipeline:
        """Preprocessor steps (zeros to missing, scale, impute) plus the model."""
        if "random_state" in estimator.get_params():
            estimator.set_params(random_state=SEEDS["random_forest"])
        steps = Preprocessor(imputer).build_pipeline().steps
        return Pipeline(steps + [("model", estimator)])

    def run_combination(self, model_name, imputer_name, X_train, y_train,
                        X_test, y_test) -> Dict[str, object]:
        """Tune on the training rows, then score once on the test rows."""
        estimator = _make(MODELS[model_name]).build_estimator()
        kind = type(estimator).__name__
        pipe = self.build_pipeline(estimator, _make(IMPUTERS[imputer_name]))
        grid = self._grids.get(kind)
        if grid:
            search = GridSearchCV(
                pipe, grid, cv=self._cv, scoring="roc_auc", n_jobs=-1, refit=True
            )
            search.fit(X_train, y_train)
            best, cv_auc = search.best_estimator_, float(search.best_score_)
            params = json.dumps(
                {k.replace("model__", ""): v for k, v in search.best_params_.items()}
            )
        else:  # the baseline has nothing to tune
            cv_auc = float(np.mean(cross_val_score(
                pipe, X_train, y_train, cv=self._cv, scoring="roc_auc")))
            best = pipe.fit(X_train, y_train)
            params = "{}"
        predicted = best.predict(X_test)
        scores = best.predict_proba(X_test)[:, 1]
        return {
            "model": model_name,
            "imputer": imputer_name,
            "is_baseline": kind == "DummyClassifier",
            "best_params": params,
            "cv_auc": round(cv_auc, 4),
            "accuracy": round(accuracy_score(y_test, predicted), 4),
            "precision": round(precision_score(y_test, predicted, zero_division=0), 4),
            "recall": round(recall_score(y_test, predicted), 4),
            "f1": round(f1_score(y_test, predicted), 4),
            "roc_auc": round(roc_auc_score(y_test, scores), 4),
            "chosen_by_cv": False,
        }

    # ----- step 5: all combinations -----------------------------------
    def run(self) -> pd.DataFrame:
        """Every model with every imputer; marks the best by CV score."""
        self.check_data()
        X_train, X_test, y_train, y_test = self.load_split()
        rows = []
        for model_name in self._model_names:
            for imputer_name in self._imputer_names:
                print(f"running {model_name} + {imputer_name} ...", flush=True)
                rows.append(self.run_combination(
                    model_name, imputer_name, X_train, y_train, X_test, y_test))
        table = pd.DataFrame(rows, columns=RESULT_COLUMNS)
        contenders = table[~table["is_baseline"]]
        if not contenders.empty:
            table.loc[contenders["cv_auc"].idxmax(), "chosen_by_cv"] = True
        self._results = table
        return table

    # ----- step 6: save -----------------------------------------------
    def save(self, table: Optional[pd.DataFrame] = None) -> Path:
        """Write final_test.csv (and a metadata file if the helper exists)."""
        table = self._results if table is None else table
        if table is None:
            raise RuntimeError("Call run() first.")
        self._out_dir.mkdir(parents=True, exist_ok=True)
        path = self._out_dir / "final_test.csv"
        table.to_csv(path, index=False)
        try:
            from src.reproducibility import write_run_metadata
        except ImportError:
            return path
        write_run_metadata(
            path,
            self._data_path,
            seeds=SEEDS,
            params={"quick": self._quick, "folds": self._cv.get_n_splits(),
                    "models": self._model_names, "imputers": self._imputer_names},
        )
        return path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full experiment.")
    parser.add_argument("--data", default="data/diabetes.csv")
    parser.add_argument("--out", default="results")
    parser.add_argument("--quick", action="store_true", help="small grids")
    args = parser.parse_args(argv)
    runner = ExperimentRunner(args.data, args.out, quick=args.quick)
    table = runner.run()
    print(table.drop(columns=["best_params"]).to_string(index=False))
    print("saved", runner.save(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
