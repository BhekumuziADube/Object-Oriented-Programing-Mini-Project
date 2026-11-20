"""X2: repeated cross-validation comparison. Author: Nicole."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_split import split_data
from src.evaluator import Evaluator
from src.imputers import IMPUTERS
from src.models import (
    BaselineModel,
    KNNModel,
    LogisticRegressionModel,
    RandomForestModel,
)
from src.preprocessor import ZeroToNaN


MODELS = {
    "logistic": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "knn": KNNModel,
    "baseline": BaselineModel,
}
METHODS = ("median", "knn", "zeros_kept")
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def build_pipeline(method, model_name):
    """Create fresh preprocessing and estimator for each fold."""
    imputer = IMPUTERS[method]()
    steps = []
    if not getattr(imputer, "keeps_zeros", False):
        steps.append(("zero_to_nan", ZeroToNaN()))
    steps.extend([
        ("scale", StandardScaler()),
        ("impute", imputer),
        ("model", MODELS[model_name]().build_estimator()),
    ])
    return Pipeline(steps)


def run_comparison(input_path, output_dir, folds=5, repeats=3, seed=42):
    data = pd.read_csv(input_path)
    original = data.copy(deep=True)
    X, X_test, y, y_test = split_data(data, random_state=seed)

    if folds < 2 or repeats < 1:
        raise ValueError("Use at least two folds and one repeat.")
    if y.value_counts().min() < folds:
        raise ValueError("Not enough training rows per class for these folds.")

    cv = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=repeats,
        random_state=seed,
    )
    splits = list(cv.split(X, y))
    evaluator = Evaluator()
    rows = []
    memberships = []

    for split_id, (training, validation) in enumerate(splits, 1):
        repeat = (split_id - 1) // folds + 1
        fold = (split_id - 1) % folds + 1
        assert set(training).isdisjoint(validation)

        memberships.append({
            "split_id": split_id,
            "repeat": repeat,
            "fold": fold,
            "training_rows": X.index[training].tolist(),
            "validation_rows": X.index[validation].tolist(),
        })

        for method in METHODS:
            for model_name in MODELS:
                pipeline = build_pipeline(method, model_name)
                pipeline.fit(X.iloc[training], y.iloc[training])

                validation_X = X.iloc[validation]
                predictions = pipeline.predict(validation_X)
                probabilities = pipeline.predict_proba(validation_X)
                positive = list(pipeline.classes_).index(1)
                scores = probabilities[:, positive]

                metrics = evaluator.evaluate(
                    y.iloc[validation], predictions, scores
                )
                matrix = np.asarray(metrics.pop("confusion_matrix"))
                if not np.isfinite(
                    [metrics[name] for name in METRICS]
                ).all():
                    raise ValueError("A fold produced a non-finite metric.")

                rows.append({
                    "method": method,
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "split_id": split_id,
                    "n_train": len(training),
                    "n_validation": len(validation),
                    **metrics,
                    "tn": int(matrix[0, 0]),
                    "fp": int(matrix[0, 1]),
                    "fn": int(matrix[1, 0]),
                    "tp": int(matrix[1, 1]),
                })

        print(
            f"Completed split {split_id}/{len(splits)}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    expected = len(METHODS) * len(MODELS) * folds * repeats
    assert len(results) == expected
    assert not results.duplicated(
        ["method", "model", "split_id"]
    ).any()
    pd.testing.assert_frame_equal(data, original)

    summary = results.groupby(["method", "model"])[
        list(METRICS)
    ].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary.columns
    ]
    summary = summary.reset_index().sort_values(
        "roc_auc_mean", ascending=False
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "fold_scores.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)

    metadata = {
        "folds": folds,
        "repeats": repeats,
        "seed": seed,
        "settings_status": "Proposed X2 settings for team review",
        "model_settings": "Defaults; no tuning in this comparison",
        "training_rows": X.index.tolist(),
        "held_out_rows": X_test.index.tolist(),
        "held_out_evaluated": False,
        "fold_memberships": memberships,
        "interpretation": (
            "Standard deviations describe variation across correlated "
            "fold scores; they are not confidence intervals."
        ),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Saved {len(results)} fold scores to {output}")
    print(f"Held-out rows remain unused: {len(y_test)}")
    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/diabetes.csv")
    parser.add_argument("--output-dir", default="results/x2")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_comparison(
        args.input,
        args.output_dir,
        args.folds,
        args.repeats,
        args.seed,
    )