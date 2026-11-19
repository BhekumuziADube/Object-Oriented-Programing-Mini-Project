"""Hyperparameter tuner tests for Q3. Author: Nicole."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.tuning import (
    HyperparameterTuner,
    LOGISTIC_GRID,
    RANDOM_FOREST_GRID,
)


def make_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def test_empty_grid_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        HyperparameterTuner(make_pipeline(), {})


def test_more_than_twenty_combinations_is_rejected():
    with pytest.raises(ValueError, match="limit"):
        HyperparameterTuner(
            make_pipeline(),
            {"C": list(range(1, 22))},
        )


def test_exactly_twenty_combinations_is_allowed():
    tuner = HyperparameterTuner(
        make_pipeline(),
        {"C": list(range(1, 21))},
    )
    assert tuner.n_combinations == 20


def test_project_grid_counts():
    assert HyperparameterTuner(
        make_pipeline(), LOGISTIC_GRID
    ).n_combinations == 10
    assert HyperparameterTuner(
        make_pipeline(), RANDOM_FOREST_GRID
    ).n_combinations == 12


@pytest.mark.parametrize(
    "method",
    ["best_params", "best_score", "best_estimator", "cv_results"],
)
def test_results_before_fit_are_rejected(method):
    tuner = HyperparameterTuner(make_pipeline(), {"C": [1]})
    with pytest.raises(RuntimeError, match="Call fit"):
        getattr(tuner, method)()


@pytest.mark.parametrize("parameter", ["C", "model__C"])
def test_pipeline_search_and_results(parameter):
    X = np.array([
        [-6, 0], [-5, 1], [-4, 0], [-3, 1], [-2, 0], [-1, 1],
        [1, 0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1],
    ])
    y = np.array([0] * 6 + [1] * 6)

    tuner = HyperparameterTuner(
        make_pipeline(),
        {parameter: [0.1, 1.0]},
        n_splits=3,
        n_jobs=1,
    )
    assert tuner.fit(X, y) is tuner

    results = tuner.cv_results()
    assert len(results) == 2
    assert np.isfinite(results["mean_test_score"]).all()
    assert results["rank_test_score"].is_monotonic_increasing
    assert tuner.best_score() == pytest.approx(
        results["mean_test_score"].max()
    )
    assert tuner.best_params()["model__C"] in [0.1, 1.0]

    best = tuner.best_estimator()
    assert isinstance(best, Pipeline)
    assert best.predict_proba(X).shape == (12, 2)


def test_single_estimator_search():
    X = np.arange(12).reshape(-1, 1)
    y = np.array([0] * 6 + [1] * 6)
    tuner = HyperparameterTuner(
        LogisticRegression(),
        {"C": [0.1, 1.0]},
        n_splits=3,
        n_jobs=1,
    ).fit(X, y)

    assert "C" in tuner.best_params()
    assert np.isfinite(tuner.best_score())