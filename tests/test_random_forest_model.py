"""Tests for RandomForestModel (card M2)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models import BaseModel, RandomForestModel


@pytest.fixture
def data():
    """60 patients: feature "signal" follows the outcome, "noise" does not."""
    rng = np.random.default_rng(0)
    y = pd.Series(np.tile([0, 1], 30))
    X = pd.DataFrame(
        {
            "signal": y * 2.0 + rng.normal(scale=0.3, size=60),
            "noise": rng.normal(size=60),
        }
    )
    return X, y


def test_random_forest_is_a_base_model():
    assert isinstance(RandomForestModel(), BaseModel)


def test_model_has_a_name():
    assert RandomForestModel().name == "Random Forest"


def test_estimator_is_a_random_forest_with_seed_42():
    estimator = RandomForestModel().build_estimator()
    assert isinstance(estimator, RandomForestClassifier)
    assert estimator.random_state == 42


def test_settings_reach_the_estimator():
    model = RandomForestModel(n_estimators=7, max_depth=3, min_samples_leaf=2)
    estimator = model.build_estimator()
    assert estimator.n_estimators == 7
    assert estimator.max_depth == 3
    assert estimator.min_samples_leaf == 2


def test_predict_before_training_raises(data):
    X, _ = data
    with pytest.raises(RuntimeError, match="not been trained"):
        RandomForestModel().predict(X)


def test_predictions_are_zero_or_one(data):
    X, y = data
    model = RandomForestModel()
    model.train(X, y)
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert set(predictions) <= {0, 1}


def test_probabilities_are_one_value_per_row_between_0_and_1(data):
    X, y = data
    model = RandomForestModel()
    model.train(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(X),)
    assert probabilities.min() >= 0.0
    assert probabilities.max() <= 1.0


def test_results_are_the_same_every_time(data):
    X, y = data
    first = RandomForestModel()
    second = RandomForestModel()
    first.train(X, y)
    second.train(X, y)
    assert np.allclose(first.predict_proba(X), second.predict_proba(X))


def test_feature_importances_are_sorted_and_add_up_to_one(data):
    X, y = data
    model = RandomForestModel()
    model.train(X, y)
    importances = model.feature_importances()
    assert list(importances.index)[0] == "signal"
    assert importances.is_monotonic_decreasing
    assert importances.sum() == pytest.approx(1.0)


def test_feature_importances_before_training_raise():
    with pytest.raises(RuntimeError):
        RandomForestModel().feature_importances()
