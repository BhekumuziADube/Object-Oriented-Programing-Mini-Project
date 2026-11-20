"""KNN model tests for Q3. Author: Nicole."""

import numpy as np
import pytest

from src.evaluator import Evaluator
from src.models import BaseModel, KNNModel


@pytest.fixture
def sample():
    X = np.array([[0, 0], [0, 1], [9, 9], [9, 10]])
    y = np.array([0, 0, 1, 1])
    test = np.array([[0.1, 0.1], [9.1, 9.1]])
    return X, y, test


def test_knn_uses_base_model():
    assert isinstance(KNNModel(), BaseModel)


@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_untrained_prediction_is_rejected(method):
    with pytest.raises(RuntimeError):
        getattr(KNNModel(), method)([[0, 0]])


def test_knn_predictions_and_probabilities(sample):
    X, y, test = sample
    model = KNNModel(n_neighbors=1)
    model.train(X, y)

    np.testing.assert_array_equal(model.predict(test), [0, 1])
    scores = model.predict_proba(test)
    assert scores.shape == (2,)
    np.testing.assert_allclose(scores, [0.0, 1.0])


def test_settings_reach_estimator():
    model = KNNModel(n_neighbors=3, weights="distance", p=1)
    settings = model.build_estimator().get_params()
    assert settings["n_neighbors"] == 3
    assert settings["weights"] == "distance"
    assert settings["p"] == 1


def test_knn_preserves_inputs(sample):
    X, y, test = sample
    originals = [value.copy() for value in (X, y, test)]

    model = KNNModel(n_neighbors=1)
    model.train(X, y)
    model.predict(test)
    model.predict_proba(test)

    for actual, original in zip((X, y, test), originals):
        np.testing.assert_array_equal(actual, original)


def test_knn_integrates_with_evaluator(sample):
    X, y, test = sample
    model = KNNModel(n_neighbors=1)
    model.train(X, y)

    result = Evaluator().evaluate_model(model, test, [0, 1])
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["confusion_matrix"] == [[1, 0], [0, 1]]