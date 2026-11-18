"""Tests for BaseModel and LogisticRegressionModel (card M1). Author: Shepard."""
import numpy as np
import pytest

from src.models import BaseModel, LogisticRegressionModel


def _toy_data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_base_model_cannot_be_created_directly():
    with pytest.raises(TypeError):
        BaseModel("x")


def test_predict_before_train_raises():
    with pytest.raises(RuntimeError):
        LogisticRegressionModel().predict(np.zeros((1, 3)))


def test_probabilities_are_between_0_and_1():
    X, y = _toy_data()
    model = LogisticRegressionModel()
    model.train(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (60,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predictions_are_0_or_1():
    X, y = _toy_data()
    model = LogisticRegressionModel()
    model.train(X, y)
    assert set(model.predict(X)) <= {0, 1}
