"""Tests for the majority-class baseline model (card X1)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.models import BaseModel, BaselineModel


@pytest.fixture
def data():
    X = pd.DataFrame({"a": range(10)})
    y = pd.Series([0] * 7 + [1] * 3)
    return X, y


def test_baseline_is_a_base_model():
    assert isinstance(BaselineModel(), BaseModel)


def test_baseline_has_a_name():
    assert BaselineModel().name == "Majority-class Baseline"


def test_predicts_the_majority_class_for_everyone(data):
    X, y = data
    model = BaselineModel()
    model.train(X, y)
    assert list(model.predict(X)) == [0] * 10


def test_probability_is_the_share_of_diabetes_in_training(data):
    X, y = data
    model = BaselineModel()
    model.train(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (10,)
    assert np.allclose(probabilities, 0.3)


def test_roc_auc_is_chance_level(data):
    X, y = data
    model = BaselineModel()
    model.train(X, y)
    assert roc_auc_score(y, model.predict_proba(X)) == pytest.approx(0.5)


def test_predict_before_training_raises(data):
    X, _ = data
    with pytest.raises(RuntimeError, match="not been trained"):
        BaselineModel().predict(X)
