"""Evaluator tests for Q3. Author: Nicole."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from src.evaluator import Evaluator
from src.models import LogisticRegressionModel


class ExampleClassifier:
    # Reversed class order checks selection by class label.
    classes_ = np.array([1, 0])

    def predict(self, X):
        return np.array([0, 1, 0, 1])

    def predict_proba(self, X):
        scores = np.array([0.1, 0.8, 0.4, 0.9])
        return np.column_stack([scores, 1 - scores])


def test_known_metrics():
    result = Evaluator().evaluate(
        [0, 0, 1, 1],
        [0, 1, 0, 1],
        [0.1, 0.8, 0.4, 0.9],
    )
    for metric in ("accuracy", "precision", "recall", "f1"):
        assert result[metric] == pytest.approx(0.5)
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]


def test_reversed_probability_columns():
    result = Evaluator().evaluate_model(
        ExampleClassifier(), np.zeros((4, 2)), [0, 0, 1, 1]
    )
    assert result["roc_auc"] == pytest.approx(0.75)


def test_actual_project_model_is_supported():
    X = np.array([[0], [1], [8], [9]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegressionModel()
    model.train(X, y)

    result = Evaluator().evaluate_model(model, X, y)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["roc_auc"] == pytest.approx(1.0)


def test_single_class_has_undefined_auc():
    result = Evaluator().evaluate([0, 0], [0, 0], [0.1, 0.2])
    assert np.isnan(result["roc_auc"])
    assert result["confusion_matrix"] == [[2, 0], [0, 0]]
    assert result["precision"] == 0.0


@pytest.mark.parametrize(
    "truth,predictions,scores",
    [
        ([], [], []),
        ([0, 2], [0, 1], [0.1, 0.9]),
        ([0, 1], [0, 2], [0.1, 0.9]),
    ],
)
def test_invalid_labels_are_rejected(truth, predictions, scores):
    with pytest.raises(ValueError):
        Evaluator().evaluate(truth, predictions, scores)


def test_comparison_preserves_names_and_scores():
    table = Evaluator().compare(
        {"first": ExampleClassifier(), "second": ExampleClassifier()},
        np.zeros((4, 2)),
        [0, 0, 1, 1],
    )
    assert table.index.tolist() == ["first", "second"]
    np.testing.assert_allclose(table["roc_auc"], [0.75, 0.75])


def test_roc_plot_creates_png(tmp_path):
    path = tmp_path / "figures" / "roc.png"
    returned = Evaluator().plot_roc(
        ExampleClassifier(), np.zeros((4, 2)), [0, 0, 1, 1], path
    )
    assert returned == path
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")