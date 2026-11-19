"""Evaluate fitted binary classifiers for the diabetes project."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class Evaluator:
    """Evaluate Outcome labels 0 and 1, treating 1 as positive."""



    @staticmethod
    def _positive_scores(model, X):
        """Accept class-1 scores or select class 1 from probability columns."""
        probabilities = np.asarray(model.predict_proba(X))

        # Project BaseModel returns one class-1 probability per row.
        if probabilities.ndim == 1:
            return probabilities

        # Scikit-learn classifiers return one column per class.
        classes = np.asarray(getattr(model, "classes_", [0, 1]))
        positive = np.flatnonzero(classes == 1)

        if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
            raise ValueError("Probability columns must match model classes.")
        if len(positive) != 1:
            raise ValueError("The fitted model must expose positive class 1.")

        return probabilities[:, positive[0]]


    def evaluate(self, y_true, y_pred, y_score):
        """Return metrics and a confusion matrix ordered by labels [0, 1]."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_true.size == 0:
            raise ValueError("Evaluation data must contain records.")
        if not np.isin(y_true, [0, 1]).all():
            raise ValueError("True labels must contain only 0 and 1.")
        if not np.isin(y_pred, [0, 1]).all():
            raise ValueError("Predicted labels must contain only 0 and 1.")

        auc = float("nan")
        if len(np.unique(y_true)) == 2:
            auc = float(roc_auc_score(y_true, y_score))

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": auc,
            "confusion_matrix": confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).tolist(),
        }

    def evaluate_model(self, model, X, y):
        """Evaluate an already fitted model or pipeline."""
        predictions = model.predict(X)
        scores = self._positive_scores(model, X)
        return self.evaluate(y, predictions, scores)

    def compare(self, models, X, y):
        """Return one row per name in a dictionary of fitted models."""
        rows = [
            {"model": name, **self.evaluate_model(model, X, y)}
            for name, model in models.items()
        ]
        return pd.DataFrame(rows).set_index("model")

    def plot_roc(self, model, X, y, output_path):
        """Save a ROC plot and return its path."""
        if set(np.asarray(y).tolist()) != {0, 1}:
            raise ValueError("A ROC plot requires both outcome classes.")

        scores = self._positive_scores(model, X)
        false_positive, true_positive, _ = roc_curve(y, scores)
        auc = roc_auc_score(y, scores)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(false_positive, true_positive, label=f"ROC-AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax.set(
            xlabel="False positive rate",
            ylabel="True positive rate",
            title="Receiver operating characteristic",
        )
        ax.legend(loc="lower right")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return path