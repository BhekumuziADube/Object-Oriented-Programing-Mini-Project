"""Hyperparameter tuning for the diabetes models (card T1).

The tuner wraps scikit-learn GridSearchCV around a whole Pipeline (ZeroToNaN,
StandardScaler, Imputer, model). In every cross-validation fold the scaler and
the imputer are fitted on that fold's training part only, so no information
leaks from the validation part. Tuning uses the training data only; the test
set is used once, at the very end.
"""

from __future__ import annotations

from math import prod

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
N_SPLITS = 5
SCORING = "roc_auc"
MAX_COMBINATIONS = 20

# Grid for logistic regression (card T1). The names are the model's own
# settings; the tuner adds the "model__" prefix when the estimator is a Pipeline.
LOGISTIC_GRID = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["lbfgs", "liblinear"],
}
