"""Train and test split for the Pima diabetes data (card W1).

Order of steps used in the whole project (Kanban plan, section 3):

1. Split the data FIRST, stratified 80/20 with random_state 42 (this module).
2. Then, inside one scikit-learn Pipeline: ZeroToNaN, StandardScaler,
   Imputer and finally the model.

Every step learns from the training data only, so the test set never
influences training (no data leakage). The test set is used once, at the end.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Outcome"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    target_column: str = TARGET_COLUMN,
):
    """Split the data into stratified training and test sets.

    Stratified means the share of patients with diabetes is almost the same
    in both sets. The original row numbers are kept so every row can be
    traced back to the CSV file.

    Returns X_train, X_test, y_train, y_test.
    """
    if target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in the data.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    features = df.drop(columns=[target_column])
    target = df[target_column]

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
