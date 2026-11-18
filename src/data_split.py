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
