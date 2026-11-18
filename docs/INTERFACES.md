# Project Interfaces

## I1 – DataLoader

`DataLoader(path)` accepts a CSV path as a string or `Path`.

| Method | Behaviour |
| --- | --- |
| `load()` | Reads and validates the CSV, then returns a copy of the resulting pandas DataFrame. |
| `summary()` | After `load()`, returns a dictionary containing `rows`, `columns`, and `positive_rate`. |

`columns` is a list of column names. `positive_rate` is the proportion
of records with `Outcome = 1`.

## C2 – Validity and Consistency Checks

Module `src/validity_consistency.py`. Interface confirmed in PR #40.

`ValidityConsistencyChecks(df)` accepts a pandas DataFrame.

| Method | Returns |
| --- | --- |
| `check_validity()` | One result dictionary for validity. |
| `check_consistency()` | One result dictionary for consistency. |
| `run_all()` | A dictionary with the keys `"Validity"` and `"Consistency"`, each mapping to its result dictionary. |

Each result contains `dimension`, `passed`, `rows_checked`, and `issues`.
`passed` is `True` when `issues` is empty. Each issue contains `column`,
`rule`, `rows_affected`, and `example_rows` (up to 10 example row indices).

The C3 report shows affected-row counts per rule because the same row
can trigger multiple rules.

## CleaningLog

- `add(step, dimension, column, issue, action, rows_affected, justification)`
  records a cleaning decision and its justification.
- `to_frame()` returns the log as a pandas DataFrame.
- The log is saved to `docs/cleaning_log.csv`.

## C4–C6 – Imputation strategies

Module `src/imputers.py`. `Imputer` defines the shared interface.

| Method | Behaviour |
| --- | --- |
| `fit(X, y=None)` | Learns from the training data and returns `self`. |
| `transform(X)` | Applies the fitted strategy and returns the transformed data. |
| `fit_transform(X, y=None)` | Fits and transforms the data using the method inherited from `TransformerMixin`. |

The plan names `MedianImputer`, `KNNImputerStrategy`, and
`ZerosKeptImputer`. `ZerosKeptImputer` is the no-op imputation strategy.

## W1 – Data split

Proposed interface from Kanban plan v2.

`split_data(df, test_size=0.2, random_state=42)` splits the data
with stratification on `Outcome`. It returns `X_train`, `X_test`,
`y_train`, and `y_test`, in that order.

## C6 – Preprocessor

Split the data first. The preprocessing pipeline applies `ZeroToNaN`,
`StandardScaler`, and an `Imputer`, in that order, before the model.
`ZeroToNaN` converts zero placeholders in designated measurement
columns to missing values.

Fit the pipeline on training data and apply the fitted preprocessing
to the held-out data. During cross-validation, fit each pipeline
on that fold's training portion.

## M1–M3 – Prediction models

Bhekumuzi proposed that `BaseModel` inherit `sklearn.base.BaseEstimator`
for compatibility with scikit-learn cloning and use `fit()` as the
training method. Shepard's confirmation for M1 is pending.

| Proposed method | Behaviour |
| --- | --- |
| `fit(X, y)` | Trains the model on features `X` and labels `y`, then returns `self`. |
| `predict(X)` | Returns a predicted class label for each row. |
| `predict_proba(X)` | Returns class probabilities for each row. |

The planned models are `LogisticRegressionModel`, `RandomForestModel`,
`KNNModel`, and the majority-class `BaselineModel`. `RandomForestModel`
also provides `feature_importances()`.

## T1 – HyperparameterTuner

Proposed approach from Kanban plan v2. Use `GridSearchCV` over the full
pipeline, with `StratifiedKFold(5)` and `roc_auc` scoring.

## E1 – Evaluator

The evaluator accepts fitted models or pipelines that expose
`predict()` and `predict_proba()`.

| Method | Behaviour |
| --- | --- |
| `evaluate_model(model, X, y)` | Gets predictions and positive-class probabilities from the model and evaluates them against `y`. |
| `compare(models, X, y)` | Evaluates models on the same data and returns a pandas DataFrame of their metrics. |

Report accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.
Use predicted labels for the class-based metrics and confusion matrix.
Use positive-class probabilities for ROC-AUC and the ROC plot.

## E2 – ExperimentRunner

Coordinates the experiment workflow. The public method signatures
are pending confirmation from the E2 owner.
