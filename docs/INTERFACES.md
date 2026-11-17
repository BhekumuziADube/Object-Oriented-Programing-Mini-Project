# Project Interfaces

## C2 – Validity and Consistency Checks

Module: `src/validity_consistency.py`  
Interface confirmed in PR #40.

`ValidityConsistencyChecks(df)` accepts a pandas DataFrame.

| Method | Returns |
| --- | --- |
| `check_validity()` | One result dictionary for validity. |
| `check_consistency()` | One result dictionary for consistency. |
| `run_all()` | A dictionary with two keys, `"Validity"` and `"Consistency"`, each mapping to its result dictionary. |

Each result contains `dimension` (string), `passed` (boolean),
`rows_checked` (integer), and `issues` (list). `passed` is `True`
when `issues` is empty.

Each issue contains `column`, `rule`, `rows_affected`, and
`example_rows` (up to 10 example row indices).

### Use in the C3 data-quality report

Call `run_all()` once and report Validity and Consistency separately.
Show the pass status, rows checked, and the number of rows affected
by each rule. The same row can fail multiple rules, so adding the
per-rule counts can double-count rows.

## C4 – Imputer Base Class

Module: `src/imputers.py`  
Status: Proposed in PR #41. Confirm against the version merged into `develop`.

`Imputer` is an abstract base class. Each imputation strategy implements
the same two methods so it can be swapped into the same pipeline.

| Method | Behaviour |
| --- | --- |
| `fit(self, X, y=None) -> Imputer` | Learns from the training data and returns `self`. |
| `transform(self, X)` | Uses the fitted imputer to fill missing values and returns a copy of `X`. |

`fit_transform()` is inherited from `TransformerMixin`. During
cross-validation, fit the imputer on the training fold, then use
`transform()` on the validation fold. This keeps validation data
out of the fitting step.

## DataLoader (I1)

- `load()` loads the dataset.
- `summary()` provides a dataset summary.

## CleaningLog

- `add(step, dimension, column, issue, action, rows_affected, justification)`
  records a cleaning decision and its justification.
- `to_frame()` returns the log as a table.
- The log is saved to `docs/cleaning_log.csv`.

## Preprocessor (C6)

Split the data first. Within each split, apply these steps in order:

1. `ZeroToNaN` converts zero placeholders in the designated measurement
   columns to missing values.
2. `StandardScaler` scales the features.
3. An `Imputer` applies the selected missing-value strategy.
4. The model fits on the processed training features.

The preprocessing pipeline contains `ZeroToNaN`, `StandardScaler`,
and the imputer, in that order. Fit the pipeline on training data
and use the fitted pipeline to transform the held-out data.
`ZerosKeptImputer` is the no-op strategy described in the plan.

## Data split (W1, proposed)

`split_data(df, test_size=0.2, random_state=42)` returns
`X_train`, `X_test`, `y_train`, and `y_test`, in that order.
The split is stratified on `Outcome`.

## Prediction models (M1–M3)

`BaseModel` provides `fit`, `predict`, and `predict_proba` for use
in a scikit-learn pipeline. Confirm the exact signatures with Shepard.

- `LogisticRegressionModel`
- `RandomForestModel`, which also provides `feature_importances()`
- `KNNModel`
- `BaselineModel`, which predicts the majority class

## HyperparameterTuner (T1, proposed)

Use `GridSearchCV` over the full pipeline, with `StratifiedKFold(5)`
and `roc_auc` scoring. This fits preprocessing within each training fold.

## Evaluator (E1)

Provide a model-based evaluation entry point, `compare()`, and a ROC plot.
Report accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix.
Use predicted labels for the class-based metrics and positive-class
scores for ROC-AUC and the ROC curve.

## ExperimentRunner (E2)

Coordinates the experiment workflow. Confirm its public methods with
the card owner.
| `predict_proba(X)` | Return class probabilities. The runner passes the probabilities for the positive class to `Evaluator` as `y_score` for ROC-AUC. |
