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

## E1 – Evaluator

Status: Proposed for team review.

`Evaluator.evaluate(y_true, y_pred, y_score)` summarises the
performance of a binary classifier.

- `y_true` contains the observed outcome labels.
- `y_pred` contains the predicted class labels.
- `y_score` contains the model scores for the positive class.

The method returns accuracy, precision, recall, F1, ROC-AUC, and
a confusion matrix. Accuracy, precision, recall, F1, and the
confusion matrix use predicted labels. ROC-AUC uses the model
scores so it can assess performance across decision thresholds.

## M1–M3 – Prediction Models

Status: Proposed for team review. Confirm these signatures with the M1 owner.

The model classes share the `BaseModel` interface so the evaluation
runner can use the same workflow for logistic regression, random
forest, and KNN classification.

| Proposed method | Behaviour |
| --- | --- |
| `fit(X, y)` | Train the model using the training features and outcome labels. Return the fitted model instance. |
| `predict(X)` | Return one predicted class label per row for `Evaluator`'s `y_pred` input. |
| `predict_proba(X)` | Return class probabilities. The runner passes the probabilities for the positive class to `Evaluator` as `y_score` for ROC-AUC. |
