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
