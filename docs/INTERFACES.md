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
