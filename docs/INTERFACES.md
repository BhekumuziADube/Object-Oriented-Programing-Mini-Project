# Project Interfaces

## C2 – Validity and Consistency Checks

Module: `src/validity_consistency.py`  
Source: open pull request #40.

- `ValidityConsistencyChecks(df)` takes a pandas DataFrame.
- An individual check result contains `dimension`, `passed`, `rows_checked`, and `issues`.
- Each issue contains `column`, `rule`, `rows_affected`, and `example_rows` (up to 10 row indices).
- `run_all()` is listed in the team handover. Confirm its return format before using it in the C3 data-quality report.
