"""DataQualityChecker: checks a dataset against the six data-quality dimensions.

Card C1 (completeness and uniqueness). Author: Shepard.

Teammates add their checks to this same class:
    C2 (Bhekumuzi): check_validity, check_consistency
    C3 (Nicole):    check_accuracy, timeliness note, report()

Every check returns a list of "findings". A finding is a plain dictionary with
the same keys, so the report and the cleaning log can treat all checks alike.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

Finding = Dict[str, object]


class DataQualityChecker:
    """Run data-quality checks on a DataFrame without changing it."""

    # In this dataset a zero in these columns is impossible for a living
    # patient, so it really means "not recorded" (a hidden missing value).
    HIDDEN_ZERO_COLUMNS: List[str] = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]
    TARGET: str = "Outcome"

    def __init__(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataQualityChecker needs a pandas DataFrame.")
        if df.empty:
            raise ValueError("Cannot check an empty DataFrame.")
        self._df = df.copy()  # work on a copy so the caller's data is safe

    @staticmethod
    def _finding(
        dimension: str,
        check: str,
        column: str,
        issues: int,
        total: int,
        note: str = "",
    ) -> Finding:
        """Build one finding: how many of `total` values/rows have the issue."""
        percent = round(100 * issues / total, 1) if total else 0.0
        return {
            "dimension": dimension,
            "check": check,
            "column": column,
            "issues": int(issues),
            "of": int(total),
            "percent": percent,
            "note": note,
        }

    def check_completeness(self) -> List[Finding]:
        """Find blank cells and hidden missing values (impossible zeros)."""
        total = len(self._df)
        findings: List[Finding] = []

        # 1) blank cells (NaN) in every column
        for col in self._df.columns:
            blanks = int(self._df[col].isna().sum())
            findings.append(
                self._finding("Completeness", "Blank (NaN) cells", col, blanks, total)
            )

        # 2) zeros that really mean "not recorded"
        present = [c for c in self.HIDDEN_ZERO_COLUMNS if c in self._df.columns]
        for col in present:
            zeros = int((self._df[col] == 0).sum())
            findings.append(
                self._finding(
                    "Completeness",
                    "Zero recorded where zero is impossible",
                    col,
                    zeros,
                    total,
                    "Hidden missing value",
                )
            )

        # 3) how many patients are affected by at least one hidden zero
        if present:
            rows_hit = int((self._df[present] == 0).any(axis=1).sum())
            findings.append(
                self._finding(
                    "Completeness",
                    "Rows with at least one hidden zero",
                    "(any of: " + ", ".join(present) + ")",
                    rows_hit,
                    total,
                    "Dropping these rows would lose a large share of the data",
                )
            )
        return findings
