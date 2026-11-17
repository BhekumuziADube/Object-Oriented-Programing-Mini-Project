"""Validity and consistency checks for the Pima diabetes data (card C2).

Validity asks: is every value allowed (right type, inside the ranges in
docs/data_dictionary.md, Outcome only 0 or 1)?
Consistency asks: do the values agree with each other (right data types,
whole numbers where expected, Pregnancies not larger than Age minus 10)?

Hidden zeros (0 meaning "missing") are reported by the completeness check,
so they are skipped here to avoid counting the same problem twice.
"""

from __future__ import annotations

import pandas as pd

HIDDEN_ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Hard limits copied from docs/data_dictionary.md
VALID_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (40, 400),
    "BloodPressure": (20, 150),
    "SkinThickness": (5, 100),
    "Insulin": (10, 900),
    "BMI": (10, 80),
    "DiabetesPedigreeFunction": (0.05, 2.5),
    "Age": (21, 100),
}

WHOLE_NUMBER_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "Age",
]

ALL_COLUMNS = list(VALID_RANGES) + ["Outcome"]


class ValidityConsistencyChecks:
    """Runs the validity and consistency checks on a diabetes DataFrame.

    Every check returns a dictionary with the keys ``dimension``, ``passed``,
    ``rows_checked`` and ``issues``. Each issue lists the ``column``, the
    ``rule`` that was broken, ``rows_affected`` and up to 10 ``example_rows``.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df.copy()

    def _issue(self, column: str, rule: str, mask: pd.Series) -> dict:
        """Build one issue entry from a True/False mask of the bad rows."""
        bad_rows = self._df.index[mask]
        return {
            "column": column,
            "rule": rule,
            "rows_affected": int(mask.sum()),
            "example_rows": [int(i) for i in bad_rows[:10]],
        }

    def _missing_column_issue(self, column: str) -> dict:
        """Issue entry used when an expected column is not in the data."""
        return {
            "column": column,
            "rule": "column is missing from the data",
            "rows_affected": len(self._df),
            "example_rows": [],
        }

    def _result(self, dimension: str, issues: list[dict]) -> dict:
        """Wrap a list of issues in the standard result format."""
        return {
            "dimension": dimension,
            "passed": len(issues) == 0,
            "rows_checked": len(self._df),
            "issues": issues,
        }

    def check_consistency(self) -> dict:
        """Check data types, whole numbers and the Pregnancies and Age rule."""
        issues = []
        all_rows = pd.Series(True, index=self._df.index)

        for column in ALL_COLUMNS:
            if column not in self._df.columns:
                issues.append(self._missing_column_issue(column))
            elif not pd.api.types.is_numeric_dtype(self._df[column]):
                issues.append(self._issue(column, "column is not numeric", all_rows))

        for column in WHOLE_NUMBER_COLUMNS:
            if column not in self._df.columns:
                continue
            values = pd.to_numeric(self._df[column], errors="coerce")
            fractional = values.notna() & (values % 1 != 0)
            if fractional.any():
                issues.append(
                    self._issue(column, "must be a whole number", fractional)
                )

        if {"Pregnancies", "Age"} <= set(self._df.columns):
            pregnancies = pd.to_numeric(self._df["Pregnancies"], errors="coerce")
            age = pd.to_numeric(self._df["Age"], errors="coerce")
            too_many = pregnancies > (age - 10)
            if too_many.any():
                issues.append(
                    self._issue(
                        "Pregnancies, Age",
                        "Pregnancies must be at most Age - 10",
                        too_many,
                    )
                )

        return self._result("Consistency", issues)

    def run_all(self) -> dict:
        """Run both checks and return the results keyed by dimension."""
        return {
            "Validity": self.check_validity(),
            "Consistency": self.check_consistency(),
        }


if __name__ == "__main__":
    # Quick manual check: python src/validity_consistency.py
    data = pd.read_csv("data/diabetes.csv")
    for name, result in ValidityConsistencyChecks(data).run_all().items():
        print(name, "passed:", result["passed"], "issues:", len(result["issues"]))
