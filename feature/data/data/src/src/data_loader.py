"""Data loading for the Pima Indians Diabetes dataset.

Responsible only for reading the CSV and checking that it has the
expected structure. Cleaning and imputation belong in preprocessing.
"""

from pathlib import Path

import pandas as pd


class DataLoader:
    """Loads and validates the diabetes dataset from a CSV file."""

    FEATURE_COLUMNS = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    TARGET_COLUMN = "Outcome"

    def _init_(self, path: str | Path):
        self._path = Path(path)
        self._data: pd.DataFrame | None = None

    @property
    def path(self) -> Path:
        """Location of the CSV file."""
        return self._path

    @property
    def expected_columns(self) -> list[str]:
        """All columns the file must contain."""
        return self.FEATURE_COLUMNS + [self.TARGET_COLUMN]

    def load(self) -> pd.DataFrame:
        """Read the CSV, validate it, and return a copy of the DataFrame."""
        if not self._path.exists():
            raise FileNotFoundError(f"Dataset not found: {self._path}")

        df = pd.read_csv(self._path)
        self._validate(df)
        self._data = df
        return df.copy()

    def _validate(self, df: pd.DataFrame) -> None:
        """Check for missing columns, an empty file and a non-binary target."""
        missing = [c for c in self.expected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

        if df.empty:
            raise ValueError("Dataset is empty.")

        if not set(df[self.TARGET_COLUMN].unique()).issubset({0, 1}):
            raise ValueError("Outcome column must contain only 0 and 1.")
