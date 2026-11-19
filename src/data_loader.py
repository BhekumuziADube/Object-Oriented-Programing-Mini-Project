"""Data loading for the Pima Indians Diabetes dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class DataLoadError(ValueError):
    """Raised when a dataset cannot be loaded or validated."""


class DataLoader:
    """Load and validate the diabetes dataset."""

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

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._data: pd.DataFrame | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def expected_columns(self) -> list[str]:
        return self.FEATURE_COLUMNS + [self.TARGET_COLUMN]

    def load(self) -> pd.DataFrame:
        if not self._path.exists():
            raise FileNotFoundError(f"Dataset not found: {self._path}")
        if not self._path.is_file():
            raise DataLoadError(f"Dataset path is not a file: {self._path}")

        try:
            df = pd.read_csv(self._path)
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError("Dataset is empty.") from exc

        self._validate(df)
        self._data = df.copy()
        return df.copy()

    def _validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise DataLoadError("Dataset is empty.")

        missing = [c for c in self.expected_columns if c not in df.columns]
        if missing:
            raise DataLoadError(f"Missing expected columns: {missing}")

        target = df[self.TARGET_COLUMN]
        if target.isna().any() or not set(target.unique()).issubset({0, 1}):
            raise DataLoadError("Outcome column must contain only 0 and 1.")

    def summary(self) -> dict:
        if self._data is None:
            raise RuntimeError("Call load() before summary().")

        return {
            "rows": len(self._data),
            "columns": list(self._data.columns),
            "positive_rate": float(self._data[self.TARGET_COLUMN].mean()),
        }


if __name__ == "__main__":
    loader = DataLoader("data/diabetes.csv")
    loader.load()
    print(loader.summary())
