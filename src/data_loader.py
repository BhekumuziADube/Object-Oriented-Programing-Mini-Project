"""DataLoader: load and validate the Pima diabetes CSV file.

Card I1. Author: Shepard.

The loader is the first step of the pipeline. It makes sure the file exists,
has the columns we expect and a binary target, so every later step can trust
the data it receives.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import pandas as pd


class DataLoadError(Exception):
    """Raised when the dataset file is missing or is not a valid dataset."""


class DataLoader:
    """Load a diabetes CSV and validate its structure.

    Example:
        loader = DataLoader("data/diabetes.csv")
        df = loader.load()
        print(loader.summary())
    """

    FEATURE_COLUMNS: List[str] = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]
    TARGET: str = "Outcome"

    def __init__(self, path: Union[str, Path]) -> None:
        self._path = Path(path)
        self._data: pd.DataFrame | None = None  # filled by load()

    @property
    def path(self) -> Path:
        """The file this loader reads (read-only)."""
        return self._path

    @property
    def required_columns(self) -> List[str]:
        """All columns the file must contain."""
        return self.FEATURE_COLUMNS + [self.TARGET]

    # ------------------------------------------------------------------
    # public methods
    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Read the CSV, validate it, and return a copy of the DataFrame."""
        self._check_file()
        df = self._read_csv()
        self._validate(df)
        self._data = df
        return df.copy()  # a copy, so callers cannot change our stored data

    def summary(self) -> Dict[str, object]:
        """Basic facts about the loaded data. Call load() first."""
        if self._data is None:
            raise DataLoadError("Call load() before summary().")
        counts = self._data[self.TARGET].value_counts().sort_index()
        return {
            "rows": int(len(self._data)),
            "columns": int(self._data.shape[1]),
            "column_names": list(self._data.columns),
            "class_counts": {int(k): int(v) for k, v in counts.items()},
            "positive_rate": round(float(self._data[self.TARGET].mean()), 3),
            "missing_cells": int(self._data.isna().sum().sum()),
        }

    # ------------------------------------------------------------------
    # private helpers (the leading underscore means "internal use only")
    # ------------------------------------------------------------------
    def _check_file(self) -> None:
        if not self._path.exists():
            raise DataLoadError(f"File not found: {self._path}")
        if not self._path.is_file():
            raise DataLoadError(f"Not a file: {self._path}")
        if self._path.suffix.lower() != ".csv":
            raise DataLoadError(f"Expected a .csv file, got: {self._path.name}")

    def _read_csv(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self._path)
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError(f"The file is empty: {self._path}") from exc
        except pd.errors.ParserError as exc:
            raise DataLoadError(f"Could not parse the CSV: {exc}") from exc

    def _validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise DataLoadError("The dataset has a header but no rows.")

        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise DataLoadError(f"Missing required columns: {missing}")

        non_numeric = [
            c for c in self.required_columns
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric:
            raise DataLoadError(f"Columns must be numeric: {non_numeric}")

        values = set(df[self.TARGET].dropna().unique())
        if not values <= {0, 1}:
            raise DataLoadError(
                f"{self.TARGET} must contain only 0 and 1, found: {sorted(values)}"
            )
