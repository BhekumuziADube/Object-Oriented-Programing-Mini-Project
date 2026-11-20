"""CSV loading and validation for the Pima diabetes dataset."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


class DataLoadError(ValueError):
    """Raised when an input file cannot be loaded as the project dataset."""


class DataLoader:
    """Load and validate a diabetes CSV, retaining only a private copy."""

    REQUIRED_COLUMNS = (
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        if not self.path.is_file():
            if self.path.name == "diabetes.csv":
                project_file = Path.cwd() / "data" / self.path.name
                if project_file.is_file():
                    self.path = project_file
                else:
                    raise FileNotFoundError(f"Data file not found: {self.path}")
            else:
                raise FileNotFoundError(f"Data file not found: {self.path}")
        try:
            df = pd.read_csv(self.path)
        except pd.errors.EmptyDataError as exc:
            raise DataLoadError("The data file is empty.") from exc
        except pd.errors.ParserError as exc:
            raise DataLoadError("The data file is not valid CSV.") from exc

        if df.empty:
            raise DataLoadError("The data file contains no data rows.")

        missing = [column for column in self.REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise DataLoadError(
                "Missing required columns: " + ", ".join(missing)
            )

        if not set(df["Outcome"].dropna().unique()).issubset({0, 1}):
            raise DataLoadError("Outcome must contain only 0 and 1.")
        if df["Outcome"].isna().any():
            raise DataLoadError("Outcome contains missing values.")

        self._df = df.copy()
        return self._df.copy()

    def summary(self) -> dict[str, object]:
        if self._df is None:
            raise RuntimeError("Data has not been loaded.")
        return {
            "rows": len(self._df),
            "columns": len(self._df.columns),
            "positive_rate": float(self._df["Outcome"].mean()),
        }
