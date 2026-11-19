"""Imputation charts: correlation heatmap and before/after imputation.

Card An2. Author: Shepard.

Run from the repository root:
    python -m src.imputation_charts

Creates three files in docs/figures/:
    correlation.png   correlation of every column with every other column
    before_after.png  Insulin and SkinThickness before and after imputation
    eda_notes.md      the numbers behind the charts, for the report

The charts use scikit-learn imputers directly, so this card does not
depend on the teammates' imputer classes. The order matches the pipeline:
zeros to missing, scale, impute, then back to the original units.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.impute import KNNImputer, SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HIDDEN_ZERO_COLUMNS: List[str] = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]
TARGET = "Outcome"
CHART_COLUMNS: List[str] = ["Insulin", "SkinThickness"]


class EdaFigures:
    """Builds the exploratory charts and the notes that explain them."""

    def __init__(self, df: pd.DataFrame, out_dir="docs/figures") -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self._df = df.copy()  # never change the caller's data
        self._out = Path(out_dir)

    # ----- data steps -------------------------------------------------
    def zeros_to_missing(self) -> pd.DataFrame:
        """Copy of the data with impossible zeros turned into NaN."""
        data = self._df.copy()
        cols = [c for c in HIDDEN_ZERO_COLUMNS if c in data.columns]
        data[cols] = data[cols].replace(0, np.nan)
        return data

    def features(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(columns=[TARGET, "PatientID"], errors="ignore")

    def impute(self, method: str) -> pd.DataFrame:
        """Fill the gaps: 'median' or 'knn' (scaled first, as in the pipeline)."""
        X = self.features(self.zeros_to_missing())
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X)  # StandardScaler ignores NaN
        if method == "median":
            imputer = SimpleImputer(strategy="median")
        elif method == "knn":
            imputer = KNNImputer(n_neighbors=5)
        else:
            raise ValueError("method must be 'median' or 'knn'")
        filled = scaler.inverse_transform(imputer.fit_transform(scaled))
        return pd.DataFrame(filled, columns=X.columns, index=X.index)

    # ----- numbers ----------------------------------------------------
    def correlations_with_outcome(self) -> pd.Series:
        """Correlation of each feature with Outcome, strongest first."""
        data = self.zeros_to_missing()
        corr = data.drop(columns=["PatientID"], errors="ignore").corr()
        return corr[TARGET].drop(TARGET).sort_values(ascending=False)

    def spread_table(self) -> pd.DataFrame:
        """Mean and standard deviation per column before/after imputation."""
        before = self.features(self.zeros_to_missing())
        tables: Dict[str, pd.DataFrame] = {
            "observed only": before,
            "median": self.impute("median"),
            "knn": self.impute("knn"),
        }
        rows = []
        for col in CHART_COLUMNS:
            for name, frame in tables.items():
                rows.append(
                    {
                        "column": col,
                        "version": name,
                        "mean": round(float(frame[col].mean()), 1),
                        "sd": round(float(frame[col].std()), 1),
                    }
                )
        return pd.DataFrame(rows)

    # ----- charts -----------------------------------------------------
    def plot_correlation(self) -> Path:
        """Heatmap of correlations (hidden zeros treated as missing)."""
        data = self.zeros_to_missing().drop(columns=["PatientID"], errors="ignore")
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(8, 6.5))
        image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr)), corr.columns)
        for i in range(len(corr)):
            for j in range(len(corr)):
                value = corr.iloc[i, j]
                colour = "white" if abs(value) > 0.6 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=8, color=colour)
        fig.colorbar(image, ax=ax, label="Correlation")
        ax.set_title("Correlation between columns (zeros treated as missing)")
        return self._save(fig, "correlation.png")

    def plot_before_after(self) -> Path:
        """Insulin and SkinThickness: observed values, median fill, KNN fill."""
        before = self.features(self.zeros_to_missing())
        median = self.impute("median")
        knn = self.impute("knn")
        fig, axes = plt.subplots(len(CHART_COLUMNS), 3, figsize=(12, 6.5))
        for row, col in enumerate(CHART_COLUMNS):
            observed = before[col].dropna()
            bins = np.linspace(observed.min(), observed.max(), 31)
            panels = [
                ("Observed only", observed),
                ("After median", median[col]),
                ("After KNN (k=5)", knn[col]),
            ]
            for ax, (title, values) in zip(axes[row], panels):
                ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white")
                ax.set_title(f"{col}: {title}", fontsize=10)
                ax.set_xlabel(col)
                ax.set_ylabel("Patients")
        fig.suptitle("Effect of imputation on the two most affected columns")
        fig.tight_layout()
        return self._save(fig, "before_after.png")

    def _save(self, fig, name: str) -> Path:
        self._out.mkdir(parents=True, exist_ok=True)
        path = self._out / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # ----- notes ------------------------------------------------------
    def write_notes(self) -> Path:
        """Write the numbers behind the charts to docs/figures/eda_notes.md."""
        corr = self.correlations_with_outcome()
        spread = self.spread_table()
        lines = ["# Exploratory analysis notes (card An2)", ""]
        lines.append("## Correlation with Outcome (zeros treated as missing)")
        lines.append("")
        lines.append("| Feature | Correlation |")
        lines.append("|---|---|")
        for name, value in corr.items():
            lines.append(f"| {name} | {value:.2f} |")
        lines += ["", "## Spread before and after imputation", ""]
        lines.append("| Column | Version | Mean | SD |")
        lines.append("|---|---|---|---|")
        for r in spread.itertuples():
            lines.append(f"| {r.column} | {r.version} | {r.mean} | {r.sd} |")
        self._out.mkdir(parents=True, exist_ok=True)
        path = self._out / "eda_notes.md"
        path.write_text("\n".join(lines) + "\n")
        return path

    def run_all(self) -> List[Path]:
        return [self.plot_correlation(), self.plot_before_after(),
                self.write_notes()]


def main() -> None:
    df = pd.read_csv("data/diabetes.csv")
    for path in EdaFigures(df).run_all():
        print("wrote", path)


if __name__ == "__main__":
    main()
