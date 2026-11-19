"""Export median- and KNN-cleaned datasets for CL2.

Author: Nicole.
Uses Shepard's Preprocessor and CleaningLog classes.

Features are exported in standardized units.
Each method has separate training and test CSV files.
Original CSV row indexes are saved as source_row.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from src.cleaning_log import CleaningLog
from src.data_split import split_data
from src.imputers import IMPUTERS
from src.src.preprocessor import Preprocessor


def export_datasets(input_path, output_dir="data/processed"):
    """Fit on training rows only, verify outputs, and export both methods."""
    data = pd.read_csv(input_path)
    original = data.copy(deep=True)
    X_train, X_test, y_train, y_test = split_data(data)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = CleaningLog()
    exported = []

    for method in ("median", "knn"):
        preprocessor = Preprocessor(
            imputer=IMPUTERS[method](),
            log=log,
        )
        preprocessor.fit(X_train)

        for split, features, target in (
            ("train", X_train, y_train),
            ("test", X_test, y_test),
        ):
            values = np.asarray(preprocessor.transform(features))

            if values.shape != features.shape:
                raise ValueError(
                    f"{method}/{split}: feature shape changed."
                )
            if not np.isfinite(values).all():
                raise ValueError(
                    f"{method}/{split}: missing or infinite values remain."
                )

            cleaned = pd.DataFrame(
                values,
                columns=preprocessor.feature_names,
                index=features.index,
            )
            cleaned["Outcome"] = target

            pd.testing.assert_series_equal(
                cleaned["Outcome"], target
            )

            path = output_dir / f"diabetes_{method}_{split}.csv"
            cleaned.to_csv(path, index=True, index_label="source_row")

            saved = pd.read_csv(path, index_col="source_row")
            expected = cleaned.copy()
            expected.index.name = "source_row"
            pd.testing.assert_frame_equal(
                saved, expected, check_exact=False
            )

            log.add(
                "Consistency",
                "All columns",
                f"Verify {method} {split} export",
                f"Saved and reloaded {path.name}",
                len(cleaned),
                "Preserved row identifiers, columns and Outcome; "
                "verified finite features and CSV round-trip.",
            )
            exported.append(path)
            print(f"Saved {path}: {len(cleaned)} rows")

    pd.testing.assert_frame_equal(data, original)
    log_path = log.save(output_dir / "cleaning_log.csv")
    print(f"Saved cleaning log: {log_path}")
    print("Passed: raw input unchanged; fitted on training rows only.")
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )
    args = parser.parse_args()
    export_datasets(args.input, args.output_dir)


if __name__ == "__main__":
    main()