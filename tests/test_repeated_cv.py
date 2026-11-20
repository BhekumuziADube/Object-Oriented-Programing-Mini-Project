"""X2 pipeline checks. Author: Nicole."""

import numpy as np
import pandas as pd
import pytest

from src.repeated_cv import build_pipeline


@pytest.mark.parametrize(
    "method,converts_zeros",
    [("median", True), ("knn", True), ("zeros_kept", False)],
)
def test_zero_conversion_matches_method(method, converts_zeros):
    pipeline = build_pipeline(method, "logistic")
    assert ("zero_to_nan" in pipeline.named_steps) == converts_zeros


def test_zeros_kept_preserves_zero():
    pipeline = build_pipeline("zeros_kept", "logistic")
    imputer = pipeline.named_steps["impute"]
    X = pd.DataFrame({"Glucose": [0.0, 100.0]})
    result = imputer.fit_transform(X)
    np.testing.assert_array_equal(np.asarray(result), X.to_numpy())


def test_zeros_kept_rejects_missing_values():
    pipeline = build_pipeline("zeros_kept", "logistic")
    imputer = pipeline.named_steps["impute"]
    with pytest.raises(ValueError, match="ZeroToNaN"):
        imputer.fit_transform([[float("nan")], [100.0]])


@pytest.mark.parametrize("method", ["median", "knn"])
def test_scaler_uses_training_rows_only(method):
    training = pd.DataFrame({
        "Glucose": [80.0, 100.0, 120.0, 140.0],
        "Age": [20.0, 30.0, 40.0, 50.0],
    })
    validation = pd.DataFrame({
        "Glucose": [10000.0],
        "Age": [1000.0],
    })
    pipeline = build_pipeline(method, "logistic")
    pipeline.fit(training, [0, 0, 1, 1])

    scaler = pipeline.named_steps["scale"]
    expected = training.mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected)

    pipeline.predict(validation)
    np.testing.assert_allclose(scaler.mean_, expected)