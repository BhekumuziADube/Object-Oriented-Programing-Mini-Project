"""Tests for ExperimentRunner (card E2). Author: Shepard.

Tests use the quick settings and a few combinations, so they stay fast.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.experiment_runner import RESULT_COLUMNS, ExperimentRunner
from src.imputers import IMPUTERS
from src.models import MODELS

DATA = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"
FIRST_IMPUTER = list(IMPUTERS)[0]


def _runner(tmp_path, models=None, imputers=None):
    return ExperimentRunner(
        DATA, tmp_path / "results", quick=True,
        models=models, imputers=imputers or [FIRST_IMPUTER],
    )


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    return _runner(out).run()


def test_split_is_80_20_and_stratified(tmp_path):
    X_train, X_test, y_train, y_test = _runner(tmp_path).load_split()
    assert (len(X_train), len(X_test)) == (614, 154)
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_one_row_per_model_and_imputer(table):
    assert len(table) == len(MODELS) * 1
    assert list(table.columns) == RESULT_COLUMNS


def test_metrics_are_between_0_and_1(table):
    metrics = ["cv_auc", "accuracy", "precision", "recall", "f1", "roc_auc"]
    assert table[metrics].ge(0).all().all()
    assert table[metrics].le(1).all().all()


def test_baseline_cannot_beat_chance_on_roc_auc(table):
    baseline = table[table["is_baseline"]]
    assert (baseline["roc_auc"] == 0.5).all()


def test_exactly_one_non_baseline_row_is_chosen(table):
    chosen = table[table["chosen_by_cv"]]
    assert len(chosen) == 1
    assert not chosen["is_baseline"].iloc[0]


def test_choice_follows_cv_score_not_test_score(table):
    contenders = table[~table["is_baseline"]]
    chosen = table[table["chosen_by_cv"]].iloc[0]
    assert chosen["cv_auc"] == contenders["cv_auc"].max()


def test_save_writes_csv_and_metadata(tmp_path, table):
    runner = _runner(tmp_path)
    path = runner.save(table)
    assert path.name == "final_test.csv"
    assert pd.read_csv(path).shape == table.shape
    assert path.with_suffix(".meta.json").exists()


def test_save_before_run_raises(tmp_path):
    with pytest.raises(RuntimeError):
        _runner(tmp_path).save()
