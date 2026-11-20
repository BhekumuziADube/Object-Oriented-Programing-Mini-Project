# Diabetes Prediction: Object-Oriented Mini Project

A collaborative Python project comparing diabetes classifiers and
missing-measurement strategies using the Pima diabetes dataset.

The project separates data preparation, imputation, model construction,
tuning, evaluation and experiment coordination into reusable components.

## Setup

Run commands from the repository root. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn matplotlib pytest
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

No pinned dependency manifest is currently supplied. Package versions
should be recorded when reproducing an experiment.

The input dataset is expected at `data/diabetes.csv`, with the binary
target column `Outcome`.

## Experiment runner

Inspect the available arguments:

```bash
python -m src.experiment_runner --help
```

The runner supports `--data`, `--out` and `--quick`.

Full-run command:

```bash
python -m src.experiment_runner --data data/diabetes.csv --out results
```

Quick-mode command:

```bash
python -m src.experiment_runner --quick --out results/quick
```

Quick mode uses smaller grids and two CV folds. It is a workflow check,
not a substitute for the full experiment.

### Current runner limitations

The help command has been verified. The commands above describe the
implemented CLI; successful execution of every configuration is not
yet confirmed for this documentation revision.

- The runner currently includes ZeroToNaN for the zeros-kept control.
  This must be corrected: that control requires
  StandardScaler → ZerosKeptImputer → model.
- The full KNN grid contains 28 combinations, exceeding the project's
  stated limit of 20.
- The runner defines its own grids and uses GridSearchCV directly;
  it does not currently delegate to HyperparameterTuner.
- It evaluates every configuration once on the same held-out split.
  It marks the winner using CV ROC-AUC, not test performance.
  Do not use the displayed test scores to revise model selection.
- CV scores are rounded before winner selection, so close scores may tie.

## Outputs

The runner saves `final_test.csv` in the selected output directory.
It includes CV ROC-AUC, selected parameters, test metrics and the
`chosen_by_cv` flag.

A neighbouring `final_test.meta.json` is written when the reproducibility
helper is available. Dataset checksum verification also depends on that
helper and an accompanying checksum file; otherwise it is skipped with
a message.

## Repeated cross-validation comparison

X2 provides a separate comparison of default model settings:

```bash
python -m src.repeated_cv --folds 5 --repeats 3 --output-dir results/x2
```

See [X2 methods and findings](results/x2/README.md).
Its saved outputs include fold scores, summary statistics and split
memberships. The held-out test set is not evaluated by X2.

## Design and class diagrams

See [Report Section 7](docs/REPORT.md#7-software-design-oop) for the
class diagrams, OOP explanation and project examples.

## Testing

Run the project tests:

```bash
python -m pytest -q
```

This command runs the tests collected by the repository's pytest
configuration. Results should be reported for the actual checkout;
a passing subset does not establish that the complete suite passes.

## Interpretation

This is an educational comparison using a historical dataset.
CV estimates and held-out results describe this experiment; they do
not establish suitability for clinical deployment.