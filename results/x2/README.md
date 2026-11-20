# X2: Repeated cross-validation comparison

## Run

From the repository root:

```bash
python -m src.repeated_cv --folds 5 --repeats 3 --output-dir results/x2
```

## Method

Compared logistic regression, Random Forest, KNN and a majority-class
baseline under median, KNN and zeros-kept preprocessing.

Used five-fold stratified cross-validation repeated three times,
with seed 42. These are proposed X2 settings for team review.
All 12 configurations used identical splits of the 614 training rows.
The 154 held-out test rows were not evaluated.

Scaling and imputation were fitted separately inside each fold.
Median and KNN pipelines converted impossible zeros to missing values.
The zeros-kept control skipped ZeroToNaN.

Models used default settings. These are untuned comparisons, not
estimates of a tuned model-selection procedure.

## Findings

- Logistic regression with KNN imputation had the highest mean
  ROC-AUC: 0.8434 (fold-score SD 0.0253), with accuracy 0.7840
  and F1 0.6503.
- Logistic regression with median imputation was very close:
  ROC-AUC 0.8425 (SD 0.0270). The difference of 0.0009 alone
  does not establish that one imputation method is better.
- Mean ROC-AUC was higher with median or KNN imputation than
  with zeros retained for each of the three predictive models.
- For Random Forest, KNN imputation had the highest ROC-AUC
  (0.8235), while retaining zeros had the highest accuracy
  (0.7579). Rankings therefore depend on the metric.
- The majority-class baseline had accuracy 0.6515, F1 0 and
  ROC-AUC 0.5 under every preprocessing method. Its accuracy
  reflects the class imbalance.

Fold scores are correlated. Their standard deviations describe
variation across validation splits, not confidence intervals.
These results do not establish statistical significance or
clinical readiness.

## Outputs

- fold_scores.csv: 180 rows, including five metrics and confusion
  matrix counts for each configuration and split.
- summary.csv: 12 configurations with metric means and standard deviations.
- run_metadata.json: settings, training/held-out row identifiers
  and fold memberships.

## Verification

The imputer and X2 pipeline test run passed 19 tests.
The full comparison generated 180 fold-score rows.
The export-validation checks cover configuration counts, finite
metrics and separation of training, validation and held-out rows.

The X2 branch also corrects a merge issue that placed KNN imputer
methods inside ZerosKeptImputer, restoring both strategies.