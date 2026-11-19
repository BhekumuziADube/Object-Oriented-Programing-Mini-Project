# CL2: Cleaned dataset exports

Generate the files from the repository root:

```bash
python -m src.export_clean_datasets --input data/diabetes.csv
```

Outputs:
- diabetes_median_train.csv
- diabetes_median_test.csv
- diabetes_knn_train.csv
- diabetes_knn_test.csv
- cleaning_log.csv

Both methods use the same stratified 80/20 split with random_state=42.
Impossible zeros become missing values, followed by scaling and imputation.
Preprocessing is fitted on training rows only.

Features are in standardized units. Outcome is unchanged.
source_row identifies the original CSV row index; exclude it from model features.

These exports use preprocessing fitted on the full training split.
For cross-validation, fit preprocessing separately inside each fold.

The script checks finite features, preserved Outcome, unchanged raw input,
and saved CSV contents.

Uses Shepard's Preprocessor and CleaningLog. CL2 includes integration
fixes to CleaningLog's file location and add() method.