# Raw data quality report

Records checked: 768

Counts are reported per check. The same row may appear under several rules.

| Dimension | Check | Column | Issues | Of | Percent | Note |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Completeness | Blank (NaN) cells | Pregnancies | 0 | 768 | 0.0 |  |
| Completeness | Blank (NaN) cells | Glucose | 0 | 768 | 0.0 |  |
| Completeness | Blank (NaN) cells | BloodPressure | 0 | 768 | 0.0 |  |
| Completeness | Blank (NaN) cells | SkinThickness | 0 | 768 | 0.0 |  |
| Completeness | Blank (NaN) cells | Insulin | 0 | 768 | 0.0 |  |
| Completeness | Blank (NaN) cells | BMI | 0 | 768 | 0.0 |  |
| Completeness | Blank (NaN) cells | DiabetesPedigreeFunction | 0 | 768 | 0.0 |  |
| Completeness | Blank (NaN) cells | Age | 0 | 768 | 0.0 |  |
| Completeness | Blank (NaN) cells | Outcome | 0 | 768 | 0.0 |  |
| Completeness | Zero recorded where zero is impossible | Glucose | 5 | 768 | 0.7 | Hidden missing value |
| Completeness | Zero recorded where zero is impossible | BloodPressure | 35 | 768 | 4.6 | Hidden missing value |
| Completeness | Zero recorded where zero is impossible | SkinThickness | 227 | 768 | 29.6 | Hidden missing value |
| Completeness | Zero recorded where zero is impossible | Insulin | 374 | 768 | 48.7 | Hidden missing value |
| Completeness | Zero recorded where zero is impossible | BMI | 11 | 768 | 1.4 | Hidden missing value |
| Completeness | Rows with at least one hidden zero | (any of: Glucose, BloodPressure, SkinThickness, Insulin, BMI) | 376 | 768 | 49.0 | Dropping these rows would lose a large share of the data |
| Uniqueness | Fully duplicated rows | (all columns) | 0 | 768 | 0.0 |  |
| Uniqueness | Duplicate rows ignoring Outcome | (feature columns) | 0 | 768 | 0.0 | The file has no patient ID, so uniqueness of patients cannot be proven from the data alone |
| Validity | All rules | (dataset) | 0 | 768 | 0.0 | Passed |
| Consistency | All rules | (dataset) | 0 | 768 | 0.0 | Passed |
| Accuracy | Verification against source records | (dataset) | — | — | — | Not assessed. No reference records were supplied to verify the recorded measurements. |
| Timeliness | Collection and update dates | (dataset) | — | — | — | Not assessed. The report was generated without verified collection or update dates. |
