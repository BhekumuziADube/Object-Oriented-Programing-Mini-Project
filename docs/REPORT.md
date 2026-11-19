# OOP Mini Project Report

## 1. Introduction and objectives
<!-- D3: Define the diabetes prediction problem, state the objectives,
and explain how the team will evaluate success. -->

## 2. Team process, Kanban and Git
<!-- Co3: Explain how the team used the board, branches, and pull
requests. Link to examples of reviews, pairing, and conflict resolution. -->

## 3. Dataset
<!-- An1: Describe the data source, number of records, predictor variables,
and meaning of Outcome. Summarise the raw data and link to the EDA findings. -->

## 4–6. Data quality, cleaning, clean datasets, EDA
<!-- REP2: Summarise the quality checks, explain how missing measurements
were handled, describe the exported datasets, and interpret the EDA figures. -->

### 4. Data quality assessment

This part checks the Pima dataset on six data quality dimensions before any cleaning. The checks were run with the project quality checker (completeness and uniqueness in C1, validity and consistency in C2, accuracy and timeliness in C3). The full output is in `docs/data_quality_report.md`. The meaning, unit and valid range of every column are in `docs/data_dictionary.md`.

#### 4.1 Results before cleaning

| Dimension | What was checked | Finding | Action |
| --- | --- | --- | --- |
| Completeness | Empty cells and hidden zeros | No empty cells, but 652 hidden zeros in five columns (10.6% of all feature values). Only 392 of 768 rows (51.0%) have no hidden zero. | Zeros in the five columns are treated as missing and imputed (part 5). |
| Validity | Values inside the allowed ranges, Outcome only 0 or 1 | No value outside the ranges in the data dictionary (hidden zeros excluded). Outcome contains only 0 and 1. | None needed. |
| Consistency | Data types, whole numbers, Pregnancies compared with Age | All nine columns are numeric, whole-number columns contain whole numbers, and Pregnancies is at most Age minus 10 in all 768 rows. | None needed. |
| Accuracy | Outliers (1.5 x IQR rule) and plausibility | Outliers flagged: Pregnancies 4, BloodPressure 14, SkinThickness 3, Insulin 24, BMI 8, DiabetesPedigreeFunction 29, Age 9, Glucose 0. The lowest non-zero BloodPressure is 24 mm Hg, which is unusual but inside the hard limits. `[[CHECK: compare these counts with docs/data_quality_report.md (C3) and use its numbers if they differ]]` | Outliers are flagged, not deleted, because none is physiologically impossible. |
| Uniqueness | Duplicate rows | 0 duplicate rows. The file has no patient ID, so it cannot show whether the same person appears twice. | A PatientID column is added in the clean datasets (part 6). |
| Timeliness | Age of the data | Collected in the late 1980s. The file has no date column, so this comes from the source documentation. | Stated as a limitation (4.4). |

#### 4.2 Hidden zeros

A value of 0 in Glucose, BloodPressure, SkinThickness, Insulin or BMI is not a real measurement, so it means "missing". The file has no empty cells, which is why the problem is hidden. Zeros in Pregnancies are valid (111 rows), because a patient can have had no pregnancies.

| Column | Rows with 0 | Share of 768 rows |
| --- | --- | --- |
| Glucose | 5 | 0.7% |
| BloodPressure | 35 | 4.6% |
| SkinThickness | 227 | 29.6% |
| Insulin | 374 | 48.7% |
| BMI | 11 | 1.4% |

Of the 768 rows, 376 (49.0%) have at least one hidden zero: 142 rows have one, 199 have two, 28 have three and 7 have four.

#### 4.3 Before and after cleaning

After imputation no missing value remains. The table shows what the zeros did to the data, using median imputation as an illustration. The medians are computed on all 768 rows here only to describe the data. In the models they are computed from the training data alone.

| Column | Median used | Mean with zeros | Mean after median fill | SD with zeros | SD after median fill |
| --- | --- | --- | --- | --- | --- |
| Glucose | 117 | 120.9 | 121.7 | 32.0 | 30.4 |
| BloodPressure | 72 | 69.1 | 72.4 | 19.4 | 12.1 |
| SkinThickness | 29 | 20.5 | 29.1 | 16.0 | 8.8 |
| Insulin | 125 | 79.8 | 140.7 | 115.2 | 86.4 |
| BMI | 32.3 | 32.0 | 32.5 | 7.9 | 6.9 |

The zeros pull the averages down and inflate the spread, most of all for Insulin and SkinThickness. Left in place they would give the models false information, which is why the project compares repairing them with leaving them alone (part 5).

#### 4.4 Limitations of the checks

- Accuracy can only be judged by plausibility, because the original patient records are not available.
- The data was collected in the late 1980s from women of Pima Indian heritage aged 21 or older, so the results may not apply to other populations, countries or years.
- With no patient ID, uniqueness is only checked on whole rows.

### 5. Data cleaning and imputation

#### 5.1 Approach

The cleaning follows the data quality findings in part 4. Hidden zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are converted to missing values and then filled by an imputer. Zeros in Pregnancies are left alone. The project compares three ways of handling the zeros, so that the effect of the choice can be measured:

| Method | What it does |
| --- | --- |
| Median imputation | Replaces each missing value with the median of that column in the training data. The median is not pulled by very large values, which matters for Insulin. |
| KNN imputation | Replaces each missing value using the 5 most similar patients in the training data. The columns are scaled first so that Insulin, which has the largest numbers, does not dominate the distances. |
| Zeros kept (control) | Leaves the zeros in the data as they are, to show what happens when they are not repaired. This pipeline skips the zero-to-missing step. |

A majority-class baseline model, which always predicts the most common outcome, gives a second reference point.

#### 5.2 Order of steps and prevention of data leakage

1. The data is split first: 80% training and 20% test (614 and 154 patients), stratified on Outcome, with `random_state` 42.
2. All other steps run inside one scikit-learn pipeline: zero-to-missing, scaling, imputation, then the model.
3. Every step learns only from the training data. During cross-validation and tuning, each fold fits the scaler and the imputer on its own training part, so the validation part never influences them. The test set is used once, at the end.

#### 5.3 Outliers

Outliers are flagged with the 1.5 x IQR rule but not deleted, unless a value is physiologically impossible. None of the flagged values was impossible, so no row was removed.

#### 5.4 Cleaning log

Every cleaning action is recorded by the CleaningLog class in `docs/cleaning_log.csv`. A summary of the main actions on the full dataset:

| Dimension | Column | Action | Rows affected |
| --- | --- | --- | --- |
| Completeness | Glucose | Zeros converted to missing, then imputed | 5 |
| Completeness | BloodPressure | Zeros converted to missing, then imputed | 35 |
| Completeness | SkinThickness | Zeros converted to missing, then imputed | 227 |
| Completeness | Insulin | Zeros converted to missing, then imputed | 374 |
| Completeness | BMI | Zeros converted to missing, then imputed | 11 |
| Uniqueness | All columns | Duplicate rows searched; none found, none removed | 0 |
| Validity | All columns | Values outside valid ranges searched; none found | 0 |
| Accuracy | Eight columns | Outliers flagged, none deleted | see 4.1 |

`[[CHECK: compare this table with docs/cleaning_log.csv and correct any row that differs. If the log has more steps, add them here.]]`

## 7. Software design, OOP
<!-- Co2: Include the class diagram and explain how the classes work
together, using examples from the project. -->

## 8. Machine-learning method
<!-- REP3: Describe the baseline, models, preprocessing order, tuning,
and evaluation. Explain which steps are fitted on training data. -->

## 9–10. Results, discussion, limitations
<!-- Co1: Present the model and imputation comparisons, interpret the
main differences, and explain the limits of what the results show. -->

## 11. Conclusion
<!-- REP3: Summarise the main finding in relation to the project objectives. -->

## 12. Contributions table
<!-- REP3: Record each member's contributions and link to the relevant
issues, commits, pull requests, or reviews. -->
