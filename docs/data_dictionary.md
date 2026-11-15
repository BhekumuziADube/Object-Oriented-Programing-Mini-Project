# Data dictionary: Pima Indians Diabetes dataset

File: `data/diabetes.csv`. 768 rows, 9 columns, no missing (empty) cells and no duplicate rows. There is no patient ID and no date column.

## Columns

| Column | Meaning | Unit | Type | Valid range (hard limits) | Observed min to max |
| --- | --- | --- | --- | --- | --- |
| Pregnancies | Number of times pregnant | count | integer | 0 to 20 | 0 to 17 |
| Glucose | Plasma glucose 2 hours into an oral glucose tolerance test | mg/dL | integer | 40 to 400 | 0 to 199 |
| BloodPressure | Diastolic blood pressure | mm Hg | integer | 20 to 150 | 0 to 122 |
| SkinThickness | Triceps skin fold thickness | mm | integer | 5 to 100 | 0 to 99 |
| Insulin | 2-hour serum insulin | µU/mL | integer | 10 to 900 | 0 to 846 |
| BMI | Body mass index | kg/m² | decimal | 10 to 80 | 0 to 67.1 |
| DiabetesPedigreeFunction | Score estimating diabetes risk from family history | no unit | decimal | 0.05 to 2.5 | 0.078 to 2.42 |
| Age | Age of the patient | years | integer | 21 to 100 | 21 to 81 |
| Outcome | Diabetes result as defined by the source: 0 = negative, 1 = positive | label | integer | 0 or 1 | 0 or 1 |

Outcome counts: 500 negative (0) and 268 positive (1), so 34.9% positive.

The valid ranges are the hard limits used by the validity check (card C2). They are proposed values the team can adjust, but only before Monday.


## Hidden zeros (missing values written as 0)

In five columns a value of 0 is not a real measurement, so it means "missing". The file has no empty cells, which is why this problem is hidden.

| Column | Rows with 0 | Share of 768 rows |
| --- | --- | --- |
| Glucose | 5 | 0.7% |
| BloodPressure | 35 | 4.6% |
| SkinThickness | 227 | 29.6% |
| Insulin | 374 | 48.7% |
| BMI | 11 | 1.4% |

Pregnancies has 111 rows with 0. These are valid, because a patient can have had no pregnancies. Do not treat them as missing.

Rule used by the project: in Glucose, BloodPressure, SkinThickness, Insulin and BMI, a value of 0 is converted to missing (NaN) before imputation.

## Consistency rule

Pregnancies must be at most Age minus 10. All 768 rows currently follow this rule.

## Source, period and population

- Origin: US National Institute of Diabetes and Digestive and Kidney Diseases, shared through the UCI Machine Learning Repository and Kaggle. Copy the exact citation from the page you downloaded the file from: https://www.kaggle.com/datasets/mragpavank/diabetes
- Population: women of Pima Indian heritage, aged 21 or older, as described in the source documentation.
- Period: collected in the late 1980s. The file has no date column, so this comes from the source documentation, not from the data itself.

## Known limitations

- Results may not apply to other populations, countries or years.
- Diagnostic practice has changed since the data was collected.
- Original patient records cannot be checked, so accuracy checks can only test whether values are plausible.
- One BloodPressure value (24 mm Hg) is unusually low. It is inside the hard limits but will be flagged by the accuracy check.
