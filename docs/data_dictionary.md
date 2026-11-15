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
