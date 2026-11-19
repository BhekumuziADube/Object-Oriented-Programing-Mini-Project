# Exploratory analysis notes (card An2)

## Correlation with Outcome (zeros treated as missing)

| Feature | Correlation |
|---|---|
| Glucose | 0.49 |
| BMI | 0.31 |
| Insulin | 0.30 |
| SkinThickness | 0.26 |
| Age | 0.24 |
| Pregnancies | 0.22 |
| DiabetesPedigreeFunction | 0.17 |
| BloodPressure | 0.17 |

## Spread before and after imputation

| Column | Version | Mean | SD |
|---|---|---|---|
| Insulin | observed only | 155.5 | 118.8 |
| Insulin | median | 140.7 | 86.4 |
| Insulin | knn | 153.7 | 97.8 |
| SkinThickness | observed only | 29.2 | 10.5 |
| SkinThickness | median | 29.1 | 8.8 |
| SkinThickness | knn | 29.0 | 9.5 |
