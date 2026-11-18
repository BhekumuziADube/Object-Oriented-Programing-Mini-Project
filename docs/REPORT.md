# OOP Mini Project Report

## 1. Introduction and objectives

### Problem statement

This project uses the eight predictors in the Pima diabetes dataset to
classify the recorded binary diabetes outcome (`Outcome`). In selected
measurement fields, recorded zeros are treated as missing. Because the
choice of imputation method can influence model performance, the project
compares median and k-nearest neighbours (KNN) imputation under the same
evaluation procedure.

### Objectives

1. Assess the quality of the raw data by identifying missing measurements
   and documenting values flagged by the validity and consistency checks.
2. Compare logistic regression, random forest, and KNN classifiers with
   a majority-class baseline.
3. Assess how median and KNN imputation affect each classifier by
   comparing them on identical validation splits.

### Success metrics

Use ROC-AUC as the primary metric for comparing how well models
distinguish positive from negative outcomes across classification
thresholds. Report accuracy, precision, recall, F1, and the confusion
matrix at the decision rule used to generate class predictions. Recall
shows the proportion of positive cases detected, while the false-negative
count shows how many positive cases the model missed.

Evaluate each model and imputation strategy on the same stratified
validation splits. Fit preprocessing and models using only the training
data within each split, then evaluate on the held-out data. Report the
mean and standard deviation of ROC-AUC across the validation splits.
Judge improvement by comparing each model with the majority-class
baseline and by examining the direction, size, and consistency of the
ROC-AUC differences between median and KNN imputation for the same
classifier. Interpret the ranking alongside recall and precision so
that any trade-offs in detecting positive cases are visible.

## 2. Team process, Kanban and Git
<!-- Co3: Explain how the team used the board, branches, and pull
requests. Link to examples of reviews, pairing, and conflict resolution. -->

## 3. Dataset
<!-- An1: Describe the data source, number of records, predictor variables,
and meaning of Outcome. Summarise the raw data and link to the EDA findings. -->

## 4–6. Data quality, cleaning, clean datasets, EDA
<!-- REP2: Summarise the quality checks, explain how missing measurements
were handled, describe the exported datasets, and interpret the EDA figures. -->

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
