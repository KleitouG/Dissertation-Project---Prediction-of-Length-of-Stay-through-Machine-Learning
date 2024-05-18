# Random Forest Regressor Explortation Script

## Overview

This document provides documentation for a Random Forest Regressor model trained to predict the length of stay of patients in a hospital using data from the provided dataset.

## Model

Random Forest Regressor is an ensemble learning method that constructs a multitude of decision trees during training and outputs the mean prediction of the individual trees. In this case, the model aims to predict the length of stay of patients based on various features.

## Model Training

The dataset is split into training and testing sets using an 80-20 train-test split. The Random Forest Regressor model is then trained on the training data.

## Scalers

Scalers can be replaced with other scalers or be removed to explore other possibilities.

## Model Evaluation

The model is evaluated using various metrics to assess its performance on both the training and testing sets.

### Metrics

- **Mean Absolute Error (MAE)**: The average absolute difference between the predicted and actual length of stay.
- **Mean Squared Error (MSE)**: The average of the squares of the differences between the predicted and actual length of stay.
- **Mean Relative Error (MRE)**: The average relative difference between the predicted and actual length of stay, expressed as a percentage.
- **R^2 Score**: The coefficient of determination, representing the proportion of the variance in the dependent variable (length of stay) that is predictable from the independent variables.


## Visualization

The following visualizations are included to analyze the model's performance:

### 1. Feature Importance
- Bar chart showing the importance of each feature in the model.

### 2. Learning Curve
- Plot showing the model's performance on the training and cross-validation sets as a function of training examples.

### 3. Residual Plot
- Scatter plot of residuals (difference between predicted and actual values) versus predicted values for both training and test data.

### 4. Actual vs. Predicted Plot
- Scatter plot of actual versus predicted values for both training and test data.
