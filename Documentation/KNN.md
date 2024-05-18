# KNN Script

## K-Nearest Neighbors (KNN) Regressor

This machine learning model predicts the length of stay of patients in a hospital using the K-Nearest Neighbors (KNN) regression algorithm.

## Model

KNN is a non-parametric algorithm that makes predictions based on the average of the 'k' nearest data points. In this case, the model aims to minimize the mean squared error between the predicted and actual length of stay.

## Model Training
The dataset is split into training and testing sets using an 80-20 train-test split.

## Scalers
Scalers can be replaced with other scalers or be removed to explore other possibilities.


## Model Evaluation

The model is evaluated using various metrics to assess its performance on both the training and testing sets.

### Metrics

- **Mean Absolute Error (MAE)**: The average absolute difference between the predicted and actual length of stay.
- **Mean Squared Error (MSE)**: The average of the squares of the differences between the predicted and actual length of stay.
- **Mean Relative Error (MRE)**: The average relative difference between the predicted and actual length of stay, expressed as a percentage.
- **R^2 Score**: The coefficient of determination, representing the proportion of the variance in the dependent variable (length of stay) that is predictable from the independent variables.

## Results

### Training Set Metrics

- **Mean Absolute Error (Training)**: {{mae_train}}
- **Mean Squared Error (Training)**: {{mse_train}}
- **Mean Relative Error (Training)**: {{mre_train}}%
- **R^2 Score (Training)**: {{r2_train}}

### Testing Set Metrics

- **Mean Absolute Error (Testing)**: {{mae_test}}
- **Mean Squared Error (Testing)**: {{mse_test}}
- **Mean Relative Error (Testing)**: {{mre_test}}%
- **R^2 Score (Testing)**: {{r2_test}}
