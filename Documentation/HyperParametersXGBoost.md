# GridSearch XGBoost Script

## XGBoost Regressor with Grid Search

This machine learning model predicts the length of stay of patients in a hospital using the XGBoost regression algorithm with Grid Search for hyperparameter tuning.

## Model

XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. The model aims to minimize the mean squared error between the predicted and actual length of stay.



## Hyperparameter Tuning

Grid search is used to find the optimal hyperparameters for the XGBoost Regressor.

## Model Evaluation

The model is evaluated using various metrics to assess its performance on both the training and testing sets.

### Metrics

- **Mean Absolute Error (MAE)**: The average absolute difference between the predicted and actual length of stay.
- **Mean Squared Error (MSE)**: The average of the squares of the differences between the predicted and actual length of stay.
- **R^2 Score**: The coefficient of determination, representing the proportion of the variance in the dependent variable (length of stay) that is predictable from the independent variables.
- **Mean Relative Error (MRE)**: The average relative difference between the predicted and actual length of stay, expressed as a percentage.

## Results

### Best Parameters (R^2)

- **Best Parameters (R^2)**: {'max_depth': {{best_params['max_depth']}}, 'min_child_weight': {{best_params['min_child_weight']}}, 'n_estimators': {{best_params['n_estimators']}}}

### Training Set Metrics

- **Mean Absolute Error (Training)**: {{mae_train}}
- **Mean Squared Error (Training)**: {{mse_train}}
- **R^2 Score (Training)**: {{r2_train}}
- **Mean Relative Error (Training)**: {{mre_train}}

### Testing Set Metrics

- **Mean Absolute Error (Testing)**: {{mae_test}}
- **Mean Squared Error (Testing)**: {{mse_test}}
- **R^2 Score (Testing)**: {{r2_test}}
- **Mean Relative Error (Testing)**: {{mre_test}}
