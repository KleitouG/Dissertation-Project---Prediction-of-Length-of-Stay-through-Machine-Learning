## XGBoost Model Script

XGBoost Regressor is used to predict the length of stay based on various features. MaxAbsScaler is applied to scale the features before training the model.

## Model Training

The dataset is split into training and testing sets using an 80-20 train-test split.

## Scalers
Scalers can be replaced with other scalers or be removed to explore other possibilities.

## Model Evaluation

The performance of the model is evaluated using various metrics on both the training and testing sets.

### Metrics

- **Mean Absolute Error (MAE)**: The average absolute difference between the predicted and actual length of stay.
- **Mean Squared Error (MSE)**: The average of the squares of the differences between the predicted and actual length of stay.
- **Mean Relative Error (MRE)**: The average relative difference between the predicted and actual length of stay, expressed as a percentage.
- **R^2 Score**: The coefficient of determination, representing the proportion of the variance in the dependent variable (length of stay) that is predictable from the independent variables.

## Model Performance

### Training Set Metrics:
- Mean Absolute Error (Training)
- Mean Squared Error (Training)
- Mean Relative Error (Training)
- R^2 Score (Training)

### Testing Set Metrics:
- Mean Absolute Error (Testing)
- Mean Squared Error (Testing)
- Mean Relative Error (Testing)
- R^2 Score (Testing)
