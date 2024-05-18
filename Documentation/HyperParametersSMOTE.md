# GridSearch RFR Documentation

## Random Forest Regressor with Grid Search

This machine learning model predicts the length of stay of patients in a hospital using the Random Forest regression algorithm with Grid Search for hyperparameter tuning.

## Model

Random Forest Regressor is an ensemble learning method that fits a number of decision tree regressors on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The model aims to minimize the mean squared error between the predicted and actual length of stay.

## Data Sampling

The dataset is sampled to reduce its size while maintaining the distribution of the data. This is done to improve the efficiency of the training process.

## Model Training

The sampled dataset is split into training and testing sets using an 80-20 train-test split. Grid search is employed to find the optimal hyperparameters for the Random Forest Regressor.

## Model Evaluation

The model is evaluated using various metrics to assess its performance on both the training and testing sets.

### Metrics

- **Mean Absolute Error (MAE)**: The average absolute difference between the predicted and actual length of stay.
- **Mean Squared Error (MSE)**: The average of the squares of the differences between the predicted and actual length of stay.
- **R^2 Score**: The coefficient of determination, representing the proportion of the variance in the dependent variable (length of stay) that is predictable from the independent variables.
- **Mean Relative Error (MRE)**: The average relative difference between the predicted and actual length of stay, expressed as a percentage.

## Results

### Best Parameters

- **Best Parameters**: {{best_params}}

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
