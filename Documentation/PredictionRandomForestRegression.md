# Prediction RFR Regression Database Script

## Random Forest Regressor

This machine learning model predicts the length of stay of patients in a hospital using the Random Forest regression algorithm.

### Data Preprocessing

Uses Smoted database. Additionaly, 30 records are excluded from the training data for further analysis.

## Model

Random Forest Regressor is an ensemble learning method that constructs a multitude of decision trees during training and outputs the mean prediction of the individual trees. In this case, the model aims to predict the length of stay of patients based on various features.


The Random Forest Regressor is trained with the remaining data using the following hyperparameters:
- **n_estimators**: 1000
- **max_depth**: None
- **min_samples_split**: 2
- **min_samples_leaf**: 1

### Model Evaluation

The model is evaluated using various metrics to assess its performance on both the training and testing sets.

#### Training Set Metrics
- **Mean Absolute Error (Training)**: {{mae_train}}
- **Mean Squared Error (Training)**: {{mse_train}}
- **Mean Relative Error (Training)**: {{mre_train}}%
- **R^2 Score (Training)**: {{r2_train}}

#### Testing Set Metrics
- **Mean Absolute Error (Testing)**: {{mae_test}}
- **Mean Squared Error (Testing)**: {{mse_test}}
- **Mean Relative Error (Testing)**: {{mre_test}}%
- **R^2 Score (Testing)**: {{r2_test}}

### Excluded Data Analysis

30 records are excluded from the training data to further analyze the model's performance. These excluded records are predicted separately, and their actual and predicted lengths of stay are compared.

Actual vs Predicted Length of Stay for Excluded Data:

| Length of Stay | Predicted Length of Stay |
|----------------|--------------------------|
| 2              | 1.732                    |
| 4              | 4.078                    |
| 2              | 3.127                    |
| 4              | 3.658                    |
| 3              | 2.942                    |


