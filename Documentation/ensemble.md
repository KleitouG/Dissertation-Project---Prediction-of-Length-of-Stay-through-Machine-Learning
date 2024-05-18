# Model Ensemble Meta-learner Script

This script trains a meta-learner using predictions from base models (Random Forest and XGBoost) and evaluates its performance.
    


## Steps


### Define Base Models
Defines Random Forest and XGBoost base models with preset parameters.

### Train Base Models
Fits the base models on the base training set.

### Get Predictions from Base Models
Obtains predictions from the base models on both training and testing data.

### Create Meta Dataset
Creates a meta dataset from the predictions of base models.

### Train Gradient Boosting Meta-Learner
Fits a Gradient Boosting meta-learner on the meta training set.

### Predictions from Meta-Learner
Obtains predictions from the meta-learner on both training and testing data.

### Calculate Metrics for Meta-Learner
Calculates Mean Absolute Error (MAE), Mean Squared Error (MSE), R^2 Score, and Mean Relative Error (MRE) for the meta-learner on both training and testing data.

### Print Metrics for Meta-Learner
Prints the evaluation metrics for the meta-learner on both training and testing data.
