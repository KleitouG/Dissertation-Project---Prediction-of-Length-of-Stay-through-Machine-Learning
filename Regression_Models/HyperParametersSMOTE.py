import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("../Regression-Smote-20k.csv")

data = data.drop(columns=["Facility ID"])

X = data.drop(columns=["Length of Stay"])
y = data["Length of Stay"]

#   Reduce dataset size while maintaining distribution with sampling
sample_data, _, sample_target, _ = train_test_split(X, y, train_size=0.05, stratify=y, random_state=42)

#   Split the sampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sample_data, sample_target, test_size=0.2, random_state=42)

#   Define parameter grid
param_grid = {
    'n_estimators': [1000],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

# Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

#   GridSearchCV
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid,
                            scoring='neg_mean_absolute_error', n_jobs=4)

#   Fit the model
grid_search.fit(X_train, y_train)

#   Getting & Printing the best parameters
best_params = grid_search.best_params_

print("Best Parameters:", best_params)

#   Predict on the training set
y_train_pred = grid_search.predict(X_train)

#   Predict on the test set
y_pred_test = grid_search.predict(X_test)

#   Calculate metrics
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_pred_test)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_pred_test)
mre_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
mre_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100


print("\nTraining Set Metrics:")
print("Mean Absolute Error (Training):", mae_train)
print("Mean Squared Error (Training):", mse_train)
print("R^2 Score (Training):", r2_train)
print("Mean Relative Error (Training):", mre_train)

print("\nTest Set Metrics:")
print("Mean Absolute Error (Testing):", mae_test)
print("Mean Squared Error (Testing):", mse_test)
print("R^2 Score (Testing):", r2_test)
print("Mean Relative Error (Testing):", mre_test)
