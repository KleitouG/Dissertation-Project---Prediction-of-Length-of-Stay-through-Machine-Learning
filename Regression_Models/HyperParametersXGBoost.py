import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

data = pd.read_csv("../Regression-Database.csv")

data = data.drop(columns=["Facility ID"])

X = data.drop(columns=["Length of Stay"])  # Features
y = data["Length of Stay"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#   Define parameter grid
param_grid = {
    'n_estimators': [300, 400, 500],
    'max_depth': [9, 12, 16],
    'min_child_weight': [1, 3, 5],
}
#   XGBoost regressor
model = XGBRegressor(random_state=42)

#   GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_relative_error', n_jobs=-1)

grid_search.fit(X_train, y_train)

#   Get & Print the best parameters
best_params = grid_search.best_params_

print("Best Parameters (R^2):", best_params)

#   Predict on the training set using the best model
y_train_pred = grid_search.predict(X_train)

#   Predict on the test set using the best model
y_pred_test = grid_search.predict(X_test)

#   Calculate evaluation metrics
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
mre_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
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
