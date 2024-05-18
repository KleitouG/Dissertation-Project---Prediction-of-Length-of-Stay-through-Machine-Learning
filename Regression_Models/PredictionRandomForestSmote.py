import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("../Regression-Smote-20k.csv")

data = data.drop(columns=["Facility ID"])

X = data.drop(columns=["Length of Stay"])
y = data["Length of Stay"]

#   30 hardcoded records
exclude_indices = [50000, 50001, 50002, 100000, 100001, 100002, 200000, 200001, 200002, 300000, 300001, 300002,
                   400000, 400001, 400002, 500000, 500001, 500002, 600000, 600001, 600002, 700000, 700001, 700002,
                   800000, 800001, 800002, 160110, 1182556, 1239771,  263582,  269120,  274598]

#   Store the excluded records
excluded_data = data.loc[exclude_indices]

#   Exclude selected records from the data
X_train_full = X.drop(index=exclude_indices)
y_train_full = y.drop(index=exclude_indices)

X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

#   Random Forest Regressor
rf_regressor = RandomForestRegressor(max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=1000, random_state=42)

rf_regressor.fit(X_train, y_train)

#   Make predictions on the training and testing data

y_train_pred = rf_regressor.predict(X_train)

y_pred_test = rf_regressor.predict(X_test)

#   Calculate metrics training and testing
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_pred_test)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_pred_test)
mre_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
mre_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100


print("Training Set Metrics:")
print("Mean Absolute Error (Training):", mae_train)
print("Mean Squared Error (Training):", mse_train)
print("Mean Relative Error (Training):", mre_train)
print("R^2 Score (Training):", r2_train)

print("\nTest Set Metrics:")
print("Mean Absolute Error (Testing):", mae_test)
print("Mean Squared Error (Testing):", mse_test)
print("Mean Relative Error (Testing):", mre_test)
print("R^2 Score (Testing):", r2_test)

#   Predict the excluded indices
X_excluded = X.loc[exclude_indices]
y_excluded_pred = rf_regressor.predict(X_excluded)

#   Display the actual vs predicted Length of Stay for the excluded data
excluded_data["Predicted Length of Stay"] = y_excluded_pred
print("\nActual vs Predicted Length of Stay for Excluded Data:")
print(excluded_data[["Length of Stay", "Predicted Length of Stay"]])

