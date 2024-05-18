import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MaxAbsScaler

data = pd.read_csv("../Regression-Smote-20k.csv")

data = data.drop(columns=["Facility ID"])

X = data.drop(columns=["Length of Stay"])
y = data["Length of Stay"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#   Replace scaler here and on the imports to change scaler as well as removing the scaler
scaler = MaxAbsScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

#   Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

rf_regressor.fit(X_train_scaled, y_train)

#   Make predictions on the training and testing data
y_train_pred = rf_regressor.predict(X_train_scaled)

y_pred_test = rf_regressor.predict(X_test_scaled)

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
