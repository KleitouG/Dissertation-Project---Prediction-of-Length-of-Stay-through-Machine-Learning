import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

data = pd.read_csv("../Regression-Encoded.csv")

data = data.drop(columns=["Facility ID_freq_encoded"])

X = data.drop(columns=["Length of Stay"])
y = data["Length of Stay"]

#   Replace scaler here and on the imports to change scaler as well as removing the scaler
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#   KNN regressor
model = KNeighborsRegressor()

model.fit(X_train, y_train)

#   Make predictions on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#   Calculate metrics for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mre_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
r2_train = r2_score(y_train, y_train_pred)

#   Calculate metrics for testing set
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
mre_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
r2_test = r2_score(y_test, y_test_pred)

print("Mean Absolute Error (Training):", mae_train)
print("Mean Squared Error (Training):", mse_train)
print("Mean Relative Error (Training):", mre_train)
print("R^2 Score (Training):", r2_train)

print("\nMean Absolute Error (Testing):", mae_test)
print("Mean Squared Error (Testing):", mse_test)
print("Mean Relative Error (Testing):", mre_test)
print("R^2 Score (Testing):", r2_test)
