import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("../Regression-Encoded.csv")

data = data.drop(columns=["Facility ID_freq_encoded"])

X = data.drop(columns=["Length of Stay"])
y = data["Length of Stay"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#   Generate polynomial features of the ElasticNet (2 should suffice to tell us the difference between
#   the relationships
poly = PolynomialFeatures(degree=2)  # You can adjust the degree as needed
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

#   Replace scaler here and on the imports to change scaler as well as removing the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

#   ElasticNet
elastic_net = ElasticNet(random_state=42)
elastic_net.fit(X_train_scaled, y_train)

#   Training Set
y_train_pred = elastic_net.predict(X_train_scaled)

#   Testing Set
y_test_pred = elastic_net.predict(X_test_scaled)

#   Calculate evaluation metrics for training set
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
mre_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100

#   Calculate evaluation metrics for testing set
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
mre_test = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

#   Print training set
print("Training Set Metrics:")
print("Mean Absolute Error (Training):", mae_train)
print("Mean Squared Error (Training):", mse_train)
print("Mean Relative Error (Training):", mre_train)
print("R^2 Score (Training):", r2_train)

#   Print testing set
print("\nTesting Set Metrics:")
print("Mean Absolute Error (Testing):", mae_test)
print("Mean Squared Error (Testing):", mse_test)
print("Mean Relative Error (Testing):", mre_test)
print("R^2 Score (Testing):", r2_test)
