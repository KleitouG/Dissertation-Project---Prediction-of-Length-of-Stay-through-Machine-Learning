import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("../Regression-Database.csv")

data = data.drop(columns=["Facility ID"])

X = data.drop(columns=["Length of Stay"])
y = data["Length of Stay"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#   Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

rf_regressor.fit(X_train, y_train)

#   Make predictions on the training and testing data
y_train_pred = rf_regressor.predict(X_train)
y_pred_test = rf_regressor.predict(X_test)

#   Calculate metrics training and testing
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)
mre_train = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
mre_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print("Training Set Metrics:")
print("Mean Absolute Error (Training):", mae_train)
print("Mean Squared Error (Training):", mse_train)
print("Root Mean Squared Error (Training):", rmse_train)
print("R^2 Score (Training):", r2_train)
print("Mean Relative Error (Training):", mre_train)

print("\nTest Set Metrics:")
print("Mean Absolute Error (Testing):", mae_test)
print("Mean Squared Error (Testing):", mse_test)
print("Root Mean Squared Error (Testing):", rmse_test)
print("R^2 Score (Testing):", r2_test)
print("Mean Relative Error (Testing):", mre_test)

# Plot feature importance, learning curve, residual plot, and actual vs. predicted plot
plt.figure(figsize=(15, 10))

# Feature Importance
plt.subplot(2, 2, 1)
feature_importance = rf_regressor.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
features = X.columns
plt.bar(range(X.shape[1]), feature_importance[sorted_idx], align="center")
plt.xticks(range(X.shape[1]), features[sorted_idx], rotation=90)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

# Learning Curve
plt.subplot(2, 2, 2)
train_sizes, train_scores, test_scores = learning_curve(rf_regressor, X_train, y_train, cv=10)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.title("Learning Curve")
plt.legend(loc="best")

# Residual Plot
plt.subplot(2, 2, 3)
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_pred_test, y_pred_test - y_test, c='green', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=min(y_train_pred.min(), y_pred_test.min()), xmax=max(y_train_pred.max(), y_pred_test.max()), color='red', lw=2)

# Actual vs. Predicted Plot
plt.subplot(2, 2, 4)
plt.scatter(y_train_pred, y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_pred_test, y_test, c='green', marker='s', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.title('Actual vs. Predicted Plot')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
