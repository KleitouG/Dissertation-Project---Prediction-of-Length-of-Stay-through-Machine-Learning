import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("Regression-Smote-20k.csv")

data = data.drop(columns=["Facility ID"])

X = data.drop(columns=["Length of Stay"])
y = data["Length of Stay"]

X_train_base, X_meta, y_train_base, y_meta = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_train_base, y_train_base, test_size=0.2, random_state=42)

#   Define preset parameters for base models
rf_params = {
    'n_estimators': 1000,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}

xgb_params = {
    'learning_rate': 0.1,
    'max_depth': 9,
    'min_child_weight': 24,
    'n_estimators': 300
}

#   Base Models
rf_model = RandomForestRegressor(**rf_params, random_state=42)
xgb_model = XGBRegressor(**xgb_params, random_state=42)


rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

#   Get predictions from base models
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

xgb_train_pred = xgb_model.predict(X_train)
xgb_test_pred = xgb_model.predict(X_test)

# Create a meta dataset from the predictions of base models
X_meta_train = np.column_stack((rf_train_pred, xgb_train_pred))
X_meta_test = np.column_stack((rf_test_pred, xgb_test_pred))

#   Gradient Boosting meta-learner
gbm_meta = GradientBoostingRegressor(random_state=42)

gbm_meta.fit(X_meta_train, y_train)

#   Predictions from the meta-learner
meta_train_pred = gbm_meta.predict(X_meta_train)
meta_test_pred = gbm_meta.predict(X_meta_test)

#   Calculate metrics for meta-learner
mae_meta_train = mean_absolute_error(y_train, meta_train_pred)
mae_meta_test = mean_absolute_error(y_test, meta_test_pred)
mse_meta_train = mean_squared_error(y_train, meta_train_pred)
mse_meta_test = mean_squared_error(y_test, meta_test_pred)
r2_meta_train = r2_score(y_train, meta_train_pred)
r2_meta_test = r2_score(y_test, meta_test_pred)
mre_meta_train = np.mean(np.abs((y_train - meta_train_pred) / y_train)) * 100
mre_meta_test = np.mean(np.abs((y_test - meta_test_pred) / y_test)) * 100

#   Print metrics for meta-learner
print("\nMeta-Learner Metrics:")
print("Mean Absolute Error (Training):", mae_meta_train)
print("Mean Squared Error (Training):", mse_meta_train)
print("R^2 Score (Training):", r2_meta_train)
print("Mean Relative Error (Training):", mre_meta_train)

print("Mean Absolute Error (Testing):", mae_meta_test)
print("Mean Squared Error (Testing):", mse_meta_test)
print("R^2 Score (Testing):", r2_meta_test)
print("Mean Relative Error (Testing):", mre_meta_test)