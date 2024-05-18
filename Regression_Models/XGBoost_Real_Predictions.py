import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("../Regression-Database.csv")

#   Segmentation Split

segment_1_7 = data[data["Length of Stay"] <= 7]
segment_8_23 = data[(data["Length of Stay"] >= 8) & (data["Length of Stay"] <= 23)]
segment_24_plus = data[data["Length of Stay"] >= 24]


X_1_7 = segment_1_7.drop(columns=["Length of Stay"])
y_1_7 = segment_1_7["Length of Stay"]

X_8_23 = segment_8_23.drop(columns=["Length of Stay"])
y_8_23 = segment_8_23["Length of Stay"]

X_24_plus = segment_24_plus.drop(columns=["Length of Stay"])
y_24_plus = segment_24_plus["Length of Stay"]

#   Split segments into training and testing sets
X_train_1_7, X_test_1_7, y_train_1_7, y_test_1_7 = train_test_split(X_1_7, y_1_7, test_size=0.2, random_state=42)
X_train_8_23, X_test_8_23, y_train_8_23, y_test_8_23 = train_test_split(X_8_23, y_8_23, test_size=0.2, random_state=42)
X_train_24_plus, X_test_24_plus, y_train_24_plus, y_test_24_plus = train_test_split(X_24_plus, y_24_plus, test_size=0.2, random_state=42)

#   Define hyperparameters
params = {
    'learning_rate': 0.1,
    'max_depth': 9,
    'min_child_weight': 24,
    'n_estimators': 300
}

#   Three XGBoost regressors for each segment with hyperparameters
xgb_model_1_7 = XGBRegressor(**params, random_state=42)
xgb_model_8_23 = XGBRegressor(**params, random_state=42)
xgb_model_24_plus = XGBRegressor(**params, random_state=42)

#   Make Predictions on training sets

xgb_model_1_7.fit(X_train_1_7, y_train_1_7)
xgb_model_8_23.fit(X_train_8_23, y_train_8_23)
xgb_model_24_plus.fit(X_train_24_plus, y_train_24_plus)

y_pred_1_7 = xgb_model_1_7.predict(X_test_1_7)
y_pred_8_23 = xgb_model_8_23.predict(X_test_8_23)
y_pred_24_plus = xgb_model_24_plus.predict(X_test_24_plus)

#   Calculate metrics training
mae_1_7 = mean_absolute_error(y_test_1_7, y_pred_1_7)
mse_1_7 = mean_squared_error(y_test_1_7, y_pred_1_7)
r2_1_7 = r2_score(y_test_1_7, y_pred_1_7)

mae_8_23 = mean_absolute_error(y_test_8_23, y_pred_8_23)
mse_8_23 = mean_squared_error(y_test_8_23, y_pred_8_23)
r2_8_23 = r2_score(y_test_8_23, y_pred_8_23)

mae_24_plus = mean_absolute_error(y_test_24_plus, y_pred_24_plus)
mse_24_plus = mean_squared_error(y_test_24_plus, y_pred_24_plus)
r2_24_plus = r2_score(y_test_24_plus, y_pred_24_plus)

print("Regression Metrics for 1-7 days segment:")
print("Mean Absolute Error:", mae_1_7)
print("Mean Squared Error:", mse_1_7)
print("R^2 Score:", r2_1_7)

print("\nRegression Metrics for 8-23 days segment:")
print("Mean Absolute Error:", mae_8_23)
print("Mean Squared Error:", mse_8_23)
print("R^2 Score:", r2_8_23)

print("\nRegression Metrics for 24+ days segment:")
print("Mean Absolute Error:", mae_24_plus)
print("Mean Squared Error:", mse_24_plus)
print("R^2 Score:", r2_24_plus)


#   Exclude 30 records
exclude_indices = [50000, 50001, 50002, 100000, 100001, 100002, 200000, 200001, 200002, 300000, 300001, 300002,
                   400000, 400001, 400002, 500000, 500001, 500002, 600000, 600001, 600002, 700000, 700001, 700002,
                   800000, 800001, 800002, 160110, 1182556, 1239771,  263582,  269120,  274598]

#   Predict LoS for excluded records
excluded_data = data.loc[exclude_indices].drop(columns=["Length of Stay"])
y_excluded_pred_1_7 = xgb_model_1_7.predict(excluded_data)
y_excluded_pred_8_23 = xgb_model_8_23.predict(excluded_data)
y_excluded_pred_24_plus = xgb_model_24_plus.predict(excluded_data)

#   Print actual vs predicted LoS for the excluded records for each segment

print("\nActual vs Predicted Length of Stay for Excluded Data for 1-7 days segment:")
print(pd.DataFrame({"Actual Length of Stay": data.loc[exclude_indices]["Length of Stay"],
                    "Predicted Length of Stay": y_excluded_pred_1_7}))

print("\nActual vs Predicted Length of Stay for Excluded Data for 8-23 days segment:")
print(pd.DataFrame({"Actual Length of Stay": data.loc[exclude_indices]["Length of Stay"],
                    "Predicted Length of Stay": y_excluded_pred_8_23}))

print("\nActual vs Predicted Length of Stay for Excluded Data for 24+ days segment:")
print(pd.DataFrame({"Actual Length of Stay": data.loc[exclude_indices]["Length of Stay"],
                    "Predicted Length of Stay": y_excluded_pred_24_plus}))
