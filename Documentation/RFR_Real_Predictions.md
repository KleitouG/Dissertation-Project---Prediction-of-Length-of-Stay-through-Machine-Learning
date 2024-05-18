## RFR Real Predictions

Random Forest Regressor is used to predict the length of stay for different segments of hospital patients based on various features. Three separate models are trained for three segments: 1-7 days, 8-23 days, and 24+ days.

## Model Training

Each segment's dataset is split into training and testing sets using an 80-20 train-test split. The Random Forest Regressor is trained with hyperparameters: n_estimators=1000, max_depth=None, min_samples_split=2, and min_samples_leaf=1.

## Segmentation

The dataset is segmented into three categories based on the length of stay:
- **1-7 days segment:** Includes patients with a length of stay between 1 and 7 days.
- **8-23 days segment:** Includes patients with a length of stay between 8 and 23 days.
- **24+ days segment:** Includes patients with a length of stay of 24 days or more.

## Model Evaluation

The performance of each segment's model is evaluated using the following regression metrics:

### Metrics for 1-7 days segment:
- **Mean Absolute Error (MAE):** 
- **Mean Squared Error (MSE):** 
- **R^2 Score:** 

### Metrics for 8-23 days segment:
- **Mean Absolute Error (MAE):** 
- **Mean Squared Error (MSE):** 
- **R^2 Score:** 

### Metrics for 24+ days segment:
- **Mean Absolute Error (MAE):** 
- **Mean Squared Error (MSE):** 
- **R^2 Score:** 

## Model Performance

### 1-7 days segment:
- Mean Absolute Error: 
- Mean Squared Error: 
- R^2 Score: 

### 8-23 days segment:
- Mean Absolute Error: 
- Mean Squared Error: 
- R^2 Score: 

### 24+ days segment:
- Mean Absolute Error: 
- Mean Squared Error: 
- R^2 Score: 

## Excluded Data Prediction

The length of stay for 10 examples excluded from training is predicted using each segment's model. Below are the actual vs. predicted length of stay for each segment:

### For 1-7 days segment:
| Actual Length of Stay | Predicted Length of Stay |
|-----------------------|---------------------------|
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |

### For 8-23 days segment:
| Actual Length of Stay | Predicted Length of Stay |
|-----------------------|---------------------------|
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |

### For 24+ days segment:
| Actual Length of Stay | Predicted Length of Stay |
|-----------------------|---------------------------|
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
|                       |                           |
