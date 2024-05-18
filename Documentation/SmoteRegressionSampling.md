# SMOTE Script

This script downsamples the dataset "Regression-Database.csv" and applies Synthetic Minority Over-sampling Technique (SMOTE) to balance the classes, then saves the downsampled and SMOTE-applied dataset as "Regression-Smote-20k.csv".

## Steps


### Set Maximum Records per Day
Defines the maximum number of records per day to downsample to.

### Downsample and Apply SMOTE
Iterates over unique lengths of stay in the dataset:
- Selects records for the current length of stay.
- If the number of records exceeds the maximum, downsamples to the maximum.
- If the number of unique classes in the target variable is greater than 1, applies SMOTE to balance the classes.
- Combines downsampled and SMOTE-applied subsets into a single DataFrame.

### Save Downsampled and SMOTE-applied Data
Saves the downsampled and SMOTE-applied dataset as "Regression-Smote-20k.csv".

