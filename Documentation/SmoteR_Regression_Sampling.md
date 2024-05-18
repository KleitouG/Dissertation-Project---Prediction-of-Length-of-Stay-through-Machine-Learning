# SMOTER Script

This script performs Synthetic Minority Over-sampling Technique for Regression (SMOTER) on the dataset "Regression-Database.csv" using the SMOGN package.


## Steps

### Resampling
Iterates over unique lengths of stay in the dataset:
- Selects records for the current length of stay.
- If the number of unique classes in the target variable is greater than 1, applies SMOTER using the `smoter` function from the SMOGN package.
- If there is only one class, no resampling is needed.
- Combines resampled subsets into a single DataFrame.

### Save Resampled Data
Saves the resampled dataset as "Regression-SMOGN.csv".

