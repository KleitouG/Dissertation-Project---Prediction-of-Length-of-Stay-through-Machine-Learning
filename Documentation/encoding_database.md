# Data Encoding Script

This script encodes categorical features in the dataset "Regression-Database.csv" using target encoding for ordinal columns and frequency encoding for non-ordinal columns.


## Steps


### Define Columns
Defines two lists of columns:
- `ordinal_cols`: Contains the names of ordinal columns to be target encoded.
- `non_ordinal_cols`: Contains the names of non-ordinal columns to be frequency encoded.

### Target Encoding
Applies target encoding to the ordinal columns using `TargetEncoder` from the `category_encoders` library.

### Frequency Encoding
Applies frequency encoding to the non-ordinal columns.

### Drop Original Columns
Drops the original non-ordinal columns from the dataset.

### Save Encoded Data
Saves the encoded dataset to a new CSV file named "Regression-Encoded.csv".

