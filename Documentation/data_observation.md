# Data Observation Script

This script reads a dataset from a CSV file named "NY-Hospital-Inpatients-2010.csv" and performs data exploration tasks.

## Data Exploration Steps

### Reading the Data
The script reads the dataset into a pandas DataFrame.

### Get Unique Values and Count of Unique Values for Each Column
The script iterates through each column of the dataset and prints the following information:
- Column name
- Number of unique values
- Unique values
- Number of '120 +' entries in the 'Length of Stay' column
- Number of 'Left Against Medical Advice' entries in all columns
- Number of NaN entries

### Count 'Y' and 'N' Values in the 'Abortion Edit Indicator' Column
The script counts the number of 'Y' and 'N' entries in the 'Abortion Edit Indicator' column.

### Count 'Y' and 'N' Values in the 'Emergency Department Indicator' Column
The script counts the number of 'Y' and 'N' entries in the 'Emergency Department Indicator' column.

### Miscellaneous
- The script also outputs the total number of columns in the dataset.

## Output
The script outputs information about unique values, counts of unique values, counts of '120 +' entries in the 'Length of Stay' column, counts of 'Left Against Medical Advice' entries, counts of NaN entries, counts of 'Y' and 'N' entries in the 'Abortion Edit Indicator' and 'Emergency Department Indicator' columns, and the total number of columns in the dataset.
