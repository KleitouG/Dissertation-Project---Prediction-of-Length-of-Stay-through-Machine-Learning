# Dataset Cleaning
The code presented on this file is dedicated in cleansing the database from junk

## Reading

 Read the CSV file into a pandas DataFrame, df, with custom missing value indicators.

## Stripping Whitespace

Apply a lambda function to strip leading and trailing whitespace from string columns using apply.


## Displaying First Few Rows

 Print the first few rows of the DataFrame using head() to ensure data was read correctly.


## Displaying Information

Print information about the DataFrame using info() to check data types and missing values.


## Generating Descriptive Statistics

Use describe() to generate descriptive statistics for numerical columns.


## Dropping Rows with Missing Values

Drop rows with missing values using dropna() and saved the result to df_cleaned.

