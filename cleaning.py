import pandas as pd

#   Create a protective mesh for cleaning
df = pd.read_csv("NY-2010-Refined.csv", na_values=['', 'NA', 'nan', 'NaN', 'N/A', 'Not Available', 'Unknown', 'U'], low_memory=False)

#   Strip whitespace from string columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

#   Display the first few rows
print(df.head())

#   Display information
print(df.info())

#   Generate descriptive statistics
print(df.describe())

#   Drop rows with missing values
df_cleaned = df.dropna()

#   Save the database
df_cleaned.to_csv("NY-2010-Refined-Cleaned.csv", index=False)
