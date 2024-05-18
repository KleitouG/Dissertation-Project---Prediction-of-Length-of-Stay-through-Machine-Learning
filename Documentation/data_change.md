# Data Change Script

This script preprocesses a dataset from a CSV file named "NY-2010-Refined-Cleaned.csv" and saves the preprocessed data to "Regression-Database.csv".


## Data Preprocessing Steps


### Mapping Encodings
Define dictionaries to map categorical variables to numerical values:
- `age_group_mapping`: Maps age groups to numerical categories.
- `gender_mapping`: Maps genders to numerical categories.
- `race_mapping`: Maps races to numerical categories.
- `ethnicity_mapping`: Maps ethnicities to numerical categories.
- `admission_mapping`: Maps types of admission to numerical categories.
- `apr_risk_mortality_mapping`: Maps APR risk of mortality to numerical categories.
- `apr_medical_surgical_description_mapping`: Maps APR medical surgical description to numerical categories.
- `emergency_department_indicator`: Maps emergency department indicator to numerical categories.

### Applying Mapping
Map the categorical columns to numerical values using the defined dictionaries.

### Data Cleaning
Remove rows where 'Length of Stay' equals '120 +' and 'Patient Disposition' equals 'Left Against Medical Advice'.
Drop the 'Patient Disposition' and 'Abortion Edit Indicator' columns.
Replace non-finite values in the 'Gender' column with the default value of 0.

### Type Conversion
Convert relevant columns to integer type.

### Save the Preprocessed Data
Print the column names.
Save the preprocessed DataFrame to "Regression-Database.csv" without the index.


