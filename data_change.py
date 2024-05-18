import pandas as pd

data = pd.read_csv("NY-2010-Refined-Cleaned.csv", low_memory=False)

#   Mapping encodings
age_group_mapping = {
    '0 to 17': 1,
    '18 to 29': 2,
    '30 to 49': 3,
    '50 to 69': 4,
    '70 or Older': 5
}
gender_mapping = {
    'F': 1,
    'M': 2,
}

race_mapping = {
    'White': 1,
    'Black/African American': 2,
    'Other Race': 3
}

ethnicity_mapping = {
    'Not Span/Hispanic': 1,
    'Spanish/Hispanic': 2
}

admission_mapping = {
    'Emergency': 1,
    'Elective': 2,
    'Urgent': 3,
    'Newborn': 4,
    'Trauma': 5,
}

apr_risk_mortality_mapping = {
    'Minor': 1,
    'Moderate': 2,
    'Major': 3,
    'Extreme': 4
}

apr_medical_surgical_description_mapping = {
    'Medical': 1,
    'Surgical': 2
}

emergency_department_indicator = {
    'Y': 1,
    'N': 0
}

data['Age Group'] = data['Age Group'].map(age_group_mapping)
data['Gender'] = data['Gender'].map(gender_mapping)
data['Race'] = data['Race'].map(race_mapping)
data['Ethnicity'] = data['Ethnicity'].map(ethnicity_mapping)
data = data[data['Length of Stay'] != '120 +']
data['Type of Admission'] = data['Type of Admission'].map(admission_mapping)

#   Dropping rows where 'Patient Disposition' is 'Left Against Medical Advice' and then removing the column
data = data[data['Patient Disposition'] != 'Left Against Medical Advice']
data.drop(columns=['Patient Disposition'], inplace=True)

data['APR Risk of Mortality'] = data['APR Risk of Mortality'].map(apr_risk_mortality_mapping)
data['APR Medical Surgical Description'] = data['APR Medical Surgical Description'].map(
    apr_medical_surgical_description_mapping)
data['Emergency Department Indicator'] = data['Emergency Department Indicator'].map(emergency_department_indicator)

#   Drop rows where 'Abortion Edit Indicator' is 'Y' and then drop the column
data = data[data['Abortion Edit Indicator'] != 'Y']
data.drop(columns=['Abortion Edit Indicator'], inplace=True)

#   Replace non-finite values in 'Gender' column with default value
data['Gender'] = data['Gender'].fillna(0)

#   Convert columns to integer type
data = data.astype({
    'Age Group': int,
    'Gender': int,
    'Race': int,
    'Ethnicity': int,
    'Type of Admission': int,
    'APR Risk of Mortality': int,
    'APR Medical Surgical Description': int,
    'Emergency Department Indicator': int
})

print(data.columns)
data.to_csv("Regression-Database.csv", index=False)
