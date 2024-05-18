import pandas as pd

data = pd.read_csv("NY-Hospital-Inpatients-2010.csv", low_memory=False)

#   Columns for the new dataset
new_data = data[['Facility ID','Age Group', 'Gender', 'Race', 'Ethnicity', 'Length of Stay', 'Type of Admission','Patient Disposition',
                 'CCS Diagnosis Code', 'CCS Procedure Code','APR MDC Code', 'APR Severity of Illness Code', 'APR Risk of Mortality',
                 'APR Medical Surgical Description', 'Abortion Edit Indicator', 'Birth Weight',
                 'Emergency Department Indicator']]

#   New dataset
new_data.to_csv("NY-2010-Refined.csv", index=False)