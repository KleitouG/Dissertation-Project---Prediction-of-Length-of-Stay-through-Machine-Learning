import pandas as pd
from category_encoders import TargetEncoder

data = pd.read_csv('Regression-Database.csv')

#   Columns
ordinal_cols = ['Age Group', 'Type of Admission', 'APR Severity of Illness Code', 'APR Risk of Mortality']
non_ordinal_cols = ['Gender', 'Race', 'Ethnicity', 'CCS Diagnosis Code', 'CCS Procedure Code',
                    'APR MDC Code', 'APR Medical Surgical Description', 'Emergency Department Indicator']

#   Targeted encoding
encoder = TargetEncoder(cols=ordinal_cols)
data[ordinal_cols] = encoder.fit_transform(data[ordinal_cols], data['Length of Stay'])

#   Frequency encoding
for col in non_ordinal_cols:
    data[col+"_freq"] = data[col].map(data[col].value_counts(normalize=True))

#   Drop original database
data.drop(non_ordinal_cols, axis=1, inplace=True)

data.to_csv('Regression-Encoded.csv', index=False)
