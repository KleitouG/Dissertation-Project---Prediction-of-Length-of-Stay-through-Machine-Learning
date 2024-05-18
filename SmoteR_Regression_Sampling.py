import pandas as pd
from smogn.smoter import smoter  # Import smoter function from SMOGN package

data = pd.read_csv("Regression-Database.csv", index_col=None)

#   List to store the resampled dataframes
resampled_data = []

#   Iterate over unique lengths of stay
for length_of_stay in data['Length of Stay'].unique():
    #   Select records for the current length of stay
    subset_data = data[data['Length of Stay'] == length_of_stay]

    #   Resample using SMOGN if the number of unique classes in target variable is greater than 1
    if len(subset_data['Length of Stay'].unique()) > 1:
        #   Resample using SMOGN
        smogn_data = smoter(data=subset_data, y='Length of Stay')
        resampled_data.append(smogn_data)
    else:
        #   If there is only one class, no resampling is needed
        resampled_data.append(subset_data)

#   Combine resampled subsets into a single DataFrame
resampled_data = pd.concat(resampled_data)

resampled_data.to_csv("Regression-SMOGN.csv", index=False)
