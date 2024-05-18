import pandas as pd
from imblearn.over_sampling import SMOTE

data = pd.read_csv("Regression-Database.csv", index_col=None)

#   Maximum number of records per day
max_records_per_day = 20000

#   List to store the downsampled dataframes
downsampled_data = []

#   Iterate over unique lengths of stay
for length_of_stay in data['Length of Stay'].unique():
    #   Select records for the current length of stay
    subset_data = data[data['Length of Stay'] == length_of_stay]

    #   Check if number of records exceeds the maximum
    if len(subset_data) > max_records_per_day:
        #   If the number of records is greater than the maximum, downsample to the maximum
        downsampled_subset = subset_data.sample(n=max_records_per_day, replace=False, random_state=42)
    else:
        #   If the number of records is less than or equal to the maximum, no downsampling needed
        downsampled_subset = subset_data.copy()

    #   If the number of unique classes in target variable is greater than 1, use SMOTE
    if len(downsampled_subset['Length of Stay'].unique()) > 1:
        smote = SMOTE(sampling_strategy={length_of_stay: max_records_per_day})
        features = downsampled_subset.drop(columns=['Length of Stay'])
        target = downsampled_subset['Length of Stay']

        #   Check if SMOTE is needed
        if len(downsampled_subset) < max_records_per_day:
            features_resampled, target_resampled = smote.fit_resample(features, target)
        else:
            features_resampled, target_resampled = features, target

        #   Combine features and target into a DataFrame
        downsampled_subset = pd.concat(
            [pd.DataFrame(features_resampled), pd.DataFrame({'Length of Stay': target_resampled})], axis=1)

    #   If the downsampled subset has fewer than 100 entries, randomly oversample to 100
    if len(downsampled_subset) < max_records_per_day:
        oversampled_subset = downsampled_subset.sample(n=max_records_per_day - len(downsampled_subset), replace=True)
        downsampled_subset = pd.concat([downsampled_subset, oversampled_subset])

    #   Append the downsampled subset to the list
    downsampled_data.append(downsampled_subset)

#   Combine downsampled subsets into a single DataFrame
downsampled_data = pd.concat(downsampled_data)

downsampled_data.to_csv("Regression-Smote-20k.csv", index=False)
