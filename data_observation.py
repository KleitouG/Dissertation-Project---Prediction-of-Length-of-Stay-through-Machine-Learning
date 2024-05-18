import pandas as pd

data = pd.read_csv("NY-Hospital-Inpatients-2010.csv", low_memory=False, dtype={"Facility ID": str})

total_nan_count = 0

#   Get unique values and count of unique values for each column
for column in data.columns:
    unique_values = data[column].unique()
    num_unique = data[column].nunique()
    print(f"Column: {column}")
    print(f"Number of unique values: {num_unique}")
    print(unique_values)
    if column == 'Length of Stay':
        count_120_plus = (data[column] == '120 +').sum()
        print(f"Number of '120 +' entries in 'Length of Stay' column: {count_120_plus}")
    if 'Left Against Medical Advice' in unique_values:
        count_left_against_medical_advice = (data[column] == 'Left Against Medical Advice').sum()
        print(f"Number of 'Left Against Medical Advice' entries: {count_left_against_medical_advice}")
    count_nan = data[column].isna().sum()
    print(f"Number of NaN entries: {count_nan}")
    total_nan_count += count_nan  # Increment total count of NaN values
    print()

print(f"Total number of NaN entries across all columns: {total_nan_count}")

#   Count 'Y' and 'N' values in the 'Abortion' column
count_Y_abortion = data['Abortion Edit Indicator'].value_counts()['Y']
count_N_abortion = data['Abortion Edit Indicator'].value_counts()['N']
print(f"Number of 'Y' entries in Abortion Edit Indicator column: {count_Y_abortion}")
print(f"Number of 'N' entries in Abortion Edit Indicator column: {count_N_abortion}")

#   Count 'Y' and 'N' values in the Emergency Department column

count_Y_emergency = data['Emergency Department Indicator'].value_counts()['Y']
count_N_emergency = data['Emergency Department Indicator'].value_counts()['N']
print(f"Number of 'Y' entries in Emergency Department Indicator column: {count_Y_emergency}")
print(f"Number of 'N' entries in Emergency Department Indicator column: {count_N_emergency}")

num_columns = len(data.columns)
print(f"Number of columns: {num_columns}")
