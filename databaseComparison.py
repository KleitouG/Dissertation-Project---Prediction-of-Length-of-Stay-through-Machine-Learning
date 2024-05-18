import pandas as pd

#   Insert Column to compare here
column_to_compare = "Length of Stay"

#   Database 1
data_1 = pd.read_csv("Regression-Database.csv")

#   Database 2
data_2 = pd.read_csv("Regression-Smote-20k.csv")

#   Database 1 column to compare
correlation_1 = data_1.groupby('Ethnicity')['Length of Stay'].mean()
print(f"Average of {correlation_1} of Database 1: ")
print(correlation_1)

#   Database 2 column to compare
correlation_2 = data_2.groupby('Ethnicity')['Length of Stay'].mean()
print(f"\nAverage of {correlation_2} of Database 2: ")
print(correlation_2)

#   Category with the smallest correlation coefficient
category_1 = correlation_1.idxmin()
category_2 = correlation_2.idxmin()

print("\nCategory with smallest average Length of Stay before undersampling and SMOTE: ", category_1)
print("Category with smallest average Length of Stay after undersampling and SMOTE: ", category_2)



#   Check if column exists in both datasets

if column_to_compare in data_1.columns and column_to_compare in data_2.columns:
    #   Get unique values of the column in each dataset
    unique_values_data1 = data_1[column_to_compare].unique()
    unique_values_data2 = data_2[column_to_compare].unique()

    #   Check if unique values in both datasets are the same
    if set(unique_values_data1) == set(unique_values_data2):
        print(f"The unique values in column '{column_to_compare}' are the same in both datasets.")
    else:
        print(f"The unique values in column '{column_to_compare}' are different between the two datasets.")
else:
    print(f"Column '{column_to_compare}' is not present in both datasets.")

