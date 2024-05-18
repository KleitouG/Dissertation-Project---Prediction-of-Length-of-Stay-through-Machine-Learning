import pandas as pd

data = pd.read_csv("Regression-SMOGN.csv")

#   Count the instances of each number in the LoS column
length_of_stay_counts = data['Length of Stay'].value_counts()

#   Sorting
length_of_stay_counts_sorted = length_of_stay_counts.sort_index()

#   Display
for length, count in length_of_stay_counts_sorted.items():
    print(f"Length of Stay: {length}, Count: {count}")
