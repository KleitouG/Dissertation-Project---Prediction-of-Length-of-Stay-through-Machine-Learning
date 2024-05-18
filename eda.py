import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Regression-Database.csv", low_memory=False)

#   Descriptive Statistics
print("Descriptive Statistics for Length of Stay:")
print(data['Length of Stay'].describe())

#   Histogram
plt.figure(figsize=(10, 6))
plt.hist(data['Length of Stay'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Length of Stay')
plt.xlabel('Length of Stay (Days)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#   Box plot of LoS by age group
plt.figure(figsize=(10, 6))
sns.boxplot(x='Age Group', y='Length of Stay', data=data, palette='muted')
plt.title('Length of Stay by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Length of Stay (Days)')
plt.show()

#   Print number of entries in each age group for the box plot
print("Number of Entries in each Age Group for the Box Plot:")
print(data['Age Group'].value_counts())

#   Pie chart of gender distribution
plt.figure(figsize=(8, 6))
gender_distribution = data['Gender'].value_counts()
print("Gender Distribution:")
print(gender_distribution)  # Print gender distribution
gender_distribution_labels = ['Female', 'Male']
plt.pie(gender_distribution, labels=gender_distribution_labels, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
plt.title('Gender Distribution')
plt.show()

#   Bar plot of LoS by gender
plt.figure(figsize=(10, 6))
gender_length_of_stay = data.groupby('Gender')['Length of Stay'].mean()
print("Average Length of Stay by Gender:")
print(gender_length_of_stay)
sns.barplot(x=gender_length_of_stay.index, y=gender_length_of_stay.values, palette='muted')
plt.title('Length of Stay by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Length of Stay (Days)')
plt.show()

#   Pie chart of ethnicity distribution
plt.figure(figsize=(8, 6))
ethnicity_distribution = data['Ethnicity'].value_counts()
print("Ethnicity Distribution:")
print(ethnicity_distribution)  # Print ethnicity distribution
ethnicity_distribution_labels = ['Not Span/Hispanic', 'Spanish/Hispanic']
plt.pie(ethnicity_distribution, labels=ethnicity_distribution_labels, autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
plt.title('Ethnicity Distribution')
plt.show()

#   Bar plot of LoS by ethnicity
plt.figure(figsize=(12, 8))
ethnicity_length_of_stay = data.groupby('Ethnicity')['Length of Stay'].mean()
print("\nAverage Length of Stay by Ethnicity:")
print(ethnicity_length_of_stay)
sns.barplot(x=ethnicity_length_of_stay.index, y=ethnicity_length_of_stay.values, palette='muted')
plt.title('Length of Stay by Ethnicity')
plt.xlabel('Ethnicity')
plt.ylabel('Average Length of Stay (Days)')
plt.show()

# Pie chart of race distribution
plt.figure(figsize=(8, 6))
race_distribution = data['Race'].value_counts()
print("Race Distribution:")
print(race_distribution)  # Print race distribution
race_distribution_labels = ['White', 'Black/African American', 'Other Race']
plt.pie(race_distribution, labels=race_distribution_labels, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Race Distribution')
plt.show()

#   Bar plot of LoS by race
plt.figure(figsize=(12, 8))
race_length_of_stay = data.groupby('Race')['Length of Stay'].mean()
print("\nAverage Length of Stay by Race:")
print(race_length_of_stay)
sns.barplot(x=race_length_of_stay.index, y=race_length_of_stay.values, palette='muted')
plt.title('Length of Stay by Race')
plt.xlabel('Race')
plt.ylabel('Average Length of Stay (Days)')
plt.show()

#   Correlation Analysis
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.xticks(rotation=65, fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
