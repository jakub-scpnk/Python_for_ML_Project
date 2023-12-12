#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 13:42:39 2023

@author: jakubsz
"""


import pandas as pd

data = pd.read_csv('data.csv', delimiter=',')

# Calculating the correlation matrix
corr_data = data.corr()

#------------------------------------------------------------------------------
# Determining wchih featurest to cut
#------------------------------------------------------------------------------
abs_corr_matrix = data.corr().abs()

# Initialize an empty list to store the pairs
high_corr_pairs = []

# Iterate over the correlation matrix
for i, coulumn in enumerate(abs_corr_matrix.columns):
    for j in range(i):
        # Check if absolute correlation is greater than 0.9 and not equal to 1
        # (to avoid self-correlation)
        if (abs_corr_matrix.iloc[i, j] > 0.98 and
            abs_corr_matrix.columns[i] != abs_corr_matrix.columns[j]):
            high_corr_pairs.append((abs_corr_matrix.columns[i],
                                    abs_corr_matrix.columns[j],
                                    abs_corr_matrix.iloc[i, j]
                                    ))

# Convert the list to a DataFrame for better readability
high_corr_pairs_df = pd.DataFrame(high_corr_pairs, columns=['Feature 1',
                                                            'Feature 2',
                                                            'Correlation'
                                                            ])

# Print the resulting DataFrame
print(high_corr_pairs_df.to_string())

# Extracting only 'Feature 2' names to remove from the dataset
features_to_remove = {pair[1] for pair in high_corr_pairs}

# Removing these features from the dataset
reduced_data = data.drop(columns=features_to_remove)

#------------------------------------------------------------------------------
# Finding Features with highest Corellation to target
#------------------------------------------------------------------------------
# Calculate the absolute correlation of each feature with the target variable
# 'Bankrupt?', then drop Bancrupt? and sort coralations by highest value
correlations = reduced_data.corr().abs()
target_correlations = correlations['Bankrupt?'].drop(labels=['Bankrupt?'])
sorted_correlations = target_correlations.sort_values(ascending=False)
top_6_features = sorted_correlations.head(6)
print('\nFeatures with the highest correlation to the target:')
print(top_6_features)
print(top_6_features.to_latex())

#------------------------------------------------------------------------------
# Creating a new data set with only 6 features and target value
#------------------------------------------------------------------------------

# Selecting only the top 6 features and the target value
selected_features = top_6_features.index.tolist()
selected_features.append('Bankrupt?')  # Adding the target variable

# Creating the new dataset with only these selected features
new_dataset = reduced_data[selected_features]

# Saving the new dataset to a CSV file (optional)
new_dataset.to_csv('prepared_data.csv', index=False)

# Display the first few rows of the new dataset
#print(new_dataset.head())
