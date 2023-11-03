#!/usr/bin/env python3

import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

# Read the data from the dataset.csv file
data = pd.read_csv('dataset.csv')

# Create a list called "y_base" to store the target column
y_base = data['Target'].tolist()

# Create a new data frame without label information
features_only = data.drop(['MD5', 'label', 'Target'], axis=1)

# Calculate the class distribution
base_class_distribution = Counter(y_base)

# Print the base class distribution
print("Base class distribution:", base_class_distribution)

# Oversample the minority class using RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_oversampled, y_oversampled = oversampler.fit_resample(features_only, y_base)

# Calculate the oversampled class distribution
oversampled_class_distribution = Counter(y_oversampled)

# Print the oversampled class distribution
print("Oversampled class distribution:", oversampled_class_distribution)
