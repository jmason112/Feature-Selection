#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, SelectPercentile, RFE
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read the data from the dataset.csv file
data = pd.read_csv('dataset.csv')

# Create a list called "y_base" to store the target column
y_base = data['Target'].tolist()

# Create a new data frame without label information
features_only = data.drop(['MD5', 'label', 'Target'], axis=1)

# Split the dataset into training and test classes
X_train, X_test, y_train, y_test = train_test_split(features_only, y_base, test_size=0.3, random_state=42)

# Select the top 15 features using the chi2 method
selector_chi2 = SelectKBest(chi2, k=15)
selector_chi2.fit(X_train, y_train)

# Get the indices of the selected features using chi2
selected_features_chi2 = X_train.columns[selector_chi2.get_support()]

# Print the list of the selected features using chi2
print("Selected features using chi2:", selected_features_chi2.tolist())

# Select the top 15 features using the mutual information method
selector_mi = SelectKBest(mutual_info_classif, k=15)
selector_mi.fit(X_train, y_train)

# Get the indices of the selected features using mutual information
selected_features_mi = X_train.columns[selector_mi.get_support()]

# Print the list of the selected features using mutual information
print("Selected features using mutual information:", selected_features_mi.tolist())

# Select the top 15 features using the selectPercentile method with chi2
selector_percentile = SelectPercentile(chi2, percentile=15)
selector_percentile.fit(X_train, y_train)

# Get the indices of the selected features using selectPercentile
selected_features_percentile = X_train.columns[selector_percentile.get_support()]

# Print the list of the selected features using selectPercentile
print("Selected features using selectPercentile and chi2:", selected_features_percentile.tolist())

# Select the top 15 features using recursive feature elimination with the SVC classifier
estimator = SVC(kernel="linear")
selector_rfe = RFE(estimator, n_features_to_select=15, step=1)
selector_rfe.fit(X_train, y_train)

# Get the indices of the selected features using recursive feature elimination
selected_features_rfe = X_train.columns[selector_rfe.get_support()]

# Print the list of the selected features using recursive feature elimination
print("Selected features using recursive feature elimination with SVC:", selected_features_rfe.tolist())

# Select the top 15 features using recursive feature elimination with the SVC classifier with a coefficient of 1
estimator = SVC(kernel="linear", C=1)
selector_rfe = RFE(estimator, n_features_to_select=15, step=1)
selector_rfe.fit(X_train, y_train)

# Get the indices of the selected features using recursive feature elimination
selected_features_rfe = X_train.columns[selector_rfe.get_support()]

# Print the list of the selected features using recursive feature elimination
print("Selected features using recursive feature elimination with BernoulliNB:", selected_features_rfe.tolist())
