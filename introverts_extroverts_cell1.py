# ================================================
# Cell 1: Import libraries and load data
# Project: Introverts Extroverts ML (IBM Kaggle Dataset)
# Description: Prepare environment, import core libraries, and load datasets
# ================================================
# --- Import essential Python libraries ---
import pandas as pd                           # Data manipulation and analysis
import numpy as np                            # Numerical operations
import matplotlib.pyplot as plt               # Data visualization
import os                                     # Operating System

# --- Set Kaggle username and key for submissions ---
os.environ['KAGGLE_USERNAME'] = 'kaggleusername'
os.environ['KAGGLE_KEY'] = 'kaggle_key'


# --- Import scikit-learn utilities for ML workflow ---
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score  # Data splitting & model validation
from sklearn.metrics import (classification_report,                    # Model evaluation tools
                             accuracy_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             mean_absolute_error)

# --- Import clustering algorithms for unsupervised learning ---
from sklearn.cluster import KMeans          # K-means clustering: partition data into K clusters based on feature similarity
from sklearn.cluster import DBSCAN          # DBSCAN: density-based clustering, useful for irregular cluster shapes and outlier detection
from sklearn.mixture import GaussianMixture # Gaussian Mixture Model: probabilistic clustering using multivariate normal distributions

# --- Load training and test datasets ---
# Assumes 'train.csv' and 'test.csv' are present in the root directory (same as this script)
train = pd.read_csv('train.csv')    # Training data
test = pd.read_csv('test.csv')      # Test data

# --- Convert string columns to integer values in both train and test datasets ---
# This step scans for columns with object (string) data types and applies Label Encoding to them.
# Null values are left unchanged.
for df in [train, test]:
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].notnull().any():  # Only encode if at least one non-null value exists
            le = LabelEncoder()
            # Fit on non-null values, transform only non-null, leave nulls untouched
            non_null = df[col].notnull()
            df[col] = df[col].where(~non_null, le.fit_transform(df.loc[non_null, col]))
            # The column will now be integer where originally string, nulls remain as NaN


# The datasets are now loaded as pandas DataFrames: `train` and `test`
# Further data exploration and preprocessing will follow in subsequent cells.

# Renumber the index to start at 1 for the test data frame
test.index = range(1, len(test) + 1)

# --- Summarize each column in the train DataFrame ---
# This provides statistics such as count, mean, std, min, max, and quartiles for numerical columns,
# and count, unique, top, and frequency for object columns.
print("Summary statistics for each column in the test DataFrame:")
print(train.describe(include='all'))  # Include all columns, not just numeric ones
