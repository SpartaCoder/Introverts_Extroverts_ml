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
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold  # Data splitting & model validation
from sklearn.metrics import (classification_report,                    # Model evaluation tools
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             ConfusionMatrixDisplay,
                             mean_absolute_error)
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif                             

# --- Import clustering algorithms for unsupervised learning ---
from sklearn.cluster import KMeans          # K-means clustering: partition data into K clusters based on feature similarity
from sklearn.cluster import DBSCAN          # DBSCAN: density-based clustering, useful for irregular cluster shapes and outlier detection
from sklearn.mixture import GaussianMixture # Gaussian Mixture Model: probabilistic clustering using multivariate normal distributions

# --- Load training and test datasets ---
# Assumes 'train.csv' and 'test.csv' are present in the root directory (same as this script)
train = pd.read_csv('train.csv')    # Training data
test = pd.read_csv('test.csv')      # Test data

# The datasets are now loaded as pandas DataFrames: `train` and `test`
# Further data exploration and preprocessing will follow in subsequent cells.

# Renumber the index to start at 1 for the test data frame
test.index = range(1, len(test) + 1)

# --- Summarize each column in the train DataFrame ---
# This provides statistics such as count, mean, std, min, max, and quartiles for numerical columns,
# and count, unique, top, and frequency for object columns.
print("Summary statistics for each column in the train DataFrame:")
print(train.describe(include='all'))  # Include all columns, not just numeric ones
