# ================================================
# Cell 1: Import libraries and load data
# Project: Introverts Extroverts ML (IBM Kaggle Dataset)
# Description: Prepare environment, import core libraries, and load datasets
# ================================================

# --- Import essential Python libraries for data science tasks ---
import pandas as pd                           # Powerful library for data manipulation and analysis
import numpy as np                            # Provides support for large, multi-dimensional arrays and matrices
import matplotlib.pyplot as plt               # Used for creating static, animated, and interactive visualizations
import os                                     # OS module for interacting with the operating system

# --- Set up Kaggle credentials for API access (if submitting or downloading datasets) ---
os.environ['KAGGLE_USERNAME'] = 'kaggleusername'   # Replace with your own Kaggle username
os.environ['KAGGLE_KEY'] = 'kaggle_key'            # Replace with your own Kaggle API key

# --- Import scikit-learn utilities for machine learning workflow ---
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder   # Encoding and scaling data
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold  # Data splitting and validation
from sklearn.metrics import (
    classification_report,      # Generates a text report showing main classification metrics
    accuracy_score,             # Calculates accuracy classification score
    precision_score,            # Calculates precision for classification tasks
    recall_score,               # Calculates recall for classification tasks
    f1_score,                   # Calculates F1 score (harmonic mean of precision and recall)
    confusion_matrix,           # Computes confusion matrix to evaluate accuracy of classification
    ConfusionMatrixDisplay,     # Utility for displaying confusion matrices
    mean_absolute_error         # Average absolute difference between predicted and true values (for regression)
)
from sklearn.svm import SVC                         # Support Vector Classifier for classification tasks
from sklearn.compose import ColumnTransformer        # Applies transformers to columns of a DataFrame
from sklearn.pipeline import Pipeline                # Assembles several steps into one ML pipeline
from sklearn.model_selection import GridSearchCV     # Hyperparameter tuning using cross-validated grid search
from sklearn.feature_selection import SelectKBest, f_classif  # Feature selection methods

# --- Import clustering algorithms for unsupervised learning ---
from sklearn.cluster import KMeans          # K-means: partitions data into K clusters based on feature similarity
from sklearn.cluster import DBSCAN          # DBSCAN: density-based clustering (detects clusters of varying shape)
from sklearn.mixture import GaussianMixture # Gaussian Mixture: probabilistic model for representing clusters

# --- Load training and test datasets (assumes CSV files are in the current directory) ---
train = pd.read_csv('train.csv')    # Load training data as a DataFrame
test = pd.read_csv('test.csv')      # Load test data as a DataFrame

# At this point, `train` and `test` DataFrames are ready for analysis

# --- Set the test DataFrame index to start at 1 instead of 0 ---
test.index = range(1, len(test) + 1)

# --- Print a summary of each column in the training data ---
# Shows statistics for numerical columns (count, mean, std, min, max, quartiles)
# and for object columns (count, unique values, top value, frequency)
print("Summary statistics for each column in the train DataFrame:")
print(train.describe(include='all'))  # 'include=all' covers both numeric and categorical columns
