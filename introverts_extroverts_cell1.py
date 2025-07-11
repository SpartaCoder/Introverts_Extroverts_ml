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

# --- Import clusters to be used ---


# --- Load training and test datasets ---
# Assumes 'train.csv' and 'test.csv' are present in the root directory (same as this script)
train = pd.read_csv('train.csv')    # Training data
test = pd.read_csv('test.csv')      # Test data

# The datasets are now loaded as pandas DataFrames: `train` and `test`
# Further data exploration and preprocessing will follow in subsequent cells.

# Renumber the index to start at 1 for the test data frame
test.index = range(1, len(test) + 1)
