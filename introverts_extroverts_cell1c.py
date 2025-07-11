# ================================================
# Cell 1C: Ensure test and train have same columns and order of columns
# ================================================
# This cell loops through each feature (column) in the `train` DataFrame (except 'Personality')
# and creates a bar chart showing the percentage of records with Personality = "Introvert" for each value/bin,
# or, for high-cardinality categorical columns, simply shows the value counts.
# Assuming 'train' and 'test' DataFrames are already loaded in your environment
# If not, import or define them as needed before running this script

# Target column name
target_column = 'Personality'

# Get columns used for modeling (exclude the target from train)
train_features = [col for col in train.columns if col != target_column]
test_features = test.columns.tolist()

# Find columns that are in train but not in test, excluding the target column
exists_in_train_not_test = [col for col in train_features if col not in test_features]

# Find columns that are in test but not in train
exists_in_test_not_train = [col for col in test_features if col not in train_features]

# Output columns that exist in train but not in test (excluding Attrition)
print("Exists in Train Not Test (excluding target column):")
print(exists_in_train_not_test)

# Output columns that exist in test but not in train
print("Exists in Test Not Train:")
print(exists_in_test_not_train)

# Optionally, remove extra columns from test that do not exist in train
test_aligned = test.copy()
if exists_in_test_not_train:
    test_aligned = test_aligned.drop(columns=exists_in_test_not_train)

# Add missing columns to test (if any), filling with 0 or NaN as appropriate
for col in exists_in_train_not_test:
    # You might want to fill with 0, NaN, or other default values depending on your use case
    test_aligned[col] = 0

# Ensure the order of columns in test matches that of train (features only)
test_aligned = test_aligned[train_features]

# Now, 'test_aligned' has the same features as train, in the same order, and can be safely used for predictions
print("Aligned test DataFrame columns (in same order as train features):")
print(test_aligned.columns.tolist())

# Save or use test_aligned as needed for modeling
# Example: predictions = model.predict(test_aligned)
