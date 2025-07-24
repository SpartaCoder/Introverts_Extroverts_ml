# ================================================
# Cell 2b: Create a Balanced Dataset and Visualize Feature with NULL values
# Context: 
#   - Balance data by matching the number of "Personality = Introvert" and "Personality = Extrovert" records.
#   - Include only features with meaningful correlation to Personality (from cor_df).
#   - Visualize the completeness (non-null counts) of features by Personality class.
# ================================================

# --- Step 0: Define columns to keep based on prior correlation analysis ---
# cor_df is assumed to be available from the previous analysis (cell2a), with "To Remove" column.
cols_to_keep = cor_df[cor_df["To Remove"] == False].index.tolist()

# Always ensure 'Personality' column is included for filtering and visualization
if 'Personality' not in cols_to_keep:
    cols_to_keep.append('Personality')

# --- Step 1: Select all records where Personality == "Introvert" (the minority class), keeping only relevant columns ---
yes_records = train.loc[train['Personality'] == 'Introvert', cols_to_keep].copy()

# --- Step 2: Randomly select an equal number of "Extrovert" records for a balanced dataset ---
num_yes = len(yes_records)
no_candidates = train.loc[train['Personality'] == 'Extrovert', cols_to_keep]
no_records = no_candidates.sample(n=num_yes, random_state=42).copy()  # random_state ensures reproducibility

# --- Step 3: Concatenate "Yes" and "No" samples to create a balanced DataFrame ---
train_balanced = pd.concat([yes_records, no_records], axis=0).reset_index(drop=True)

# --- Step 4: Create an unbalanced dataset with the same selected columns for comparison ---
train_unbalanced = train[cols_to_keep].copy()

# --- Step 5: Create rf_test DataFrame for inference (copy of test with selected columns, excluding 'Personality') ---
# This DataFrame will be used for model predictions, matching the features of the training set except for the target.
rf_test = test[[col for col in cols_to_keep if col != 'Personality']].copy()

# --- Step 6: Calculate counts of non-null values for each feature, broken down by Personality value ---
# This helps visualize the "missingness" or completeness of each feature across classes.
features = [col for col in cols_to_keep if col != 'Personality']  # Exclude target column from feature list
counts_by_personality = (
    train_balanced.groupby('Personality')[features]
    .apply(lambda df: df.notnull().sum())
    .T  # Transpose so features are on the x-axis for plotting
)

# --- Step 7: Visualize feature missingness as a stacked bar chart ---
counts_by_personality.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    color=['#1f77b4', '#ff7f0e'],  # Color for "No" and "Yes"
    edgecolor='black'
)
plt.title('Non-Null Record Count by Feature (Stacked by Personality)')
plt.xlabel('Feature')
plt.ylabel('Non-Null Record Count')
plt.legend(title='Personality')
plt.tight_layout()
plt.show()
