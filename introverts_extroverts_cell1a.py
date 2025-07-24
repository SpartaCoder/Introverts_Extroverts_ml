# ================================================
# Cell 1a: Null Value Check, Imputation, and Visualization
# ================================================
def encode_string_columns(df, df_name="DataFrame"):
    """
    Encodes string columns by mapping unique string values to integers (sorted alphabetically).
    Leaves null values unchanged.
    Prints a summary of mappings for each string column.
    Returns a new DataFrame with encoded values.
    """
    df_encoded = df.copy()
    print(f"\nString Column Encoding Summary for {df_name}:")
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            unique_vals = sorted([v for v in df[col].dropna().unique()])
            mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}  # Start at 1
            # Output summary of mappings
            print(f"\nColumn: '{col}'")
            for val, idx in mapping.items():
                print(f"  '{val}': {idx}")
            # Replace string values with their integer code, leave nulls as is
            df_encoded[col] = df[col].map(mapping)
    return df_encoded

# --- Apply the encoding to your dataframes ---
train_encoded = encode_string_columns(train, df_name="train")
test_encoded = encode_string_columns(test, df_name="test")

# --- Proceed with your existing null imputation code ---
nulls_imputation_summary(train_encoded, df_name="train (encoded)")
nulls_imputation_summary(test_encoded, df_name="test (encoded)")

# This function checks a DataFrame for null values,
# imputes missing values (mean for numerics, mode for categoricals),
# and generates a summary table describing the imputation.
def nulls_imputation_summary(df, df_name="DataFrame"):
    summary = []  # Collects summary rows for columns with missing values

    for col in df.columns:
        null_count = df[col].isnull().sum()        # Number of missing values
        non_null_count = df[col].notnull().sum()   # Number of present values
        percent_null = (null_count / len(df)) * 100  # Percent of missing values
        
        # If we have missing values in this column, perform imputation
        if null_count > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                # For numeric columns, use the mean for imputation
                impute_value = df[col].mean()
                impute_type = "mean"
            else:
                # For categorical columns, use the mode for imputation
                impute_value = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
                impute_type = "mode"
            # Fill missing values in place
            df[col].fillna(impute_value, inplace=True)
        else:
            impute_value = np.nan
            impute_type = None
        
        # Always calculate column mean and mode for reporting purposes
        col_mean = df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else np.nan
        col_mode = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
        
        # Only add columns with original nulls to the summary table
        if null_count > 0:
            summary.append({
                'column': col,
                'null_count': null_count,
                'non_null_count': non_null_count,
                'percent_null': percent_null,
                'mean_used_for_imputation': col_mean,
                'mode_used_for_imputation': col_mode
            })

    # Create a summary DataFrame for visualization
    summary_df = pd.DataFrame(summary, columns=[
        'column', 'null_count', 'non_null_count', 'percent_null',
        'mean_used_for_imputation', 'mode_used_for_imputation'
    ])
    print(f"\nNull Value Imputation Summary for {df_name}:")
    if not summary_df.empty:
        # Use display() in Jupyter; use print(summary_df) otherwise
        display(summary_df)
    else:
        print("No null values found.")

# Apply null checking and imputation on both train and test DataFrames
nulls_imputation_summary(train, df_name="train")
nulls_imputation_summary(test, df_name="test")
