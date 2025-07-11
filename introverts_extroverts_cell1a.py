# ================================================
# Cell 1a: Null Value Check, Imputation, and Visualization
# ================================================
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
