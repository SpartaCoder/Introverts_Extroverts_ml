# ================================================
# Cell 1b: Bar Charts - % Personality ("Introverts") by Feature
# ================================================
# This cell loops through each feature (column) in the `train` DataFrame (except 'Personality')
# and creates a bar chart showing the percentage of records with Personality = "Introvert" for each value/bin,
# or, for high-cardinality categorical columns, simply shows the value counts.

for col in train.columns:
    if col == 'Personality':
        continue  # Skip the target column itself
    
    col_data = train[col]  # Extract the current column's data

    # If the column is integer type and has more than 10 unique values, bin into 10 ranges for readability
    if pd.api.types.is_integer_dtype(col_data) and col_data.nunique() > 10:
        binned = pd.cut(col_data, bins=10)
        groupby_col = binned
        plot_type = 'Personality'
    # If the column is categorical (not integer) and has 30 or fewer unique values, group by value
    elif not pd.api.types.is_integer_dtype(col_data) and col_data.nunique() <= 30:
        groupby_col = col_data
        plot_type = 'Personality'
    # If the column is categorical with more than 30 unique values, plot value counts (not attrition)
    elif not pd.api.types.is_integer_dtype(col_data) and col_data.nunique() > 30:
        value_counts = col_data.value_counts().sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        value_counts.plot(kind='bar', color='orange')
        plt.ylabel('Count')
        plt.title(f'Value Counts for {col} (Top {len(value_counts)})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        continue  # Skip to next column after plotting value counts
    else:
        # Integer columns with 10 or fewer unique values, treat like categorical
        groupby_col = col_data
        plot_type = 'Personality'
    
    # For columns marked for attrition plotting, calculate percent "Yes" for each group/bin
    if plot_type == 'attrition':
        attrition_rate = (
            train.groupby(groupby_col)['Personality']
            .apply(lambda x: (x == "Introvert").mean() * 100)
            .sort_index()
        )
        plt.figure(figsize=(8, 5))
        attrition_rate.plot(kind='bar', color='teal')
        plt.ylabel('% Personality = "Introvert"')
        plt.title(f'Personality Rate by {col}')
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
