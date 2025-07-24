# ================================================
# Cell 2a: Identify and Visualize Features with Weak Correlation to Attrition
# Purpose: Remove features that have very weak or NaN correlation with Attrition, and visualize this process.
# ================================================
# --- Define weak correlation mask ---
# Features are marked for removal if their correlation with Attrition is:
#   - NaN (not a number, i.e., undefined correlation)
#   - Between -0.1 and 0.1 (exclusive), i.e., very weak correlation
corr_mask = corr_with_personality.isna() | ((corr_with_personality > -0.01) & (corr_with_personality < 0.01))
cols_to_remove = corr_with_personality[corr_mask].index.tolist()

# --- Print the features identified for removal ---
if cols_to_remove:
    print(f"Columns with weak or NaN correlation with Personality: {cols_to_remove}")

# --- Create a new DataFrame for modeling that excludes weakly correlated features ---
# This helps to improve model efficiency and possibly performance by removing irrelevant features.
model_train = train.drop(columns=[col for col in cols_to_remove if col in train.columns])

# --- Prepare a styled DataFrame to visualize correlation values ---
# The table highlights features to be removed in red for easy identification.
cor_df = corr_with_personality.to_frame(name='Correlation')
cor_df['To Remove'] = cor_df.index.isin(cols_to_remove)
styled = cor_df.style.apply(
    lambda col: ['color: red' if rem else '' for rem in cor_df['To Remove']],
    subset=['Correlation']
).format({'Correlation': "{:.2f}"})

# --- Display the styled correlation table (red = marked for removal) ---
from IPython.display import display
display(styled)
