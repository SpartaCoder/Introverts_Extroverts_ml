# ================================================
# Cell 2: Compute and Visualize Correlation Matrix
# Purpose: Identify features with little or no correlation to Personality
# Dependencies: Uses only matplotlib (no seaborn needed)
# ================================================
# --- Convert Personality to numeric if necessary ---
# This ensures the Personality column is numeric for correlation analysis:
# 'Introvert' becomes 1, 'Extrovert' becomes 0.
if train['Personality'].dtype == object:
    train['Personality_numeric'] = train['Personality'].map({'Introvert': 1, 'Extrovert': 0})
    target_col = 'Personality_numeric'
else:
    target_col = 'Personality'

# --- Compute correlation matrix (numeric columns only) ---
cor_matrix = train.corr(numeric_only=True)

# --- Extract correlation of each feature with Personality ---
# Remove self-correlation and sort features by absolute correlation value (strongest to weakest)
corr_with_personality = (
    cor_matrix[target_col]
    .drop(target_col)
    .sort_values(key=abs, ascending=False)
)

print("Correlation of features with Personality:")
print(corr_with_personality)

# --- Identify features with very low correlation (absolute value < 0.01) ---
no_corr_features = corr_with_personality[abs(corr_with_personality) < 0.01].index.tolist()
print("\nFeatures with near-zero correlation to Personality (|corr| < 0.01):")
print(no_corr_features)

# --- Visualize the entire correlation matrix as a heatmap ---
plt.figure(figsize=(12, 8))
plt.imshow(cor_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(cor_matrix.columns)), cor_matrix.columns, rotation=90)
plt.yticks(range(len(cor_matrix.index)), cor_matrix.index)
plt.title('Correlation Matrix (All Features)')
plt.tight_layout()
plt.show()

# --- Visualize correlations with Personality as a horizontal bar chart ---
plt.figure(figsize=(4, len(corr_with_attrition) * 0.4))
plt.barh(corr_with_personality.index, corr_with_personality.values)
plt.title('Feature Correlation with Personality')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()
