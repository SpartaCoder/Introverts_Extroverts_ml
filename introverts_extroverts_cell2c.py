# ================================================
# Cell 2c: ML Model Metrics DataFrame Creation
# Purpose: Provide a standardized DataFrame to store and compare evaluation metrics
#          for each machine learning model used in the attrition prediction workflow.
# ================================================
# --- Step 1: Define the columns for model evaluation metrics ---
# These columns will hold the key metrics for model comparison.
metrics_columns = [
    "ML Model",                  # Name or type of the ML model (e.g., RandomForest)
    "accuracy",                  # Overall accuracy of the model
    "specificity",               # True negative rate
    "sensitivity",               # True positive rate (recall)
    "precision",                 # Precision (positive predictive value)
    "root mean absolute error",  # Error metric (regression-style)
    "mean cv accuracy"           # Mean cross-validation accuracy
]

# --- Step 2: Create an empty DataFrame for storing metrics of each ML model ---
model_metrics_df = pd.DataFrame(columns=metrics_columns)

# --- Step 3: Example for adding results to the DataFrame ---
# To add a row/results for a specific model, use the following pattern:
# model_metrics_df = model_metrics_df.append({
#     "ML Model": "RandomForest",
#     "accuracy": accuracy_value,
#     "specificity": specificity_value,
#     "sensitivity": sensitivity_value,
#     "precision": precision_value,
#     "root mean absolute error": rmae_value,
#     "mean cv accuracy": mean_cv_accuracy_value
# }, ignore_index=True)
#
# Replace the placeholders with actual computed metric values from your model evaluation.

# --- Step 4: Usage ---
# The DataFrame `model_metrics_df` serves as a central location to collect and display
# all metrics for each trained model, making it easy to compare performance at a glance.
