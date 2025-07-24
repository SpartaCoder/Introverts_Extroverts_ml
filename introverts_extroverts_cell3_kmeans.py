# ================================================
# Cell 3: KMeans - Model Training, Evaluation, and Metrics Collection
# Purpose:
#   - Train a K Means clustering model for introvert/extrovert identification
#   - Evaluate model performance using multiple metrics
#   - Store results for comparison with other models
# ================================================
# --- Prepare features (X) and target (y) ---
# The model uses the 'train_unbalanced' DataFrame with relevant features and the target column 'Personality'.
X = train_unbalanced.drop('Prediction', axis=1)
y = train_unbalanced['Prediction']

# --- Split the data into training and test sets (80% train, 20% test) ---
# Stratify ensures the class distribution is similar in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Initialize kmeans clustering model ---
kmeans = KMeans(n_clusters=2, random_state=42, max_iter=3000)

# Train (fit) the model on the training data
kmeans.fit(X_train)

# --- Make predictions on the test set ---
y_pred = kmeans.predict(X_test)

# --- Print accuracy and detailed classification report ---
print("KMeans Clustering Test Accuracy:", accuracy_score(y_test, y_pred))
print("Cluster Report:\n", classification_report(y_test, y_pred))

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Annotate each cell with the numeric value
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# --- Display the confusion matrix visually ---
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - KMeans Clustering')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_test)))
plt.xticks(tick_marks, np.unique(y_test))
plt.yticks(tick_marks, np.unique(y_test))

# --- Perform 10-fold cross-validation and display accuracy results ---
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train)
    y_pred = kmeans.predict(X_test)
    
    # KMeans labels are arbitrary; accuracy may need aligning labels
    # Align cluster labels with true labels for best accuracy
    from scipy.stats import mode
    labels = np.zeros_like(y_pred)
    for i in range(2):
        mask = (y_pred == i)
        labels[mask] = mode(y_test[mask])[0]
    
    acc = accuracy_score(y_test, labels)
    accuracies.append(acc)

print("10-fold cross-validation accuracies for KMeans:")
for i, acc in enumerate(accuracies, 1):
    print(f"Fold {i}: {acc:.2f}")

print(f"Mean Accuracy: {np.mean(accuracies):.2f}")

# --- Encode the labels for error calculation ---
# LabelEncoder ensures string labels are converted to integers for MAE calculation.
le = LabelEncoder()
y_test_num = le.fit_transform(y_test)
y_pred_num = le.transform(y_pred)  # Use transform to match the same mapping

# --- Calculate and print Root Mean Absolute Error (RMAE) ---
mae = mean_absolute_error(y_test_num, y_pred_num)
rmae = np.sqrt(mae)
print("Root Mean Absolute Error (RMAE):", rmae)

# --- Compute additional model metrics for reporting and comparison ---
# Extract confusion matrix components for metric calculations
# cm is [[TN, FP], [FN, TP]]
TN, FP, FN, TP = cm.ravel()
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# --- Prepare results dictionary for this model ---
new_metrics = {
    "ML Model": "Kmeans Cluster",
    "accuracy": accuracy,
    "specificity": specificity,
    "sensitivity": sensitivity,
    "precision": precision,
    "root mean absolute error": np.sqrt(mae),
    "mean cv accuracy": np.mean(cv_scores)
}

# --- Append these metrics to the central model_metrics_df DataFrame for comparison ---
model_metrics_df = pd.concat(
    [model_metrics_df, pd.DataFrame([new_metrics])],
    ignore_index=True
)

# --- (Optional) Save the updated metrics DataFrame to disk for later use ---
# model_metrics_df.to_pickle('model_metrics_df.pkl')
# model_metrics_df.to_csv('model_metrics_df.csv', index=False)

# --- Predict Personality on the Test DataFrame from Cell 1 and Store Results ---
# Ensure 'test' has the same features/columns as X_train (may require preprocessing)
KMeansOutput = test.copy()
KMeansOutput['Personality_Prediction'] = logreg.predict(test[X_train.columns])
