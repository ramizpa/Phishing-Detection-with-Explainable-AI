import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load cleaned dataset
df = pd.read_csv("../output/cleaned_multi_class_url_dataset.csv")

# Drop original text labels
X = df.drop(columns=["Label", "URL_Type_obf_Type"])
y = df["Label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize Logistic Regression
clf = LogisticRegression(
    max_iter=500,
    solver='lbfgs',
    multi_class='multinomial',
    random_state=42
)

# Train model
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(clf, "../models/logistic_model.pkl")
print("Model saved to /models/logistic_model.pkl")
