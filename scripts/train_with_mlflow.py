import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("output/cleaned_multi_class_url_dataset.csv")  # Update path if needed
X = df.drop(columns=["Label", "URL_Type_obf_Type"])
y = df["Label"]

# Clean NaNs and Infs
X = X.replace([np.inf, -np.inf], np.nan).dropna()
y = y.loc[X.index]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model list
models = {
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=300),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
}

# Set experiment
mlflow.set_experiment("Phishing Detection Models")

# Train and log each model
for model_name, model in models.items():
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Log model and metrics
        mlflow.log_param("model", model_name)
        mlflow.log_metrics({
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1
        })

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)

        print(f"{model_name} logged to MLflow.")
