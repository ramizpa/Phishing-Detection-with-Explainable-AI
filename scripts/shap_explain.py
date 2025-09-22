import pandas as pd
import shap
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("../output/cleaned_multi_class_url_dataset.csv")

# Prepare features and label
X = df.drop(columns=["Label", "URL_Type_obf_Type"])
y = df["Label"]

# Train/test split (must match your previous split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load saved XGBoost model
model = joblib.load("../models/xgboost_model.pkl")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values for the test set
print("Calculating SHAP values...")
shap_values = explainer.shap_values(X_test)

# Summary Plot (global feature importance)
print("Creating summary plot...")
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("../reports/shap_summary_plot.png", dpi=300)
print("SHAP summary plot saved to /reports/shap_summary_plot.png")

# Force Plot (individual prediction)
print("Creating force plot...")
i = 0  # Index of the sample to explain

# For multiclass models, pick the target class (e.g., class 0)
expected_value = explainer.expected_value[0]
shap_value = shap_values[0][i]

# Create SHAP Explanation object
explanation = shap.Explanation(
    values=shap_value,
    base_values=expected_value,
    data=X_test.iloc[i],
    feature_names=X_test.columns.tolist()
)

# Create waterfall plot
plt.figure()
shap.plots.waterfall(explanation, max_display=15, show=False)
plt.tight_layout()
plt.savefig("../reports/shap_waterfall_plot.png", dpi=300)
print("SHAP waterfall plot saved to /reports/shap_waterfall_plot.png")