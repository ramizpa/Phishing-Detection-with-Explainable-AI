import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sys
import os

# Import the feature extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
from feature_extractor import extract_features

# Load model
model = joblib.load("../models/xgboost_model.pkl")

# Load dataset to get column order
df = pd.read_csv("../output/cleaned_multi_class_url_dataset.csv")
X = df.drop(columns=["Label", "URL_Type_obf_Type"])

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Label mapping
numeric_to_text = {0: "benign", 1: "phishing", 2: "spam", 3: "malware", 4: "defacement"}

# Page config
st.set_page_config(page_title="Phishing URL Classifier", layout="wide")

# Title
st.title("🔍 Phishing URL Detection (Enter URL)")

st.markdown("""
Enter a URL below. The system will extract features and predict whether it is:
- benign
- phishing
- spam
- malware
- defacement
""")

url_input = st.text_input("Enter URL:", "")

if url_input:
    st.write("✅ Extracting features...")
    features_dict = extract_features(url_input)

    # Convert to DataFrame
    features_df = pd.DataFrame([features_dict])

    # Reindex to match training columns (fill missing with 0)
    features_df = features_df.reindex(columns=X.columns, fill_value=0)

    # Predict
    pred = model.predict(features_df)[0]
    proba = model.predict_proba(features_df)[0]

    pred_label = numeric_to_text.get(pred, "Unknown")

    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {pred_label}")

    # Show probabilities
    class_labels = model.classes_
    text_labels = [numeric_to_text[c] for c in class_labels]
    proba_df = pd.DataFrame({"Probability": proba}, index=text_labels)
    st.dataframe(proba_df)

    # SHAP explanation
    shap_result = explainer.shap_values(features_df)
    pred_class_idx = np.where(model.classes_ == pred)[0][0]

    # Detect whether shap_result is list or array
    if isinstance(shap_result, list):
        # Classic format: list of arrays
        shap_values_for_class = shap_result[pred_class_idx][0]
        base_value = explainer.expected_value[pred_class_idx]
    else:
        # New format: single array (1, features, classes)
        shap_values_for_class = shap_result.values[0, :, pred_class_idx]
        base_value = shap_result.base_values[0][pred_class_idx]

    # Build Explanation object
    explanation = shap.Explanation(
        values=shap_values_for_class,
        base_values=base_value,
        data=features_df.iloc[0],
        feature_names=features_df.columns.tolist()
    )

    st.write("**Feature Contribution (SHAP):**")
    fig, ax = plt.subplots(figsize=(6, 6))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    st.pyplot(fig)
