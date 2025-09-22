# Explainable AI for Cyber Threats: Phishing Detection Using SHAP Interpretations

## Project Overview

This project is part of my M.Tech dissertation at BITS Pilani, focusing on the application of Explainable Artificial Intelligence (XAI) to improve cybersecurity in detecting phishing websites.

While machine learning models achieve high accuracy in phishing detection, they often function as "black boxes," leaving security analysts uncertain about why a URL is classified as phishing or benign. To address this, the project integrates SHAP (SHapley Additive exPlanations) to provide transparent, interpretable, and trustworthy insights into model decisions.

The project is deployed as a Streamlit web application, allowing users to test URLs and understand the reasoning behind predictions.

## Key Features

Phishing Detection Model trained on benchmark datasets (ISCX-URL2016, PhishTank, etc.)

Explainable AI Integration using SHAP to visualize feature importance and decision paths

End-to-End Workflow: Data preprocessing → Feature engineering → Model training → MLflow evaluation → SHAP integration → Web deployment

Streamlit Web Application for real-time phishing detection with interpretability

Reproducible Results with MLflow experiment tracking

## System Architecture

Data Processing: URL parsing, feature extraction (lexical, host-based, content-based features)

Model Training: ML models (Random Forest, XGBoost, LightGBM, etc.) evaluated and optimized

Explainability Layer: SHAP visualizations to highlight feature contributions

Web Application: Streamlit interface for input, prediction, and explanation

Deployment: Containerized deployment with support for cloud hosting

## Results & Insights

Achieved high accuracy (>95%) in phishing URL detection

SHAP visualizations revealed critical features like:

URL length

Use of special characters (@, -, //)

Domain age & reputation

Improved trust and adoption of AI in cybersecurity by making predictions interpretable
