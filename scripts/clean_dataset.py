# clean_dataset.py
import pandas as pd

# Load the dataset
df = pd.read_csv("../data/multi_class_url_dataset.csv")

# Preview first rows
print("Preview of dataset:")
print(df.head())

# List columns
print("Columns in the dataset:")
print(df.columns)

# Label encoding mapping
label_mapping = {
    "benign": 0,
    "phishing": 1,
    "spam": 2,
    "malware": 3,
    "defacement": 4
}

# Encode 'labels' column
df["Label"] = df["URL_Type_obf_Type"].map(label_mapping)

# Drop rows with unmapped labels
df = df.dropna(subset=["Label"])

# Drop any rows with missing features
df = df.dropna()

# Reset index
df = df.reset_index(drop=True)

# Show label counts
print("Label counts:")
print(df["Label"].value_counts())

# Save cleaned dataset
df.to_csv("../output/cleaned_multi_class_url_dataset.csv", index=False)

print("Cleaned dataset saved to /output/cleaned_multi_class_url_dataset.csv")
