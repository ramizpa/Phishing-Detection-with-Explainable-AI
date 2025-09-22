# inspect_columns.py

import pandas as pd

# Load cleaned dataset
df = pd.read_csv("../output/cleaned_multi_class_url_dataset.csv")

# Show dtypes
print("Data types of each column:")
print(df.dtypes)

# Show unique values for any non-numeric columns
non_numeric = df.select_dtypes(include=["object"])
print("\n Non-numeric columns and sample values:")
for col in non_numeric.columns:
    print(f"\nColumn: {col}")
    print(df[col].unique()[:10])
