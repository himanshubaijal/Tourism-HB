import pandas as pd

RAW_PATH = "/content/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that key expected columns are present
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact",
    "CityTier", "DurationOfPitch", "Occupation", "Gender",
    "MaritalStatus", "Designation", "MonthlyIncome"
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("Target distribution (ProdTaken):")
print(df["ProdTaken"].value_counts())
