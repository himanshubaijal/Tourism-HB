import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure paths
RAW_PATH = "/content/tourism.csv"
OUTPUT_DIR = "/content/tourism_project/model_building/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Apply standardizations we established earlier
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

if 'MaritalStatus' in df.columns:
    df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')

# Drop unnecessary columns that shouldn't be fed into model
columns_to_drop = ["Unnamed: 0", "CustomerID"]
df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Split features and target
X = df_cleaned.drop(columns=["ProdTaken"])
y = df_cleaned["ProdTaken"]

# Split dataset into 80% train and 20% test, stratifying on target balance
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Combine back to include target column for training/testing phases
train_df = pd.concat([Xtrain, ytrain], axis=1)
test_df = pd.concat([Xtest, ytest], axis=1)

# Save the final preprocessed train and test split sets
train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

print("Data prepared: train/test splits successfully saved.")
print(f"Train Shape: {train_df.shape} | Test Shape: {test_df.shape}")
