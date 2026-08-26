import pandas as pd

DATA_PATH = "data/emi_prediction_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("EMIPREDICT AI - DATASET AUDIT")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nEMI Eligibility:")
print(df["emi_eligibility"].value_counts())

print("\nMaximum Monthly EMI:")
print(df["max_monthly_emi"].describe())

print("\nAudit completed!")