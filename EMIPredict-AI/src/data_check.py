import pandas as pd

# ==========================================
# EMIPredict AI - Dataset Audit
# ==========================================

DATA_PATH = "data/emi_prediction_dataset.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("              EMIPREDICT AI - DATASET AUDIT")
print("=" * 70)

# 1. Dataset shape
print("\n1. DATASET SHAPE")
print("-" * 40)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Column names
print("\n2. COLUMN NAMES")
print("-" * 40)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# 3. Data types
print("\n3. DATA TYPES")
print("-" * 40)
print(df.dtypes)

# 4. Missing values
print("\n4. MISSING VALUES")
print("-" * 40)

missing = df.isnull().sum()

for column, count in missing.items():
    if count > 0:
        percentage = (count / len(df)) * 100
        print(f"{column}: {count} ({percentage:.2f}%)")

# 5. Duplicate rows
print("\n5. DUPLICATE ROWS")
print("-" * 40)
print("Duplicates:", df.duplicated().sum())

# 6. Classification target
print("\n6. CLASSIFICATION TARGET")
print("-" * 40)

print("Target column: emi_eligibility")
print(df["emi_eligibility"].value_counts())

print("\nClass percentages:")
print(
    (df["emi_eligibility"].value_counts(normalize=True) * 100)
    .round(2)
)

# 7. Regression target
print("\n7. REGRESSION TARGET")
print("-" * 40)

print("Target column: max_monthly_emi")
print(df["max_monthly_emi"].describe())

# 8. Unique values
print("\n8. UNIQUE VALUES")
print("-" * 40)

for column in df.columns:
    print(f"{column}: {df[column].nunique()}")

# 9. Numerical columns
print("\n9. NUMERICAL COLUMNS")
print("-" * 40)

numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()

print(numerical_columns)

# 10. Categorical columns
print("\n10. CATEGORICAL COLUMNS")
print("-" * 40)

categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

print(categorical_columns)

# 11. Summary statistics
print("\n11. NUMERICAL SUMMARY")
print("-" * 40)

print(df[numerical_columns].describe())

print("\n" + "=" * 70)
print("                 AUDIT COMPLETED")
print("=" * 70)