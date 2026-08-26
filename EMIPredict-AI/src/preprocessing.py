import pandas as pd

# ==========================================
# EMIPredict AI - Data Preprocessing
# ==========================================

DATA_PATH = "data/emi_prediction_dataset.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# ==========================================
# 1. CONVERT NUMERIC COLUMNS
# ==========================================

numeric_columns = [
    "age",
    "monthly_salary",
    "bank_balance"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# ==========================================
# 2. HANDLE MISSING VALUES
# ==========================================

# Categorical column
df["education"] = df["education"].fillna(df["education"].mode()[0])

# Numerical columns
median_columns = [
    "monthly_rent",
    "credit_score",
    "bank_balance",
    "emergency_fund"
]

for column in median_columns:
    df[column] = df[column].fillna(df[column].median())

# ==========================================
# 3. CHECK MISSING VALUES
# ==========================================

print("\nMissing values after preprocessing:")

missing = df.isnull().sum()

print(missing[missing > 0])

# ==========================================
# 4. CHECK DATA TYPES
# ==========================================

print("\nData types:")

print(df.dtypes)

# ==========================================
# 5. FINAL SHAPE
# ==========================================

print("\nFinal shape:", df.shape)

print("\nMissing-value handling completed successfully!")


# ==========================================
# 6. CHECK CATEGORICAL VALUES
# ==========================================

categorical_columns = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
    "emi_eligibility"
]

print("\nCategorical column values:")

for column in categorical_columns:
    print("\n" + "=" * 50)
    print(column)
    print("=" * 50)
    print(df[column].value_counts())