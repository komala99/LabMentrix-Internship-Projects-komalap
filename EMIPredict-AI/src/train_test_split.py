import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# EMIPredict AI - Train/Test Split
# ==========================================

DATA_PATH = "data/emi_prediction_dataset.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

# ==========================================
# 1. NUMERIC CONVERSION
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

df["education"] = df["education"].fillna(
    df["education"].mode()[0]
)

median_columns = [
    "monthly_rent",
    "credit_score",
    "bank_balance",
    "emergency_fund"
]

for column in median_columns:
    df[column] = df[column].fillna(df[column].median())

# ==========================================
# 3. REMOVE TARGET COLUMNS FROM FEATURES
# ==========================================

target_classification = "emi_eligibility"
target_regression = "max_monthly_emi"

X = df.drop(
    columns=[
        target_classification,
        target_regression
    ]
)

y_classification = df[target_classification]

y_regression = df[target_regression]

# ==========================================
# 4. CLASSIFICATION TRAIN/TEST SPLIT
# ==========================================

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X,
    y_classification,
    test_size=0.20,
    random_state=42,
    stratify=y_classification
)

# ==========================================
# 5. REGRESSION TRAIN/TEST SPLIT
# ==========================================

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X,
    y_regression,
    test_size=0.20,
    random_state=42
)

# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("\nClassification:")
print("X_train:", X_train_cls.shape)
print("X_test :", X_test_cls.shape)
print("y_train:", y_train_cls.shape)
print("y_test :", y_test_cls.shape)

print("\nRegression:")
print("X_train:", X_train_reg.shape)
print("X_test :", X_test_reg.shape)
print("y_train:", y_train_reg.shape)
print("y_test :", y_test_reg.shape)

print("\nClassification distribution:")
print(y_train_cls.value_counts())

print("\nTrain/test split completed successfully!")