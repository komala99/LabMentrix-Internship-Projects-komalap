import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ==========================================
# EMIPredict AI - ML Preprocessing Pipeline
# ==========================================

DATA_PATH = "data/emi_prediction_dataset.csv"

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)


# ==========================================
# 2. CONVERT NUMERIC-LOOKING COLUMNS
# ==========================================

numeric_conversion_columns = [
    "age",
    "monthly_salary",
    "bank_balance"
]

for column in numeric_conversion_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# ==========================================
# 3. DEFINE TARGETS
# ==========================================

classification_target = "emi_eligibility"
regression_target = "max_monthly_emi"


# ==========================================
# 4. CREATE FEATURES
# ==========================================

X = df.drop(
    columns=[
        classification_target,
        regression_target
    ]
)


# ==========================================
# 5. IDENTIFY NUMERICAL FEATURES
# ==========================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# ==========================================
# 6. IDENTIFY CATEGORICAL FEATURES
# ==========================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ==========================================
# 7. NUMERICAL PIPELINE
# ==========================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ==========================================
# 8. CATEGORICAL PIPELINE
# ==========================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# ==========================================
# 9. COMBINE BOTH PIPELINES
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ==========================================
# 10. FIT AND TRANSFORM DATA
# ==========================================

X_transformed = preprocessor.fit_transform(X)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\nOriginal feature shape:")
print(X.shape)

print("\nTransformed feature shape:")
print(X_transformed.shape)

print("\nPreprocessing pipeline completed successfully!")