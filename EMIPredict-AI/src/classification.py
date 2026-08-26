import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier


# ==========================================
# EMIPredict AI - XGBoost Classification
# ==========================================

DATA_PATH = "data/emi_prediction_dataset.csv"

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# ==========================================
# 2. CONVERT NUMERIC COLUMNS
# ==========================================

numeric_conversion_columns = [
    "age",
    "monthly_salary",
    "bank_balance"
]

for column in numeric_conversion_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = df.drop(
    columns=[
        "emi_eligibility",
        "max_monthly_emi"
    ]
)

y = df["emi_eligibility"]

# ==========================================
# 4. IDENTIFY FEATURES
# ==========================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# ==========================================
# 5. PREPROCESSING
# ==========================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

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
# 6. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==========================================
# 7. CONVERT TARGET TO NUMERIC
# ==========================================

# XGBoost requires numeric target labels.

target_mapping = {
    "Not_Eligible": 0,
    "Eligible": 1,
    "High_Risk": 2
}

y_train_encoded = y_train.map(target_mapping)
y_test_encoded = y_test.map(target_mapping)

# ==========================================
# 8. XGBOOST MODEL
# ==========================================

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# ==========================================
# 9. COMPLETE PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ==========================================
# 10. TRAIN
# ==========================================

print("\nTraining XGBoost...")

pipeline.fit(
    X_train,
    y_train_encoded
)

# ==========================================
# 11. PREDICT
# ==========================================

y_pred_encoded = pipeline.predict(X_test)

# Convert predictions back to original labels

reverse_mapping = {
    0: "Not_Eligible",
    1: "Eligible",
    2: "High_Risk"
}

y_pred = pd.Series(
    y_pred_encoded
).map(reverse_mapping)

# ==========================================
# 12. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)

print("\nAccuracy:", round(accuracy, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==========================================
# SAVE CLASSIFICATION MODEL
# ==========================================

MODEL_PATH = "models/emi_classification_model.joblib"

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nClassification model saved successfully!")
print("Saved to:", MODEL_PATH)