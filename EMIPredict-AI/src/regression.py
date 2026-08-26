import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# EMIPredict AI - Regression
# ==========================================

DATA_PATH = "data/emi_prediction_dataset.csv"

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

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

y = df["max_monthly_emi"]


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
# 5. NUMERICAL PIPELINE
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
# 6. CATEGORICAL PIPELINE
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
# 7. PREPROCESSOR
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
# 8. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 9. RANDOM FOREST REGRESSOR
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 10. COMPLETE PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ==========================================
# 11. TRAIN
# ==========================================

print("\nTraining Random Forest Regressor...")

pipeline.fit(
    X_train,
    y_train
)


# ==========================================
# 12. PREDICT
# ==========================================

y_pred = pipeline.predict(X_test)


# ==========================================
# 13. EVALUATE
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


# ==========================================
# 14. RESULTS
# ==========================================

print("\n" + "=" * 60)
print("RANDOM FOREST REGRESSION RESULTS")
print("=" * 60)

print("\nMAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))

print("\nRegression completed successfully!")

# ==========================================
# 15. SAVE MODEL
# ==========================================

MODEL_PATH = "models/emi_regression_model.joblib"

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Saved to:", MODEL_PATH)