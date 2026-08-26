import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


# ==========================================
# LOAD DATA
# ==========================================

DATA_PATH = "data/emi_prediction_dataset.csv"

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

# Convert numeric-looking columns
for column in ["age", "monthly_salary", "bank_balance"]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Features and target
X = df.drop(
    columns=[
        "emi_eligibility",
        "max_monthly_emi"
    ]
)

y = df["emi_eligibility"]


# ==========================================
# FEATURES
# ==========================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# ==========================================
# PREPROCESSOR
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
            OneHotEncoder(handle_unknown="ignore")
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
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# TARGET ENCODING FOR XGBOOST
# ==========================================

target_mapping = {
    "Not_Eligible": 0,
    "Eligible": 1,
    "High_Risk": 2
}

y_train_xgb = y_train.map(target_mapping)


# ==========================================
# MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
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
}


# ==========================================
# TRAIN AND COMPARE
# ==========================================

results = []

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    if name == "XGBoost":

        pipeline.fit(
            X_train,
            y_train_xgb
        )

        predictions_encoded = pipeline.predict(
            X_test
        )

        reverse_mapping = {
            0: "Not_Eligible",
            1: "Eligible",
            2: "High_Risk"
        }

        predictions = pd.Series(
            predictions_encoded
        ).map(reverse_mapping)

    else:

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "Macro Precision": precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "Macro Recall": recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),
        "Macro F1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )
    })


# ==========================================
# RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 70)

print(
    results_df.round(4).to_string(index=False)
)

print("\nComparison completed!")