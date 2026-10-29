from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_products.csv"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "feature_set_comparison.csv"

TARGET = "units_sold"

FEATURE_SETS = {
    "All Features": [
        "base_price_inr",
        "discount_percent",
        "selling_price_inr",
        "cost_price_inr",
        "inventory_available",
        "promotion_active",
        "product_rating",
        "category",
        "region",
        "sales_channel",
    ],
    "Without Inventory": [
        "base_price_inr",
        "discount_percent",
        "selling_price_inr",
        "cost_price_inr",
        "promotion_active",
        "product_rating",
        "category",
        "region",
        "sales_channel",
    ],
    "Product and Market Features": [
        "base_price_inr",
        "cost_price_inr",
        "product_rating",
        "category",
        "region",
        "sales_channel",
    ],
}


def build_pipeline(X):
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_features = [
        column
        for column in X.columns
        if column not in categorical_features
    ]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )),
    ])

    return model


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = set().union(
        *(set(features) for features in FEATURE_SETS.values())
    ) | {TARGET}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {sorted(missing_columns)}"
        )

    df = df.dropna(subset=[TARGET]).copy()
    df = df[df[TARGET] >= 0]

    if len(df) < 5:
        raise ValueError(
            "At least five valid rows are required for five-fold "
            "cross-validation."
        )

    y = df[TARGET]

    cross_validation = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    for feature_set_name, features in FEATURE_SETS.items():
        X = df[features]

        model = build_pipeline(X)

        scores = cross_validate(
            model,
            X,
            y,
            cv=cross_validation,
            scoring={
                "mae": "neg_mean_absolute_error",
                "mse": "neg_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=1,
            return_train_score=False,
        )

        fold_mae = -scores["test_mae"]
        fold_rmse = np.sqrt(-scores["test_mse"])
        fold_r2 = scores["test_r2"]

        results.append({
            "feature_set": feature_set_name,
            "mean_mae": fold_mae.mean(),
            "std_mae": fold_mae.std(),
            "mean_rmse": fold_rmse.mean(),
            "mean_r2": fold_r2.mean(),
        })

    results_df = pd.DataFrame(results).sort_values(
        by="mean_mae",
        ascending=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(REPORT_PATH, index=False)

    print("\nCROSS-VALIDATION FEATURE COMPARISON")
    print("=" * 75)
    print(results_df.round(4).to_string(index=False))

    print(f"\nReport saved to: {REPORT_PATH}")

    print("\nINTERPRETATION")
    print("=" * 75)
    print("Lower MAE and RMSE indicate smaller prediction errors.")
    print("Higher R² generally indicates better predictive performance.")
    print(
        "\nImportant: These results use random cross-validation on "
        "synthetic data. They do not prove future forecasting accuracy "
        "or rule out all forms of data leakage."
    )


if __name__ == "__main__":
    main()