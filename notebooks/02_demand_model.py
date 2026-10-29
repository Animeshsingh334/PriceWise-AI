import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_products.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "demand_model.joblib"
CHARTS_PATH = PROJECT_ROOT / "outputs" / "charts"
REPORTS_PATH = PROJECT_ROOT / "outputs" / "reports"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
CHARTS_PATH.mkdir(parents=True, exist_ok=True)
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\nDATASET LOADED")
    print("=" * 45)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    return df


def prepare_features(df):
    target = "units_sold"

    feature_columns = [
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
    ]

    missing_columns = [
        column
        for column in feature_columns + [target]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    model_df = df[feature_columns + [target]].copy()

    model_df = model_df.dropna(subset=[target])
    model_df = model_df[model_df[target] >= 0]

    X = model_df[feature_columns]
    y = model_df[target]

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
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return X, y, preprocessor


def train_models(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    baseline_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", DummyRegressor(strategy="mean"))
    ])

    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )

    baseline_r2 = r2_score(
        y_test,
        baseline_predictions
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    predictions = np.maximum(predictions, 0)

    model_mae = mean_absolute_error(
        y_test,
        predictions
    )

    model_rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    model_r2 = r2_score(
        y_test,
        predictions
    )

    results = pd.DataFrame({
        "model": ["Mean Baseline", "Random Forest"],
        "MAE": [baseline_mae, model_mae],
        "RMSE": [baseline_rmse, model_rmse],
        "R2": [baseline_r2, model_r2]
    })

    print("\nMODEL EVALUATION")
    print("=" * 45)
    print(results.round(3).to_string(index=False))

    results.to_csv(
        REPORTS_PATH / "demand_model_evaluation.csv",
        index=False
    )

    return model, X_test, y_test, predictions, results


def plot_predictions(y_test, predictions):
    comparison = pd.DataFrame({
        "actual_units_sold": y_test.to_numpy(),
        "predicted_units_sold": predictions
    })

    comparison.to_csv(
        REPORTS_PATH / "demand_predictions.csv",
        index=False
    )

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=comparison,
        x="actual_units_sold",
        y="predicted_units_sold",
        alpha=0.7
    )

    maximum = max(
        comparison["actual_units_sold"].max(),
        comparison["predicted_units_sold"].max()
    )

    plt.plot(
        [0, maximum],
        [0, maximum],
        linestyle="--"
    )

    plt.title("Actual vs Predicted Product Demand")
    plt.xlabel("Actual Units Sold")
    plt.ylabel("Predicted Units Sold")
    plt.tight_layout()

    plt.savefig(
        CHARTS_PATH / "actual_vs_predicted_demand.png",
        dpi=300
    )

    plt.close()


def save_model(model):
    joblib.dump(model, MODEL_PATH)

    print("\nMODEL SAVED")
    print("=" * 45)
    print(MODEL_PATH)


def main():
    df = load_data()

    X, y, preprocessor = prepare_features(df)

    print("\nFEATURES USED:", X.columns.tolist())
    print("TARGET VARIABLE: units_sold")

    model, X_test, y_test, predictions, results = train_models(
        X,
        y,
        preprocessor
    )

    plot_predictions(y_test, predictions)

    save_model(model)

    print("\nDEMAND MODEL PIPELINE COMPLETED")
    print("=" * 45)
    print("Evaluation report:", REPORTS_PATH / "demand_model_evaluation.csv")
    print("Predictions:", REPORTS_PATH / "demand_predictions.csv")
    print("Prediction chart:", CHARTS_PATH / "actual_vs_predicted_demand.png")


if __name__ == "__main__":
    main()