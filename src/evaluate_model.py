from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_products.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "demand_model.joblib"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "model_evaluation.csv"


FEATURES = [
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

TARGET = "units_sold"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    missing = [
        column for column in FEATURES + [TARGET]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] >= 0]

    return df, model


def evaluate_predictions(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "Negative prediction rate (%)": (
            np.mean(predicted < 0) * 100
        ),
    }


def main():
    df, saved_model = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)
    model_predictions = saved_model.predict(X_test)

    baseline_metrics = evaluate_predictions(
        y_test,
        baseline_predictions,
    )

    model_metrics = evaluate_predictions(
        y_test,
        model_predictions,
    )

    results = pd.DataFrame([
        {
            "Model": "Mean Baseline",
            **baseline_metrics,
        },
        {
            "Model": "Saved Random Forest",
            **model_metrics,
        },
    ])

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(REPORT_PATH, index=False)

    print("\nMODEL EVALUATION")
    print("=" * 65)
    print(results.round(3).to_string(index=False))
    print(f"\nReport saved to: {REPORT_PATH}")

    print("\nINTERPRETATION")
    print("=" * 65)

    if model_metrics["MAE"] < baseline_metrics["MAE"]:
        print("The saved model has lower MAE than the baseline.")
    else:
        print("The saved model does not beat the baseline on MAE.")

    if model_metrics["R2"] < 0:
        print(
            "The model has negative R² on this test split. "
            "Its predictions are worse than the test-set mean "
            "under the R² metric."
        )

    print(
        "\nImportant: These metrics use a random split of synthetic data. "
        "They do not establish real-world forecasting accuracy."
    )


if __name__ == "__main__":
    main()