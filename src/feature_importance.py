from pathlib import Path

import joblib
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_products.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "demand_model.joblib"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "feature_importance.csv"

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


def main():
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] >= 0]

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    results = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    })

    results = results.sort_values(
        "importance_mean",
        ascending=False,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(REPORT_PATH, index=False)

    print("\nFEATURE IMPORTANCE")
    print("=" * 65)
    print(results.round(4).to_string(index=False))
    print(f"\nReport saved to: {REPORT_PATH}")

    print("\nINTERPRETATION")
    print("=" * 65)
    print(
        "A higher positive importance generally means that shuffling "
        "the feature worsens model performance."
    )
    print(
        "Near-zero or negative importance means the feature did not "
        "help consistently on this test split."
    )
    print(
        "\nFeature importance measures predictive usefulness, "
        "not causation."
    )


if __name__ == "__main__":
    main()