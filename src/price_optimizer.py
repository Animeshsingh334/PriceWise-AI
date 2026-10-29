from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_products.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "demand_model.joblib"
REPORTS_PATH = PROJECT_ROOT / "outputs" / "reports"
CHARTS_PATH = PROJECT_ROOT / "outputs" / "charts"

REPORTS_PATH.mkdir(parents=True, exist_ok=True)
CHARTS_PATH.mkdir(parents=True, exist_ok=True)


def load_resources():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained demand model not found: {MODEL_PATH}. "
            "Run 02_demand_model.py first."
        )

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    return df, model


def select_product(df):
    required_columns = [
        "product_id",
        "product_name",
        "base_price_inr",
        "selling_price_inr",
        "cost_price_inr",
        "discount_percent",
        "units_sold",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    products = df.dropna(
        subset=[
            "product_id",
            "base_price_inr",
            "selling_price_inr",
            "cost_price_inr",
        ]
    ).copy()

    products = products[
        (products["base_price_inr"] > 0)
        & (products["selling_price_inr"] > 0)
        & (products["cost_price_inr"] >= 0)
    ]

    if products.empty:
        raise ValueError("No valid products were found.")

    print("\nAVAILABLE PRODUCTS")
    print("=" * 65)

    display_columns = [
        "product_id",
        "product_name",
        "selling_price_inr",
        "cost_price_inr",
    ]

    print(
        products[display_columns]
        .head(20)
        .to_string(index=False)
    )

    product_id = input(
        "\nEnter the product ID to optimize: "
    ).strip()

    selected = products[
        products["product_id"].astype(str) == product_id
    ]

    if selected.empty:
        raise ValueError(
            f"Product ID '{product_id}' was not found."
        )

    return selected.iloc[0].copy()


def generate_price_scenarios(product, model):
    base_price = float(product["base_price_inr"])
    current_price = float(product["selling_price_inr"])
    cost_price = float(product["cost_price_inr"])

    minimum_price = max(
        base_price * 0.50,
        cost_price + 0.01
    )

    maximum_price = base_price

    if minimum_price > maximum_price:
        raise ValueError(
            "No profitable candidate price range exists below "
            "the product's base price."
        )

    candidate_prices = np.linspace(
        minimum_price,
        maximum_price,
        25
    )

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

    missing_features = [
        column for column in feature_columns
        if column not in product.index
    ]

    if missing_features:
        raise ValueError(
            f"Model features are missing: {missing_features}"
        )

    scenarios = []

    for candidate_price in candidate_prices:
        discount = (
            (base_price - candidate_price) / base_price
        ) * 100

        input_row = product[feature_columns].to_frame().T.copy()

        input_row["selling_price_inr"] = candidate_price
        input_row["discount_percent"] = discount

        predicted_demand = float(
            model.predict(input_row)[0]
        )

        predicted_demand = max(0, predicted_demand)

        estimated_revenue = (
            candidate_price * predicted_demand
        )

        estimated_profit = (
            (candidate_price - cost_price)
            * predicted_demand
        )

        scenarios.append({
            "candidate_price_inr": round(
                candidate_price, 2
            ),
            "discount_percent": round(discount, 2),
            "predicted_units_sold": round(
                predicted_demand, 2
            ),
            "estimated_revenue_inr": round(
                estimated_revenue, 2
            ),
            "estimated_profit_inr": round(
                estimated_profit, 2
            ),
        })

    results = pd.DataFrame(scenarios)

    results["current_price_inr"] = current_price

    return results


def save_results(product, results):
    product_id = str(product["product_id"])
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", product_id)

    csv_path = (
        REPORTS_PATH
        / f"price_optimization_{safe_id}.csv"
    )

    chart_path = (
        CHARTS_PATH
        / f"price_optimization_{safe_id}.png"
    )

    results.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))

    plt.plot(
        results["candidate_price_inr"],
        results["estimated_profit_inr"],
        marker="o",
        label="Estimated Profit"
    )

    best_row = results.loc[
        results["estimated_profit_inr"].idxmax()
    ]

    plt.scatter(
        best_row["candidate_price_inr"],
        best_row["estimated_profit_inr"],
        s=120,
        label="Best Candidate"
    )

    plt.xlabel("Candidate Selling Price (INR)")
    plt.ylabel("Estimated Total Profit (INR)")
    plt.title("Price Optimization Analysis")
    plt.legend()
    plt.tight_layout()

    plt.savefig(chart_path, dpi=300)
    plt.close()

    return csv_path, chart_path


def main():
    df, model = load_resources()

    product = select_product(df)

    print("\nSELECTED PRODUCT")
    print("=" * 65)
    print("Product ID:", product["product_id"])
    print("Product Name:", product["product_name"])
    print(
        "Current Selling Price: ₹",
        round(float(product["selling_price_inr"]), 2)
    )
    print(
        "Cost Price: ₹",
        round(float(product["cost_price_inr"]), 2)
    )

    results = generate_price_scenarios(product, model)

    best_row = results.loc[
        results["estimated_profit_inr"].idxmax()
    ]

    print("\nPRICE OPTIMIZATION RESULTS")
    print("=" * 65)

    print(
        results[
            [
                "candidate_price_inr",
                "discount_percent",
                "predicted_units_sold",
                "estimated_revenue_inr",
                "estimated_profit_inr",
            ]
        ].round(2).to_string(index=False)
    )

    print("\nRECOMMENDED CANDIDATE PRICE")
    print("=" * 65)
    print(
        f"Estimated best price: "
        f"₹{best_row['candidate_price_inr']:.2f}"
    )
    print(
        f"Discount: "
        f"{best_row['discount_percent']:.2f}%"
    )
    print(
        f"Predicted demand: "
        f"{best_row['predicted_units_sold']:.2f} units"
    )
    print(
        f"Estimated revenue: "
        f"₹{best_row['estimated_revenue_inr']:.2f}"
    )
    print(
        f"Estimated profit: "
        f"₹{best_row['estimated_profit_inr']:.2f}"
    )

    csv_path, chart_path = save_results(
        product,
        results
    )

    print("\nFILES SAVED")
    print("=" * 65)
    print("Scenario report:", csv_path)
    print("Optimization chart:", chart_path)


if __name__ == "__main__":
    main()