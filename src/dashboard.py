from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_products.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "demand_model.joblib"


st.set_page_config(
    page_title="PriceWise AI",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. Run the data cleaning script first."
        )

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Demand model not found. Run 02_demand_model.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_demand(model, product, candidate_price):
    base_price = float(product["base_price_inr"])

    discount = (
        (base_price - candidate_price) / base_price
    ) * 100

    features = pd.DataFrame([{
        "base_price_inr": base_price,
        "discount_percent": discount,
        "selling_price_inr": candidate_price,
        "cost_price_inr": float(product["cost_price_inr"]),
        "inventory_available": product["inventory_available"],
        "promotion_active": product["promotion_active"],
        "product_rating": product["product_rating"],
        "category": product["category"],
        "region": product["region"],
        "sales_channel": product["sales_channel"],
    }])

    predicted_units = float(model.predict(features)[0])

    return max(0, predicted_units), discount


def main():
    st.title("📊 PriceWise AI")
    st.subheader("Product Demand Prediction & Pricing Analysis")

    st.write(
        "Explore product demand predictions and compare estimated "
        "revenue and profit across different selling prices."
    )

    try:
        df = load_data()
        model = load_model()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    total_products = len(df)
    average_price = df["selling_price_inr"].mean()
    total_units = df["units_sold"].sum()
    total_revenue = (
        df["selling_price_inr"] * df["units_sold"]
    ).sum()

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Dataset Records", f"{total_products:,}")
    metric2.metric("Average Selling Price", f"₹{average_price:,.2f}")
    metric3.metric("Recorded Units Sold", f"{total_units:,.0f}")
    metric4.metric("Recorded Revenue", f"₹{total_revenue:,.2f}")

    st.divider()

    st.header("Product Demand Prediction")

    valid_products = df.dropna(
        subset=[
            "product_id",
            "product_name",
            "base_price_inr",
            "selling_price_inr",
            "cost_price_inr",
        ]
    )

    valid_products = valid_products[
        (valid_products["base_price_inr"] > 0)
        & (valid_products["selling_price_inr"] > 0)
        & (valid_products["cost_price_inr"] >= 0)
    ]

    if valid_products.empty:
        st.warning("No valid products are available.")
        st.stop()

    product_options = valid_products[
        ["product_id", "product_name"]
    ].drop_duplicates()

    product_options["label"] = (
        product_options["product_name"].astype(str)
        + " ("
        + product_options["product_id"].astype(str)
        + ")"
    )

    selected_label = st.selectbox(
        "Select a product",
        product_options["label"].tolist()
    )

    selected_id = product_options.loc[
        product_options["label"] == selected_label,
        "product_id"
    ].iloc[0]

    product = valid_products[
        valid_products["product_id"].astype(str)
        == str(selected_id)
    ].iloc[0]

    current_price = float(product["selling_price_inr"])
    base_price = float(product["base_price_inr"])
    cost_price = float(product["cost_price_inr"])

    st.write(f"**Product:** {product['product_name']}")
    st.write(f"**Current selling price:** ₹{current_price:,.2f}")
    st.write(f"**Cost price:** ₹{cost_price:,.2f}")

    minimum_price = max(base_price * 0.5, cost_price + 0.01)

    if minimum_price >= base_price:
        st.warning(
            "There is no valid price range above cost and at or below "
            "the base price for this product."
        )
        st.stop()

    candidate_price = st.slider(
        "Choose a candidate selling price (₹)",
        min_value=float(minimum_price),
        max_value=float(base_price),
        value=float(np.clip(current_price, minimum_price, base_price)),
        step=1.0
    )

    predicted_units, discount = predict_demand(
        model,
        product,
        candidate_price
    )

    estimated_revenue = candidate_price * predicted_units
    estimated_profit = (
        candidate_price - cost_price
    ) * predicted_units

    st.subheader("Prediction Results")

    result1, result2, result3, result4 = st.columns(4)

    result1.metric("Candidate Price", f"₹{candidate_price:,.2f}")
    result2.metric("Calculated Discount", f"{discount:.2f}%")
    result3.metric("Predicted Demand", f"{predicted_units:.2f} units")
    result4.metric("Estimated Profit", f"₹{estimated_profit:,.2f}")

    st.caption(
        "Predictions are experimental and based on synthetic training data. "
        "Estimated profit excludes operating expenses and is not a "
        "guaranteed business outcome."
    )

    st.divider()

    st.header("Price Scenario Comparison")

    candidate_prices = np.linspace(
        minimum_price,
        base_price,
        25
    )

    scenario_rows = []

    for price in candidate_prices:
        demand, scenario_discount = predict_demand(
            model,
            product,
            float(price)
        )

        scenario_rows.append({
            "Selling Price (₹)": round(float(price), 2),
            "Discount (%)": round(float(scenario_discount), 2),
            "Predicted Units": round(demand, 2),
            "Estimated Revenue (₹)": round(
                float(price) * demand, 2
            ),
            "Estimated Profit (₹)": round(
                (float(price) - cost_price) * demand, 2
            ),
        })

    scenarios = pd.DataFrame(scenario_rows)

    st.dataframe(
        scenarios,
        use_container_width=True,
        hide_index=True
    )

    best_scenario = scenarios.loc[
        scenarios["Estimated Profit (₹)"].idxmax()
    ]

    st.success(
        f"Best tested candidate: ₹{best_scenario['Selling Price (₹)']:.2f} "
        f"with estimated profit of "
        f"₹{best_scenario['Estimated Profit (₹)']:.2f}."
    )

    chart_data = scenarios.set_index("Selling Price (₹)")[
        ["Estimated Revenue (₹)", "Estimated Profit (₹)"]
    ]

    st.line_chart(chart_data)

    st.download_button(
        label="Download Scenario Results (CSV)",
        data=scenarios.to_csv(index=False).encode("utf-8"),
        file_name="pricewise_scenarios.csv",
        mime="text/csv"
    )

    st.divider()

    st.header("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.download_button(
        label="Download Cleaned Dataset (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="pricewise_cleaned_data.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()