import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pricewise_ai_10000_products.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_products.csv"
CHARTS_PATH = PROJECT_ROOT / "outputs" / "charts"
REPORTS_PATH = PROJECT_ROOT / "outputs" / "reports"

CHARTS_PATH.mkdir(parents=True, exist_ok=True)
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


def load_data():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    print("\nDATASET OVERVIEW")
    print("=" * 40)
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print("\nFirst five records:")
    print(df.head())
    print("\nDataset information:")
    df.info()

    return df


def analyze_data_quality(df):
    print("\nDATA QUALITY ANALYSIS")
    print("=" * 40)

    missing_values = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percentage": (df.isnull().mean() * 100).round(2)
    })

    print("\nMissing values:")
    print(missing_values)

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nNumerical summary:")
    print(df.describe().round(2))

    missing_values.to_csv(REPORTS_PATH / "missing_values_report.csv")

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
    plt.title("Missing Values in Raw Dataset")
    plt.xlabel("Columns")
    plt.tight_layout()
    plt.savefig(CHARTS_PATH / "missing_values_heatmap.png", dpi=300)
    plt.close()


def clean_data(df):
    cleaned_df = df.copy()

    initial_rows = len(cleaned_df)

    cleaned_df = cleaned_df.drop_duplicates()

    cleaned_df = cleaned_df.dropna()

    cleaned_df["date"] = pd.to_datetime(
        cleaned_df["date"],
        errors="coerce"
    )

    cleaned_df = cleaned_df.dropna(subset=["date"])

    cleaned_df = cleaned_df[
        (cleaned_df["base_price_inr"] > 0)
        & (cleaned_df["selling_price_inr"] > 0)
        & (cleaned_df["cost_price_inr"] > 0)
        & (cleaned_df["units_sold"] >= 0)
        & (cleaned_df["inventory_available"] >= 0)
        & (cleaned_df["discount_percent"].between(0, 100))
        & (cleaned_df["product_rating"].between(0, 5))
    ]

    cleaned_df = cleaned_df.reset_index(drop=True)

    print("\nDATA CLEANING RESULTS")
    print("=" * 40)
    print("Original rows:", initial_rows)
    print("Cleaned rows:", len(cleaned_df))
    print("Rows removed:", initial_rows - len(cleaned_df))
    print(
        "Rows retained:",
        round(len(cleaned_df) / initial_rows * 100, 2),
        "%"
    )

    return cleaned_df


def feature_engineering(df):
    df = df.copy()

    df["revenue_inr"] = (
        df["selling_price_inr"] * df["units_sold"]
    )

    df["profit_per_unit_inr"] = (
        df["selling_price_inr"] - df["cost_price_inr"]
    )

    df["total_profit_inr"] = (
        df["profit_per_unit_inr"] * df["units_sold"]
    )

    df["profit_margin_percent"] = np.where(
        df["selling_price_inr"] > 0,
        df["profit_per_unit_inr"]
        / df["selling_price_inr"] * 100,
        np.nan
    )

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.day_name()

    return df


def analyze_distributions(df):
    columns = [
        "base_price_inr",
        "selling_price_inr",
        "discount_percent",
        "units_sold",
        "inventory_available",
        "product_rating"
    ]

    for column in columns:
        plt.figure(figsize=(8, 5))

        sns.histplot(data=df, x=column, kde=True)

        plt.title(f"Distribution of {column}")
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(
            CHARTS_PATH / f"distribution_{column}.png",
            dpi=300
        )

        plt.close()


def analyze_categories(df):
    category_analysis = (
        df.groupby("category")
        .agg(
            total_units_sold=("units_sold", "sum"),
            total_revenue_inr=("revenue_inr", "sum"),
            total_profit_inr=("total_profit_inr", "sum"),
            average_selling_price_inr=("selling_price_inr", "mean"),
            average_discount_percent=("discount_percent", "mean")
        )
        .round(2)
        .sort_values("total_revenue_inr", ascending=False)
    )

    print("\nCATEGORY PERFORMANCE")
    print("=" * 40)
    print(category_analysis)

    category_analysis.to_csv(
        REPORTS_PATH / "category_analysis.csv"
    )

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=category_analysis.reset_index(),
        x="category",
        y="total_revenue_inr"
    )

    plt.title("Revenue by Product Category")
    plt.xlabel("Category")
    plt.ylabel("Revenue (INR)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(CHARTS_PATH / "revenue_by_category.png", dpi=300)
    plt.close()


def analyze_pricing(df):
    discount_analysis = (
        df.groupby("discount_percent")
        .agg(
            average_units_sold=("units_sold", "mean"),
            total_units_sold=("units_sold", "sum"),
            average_selling_price_inr=("selling_price_inr", "mean")
        )
        .round(2)
    )

    print("\nDISCOUNT ANALYSIS")
    print("=" * 40)
    print(discount_analysis.head(15))

    discount_analysis.to_csv(
        REPORTS_PATH / "discount_analysis.csv"
    )

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="discount_percent",
        y="units_sold",
        hue="promotion_active",
        alpha=0.7
    )

    plt.title("Discount Percentage vs Units Sold")
    plt.xlabel("Discount Percentage")
    plt.ylabel("Units Sold")
    plt.tight_layout()

    plt.savefig(CHARTS_PATH / "discount_vs_units_sold.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="selling_price_inr",
        y="units_sold",
        hue="category",
        alpha=0.7
    )

    plt.title("Selling Price vs Units Sold")
    plt.xlabel("Selling Price (INR)")
    plt.ylabel("Units Sold")
    plt.tight_layout()

    plt.savefig(CHARTS_PATH / "price_vs_units_sold.png", dpi=300)
    plt.close()


def analyze_correlations(df):
    numeric_df = df.select_dtypes(include=np.number)

    correlation_matrix = numeric_df.corr()

    print("\nCORRELATION MATRIX")
    print("=" * 40)
    print(correlation_matrix.round(2))

    correlation_matrix.to_csv(
        REPORTS_PATH / "correlation_matrix.csv"
    )

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Matrix")
    plt.tight_layout()

    plt.savefig(CHARTS_PATH / "correlation_matrix.png", dpi=300)
    plt.close()


def analyze_sales_channels(df):
    channel_analysis = (
        df.groupby("sales_channel")
        .agg(
            total_units_sold=("units_sold", "sum"),
            total_revenue_inr=("revenue_inr", "sum"),
            total_profit_inr=("total_profit_inr", "sum")
        )
        .round(2)
        .sort_values("total_revenue_inr", ascending=False)
    )

    print("\nSALES CHANNEL PERFORMANCE")
    print("=" * 40)
    print(channel_analysis)

    channel_analysis.to_csv(
        REPORTS_PATH / "sales_channel_analysis.csv"
    )

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=channel_analysis.reset_index(),
        x="sales_channel",
        y="total_revenue_inr"
    )

    plt.title("Revenue by Sales Channel")
    plt.xlabel("Sales Channel")
    plt.ylabel("Revenue (INR)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(
        CHARTS_PATH / "revenue_by_sales_channel.png",
        dpi=300
    )

    plt.close()


def generate_summary(df, original_rows):
    summary = {
        "original_rows": original_rows,
        "cleaned_rows": len(df),
        "rows_removed": original_rows - len(df),
        "total_units_sold": df["units_sold"].sum(),
        "total_revenue_inr": df["revenue_inr"].sum(),
        "total_profit_inr": df["total_profit_inr"].sum(),
        "average_selling_price_inr": df["selling_price_inr"].mean(),
        "average_discount_percent": df["discount_percent"].mean()
    }

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["metric", "value"]
    )

    summary_df.to_csv(
        REPORTS_PATH / "eda_summary_report.csv",
        index=False
    )

    print("\nFINAL SUMMARY")
    print("=" * 40)
    print(summary_df.to_string(index=False))


def main():
    df = load_data()

    analyze_data_quality(df)

    cleaned_df = clean_data(df)

    cleaned_df = feature_engineering(cleaned_df)

    cleaned_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    analyze_distributions(cleaned_df)
    analyze_categories(cleaned_df)
    analyze_pricing(cleaned_df)
    analyze_correlations(cleaned_df)
    analyze_sales_channels(cleaned_df)

    generate_summary(cleaned_df, len(df))

    print("\nData cleaning and EDA completed successfully.")
    print("Cleaned data:", PROCESSED_DATA_PATH)
    print("Charts:", CHARTS_PATH)
    print("Reports:", REPORTS_PATH)


if __name__ == "__main__":
    main()