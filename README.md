# PriceWise AI

An educational data science and machine learning project exploring
product demand prediction, pricing analysis, and customer churn prediction.

## Project Structure

- data/raw: Original datasets
- data/processed: Cleaned datasets
- notebooks: Data cleaning, EDA, and ML experiments
- models: Trained machine learning models
- outputs/charts: Generated visualizations
- outputs/reports: Reports and evaluation results
- src: Reusable Python source code
- tests: Automated tests
- docs: Project documentation

## Setup

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

Open the project in VS Code and select .venv/bin/python
as the Python interpreter and notebook kernel.

## Dataset

Place the original CSV at:

    data/raw/pricewise_ai_10000_products.csv

The initial practice dataset is synthetic and should not be
treated as real-world market evidence.
