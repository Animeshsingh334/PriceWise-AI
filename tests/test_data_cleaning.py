import pandas as pd
from src.data_cleaning import load_dataset


def test_load_dataset(tmp_path):
    file_path = tmp_path / "sample.csv"
    pd.DataFrame({"product": ["Sample"], "units_sold": [10]}).to_csv(
        file_path, index=False
    )

    result = load_dataset(file_path)

    assert len(result) == 1
    assert result.loc[0, "product"] == "Sample"
