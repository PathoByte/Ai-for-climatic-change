# tests/test_data_processing.py

import os
import pandas as pd
from src.data_preprocessing import load_and_clean_data

def test_load_and_clean_data_returns_dataframe():
    df = load_and_clean_data("Data/historical_emissions.csv")
    assert isinstance(df, pd.DataFrame)

def test_no_missing_values_after_cleaning():
    df = load_and_clean_data("Data/historical_emissions.csv")
    assert not df.isnull().values.any()

def test_columns_present_after_cleaning():
    df = load_and_clean_data("Data/historical_emissions.csv")
    expected_columns = ["Country", "Year", "Emissions"]
    for col in expected_columns:
        assert col in df.columns
