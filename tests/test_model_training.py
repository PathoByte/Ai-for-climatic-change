# tests/test_model.py

import pandas as pd
from src.model_training import train_model

def test_train_model_returns_outputs():
    df = pd.DataFrame({
        "Feature1": [1, 2, 3, 4, 5],
        "Emissions": [2.1, 4.1, 6.1, 8.1, 10.1]
    })

    model, X_test, y_test, y_pred = train_model(df, "Emissions")
    
    assert model is not None
    assert X_test.shape[0] == y_test.shape[0] == y_pred.shape[0]

def test_predictions_are_floats():
    df = pd.DataFrame({
        "Feature1": [10, 20, 30, 40, 50],
        "Emissions": [21, 41, 61, 81, 101]
    })
    _, _, _, y_pred = train_model(df, "Emissions")

    assert all(isinstance(y, float) for y in y_pred)
