# tests/test_utils.py

from src.utils import calculate_metrics
from sklearn.metrics import mean_squared_error, r2_score

def test_calculate_metrics_accuracy():
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]

    metrics = calculate_metrics(y_true, y_pred)

    assert abs(metrics["mse"] - mean_squared_error(y_true, y_pred)) < 1e-6
    assert abs(metrics["r2"] - r2_score(y_true, y_pred)) < 1e-6
    assert abs(metrics["mae"] - sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)) < 1e-6

def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
