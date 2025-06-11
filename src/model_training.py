import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_features(df):
    """Prepare features (X) and target (y) from cleaned dataframe."""
    # Drop rows with missing emissions just in case
    df = df.dropna(subset=['Emissions'])

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df[['Country', 'Sector', 'Gas', 'Year']])

    # Target variable
    y = df['Emissions']

    return df_encoded, y
from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


import joblib
from sklearn.metrics import mean_squared_error, r2_score
import json
import os
import numpy as np

def save_model(model, path):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def evaluate_model(model, X_test, y_test, metrics_path=None):
    """Evaluate model and optionally save metrics to JSON."""
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    metrics = {"r2_score": r2, "rmse": rmse}

    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics
