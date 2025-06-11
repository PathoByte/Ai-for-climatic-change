import os
import matplotlib.pyplot as plt
import pandas as pd



def plot_actual_vs_predicted(y_true, y_pred, output_path):
    """Plot actual vs predicted emissions and save the figure."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='teal')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # ideal line
    plt.xlabel("Actual Emissions")
    plt.ylabel("Predicted Emissions")
    plt.title("Actual vs Predicted Emissions")
    plt.grid(True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_correlation_heatmap(df, save_path):
    """Plot and save a correlation heatmap from a DataFrame."""
    correlation_matrix = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_data_distribution(df, column, save_path):
    """Plot and save the distribution of a specific numerical column."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True, color='skyblue')
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_emissions_over_time(df, save_path):
    """Plot total CO₂ emissions over time and save the figure."""
    emissions_per_year = df.groupby("Year")["Emissions"].sum()

    plt.figure(figsize=(12, 6))
    emissions_per_year.plot(kind="line", marker="o", color="coral")
    plt.title("Total CO₂ Emissions Over Time")
    plt.xlabel("Year")
    plt.ylabel("Total Emissions")
    plt.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(model, feature_names, save_path):
    """Plot and save the feature importances from a linear regression model."""
    import numpy as np

    coefficients = model.coef_
    importance = pd.Series(coefficients, index=feature_names).sort_values()

    plt.figure(figsize=(10, 6))
    importance.plot(kind='barh', color='skyblue')
    plt.title("Feature Importance (Model Coefficients)")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

