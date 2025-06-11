import pandas as pd

def load_data(filepath):
    """Load dataset from CSV"""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean CO2 emissions data"""
    df = df.dropna(axis=1)  # Remove columns with NaN values

    df_long = df.melt(
        id_vars=['Country', 'Data source', 'Sector', 'Gas', 'Unit'],
        var_name='Year',
        value_name='Emissions'
    )

    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    df_long = df_long.dropna(subset=['Year', 'Emissions'])

    return df_long


