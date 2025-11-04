import pandas as pd

# Read the Parquet file
df = pd.read_parquet("Data/dataset.parquet")


# Show the first few rows
print(df.head())

# See info about columns and data types
print(df.info())
