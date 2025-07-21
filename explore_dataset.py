import pandas as pd

# Load dataset
df = pd.read_csv("adult 3.csv")

# Show first few rows
print("First 5 Rows:\n", df.head())

# Basic info
print("\nDataset Info:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check data types
print("\nData Types:")
print(df.dtypes)

# Unique values in target column
print("\nTarget Column Unique Values (e.g., Salary):")
print(df.iloc[:, -1].value_counts())  # Assuming last column is target