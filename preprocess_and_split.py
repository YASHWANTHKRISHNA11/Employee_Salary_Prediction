import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("adult 3.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Replace '?' with NaN and drop rows with missing values
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)

# Strip whitespace from string values
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop("salary", axis=1)
y = df["salary"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save datasets
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("✅ Preprocessing complete. Data ready for regression model.")