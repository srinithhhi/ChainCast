# train_crypto_model.py

import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# -----------------------------
# ✅ Step 1: Load and merge all CSVs
# -----------------------------
file_paths = glob.glob("crypto-coins/*.csv")  # Your folder with 26+ CSVs
dfs = []

for file in file_paths:
    df = pd.read_csv(file)
    coin_name = os.path.basename(file).replace("coin_", "").replace(".csv", "")
    df["Symbol"] = coin_name
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)

# -----------------------------
# ✅ Step 2: Feature engineering
# -----------------------------
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df = merged_df.sort_values(['Symbol', 'Date'])

merged_df['Daily_Return'] = merged_df.groupby('Symbol')['Close'].pct_change()
merged_df['MA_7'] = merged_df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(7).mean())
merged_df['MA_30'] = merged_df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(30).mean())
merged_df['Volatility'] = merged_df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(7).std())

merged_df.dropna(inplace=True)  # Clean up NA rows from moving averages

# -----------------------------
# ✅ Step 3: Model training
# -----------------------------
target = 'Close'
cat_features = ['Symbol']
num_features = ['Open', 'High', 'Low', 'Volume', 'Daily_Return', 'MA_7', 'MA_30', 'Volatility']

X = merged_df[num_features + cat_features]
y = merged_df[target]

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
pipeline.fit(X_train, y_train)

# -----------------------------
# ✅ Step 4: Save the model
# -----------------------------
joblib.dump(pipeline, 'crypto_regression_model.pkl')
joblib.dump(num_features + cat_features, 'model_features.pkl')

print("Model trained and saved!")
