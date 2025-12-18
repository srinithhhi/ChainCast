# ChainCast :Crypto Price Predictor
A professional, end-to-end data science project for predicting cryptocurrency closing prices using machine learning and financial time series analysis. Built using Python, scikit-learn, and Streamlit, this tool enables users to explore market behavior and generate real-time predictions with an interactive dashboard.
---
## Project Overview
This application forecasts the next closing price of various cryptocurrencies by leveraging historical OHLCV (Open, High, Low, Close, Volume) data and engineered features like moving averages, volatility, and daily return. Features are fed into a trained regression model, and insights are displayed via an intuitive, modern dashboard.
---
## Features
- Multi-coin support from dynamic datasets (25+ crypto assets)
- Technical feature engineering (MA, volatility, return)
- Trained regression model using RandomForestRegressor
- Fully interactive dashboard with input sliders and live predictions
- Plotly-powered visualizations: price trends, correlation, feature importances
- Professional UI with theme customization (dark mode, monospace font)
- Session history tracking for forecast comparisons

---
## Technologies Used
- Python 3.9+
- Streamlit – UI and dashboard framework
- scikit-learn – Regression modeling
- pandas / NumPy – Data processing
- plotly / seaborn / matplotlib – Visual analytics
- joblib – Model serialization

---

## Folder Structure
crypto-price-predictor/
├── crypto-coins/ # Folder containing all coin .csv files
├── .streamlit/ # Streamlit theme config (dark mode + monospace)
│ └── config.toml
├── app.py # Streamlit app (visualization and prediction)
├── train_crypto_model.py # Model training pipeline
├── crypto_regression_model.pkl # Trained model (Random Forest)
├── model_features.pkl # Pickled feature list
└── README.md # Project documentation


---

## How It Works

### 1. Data

Each cryptocurrency has an individual CSV file under `/crypto-coins/`, containing historical pricing data (OHLCV).

### 2. Feature Engineering

The model uses:
- MA_7: 7-day Moving Average
- MA_30: 30-day Moving Average
- Volatility: 7-day Standard Deviation
- Daily Return: Percent change between days

### 3. Modeling
A RandomForestRegressor is trained on user-defined features and saved as `crypto_regression_model.pkl`.
### 4. Deployment
The interactive app (`app.py`) allows the user to:
- Select a coin and date range
- View price trends and charts
- Adjust input parameters
- Predict the next closing price instantly
---
## Screenshot

<img width="1916" height="996" alt="image" src="https://github.com/user-attachments/assets/779b9b1e-986a-401d-81f6-656c7bdd985f" />
<img width="1917" height="994" alt="image" src="https://github.com/user-attachments/assets/81a53362-3c0c-4288-9897-495a7edb38df" />

## Conclusion

This project showcases a complete machine learning workflow applied to real-world financial data. From feature engineering and model training to interactive deployment with Streamlit, the Crypto Price Predictor demonstrates the practical use of data science tools to build intelligent, user-friendly applications.



