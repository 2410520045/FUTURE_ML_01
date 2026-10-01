# Sales & Demand Forecasting for Businesses
# Complete project generated from a realistic sample retail dataset.
# Replace the generated historical dataset with your own CSV if required.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load historical_monthly_sales.csv
monthly = pd.read_csv("historical_monthly_sales.csv", parse_dates=["Date"])

def add_features(data):
    x = data.copy()
    x["Year"] = x["Date"].dt.year
    x["Month"] = x["Date"].dt.month
    x["Quarter"] = x["Date"].dt.quarter
    x["Month_Sin"] = np.sin(2*np.pi*x["Month"]/12)
    x["Month_Cos"] = np.cos(2*np.pi*x["Month"]/12)
    x["Lag_1"] = x["Monthly_Sales"].shift(1)
    x["Lag_2"] = x["Monthly_Sales"].shift(2)
    x["Lag_3"] = x["Monthly_Sales"].shift(3)
    x["Rolling_3"] = x["Monthly_Sales"].shift(1).rolling(3).mean()
    return x

features = [
    "Year","Month","Quarter","Month_Sin","Month_Cos",
    "Lag_1","Lag_2","Lag_3","Rolling_3"
]

# Chronological validation
for product in monthly["Product"].unique():
    p = monthly[monthly["Product"] == product].sort_values("Date")
    p = add_features(p).dropna()

    train = p.iloc[:-6]
    test = p.iloc[-6:]

    model = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=42
    )
    model.fit(train[features], train["Monthly_Sales"])
    pred = model.predict(test[features])

    mae = mean_absolute_error(test["Monthly_Sales"], pred)
    rmse = np.sqrt(mean_squared_error(test["Monthly_Sales"], pred))
    mape = np.mean(
        np.abs((test["Monthly_Sales"] - pred) / test["Monthly_Sales"])
    ) * 100

    print(product)
    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("MAPE:", round(mape, 2), "%")
