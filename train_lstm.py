import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("btc_usdt_5m_technical_large.csv")  # Ensure the dataset includes FA + TA

# Feature Engineering: Create new indicators
df["ATR"] = df["high"] - df["low"]  # Simplified ATR
df["RSI"] = 100 - (100 / (1 + (df["close"].pct_change() + 1).rolling(14).mean()))
df["EMA_10"] = df["close"].ewm(span=10).mean()
df["MACD"] = df["EMA_10"] - df["close"].ewm(span=26).mean()

# Select features & target
features = ["close", "volume", "ATR", "RSI", "EMA_10", "MACD"]
target = "close"  # Predict future price

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Prepare training data
X, y = [], []
lookback = 50  # Number of past timesteps to use

for i in range(lookback, len(df) - 1):
    X.append(df[features].iloc[i - lookback:i].values)
    y.append(df[target].iloc[i + 1])  # Predict next step

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ✅ Improved LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(lookback, len(features))),
    Dropout(0.2),
    BatchNormalization(),
    
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    
    Dense(32, activation="relu"),
    Dense(1)  # Output layer
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save the trained model
model.save("optimized_lstm_model.h5")
