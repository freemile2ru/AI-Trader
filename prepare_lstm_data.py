import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Load Data
file_path = "btc_usdt_1h_technical_larger.csv"  # Ensure the correct file path
df = pd.read_csv(file_path)

# Select Features (Price + Indicators)
features = ['close', 'volume', 'ATR', 'RSI', 'EMA_10', 'MACD' ,'bollinger_h','bollinger_l','SMA_50','SMA_200']
df = df[features]

# Normalize Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Create Sequences for LSTM
def create_sequences(data, seq_length=50):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):
        X.append(data[i : i + seq_length])  # Past 50 timestamps as features
        y.append(data[i + seq_length, 0])   # Next closing price as label
    return np.array(X), np.array(y)

SEQ_LENGTH = 50  # Use the past 50 hours for prediction
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split into Train & Test Sets
split = int(0.8 * len(X))  # 80% Training, 20% Testing
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Save Data for Training
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ LSTM Training Data Prepared & Saved!")
