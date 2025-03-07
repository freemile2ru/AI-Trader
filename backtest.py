import pandas as pd
import numpy as np
import tensorflow as tf

# Load trained LSTM model
model = tf.keras.models.load_model("lstm_model_fixed.h5")

# Load historical dataset
df = pd.read_csv("btc_usdt_technical_large.csv")

# Initialize variables
initial_balance = 10000
balance = initial_balance
open_trade = None
trade_log = []

# Function to preprocess input data for LSTM
def preprocess_input(data):
    min_vals = data.min()
    max_vals = data.max()
    return (data - min_vals) / (max_vals - min_vals)

# Function to predict price movement
def predict_price(symbol_data):
    input_data = preprocess_input(symbol_data).values.reshape(1, -1, symbol_data.shape[1])
    prediction = model.predict(input_data)[0][0]
    min_price, max_price = symbol_data["low"].min(), symbol_data["high"].max()
    return prediction * (max_price - min_price) + min_price

# Function to calculate trade confidence (Mock FA, since past FA data isn't available)
def get_trade_confidence():
    return np.random.uniform(50, 100)  # Simulate confidence score

# Simulate AI trading
for index, row in df.iterrows():
    if index < 50:
        continue

    historical_data = df.iloc[index-50:index]
    predicted_price = predict_price(historical_data)
    current_price = row["close"]
    atr = row["ATR_1h"]
    support, resistance = row["bollinger_l_1h"], row["bollinger_h_1h"]
    
    side = "BUY" if predicted_price > current_price * 1.005 else "SELL"

    # **Trade Confidence Check**
    confidence = get_trade_confidence()
    if confidence < 60:
        continue  # Skip trade if confidence is low

    # SL & TP Logic
    stop_loss = max(current_price - 1.5 * atr, support * 0.99) if side == "BUY" else min(current_price + 1.5 * atr, resistance * 1.01)
    take_profit = min(current_price + 3 * atr, resistance * 1.01) if side == "BUY" else max(current_price - 3 * atr, support * 0.99)

    # **DCA Logic**
    if open_trade:
        unrealized_pnl = (current_price - open_trade["entry_price"]) * open_trade["position_size"] if open_trade["side"] == "BUY" else (open_trade["entry_price"] - current_price) * open_trade["position_size"]

        if unrealized_pnl < -0.02 * balance:
            new_position_size = open_trade["position_size"] * 1.5
            avg_entry_price = ((open_trade["position_size"] * open_trade["entry_price"]) + (new_position_size * current_price)) / (open_trade["position_size"] + new_position_size)
            
            open_trade.update({"position_size": new_position_size, "entry_price": avg_entry_price, "stop_loss": stop_loss, "take_profit": take_profit})
            continue

    # Execute trade
    position_size = balance * 0.02 / current_price
    exit_price = np.random.choice([take_profit, stop_loss])
    profit = (exit_price - current_price) * position_size if side == "BUY" else (current_price - exit_price) * position_size
    balance += profit

    trade_log.append({"Entry Price": current_price, "Exit Price": exit_price, "Side": side, "Profit": profit})

    open_trade = {"side": side, "position_size": position_size, "entry_price": current_price, "stop_loss": stop_loss, "take_profit": take_profit}

# Save Results
pd.DataFrame(trade_log).to_csv("backtest_results.csv", index=False)
print("✅ Backtesting Completed. Results saved to backtest_results.csv")
