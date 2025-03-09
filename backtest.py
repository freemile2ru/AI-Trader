import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

custom_objects = {"mse": MeanSquaredError()}

# Load Trained LSTM Models
model5m = tf.keras.models.load_model("optimized_5m_lstm_model.h5", custom_objects=custom_objects)
model1h = tf.keras.models.load_model("optimized_1h_lstm_model.h5", custom_objects=custom_objects)
model4h = tf.keras.models.load_model("optimized_4h_lstm_model.h5", custom_objects=custom_objects)

# Load dataset
df = pd.read_csv("btc_usdt_1h_technical_large.csv", dtype={"close": float, "ATR": float, "bollinger_l": float, "bollinger_h": float})
df.dropna(inplace=True)

# ✅ Bot Configuration
initial_balance = 100000
balance = initial_balance
open_trade = None
trade_log = []
consecutive_losses = 0

# ✅ Feature Set
FEATURE_COLUMNS = ['close', 'volume', 'ATR', 'RSI', 'EMA_10', 'MACD', 'bollinger_h', 'bollinger_l', 'SMA_50', 'SMA_200']

def preprocess_input(data):
    data = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    min_vals = data.iloc[-50:].min()
    max_vals = data.iloc[-50:].max()
    normalized_data = (data.iloc[-50:] - min_vals) / (max_vals - min_vals)
    normalized_data.fillna(0, inplace=True)
    return normalized_data

def predict_price(symbol_data):
    input_data = preprocess_input(symbol_data)
    if input_data.shape != (50, 10):
        raise ValueError(f"🚨 Shape Mismatch: Expected (50,10), but got {input_data.shape}")
    input_data = input_data.values.reshape(1, 50, 10)
    predicted_change = final_prediction(input_data)
    last_close_price = symbol_data["close"].iloc[-1]
    return last_close_price + (predicted_change * last_close_price)

def final_prediction(input_data):
    pred_5m = model5m.predict(input_data)[0][0]
    pred_1h = model1h.predict(input_data)[0][0]
    pred_4h = model4h.predict(input_data)[0][0]
    return (0.25 * pred_5m) + (0.35 * pred_1h) + (0.4 * pred_4h)

def get_trade_confidence(atr, market_trend="neutral"):
    base_confidence = np.random.uniform(50, 100) if atr > 50 else np.random.uniform(40, 80)
    if market_trend == "bullish":
        base_confidence += 5
    elif market_trend == "bearish":
        base_confidence -= 5
    return min(100, max(40, base_confidence))

# ✅ Risk-Based Position Sizing
def get_dynamic_position_size(balance, max_risk=0.02, leverage=10):
    risk_adjusted_balance = max(balance, initial_balance * 0.2)  # Prevent extreme size reduction
    return (risk_adjusted_balance * max_risk * leverage) / balance

# ✅ Backtesting Logic
for index, row in df.iterrows():
    if index < 50:
        continue

    if consecutive_losses >= 3:
        print(f"⚠️ Circuit Breaker: Skipping trade {index} due to 3 consecutive losses")
        consecutive_losses = 0
        continue

    historical_data = df.iloc[index-50:index]
    predicted_price = predict_price(historical_data)
    current_price = row["close"]
    atr = row["ATR"]

    side = "BUY" if predicted_price > current_price * 1.005 else "SELL"
    confidence = get_trade_confidence(atr)

    if confidence < 50:
        continue

    atr_multiplier = 1.5 if atr > 50 else 2.0
    stop_loss = max(current_price * 0.995, current_price - atr_multiplier * atr) if side == "BUY" else min(current_price * 1.005, current_price + atr_multiplier * atr)
    take_profit = min(current_price * 1.01, current_price + (atr_multiplier * 2.5 * atr)) if side == "BUY" else max(current_price * 0.99, current_price - (atr_multiplier * 2.5 * atr))

    # ✅ **Calculate Risk-to-Reward Ratio (RRR)**
    risk = abs(current_price - stop_loss)
    reward = abs(take_profit - current_price)
    rrr = reward / risk if risk > 0 else 0

    print(f"📊 Paper Trade RRR Calculation: Risk = {risk:.2f}, Reward = {reward:.2f}, RRR = {rrr:.2f}")

    if rrr < 2.0:
        print(f"❌ Paper Trade Rejected: RRR {rrr:.2f} is too low (needs at least 2:1).")
        continue

    if balance < 100:
        print(f"🚨 Insufficient Funds! Stopping at trade {index}. Final Balance: ${balance:.2f}")
        break

    position_size = get_dynamic_position_size(balance)
    exit_price = row["close"]
    profit = (exit_price - current_price) * position_size if side == "BUY" else (current_price - exit_price) * position_size

    balance += profit
    trade_log.append({"symbol": "BTCUSDT", "entry_price": current_price, "exit_price": exit_price, "Side": side, "stop_loss": stop_loss, "Final Balance": balance, "take_profit": take_profit, "result": "WIN" if profit > 0 else "LOST", "pnl": profit, "position_size": position_size})

    consecutive_losses = 0 if profit > 0 else consecutive_losses + 1

# ✅ Performance Metrics
total_trades = len(trade_log)
winning_trades = sum(1 for trade in trade_log if trade["pnl"] > 0)
win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
average_profit = sum(trade["pnl"] for trade in trade_log) / total_trades if total_trades > 0 else 0
balances = [initial_balance] + [trade["Final Balance"] for trade in trade_log]
max_drawdown = max([max(balances[:i+1]) - balance for i, balance in enumerate(balances)])

print(f"✅ Backtest Complete. Win Rate: {win_rate:.2f}% | Avg Profit/Trade: ${average_profit:.2f} | Max Drawdown: ${max_drawdown:.2f}")

# Save Results
pd.DataFrame(trade_log).to_csv("backtest_results.csv", index=False)
print("✅ Backtesting Completed. Results saved to backtest_results.csv")

# ✅ **Plot Performance**
results_df = pd.DataFrame(trade_log)
results_df["Cumulative Balance"] = initial_balance + results_df["pnl"].cumsum()

plt.figure(figsize=(12, 6))
plt.plot(results_df["Cumulative Balance"], label="Balance Over Time", marker="o")
plt.xlabel("Trades")
plt.ylabel("Balance ($)")
plt.title("Backtest Performance")
plt.legend()
plt.show()
