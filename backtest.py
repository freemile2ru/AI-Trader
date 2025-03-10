import pandas as pd
import numpy as np
import tensorflow as tf
import ta
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError

# ✅ Load Trained LSTM Models
custom_objects = {"mse": MeanSquaredError()}
model5m = tf.keras.models.load_model("optimized_5m_lstm_model.h5", custom_objects=custom_objects)
model1h = tf.keras.models.load_model("optimized_1h_lstm_model.h5", custom_objects=custom_objects)
model4h = tf.keras.models.load_model("optimized_4h_lstm_model.h5", custom_objects=custom_objects)

# ✅ Load dataset
df = pd.read_csv("btc_usdt_1h_technical_large.csv").dropna()

# ✅ Bot Configuration
initial_balance = 100000
balance = initial_balance
consecutive_losses = 0
trade_log = []

# ✅ Feature Set
FEATURE_COLUMNS = ['close', 'volume', 'ATR', 'RSI', 'EMA_10', 'MACD', 'bollinger_h', 'bollinger_l', 'SMA_50', 'SMA_200']

def preprocess_input(data):
    data = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    min_vals = data.iloc[-50:].min()
    max_vals = data.iloc[-50:].max()
    return (data.iloc[-50:] - min_vals) / (max_vals - min_vals)

def final_prediction(input_data):
    pred_5m = model5m.predict(input_data)[0][0]
    pred_1h = model1h.predict(input_data)[0][0]
    pred_4h = model4h.predict(input_data)[0][0]
    return (0.25 * pred_5m) + (0.35 * pred_1h) + (0.4 * pred_4h)

def predict_price(symbol_data):
    input_data = preprocess_input(symbol_data)
    if input_data.shape != (50, 10):
        raise ValueError(f"🚨 Shape Mismatch: Expected (50,10), got {input_data.shape}")
    return symbol_data['close'].iloc[-1] + (final_prediction(input_data.values.reshape(1, 50, 10)) * symbol_data['close'].iloc[-1])

def get_trade_confidence(atr, trend, win_rate):
    confidence = 50 + (trend * 20) + ((win_rate - 50) * 0.5) + ((atr / 10) * 5)
    return min(100, max(40, confidence))

def get_dynamic_position_size(balance, risk_per_trade=0.02, leverage=10):
    return (balance * risk_per_trade * leverage) / balance

def simulate_trade_execution(row, stop_loss, take_profit, side):
    """Simulates price movements to determine trade exit price."""
    if side == "BUY":
        if row["low"] <= stop_loss:
            return stop_loss, "LOSS"
        elif row["high"] >= take_profit:
            return take_profit, "WIN"
    else:  # SELL trade
        if row["high"] >= stop_loss:
            return stop_loss, "LOSS"
        elif row["low"] <= take_profit:
            return take_profit, "WIN"
    return row["close"], "LOSS"  # Default exit at closing price

# ✅ Backtesting Logic
for index, row in df.iterrows():
    if index < 50:
        continue
    
    if consecutive_losses >= 3:
        consecutive_losses = 0
        continue
    
    historical_data = df.iloc[index-50:index]
    predicted_price = predict_price(historical_data)
    current_price = row["close"]
    atr = row["ATR"]
    trend = np.sign(row["SMA_50"] - row["SMA_200"])
    win_rate = 60  # Placeholder, should be replaced with actual backtest results
    confidence = get_trade_confidence(atr, trend, win_rate)
    
    if confidence < 50:
        continue
    
    side = "BUY" if predicted_price > current_price * 1.005 else "SELL"
    atr_multiplier = 1.5 if atr > 50 else 2.0
    stop_loss = current_price - atr_multiplier * atr if side == "BUY" else current_price + atr_multiplier * atr
    take_profit = current_price + (atr_multiplier * 2.5 * atr) if side == "BUY" else current_price - (atr_multiplier * 2.5 * atr)
    
    risk = abs(current_price - stop_loss)
    reward = abs(take_profit - current_price)
    rrr = reward / risk if risk > 0 else 0
    
    if rrr < 2.0:
        continue
    
    if balance < 100:
        break
    
    position_size = get_dynamic_position_size(balance)
    exit_price, result = simulate_trade_execution(row, stop_loss, take_profit, side)
    profit = (exit_price - current_price) * position_size if side == "BUY" else (current_price - exit_price) * position_size
    
    balance += profit
    trade_log.append({"symbol": "BTCUSDT", "entry_price": current_price, "exit_price": exit_price, "side": side, "stop_loss": stop_loss, "final_balance": balance, "take_profit": take_profit, "result": result, "pnl": profit, "position_size": position_size})
    consecutive_losses = 0 if profit > 0 else consecutive_losses + 1

# ✅ Performance Metrics
total_trades = len(trade_log)
winning_trades = sum(1 for trade in trade_log if trade["pnl"] > 0)
win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
balances = [initial_balance] + [trade["final_balance"] for trade in trade_log]
max_drawdown = max([max(balances[:i+1]) - balance for i, balance in enumerate(balances)])

print(f"✅ Backtest Complete. Win Rate: {win_rate:.2f}% | Max Drawdown: ${max_drawdown:.2f}")

# Save Results
pd.DataFrame(trade_log).to_csv("backtest_results.csv", index=False)

# ✅ Plot Performance
results_df = pd.DataFrame(trade_log)
results_df["Cumulative Balance"] = initial_balance + results_df["pnl"].cumsum()
plt.figure(figsize=(12, 6))
plt.plot(results_df["Cumulative Balance"], label="Balance Over Time", marker="o")
plt.xlabel("Trades")
plt.ylabel("Balance ($)")
plt.title("Backtest Performance")
plt.legend()
plt.show()
