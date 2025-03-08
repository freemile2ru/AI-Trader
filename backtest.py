import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt


custom_objects = {"mse": MeanSquaredError()}

# Load Trained LSTM Model
model5m = tf.keras.models.load_model("optimized_5m_lstm_model.h5", custom_objects=custom_objects)
model1h = tf.keras.models.load_model("optimized_1h_lstm_model.h5", custom_objects=custom_objects)
model4h = tf.keras.models.load_model("optimized_4h_lstm_model.h5", custom_objects=custom_objects)


# Load historical dataset
# Load historical dataset with correct types
df = pd.read_csv("btc_usdt_1h_technical_large.csv", dtype={"close": float, "ATR": float, "bollinger_l": float, "bollinger_h": float})


# Initialize variables
initial_balance = 100000
balance = initial_balance
open_trade = None
trade_log = []

# Function to preprocess input data for LSTM
# ✅ Ensure input data matches the model's expected feature set
FEATURE_COLUMNS = ['close', 'volume', 'ATR', 'RSI', 'EMA_10', 'MACD', 'bollinger_h', 'bollinger_l', 'SMA_50', 'SMA_200']
def preprocess_input(data):
    # ✅ Select only the required 10 features
    data = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

    # ✅ Normalize only on the last 50 rows
    min_vals = data.iloc[-50:].min()
    max_vals = data.iloc[-50:].max()

    # ✅ Prevent division by zero
    normalized_data = (data.iloc[-50:] - min_vals) / (max_vals - min_vals)
    normalized_data.fillna(0, inplace=True)  # Replace NaNs with 0

    return normalized_data




def predict_price(symbol_data):
    input_data = preprocess_input(symbol_data)

    # ✅ Ensure the input has exactly 50 rows and 10 columns
    if input_data.shape != (50, 10):
        raise ValueError(f"🚨 Shape Mismatch: Expected (50,10), but got {input_data.shape}")

    input_data = input_data.values.reshape(1, 50, 10)  # ✅ Reshape correctly

    predicted_change = final_prediction(input_data)
    last_close_price = symbol_data["close"].iloc[-1]  # Use last known close price

    return last_close_price + (predicted_change * last_close_price)  # Apply change


def final_prediction(input_data):
    pred_5m = model5m.predict(input_data)[0][0]
    pred_1h = model1h.predict(input_data)[0][0]
    pred_4h = model4h.predict(input_data)[0][0]

    print(f"🔮 Predictions: 5m={pred_5m}, 1h={pred_1h}, 4h={pred_4h}")

    predicted_change = (0.25 * pred_5m) + (0.35 * pred_1h) + (0.4 * pred_4h)
    return predicted_change  # Return change, not direct price

def get_market_trend(row):
    """Determines market trend based on SMA, MACD, and RSI."""
    if row["SMA_50"] > row["SMA_200"] and row["MACD"] > 0 and row["RSI"] > 60:
        return "bullish"
    elif row["SMA_50"] < row["SMA_200"] and row["MACD"] < 0 and row["RSI"] < 40:
        return "bearish"
    else:
        return "neutral"


def get_trade_confidence(atr, market_trend="neutral"):
    base_confidence = np.random.uniform(60, 100) if atr > 50 else np.random.uniform(40, 80)  # Higher confidence in high volatility

    if market_trend == "bullish":
        base_confidence += 5  # Small boost in bullish markets
    elif market_trend == "bearish":
        base_confidence -= 5  # Reduce confidence in bearish conditions

    return min(100, max(40, base_confidence))  # Keep between 40-100

def execute_trade(index, side, stop_loss, take_profit, df):
    """
    Simulate trade execution by checking historical price movements.
    - If SL is hit first, exit at SL.
    - If TP is hit first, exit at TP.
    - If neither is hit, exit at the last available price.
    """

    trade_data = df.iloc[index : min(index + 50, len(df))]  # Look ahead 50 candles max

    if side == "BUY":
        # Check if stop-loss was hit first
        if (trade_data["low"] <= stop_loss).any():
            exit_price = trade_data.loc[trade_data["low"] <= stop_loss, "low"].iloc[0]
        # Check if take-profit was hit first
        elif (trade_data["high"] >= take_profit).any():
            exit_price = trade_data.loc[trade_data["high"] >= take_profit, "high"].iloc[0]
        else:
            exit_price = trade_data["close"].iloc[-1]  # Exit at last known price

    elif side == "SELL":
        # Check if stop-loss was hit first
        if (trade_data["high"] >= stop_loss).any():
            exit_price = trade_data.loc[trade_data["high"] >= stop_loss, "high"].iloc[0]
        # Check if take-profit was hit first
        elif (trade_data["low"] <= take_profit).any():
            exit_price = trade_data.loc[trade_data["low"] <= take_profit, "low"].iloc[0]
        else:
            exit_price = trade_data["close"].iloc[-1]  # Exit at last known price

    return exit_price


# Simulate AI trading
for index, row in df.iterrows():
    if index < 50:
        continue

    historical_data = df.iloc[index-50:index]
    predicted_price = predict_price(historical_data)
    current_price = row["close"]
    atr = row["ATR"]
    support, resistance = row["bollinger_l"], row["bollinger_h"]
    
    side = "BUY" if predicted_price > current_price * 1.005 else "SELL"

    # **Trade Confidence Check**
    confidence = get_trade_confidence(atr, get_market_trend(row) )
    if confidence < 60:
        continue  # Skip trade if confidence is low

    atr_multiplier = 1.5 if atr > 50 else 2.0  # Adjust based on ATR
    tp_multiplier = 2.5 if atr > 50 else 3.0  # Dynamic TP based on volatility

    stop_loss = max(current_price * 0.995, current_price - atr_multiplier * atr) if side == "BUY" else min(current_price * 1.005, current_price + atr_multiplier * atr)
    take_profit = min(current_price * 1.01, current_price + (tp_multiplier * atr)) if side == "BUY" else max(current_price * 0.99, current_price - (tp_multiplier * atr))



    # **DCA Logic**
    if open_trade:
        unrealized_pnl = (current_price - open_trade["entry_price"]) * open_trade["position_size"] if open_trade["side"] == "BUY" else (open_trade["entry_price"] - current_price) * open_trade["position_size"]

        if unrealized_pnl < -0.02 * balance:  # If loss is more than 2% of balance, apply DCA
            atr_dca_multiplier = 0.3 if atr < 20 else 0.5  # Adjust DCA size dynamically
            additional_size = open_trade["position_size"] * atr_dca_multiplier
            new_total_size = open_trade["position_size"] + additional_size

            avg_entry_price = ((open_trade["position_size"] * open_trade["entry_price"]) + 
                            (additional_size * current_price)) / new_total_size

            open_trade.update({
                "position_size": new_total_size, 
                "entry_price": avg_entry_price, 
                "stop_loss": stop_loss, 
                "take_profit": take_profit
            })

            continue  # Skip opening a new trade

    # ✅ Ensure the user has enough balance before placing a trade
    if balance < 10:  # Stop trading if balance is below $10
        print(f"🚨 Insufficient Funds! Stopping backtest at trade {index}. Final Balance: ${balance:.2f}")
        break
    
    max_loss_per_trade = balance * 0.05  # Risk max 5% per trade

    # ✅ Ensure Stop-Loss does not exceed 5% of balance
    
    
    # Execute trade
    leverage = 10  # Adjust based on your strategy
    risk_per_trade = 0.02  # 2% of balance risked per trade

    position_size = (balance * risk_per_trade * leverage) / current_price
    
    if abs(stop_loss - current_price) * position_size > max_loss_per_trade:
        print(f"⚠️ Adjusting stop-loss to prevent excessive loss")
        stop_loss = current_price - (max_loss_per_trade / position_size) if side == "BUY" else current_price + (max_loss_per_trade / position_size)
        
    exit_price = execute_trade(index, side, stop_loss, take_profit, df)
  
    profit = (exit_price - current_price) * position_size if side == "BUY" else (current_price - exit_price) * position_size
    balance += profit

    trade_log.append({ "symbol": "BTCUSDT", "entry_price": current_price, "exit_price": exit_price, "Side": side,  "stop_loss": stop_loss,  "Final Balance": balance, "take_profit": take_profit, "result": "WIN" if profit > 0 else "LOST", "pnl": profit,  "position_size": position_size})
    open_trade = {"side": side, "position_size": position_size, "entry_price": current_price, "stop_loss": stop_loss, "take_profit": take_profit}


# Win rate
total_trades = len(trade_log)
winning_trades = sum(1 for trade in trade_log if trade["pnl"] > 0)
win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

# Average Profit per Trade
average_profit = sum(trade["pnl"] for trade in trade_log) / total_trades if total_trades > 0 else 0

# Max Drawdown Calculation
balances = [initial_balance] + [trade["Final Balance"] for trade in trade_log]
max_drawdown = max([max(balances[:i+1]) - balance for i, balance in enumerate(balances)])

# Print Summary
print(f"✅ Backtest Complete. Results Saved!")
print(f"📊 Win Rate: {win_rate:.2f}%")
print(f"💰 Average Profit per Trade: ${average_profit:.2f}")
print(f"📉 Max Drawdown: ${max_drawdown:.2f}")

# Save Results
pd.DataFrame(trade_log).to_csv("backtest_results.csv", index=False)
print("✅ Backtesting Completed. Results saved to backtest_results.csv")
if balance <= 0:
    print("❌ The bot lost all funds! Consider adjusting risk management.")

results_df = pd.DataFrame(trade_log)
results_df["Cumulative Balance"] = initial_balance + results_df["pnl"].cumsum()

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(results_df["Cumulative Balance"], label="Balance Over Time", marker="o")
plt.xlabel("Trades")
plt.ylabel("Balance ($)")
plt.title("Backtest Performance")
plt.legend()

plt.subplot(2, 1, 2)
plt.bar(range(len(results_df)), results_df["pnl"], color=['green' if p > 0 else 'red' for p in results_df["pnl"]])
plt.xlabel("Trades")
plt.ylabel("Profit/Loss")
plt.title("Trade Profitability")

plt.tight_layout()
plt.show()

