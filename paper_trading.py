import pandas as pd
import requests
import time
import numpy as np
import tensorflow as tf
import os

# Load trained LSTM model
model = tf.keras.models.load_model("lstm_model_fixed.h5")

# Initialize variables
paper_balance = 10000  # USD
open_trade = None
trade_log = []
BINANCE_FUTURES_URL = "https://fapi.binance.com"
CRYPTO_PANIC_API_KEY = os.getenv("CRYPTO_PANIC_API_KEY")

# Function to fetch live price
def get_live_price(symbol="BTCUSDT"):
    url = f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/price"
    response = requests.get(url, params={"symbol": symbol}).json()
    return float(response["price"])

# Function to fetch FA data
def get_news_sentiment():
    response = requests.get(f"https://cryptopanic.com/api/v1/posts/?auth_token=" + CRYPTO_PANIC_API_KEY).json()
    sentiment = np.mean([post["votes"]["positive"] - post["votes"]["negative"] for post in response["results"]])
    return sentiment

# Function to calculate trade confidence
def get_trade_confidence():
    sentiment_score = get_news_sentiment()
    confidence = 50 + (sentiment_score * 5)
    return min(max(confidence, 0), 100)

# Paper Trading Loop
while True:
    current_price = get_live_price()
    predicted_price = current_price * np.random.uniform(0.98, 1.02)
    side = "BUY" if predicted_price > current_price * 1.005 else "SELL"

    # **Trade Confidence Check**
    confidence = get_trade_confidence()
    if confidence < 60:
        print(f"⚠️ Trade Skipped: Confidence {confidence:.2f}% Too Low")
        time.sleep(30)
        continue

    position_size = paper_balance * 0.02 / current_price
    stop_loss = current_price * 0.98 if side == "BUY" else current_price * 1.02
    take_profit = current_price * 1.03 if side == "BUY" else current_price * 0.97

    # **DCA Logic**
    if open_trade:
        unrealized_pnl = (current_price - open_trade["entry_price"]) * open_trade["position_size"] if open_trade["side"] == "BUY" else (open_trade["entry_price"] - current_price) * open_trade["position_size"]

        if unrealized_pnl < -0.02 * paper_balance:
            new_position_size = open_trade["position_size"] * 1.5
            avg_entry_price = ((open_trade["position_size"] * open_trade["entry_price"]) + (new_position_size * current_price)) / (open_trade["position_size"] + new_position_size)

            open_trade.update({"position_size": new_position_size, "entry_price": avg_entry_price, "stop_loss": stop_loss, "take_profit": take_profit})
            print(f"🔄 DCA Applied: New Entry Price = {avg_entry_price:.2f}")
            time.sleep(30)
            continue

    # Place Trade
    open_trade = {"side": side, "position_size": position_size, "entry_price": current_price, "stop_loss": stop_loss, "take_profit": take_profit}
    print(f"📊 AI Paper Trade | {side} | Entry: {current_price:.2f} | Confidence: {confidence:.2f}%")

    # Simulated exit
    time.sleep(30)
