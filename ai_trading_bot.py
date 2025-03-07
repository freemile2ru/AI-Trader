import os
import time
import numpy as np
import tensorflow as tf
import requests
import ccxt
import re
from dotenv import load_dotenv
from urllib.parse import urlencode
import hashlib
import hmac
import json
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
import nltk
import csv
import datetime
import threading
import json
import pandas as pd
from websocket import WebSocketApp
import tweepy
import random


# Download VADER model
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load Binance API Keys
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY")
CRYPTO_PANIC_API_KEY = os.getenv("CRYPTO_PANIC_API_KEY")

# Twitter API credentials (Ensure you have a valid Twitter Developer API Key)
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Initialize Twitter API Client
client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)

# Binance Futures API Base URL
BINANCE_FUTURES_URL = "https://fapi.binance.com"

# Load Trained LSTM Model
model = tf.keras.models.load_model("lstm_model_fixed.h5")

# Initialize Binance Futures Client
binance = ccxt.binance({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "options": {"defaultType": "future", "adjustForTimeDifference": True},
    "rateLimit": 1200,
})
binance.load_markets()

def get_whale_exchange_flows(symbol="BTCUSDT"):
    try:
        endpoint = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(endpoint, params={"symbol": symbol}).json()

        inflow = float(response["quoteVolume"])  # Total USDT volume traded in 24h
        buy_volume = float(response["volume"])   # Total BTC volume traded in 24h

        print(f"📊 BTC Exchange Volume: {inflow:.2f} USDT | Buy Volume: {buy_volume:.4f} BTC")
        return inflow, buy_volume

    except Exception as e:
        print(f"❌ Error Fetching Whale Exchange Flow: {e}")
        return 0, 0


def get_news_sentiment():
    try:
        total_sentiment = 0
        news_count = 0

        # ✅ Fetch news from CryptoPanic
        response = requests.get(f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTO_PANIC_API_KEY}&filter=hot").json()

        for news in response['results']:
            title = news['title'].lower()
            sentiment_score = sia.polarity_scores(title)["compound"]  # AI Sentiment Analysis
            total_sentiment += sentiment_score
            news_count += 1

        # ✅ Compute average sentiment score
        avg_sentiment = total_sentiment / news_count if news_count > 0 else 0

        print(f"📰 News Sentiment Score (CryptoPanic): {avg_sentiment:.2f}")
        return avg_sentiment

    except Exception as e:
        print(f"❌ Error Fetching News Sentiment: {e}")
        return 0  # Default to neutral if error occurs

    
# Function to generate Binance API signature
def generate_signature(params):
    query_string = urlencode(params)
    return hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def get_fundamental_score():
    news_sentiment = get_news_sentiment()
    exchange_inflow, buy_volume = get_whale_exchange_flows()  # ✅ New Whale Tracking
    funding_rate, open_interest = get_funding_rate()

    fundamental_score = 50  # Start at neutral 50%

    # ✅ News Sentiment Weight (40%)
    fundamental_score += min(max(news_sentiment * 40, -20), 20)

    # ✅ Whale Activity (Exchange Inflow) Weight (30%)
    if exchange_inflow > 1_000_000_000:  # If total inflow > $1B in 24h
        fundamental_score += 15  # Indicates strong market interest
    elif exchange_inflow < 500_000_000:  # If total inflow < $500M
        fundamental_score -= 10  # Market slowing down

    # ✅ Buy Volume Weight (Whale Buying Activity)
    if buy_volume > 10_000:  # If more than 10K BTC bought in 24h
        fundamental_score += 10  # Whales accumulating BTC
    elif buy_volume < 5_000:  # If fewer than 5K BTC bought
        fundamental_score -= 5  # Weak buy pressure

    # ✅ Funding Rate Weight (20%)
    if abs(funding_rate) > 0.03:  # Extreme funding rate
        fundamental_score -= 15  # Market imbalance detected

    # ✅ Open Interest Weight (10%)
    if open_interest > 100_000:
        fundamental_score += 10  # More open positions = strong market

    return max(0, min(100, fundamental_score))  # Keep score between 0-100


def get_funding_rate(symbol="BTCUSDT"):
    try:
        response = requests.get(f"{BINANCE_FUTURES_URL}/fapi/v1/premiumIndex", params={"symbol": symbol}).json()
        
        funding_rate = float(response["lastFundingRate"])
        open_interest = float(response.get("openInterest", 0))

        print(f"📊 Funding Rate: {funding_rate:.4f} | Open Interest: {open_interest:.2f}")
        return funding_rate, open_interest

    except Exception as e:
        print(f"❌ Error Fetching Funding Rate: {e}")
        return 0, 0


# Function to determine market volatility and adjust trading frequency
def get_market_volatility(symbol="BTC/USDT"):
    binance_symbol = symbol.replace("/", "")
    endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr"
    
    try:
        response = requests.get(endpoint, params={"symbol": binance_symbol}).json()
        price_change_pct = abs(float(response["priceChangePercent"]))  # 24h price change percentage
        
        # Adjust trading frequency based on volatility
        if price_change_pct > 5:  # High volatility
            return 60  # Check every 1 minute
        elif price_change_pct > 2:  # Medium volatility
            return 300  # Check every 5 minutes
        else:  # Low volatility
            return 900  # Check every 15 minutes
        
    except Exception as e:
        print(f"❌ Error Fetching Market Volatility: {e}")
        return 900  # Default to 15 minutes if API fails
    
# Function to get min/max price from Binance 24hr stats
def get_price_min_max(symbol="BTC/USDT"):
    binance_symbol = symbol.replace("/", "")
    endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr"
    
    try:
        response = requests.get(endpoint, params={"symbol": binance_symbol}).json()
        min_price = float(response["lowPrice"])  # Lowest price in last 24h
        max_price = float(response["highPrice"])  # Highest price in last 24h
        return min_price, max_price
    except Exception as e:
        print(f"❌ Error Fetching Min/Max Price: {e}")
        return None, None


# Function to fetch market-wide indicators
def get_market_indicators():
    try:
        response = requests.get("https://api.coingecko.com/api/v3/global").json()
        btc_dominance = response["data"]["market_cap_percentage"]["btc"]
        return btc_dominance
    except Exception as e:
        print(f"❌ Error Fetching Market Indicators: {e}")
        return None

# Function to fetch real-time Binance Futures market data
def get_real_time_data(symbol="BTC/USDT"):
    binance_symbol = symbol.replace("/", "")
    endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
    params = {"symbol": binance_symbol, "interval": "1h", "limit": 50}
    
    response = requests.get(endpoint, params=params)
    data = response.json()
    
    ohlcv = np.array([[float(entry[i]) for i in range(1, 6)] for entry in data])
    close_open_diff = (ohlcv[:, 3] - ohlcv[:, 0]).reshape(-1, 1)
    high_low_diff = (ohlcv[:, 1] - ohlcv[:, 2]).reshape(-1, 1)
    volume_change = np.append([0], np.diff(ohlcv[:, 4])).reshape(-1, 1)
    
    full_features = np.hstack([ohlcv, close_open_diff, high_low_diff, volume_change])
    min_vals, max_vals = full_features.min(axis=0), full_features.max(axis=0)
    normalized_features = (full_features - min_vals) / (max_vals - min_vals)
    
    return np.expand_dims(normalized_features, axis=0)

# Function to predict price movement using LSTM model
def predict_price_movement(symbol="BTC/USDT"):
    input_data = get_real_time_data(symbol)
    prediction = model.predict(input_data)[0][0]  # Get single prediction
    
    # Get real-time price range from Binance
    min_price, max_price = get_price_min_max(symbol)

    # Proper denormalization formula
    predicted_price = prediction * (max_price - min_price) + min_price  

    # Get real-time futures price
    futures_price = float(requests.get(f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/price", params={"symbol": symbol.replace("/", "")}).json()["price"])

    print(f"📙 Predicted Price: {predicted_price:.2f} | Current Price: {futures_price:.2f}")
    
    return predicted_price, futures_price


def get_dynamic_sl_tp(symbol="BTC/USDT", current_price=0, side="BUY"):
    try:
        binance_symbol = symbol.replace("/", "")
        endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
        params = {"symbol": binance_symbol, "interval": "1h", "limit": 15}  # Fetch 15 candles to ensure enough data

        response = requests.get(endpoint, params=params).json()

        if not response or len(response) < 15:
            print("❌ Error: Binance API did not return enough data for ATR calculation.")
            return None, None  # Prevents SL/TP from being invalid

        highs = np.array([float(entry[2]) for entry in response[-14:]])  # Take last 14 highs
        lows = np.array([float(entry[3]) for entry in response[-14:]])   # Take last 14 lows
        closes = np.array([float(entry[4]) for entry in response[-15:]]) # Take last 15 closes to ensure match

        if len(highs) != 14 or len(lows) != 14 or len(closes) != 15:
            print(f"❌ ATR Calculation Error: Data length mismatch (Highs: {len(highs)}, Lows: {len(lows)}, Closes: {len(closes)})")
            return None, None  # Prevents using invalid SL/TP

        # ✅ Fix: Ensure shape alignment before ATR calculation
        tr = np.maximum(highs - lows, np.maximum(abs(highs - closes[:-1]), abs(lows - closes[:-1])))

        atr = np.mean(tr)

        # Get support/resistance levels
        support, resistance = get_support_resistance(symbol)

        if support is None or resistance is None:
            print("❌ Error: Support/Resistance levels not found.")
            return None, None  # Prevents using invalid SL/TP

        # Adjust SL & TP based on ATR & S/R levels
        if side == "BUY":
            stop_loss = max(current_price - 1.5 * atr, support * 0.99)
            take_profit = min(current_price + 3 * atr, resistance * 1.01)
        else:
            stop_loss = min(current_price + 1.5 * atr, resistance * 1.01)
            take_profit = max(current_price - 3 * atr, support * 0.99)

        return round(stop_loss, 2), round(take_profit, 2)

    except Exception as e:
        print(f"❌ Error Fetching ATR-based SL/TP: {e}")
        return None, None


def log_trade(symbol, side, position_size, entry_price, stop_loss, take_profit, exit_price, result, pnl, trade_type="regular"):
    log_file = "trade_logs.csv"
    headers = ["timestamp", "symbol", "side", "size", "entry_price", "exit_price", "stop_loss", "take_profit", "result", "pnl", "trade_type"]

    # Create the log file if it doesn't exist
    try:
        with open(log_file, "r") as file:
            pass  # File exists, do nothing
    except FileNotFoundError:
        with open(log_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    # Log the trade only after it closes
    with open(log_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol, side, position_size, entry_price, exit_price, stop_loss, take_profit, result, round(pnl, 4), trade_type
        ])

    print(f"📊 Trade Closed & Logged: {trade_type.upper()} {side} {symbol} | PnL: {pnl}")

# Function to determine dynamic trade size based on risk factors
def get_trade_size(symbol="BTC/USDT", risk_per_trade=0.02):
    try:
        # Fetch Binance Futures account balance
        account_info = binance.fetch_balance({"type": "future"})
        print(account_info["info"]["totalWalletBalance"])
        balance = float(account_info["info"]["totalWalletBalance"])  # Available balance in Futures

        # Fetch active positions
        positions = account_info["info"]["positions"]
        active_positions = [pos for pos in positions if float(pos["positionAmt"]) != 0]
        num_active_trades = len(active_positions)

        # Calculate unrealized PnL
        unrealized_pnl = sum(float(pos["unrealizedProfit"]) for pos in active_positions)

        # Get market volatility
        volatility = abs(float(requests.get(f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr", params={"symbol": symbol.replace("/", "")}).json()["priceChangePercent"])) / 100

        # Base position size (Risk % of balance)
        position_size = (risk_per_trade * balance) / volatility if volatility > 0 else (risk_per_trade * balance)

        # Adjust for active trades to prevent overexposure
        if num_active_trades > 3:
            position_size *= 0.7  # Reduce size if too many trades are open
        
        # Adjust for unrealized PnL (Reduce size if floating losses are high)
        if unrealized_pnl < -0.03 * balance:  # If unrealized losses exceed 3% of balance
            position_size *= 0.5  # Reduce trade size by 50%

        # Ensure minimum allowable trade size
        min_size = max(0.005, round((100 / get_price_min_max(symbol)[1]), 6))
        position_size = max(min_size, position_size)

        print(f"📊 Trade Size: {position_size:.6f} BTC | Active Trades: {num_active_trades} | Unrealized PnL: {unrealized_pnl:.2f}")
        return position_size
    
    except Exception as e:
        print(f"❌ Error Fetching Trade Size: {e}")
        return 0.005  # Default to minimum if API fails

def get_open_trades(symbol="BTC/USDT"):
    try:
        account_info = binance.fetch_balance({"type": "future"})
        positions = account_info["info"]["positions"]

        open_positions = [
            pos for pos in positions 
            if pos["symbol"] == symbol.replace("/", "") and float(pos["positionAmt"]) != 0
        ]

        return open_positions  # Returns list of open trades
    except Exception as e:
        print(f"❌ Error Fetching Open Trades: {e}")
        return []
 
def should_dca(symbol="BTC/USDT"):
    open_trades = get_open_trades(symbol)
    if not open_trades:
        return False, 0  # No existing trade → No DCA needed

    for pos in open_trades:
        position_amt = float(pos["positionAmt"])
        unrealized_pnl = float(pos["unrealizedProfit"])

        # ✅ DCA only if unrealized losses exceed 2% of balance AND AI confirms a trend reversal
        account_info = binance.fetch_balance({"type": "future"})
        balance = float(account_info["totalWalletBalance"])

        # AI checks for trend reversal confirmation
        predicted_price, current_price = predict_price_movement(symbol)
        trend_reversal = (predicted_price > current_price and pos["positionSide"] == "SHORT") or (predicted_price < current_price and pos["positionSide"] == "LONG")

        if unrealized_pnl < -0.02 * balance and trend_reversal:  
            new_position_size = abs(position_amt) * 0.5  # Add 50% of current position
            return True, round(new_position_size, 6)  

    return False, 0  # Default to no DCA

def monitor_trade(symbol="BTC/USDT", side="BUY", entry_price=0, stop_loss=0, take_profit=0, trade_type="regular"):
    while True:
        time.sleep(10)  # Check every 10 seconds

        open_position = get_open_position(symbol)
        if not open_position:
            exit_price = float(requests.get(f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/price", params={"symbol": symbol.replace("/", "")}).json()["price"])

            # Determine if it hit SL or TP
            if exit_price >= take_profit:
                result = "WIN"
                pnl = (take_profit - entry_price) * float(open_position["positionAmt"]) if side == "BUY" else (entry_price - take_profit) * float(open_position["positionAmt"])
            elif exit_price <= stop_loss:
                result = "LOSS"
                pnl = (stop_loss - entry_price) * float(open_position["positionAmt"]) if side == "BUY" else (entry_price - stop_loss) * float(open_position["positionAmt"])
            else:
                result = "CLOSED MANUALLY"
                pnl = (exit_price - entry_price) * float(open_position["positionAmt"]) if side == "BUY" else (entry_price - exit_price) * float(open_position["positionAmt"])

            # ✅ Log the closed trade
            log_trade(symbol, side, float(open_position["positionAmt"]), entry_price, stop_loss, take_profit, exit_price, result, pnl, trade_type)
            break  # Exit loop when trade is closed

def is_profitable_setup(symbol="BTC/USDT", side="BUY"):
    try:
        df = pd.read_csv("trade_logs.csv")  # Load past trade logs
        df = df[(df["symbol"] == symbol) & (df["side"] == side)]

        if len(df) < 10:
            return True  # Not enough data, allow trade
        
        win_rate = df["result"].value_counts(normalize=True).get("WIN", 0)

        return win_rate >= 0.6  # Only allow trades with ≥60% historical win rate

    except Exception as e:
        print(f"❌ Error Checking Trade Setup Profitability: {e}")
        return True  # Default to allowing the trade if error

def get_historical_win_rate(symbol="BTC/USDT"):
    try:
        df = pd.read_csv("trade_logs.csv")  # Load past trade logs
        df = df[(df["symbol"] == symbol)]

        if len(df) < 10:
            return 0.5  # Default to 50% win rate if not enough data

        win_rate = df["result"].value_counts(normalize=True).get("WIN", 0)

        return round(win_rate, 2)  # Return win rate as a decimal (e.g., 0.65 for 65%)

    except Exception as e:
        print(f"❌ Error Fetching Historical Win Rate: {e}")
        return 0.5  # Default to 50% if an error occurs


def get_atr(symbol="BTC/USDT", period=14):
    try:
        binance_symbol = symbol.replace("/", "")
        endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
        params = {"symbol": binance_symbol, "interval": "1h", "limit": period + 1}
        
        response = requests.get(endpoint, params=params).json()
        
        highs = np.array([float(entry[2]) for entry in response])
        lows = np.array([float(entry[3]) for entry in response])
        closes = np.array([float(entry[4]) for entry in response])

        # ✅ Ensure arrays are aligned
        highs = highs[-period:]
        lows = lows[-period:]
        closes = closes[-(period + 1):]

        if len(highs) != period or len(lows) != period or len(closes) != period + 1:
            print(f"❌ ATR Calculation Failed: Data mismatch (Highs: {len(highs)}, Lows: {len(lows)}, Closes: {len(closes)})")
            return 0  # Instead of None, return 0 so SL/TP defaults to a minimum

        # Compute True Range (TR)
        tr = np.maximum(highs - lows, np.maximum(abs(highs - closes[:-1]), abs(lows - closes[:-1])))
        atr = np.mean(tr)

        return round(atr, 2)

    except Exception as e:
        print(f"❌ Error Fetching ATR: {e}")
        return 0  # ✅ Return 0 instead of None to prevent SL/TP issues



 
def get_trend_strength(symbol="BTC/USDT"):
    try:
        binance_symbol = symbol.replace("/", "")
        endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
        params = {"symbol": binance_symbol, "interval": "1h", "limit": 50}  # Fetch last 50 candles

        response = requests.get(endpoint, params=params).json()
        closes = np.array([float(entry[4]) for entry in response])  # Extract closing prices

        # Compute short-term (10-period) and long-term (50-period) moving averages
        short_ma = np.mean(closes[-10:])  # Short-term trend (10 periods)
        long_ma = np.mean(closes)  # Long-term trend (50 periods)

        # Compute trend strength as a percentage difference
        trend_strength = ((short_ma - long_ma) / long_ma) * 100

        return round(trend_strength, 2)  # Return trend strength as a percentage

    except Exception as e:
        print(f"❌ Error Fetching Trend Strength: {e}")
        return 0  # Default to neutral trend if API fails
   
def get_trade_confidence(symbol="BTC/USDT"):
    support, resistance = get_support_resistance(symbol)
    atr = get_atr(symbol)
    trend_strength = get_trend_strength(symbol)
    historical_win_rate = get_historical_win_rate(symbol)
    fundamental_score = get_fundamental_score()  # ✅ Now includes FA

    confidence = 50  # Base confidence at 50%

    # ✅ Technical Analysis (TA) Weights (60%)
    confidence += min(max(trend_strength * 30, 0), 30)
    confidence += min(max((atr / get_real_time_price(symbol)) * 10, 0), 10)
    confidence += min(max((1 - (abs(get_real_time_price(symbol) - support) / (resistance - support))) * 20, 0), 20)

    # ✅ Fundamental Analysis (FA) Weight (30%)
    confidence += min(max(fundamental_score * 0.3, -15), 15)

    # ✅ Historical Trade Performance (10%)
    confidence += min(max(historical_win_rate * 10, 0), 10)

    return max(0, min(100, confidence))



# Function to place an AI-driven trade with corrected trade direction
def place_ai_trade(symbol="BTC/USDT", leverage=10):
    predicted_price, current_price = predict_price_movement(symbol)
    sentiment_score = get_news_sentiment()
    support, resistance = get_support_resistance(symbol)

    if sentiment_score < 0:
        print("⚠️ FA Signals Bearish Market, Skipping Trade")
        return

    # ✅ Fix: Ensure the bot places the correct order type
    if predicted_price > current_price:  # AI predicts bullish move
        side = "BUY"
        position_side = "LONG"  # Hedge mode requires positionSide
    else:  # AI predicts bearish move
        side = "SELL"
        position_side = "SHORT"
    
    if side == "BUY" and current_price >= resistance * 0.99:  # Near resistance
        print(f"⚠️ Skipping Trade: {symbol} is near resistance at {resistance:.2f}")
        return
    if side == "SELL" and current_price <= support * 1.01:  # Near support
        print(f"⚠️ Skipping Trade: {symbol} is near support at {support:.2f}")
        return
    
    if not is_profitable_setup(symbol, side):
        print(f"⚠️ Skipping Trade: Similar setups had <60% win rate")
        return
    
    confidence = get_trade_confidence(symbol)
    if confidence < 75:
        print(f"⚠️ Skipping Trade: Confidence too low ({confidence}%)")
        return

    open_trades = get_open_trades(symbol)
    should_dca_now, dca_size = should_dca(symbol)

    if open_trades and not should_dca_now:
        print(f"⚠️ Skipping Trade: Already an Open Position for {symbol}")
        return False, 0  # No
    
    inflow, buy_volume = get_whale_exchange_flows()

    print(f"""
    📊 Market Analysis:
    - 🔹 Total Exchange Inflow: ${inflow:,.2f} USDT
    - 🔹 BTC Buy Volume: {buy_volume:.2f} BTC
    - 🔹 Whale Activity: {"HIGH" if buy_volume > 10_000 else "LOW"}
    """)
    
    position_size = dca_size if should_dca_now else get_trade_size(symbol)
    stop_loss, take_profit = get_dynamic_sl_tp(symbol, current_price, side)
    
    if stop_loss is None or take_profit is None:
        print(f"❌ Error: Stop-Loss or Take-Profit calculation failed. Skipping trade.")
        return

    print(f"🚀 AI Trading Signal: {side} | Order Amount: {position_size:.6f} BTC | SL: {stop_loss:.2f} | TP: {take_profit:.2f} | Mode: Hedge")

    headers = {"X-MBX-APIKEY": API_KEY}

    # Step 1: Place Market Order First
    market_order_params = {
        "symbol": symbol.replace("/", ""),
        "side": side.upper(),
        "type": "MARKET",
        "quantity": position_size,
        "positionSide": position_side,  # ✅ Required in Hedge Mode
        "timestamp": int(time.time() * 1000)
    }

    market_order_params["signature"] = generate_signature(market_order_params)
    market_response = requests.post(f"{BINANCE_FUTURES_URL}/fapi/v1/order", headers=headers, params=market_order_params)
    print(f"✅ Market Order Executed: {market_response.json()}")

    if market_response.status_code == 200:
        # Step 2: Place Stop-Loss Order
        sl_order_params = {
            "symbol": symbol.replace("/", ""),
            "side": "SELL" if side == "BUY" else "BUY",
            "type": "STOP_MARKET",
            "stopPrice": stop_loss,
            "positionSide": position_side,  # ✅ Ensures SL is for the correct position
            "closePosition": "true",
            "timestamp": int(time.time() * 1000)
        }

        sl_order_params["signature"] = generate_signature(sl_order_params)
        sl_response = requests.post(f"{BINANCE_FUTURES_URL}/fapi/v1/order", headers=headers, params=sl_order_params)
        print(f"✅ Stop-Loss Order Executed: {sl_response.json()}")

        # Step 3: Place Take-Profit Order
        tp_order_params = {
            "symbol": symbol.replace("/", ""),
            "side": "SELL" if side == "BUY" else "BUY",
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": take_profit,
            "positionSide": position_side,  # ✅ Ensures TP is for the correct position
            "closePosition": "true",
            "timestamp": int(time.time() * 1000)
        }

        tp_order_params["signature"] = generate_signature(tp_order_params)
        tp_response = requests.post(f"{BINANCE_FUTURES_URL}/fapi/v1/order", headers=headers, params=tp_order_params)
        print(f"✅ Take-Profit Order Executed: {tp_response.json()}")
        
        if should_dca_now and open_trade:
            old_position_size = open_trade["position_size"]
            new_position_size = old_position_size + position_size  # Accumulate position

            # Recalculate weighted average entry price
            avg_entry_price = ((old_position_size * open_trade["entry_price"]) + (position_size * current_price)) / new_position_size

            # Update Stop-Loss & Take-Profit dynamically
            stop_loss, take_profit = get_dynamic_sl_tp(symbol, avg_entry_price, side)

            open_trade.update({
                "position_size": new_position_size,
                "entry_price": avg_entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            })

            print(f"🔄 DCA Applied: New Position Size = {new_position_size:.6f} BTC | New Entry Price = {avg_entry_price:.2f}")
        
        else:
            # ✅ Register new trade in memory for real-time tracking
            open_trades[symbol] = {
                "side": side,
                "position_size": position_size,
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trade_type": trade_type
            }

def get_support_resistance(symbol="BTC/USDT", lookback=50):
    binance_symbol = symbol.replace("/", "")
    endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
    params = {"symbol": binance_symbol, "interval": "1h", "limit": lookback}
    
    response = requests.get(endpoint, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "ignore1", "ignore2", "ignore3", "ignore4", "ignore5", "ignore6"])
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)

    # Identify resistance (recent highs)
    resistance = df["high"].max()

    # Identify support (recent lows)
    support = df["low"].min()

    return support, resistance

def get_real_time_price(symbol="BTC/USDT"):
    try:
        binance_symbol = symbol.replace("/", "")
        endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/price"
        response = requests.get(endpoint, params={"symbol": binance_symbol}).json()

        return float(response["price"])  # Return the latest price

    except Exception as e:
        print(f"❌ Error Fetching Real-Time Price: {e}")
        return None  # Return None if there's an error

def on_message(ws, message):
    data = json.loads(message)

    if "e" in data and data["e"] == "ORDER_TRADE_UPDATE":
        symbol = data["o"]["s"]
        status = data["o"]["X"]  # Order status
        exit_price = float(data["o"]["ap"]) if "ap" in data["o"] else None  # Filled average price

        if status in ["FILLED", "CANCELED"]:
            if symbol in open_trades:
                trade = open_trades.pop(symbol)

                # Determine if it hit SL or TP
                result = "WIN" if exit_price >= trade["take_profit"] else "LOSS" if exit_price <= trade["stop_loss"] else "CLOSED MANUALLY"
                pnl = (exit_price - trade["entry_price"]) * trade["position_size"] if trade["side"] == "BUY" else (trade["entry_price"] - exit_price) * trade["position_size"]

                # Log trade after it closes
                log_trade(symbol, trade["side"], trade["position_size"], trade["entry_price"], trade["stop_loss"], trade["take_profit"], exit_price, result, pnl, trade["trade_type"])

                print(f"✅ Trade Closed: {symbol} | {result} | PnL: {pnl:.2f}")

# Function to start Binance WebSocket for real-time trade tracking
def start_trade_monitoring():
    def on_open(ws):
        payload = {
            "method": "SUBSCRIBE",
            "params": [f"userData"],
            "id": 1
        }
        ws.send(json.dumps(payload))

    ws = WebSocketApp(f"wss://fstream.binance.com/ws/{API_KEY}", on_message=on_message, on_open=on_open)
    ws.run_forever()
    
trade_monitor_thread = threading.Thread(target=start_trade_monitoring, daemon=True)
trade_monitor_thread.start()

while True:
    print("🔄 Checking Market for AI Trade Signal...")
    place_ai_trade()
    time.sleep(get_market_volatility())
