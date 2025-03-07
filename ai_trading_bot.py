import os
import time
import numpy as np
import tensorflow as tf
import requests
import ccxt
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
import ta



# Download VADER model
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load Binance API Keys
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY")
CRYPTO_PANIC_API_KEY = os.getenv("CRYPTO_PANIC_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

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

open_trades = {}

def get_whale_exchange_flows(symbol="BTCUSDT"):
    try:
        endpoint = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(endpoint, params={"symbol": symbol}).json()

        inflow = float(response["quoteVolume"])  # Total USDT volume traded in 24h
        buy_volume = float(response["volume"])   # Total BTC volume traded in 24h
        return inflow, buy_volume

    except Exception as e:
        print(f"❌ Error Fetching Whale Exchange Flow: {e}")
        return 0, 0


def get_news_sentiment():
    try:
        categories = ["hot", "bullish", "bearish", "news", "media"]
        total_sentiment = 0
        news_count = 0

        for category in categories:
            response = requests.get(
                f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTO_PANIC_API_KEY}&filter={category}"
            ).json()

            for news in response.get("results", []):
                title = news["title"].lower()
                sentiment_score = sia.polarity_scores(title)["compound"]
                total_sentiment += sentiment_score
                news_count += 1

        avg_sentiment = round(total_sentiment / news_count, 2) if news_count > 0 else np.nan
        return avg_sentiment

    except Exception as e:
        print(f"❌ Error Fetching News Sentiment: {e}")
        return np.nan

def get_reddit_sentiment():
    try:
        url = "https://www.reddit.com/r/cryptocurrency/top/.json?limit=10"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers).json()

        total_sentiment = 0
        post_count = 0

        for post in response["data"]["children"]:
            title = post["data"]["title"].lower()
            sentiment_score = sia.polarity_scores(title)["compound"]
            total_sentiment += sentiment_score
            post_count += 1

        avg_sentiment = round(total_sentiment / post_count, 2) if post_count > 0 else np.nan
        return avg_sentiment

    except Exception as e:
        print(f"❌ Error Fetching Reddit Sentiment: {e}")
        return np.nan

def get_combined_news_sentiment():
    cryptopanic_score = get_news_sentiment()
    newsapi_score = get_news_sentiment_newsapi()
    reddit_score = get_reddit_sentiment()

    sentiment_scores = [cryptopanic_score, newsapi_score, reddit_score]
    sentiment_scores = [score for score in sentiment_scores if not np.isnan(score)]

    avg_sentiment = round(sum(sentiment_scores) / len(sentiment_scores), 2) if sentiment_scores else np.nan
    print(f"🔹 Combined Multi-Source Sentiment Score: {avg_sentiment}")
    return avg_sentiment


def get_news_sentiment_newsapi():
    try:
        url = f"https://newsapi.org/v2/everything?q=crypto&apiKey={NEWS_API_KEY}"
        response = requests.get(url).json()

        total_sentiment = 0
        news_count = 0

        for article in response.get("articles", []):
            title = article["title"].lower()
            sentiment_score = sia.polarity_scores(title)["compound"]
            total_sentiment += sentiment_score
            news_count += 1

        avg_sentiment = round(total_sentiment / news_count, 2) if news_count > 0 else np.nan
        return avg_sentiment

    except Exception as e:
        print(f"❌ Error Fetching NewsAPI Sentiment: {e}")
        return np.nan


    
# Function to generate Binance API signature
def generate_signature(params):
    query_string = urlencode(params)
    return hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def get_fundamental_score():
    news_sentiment = get_combined_news_sentiment()
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
        
        funding_rate = round(float(response["lastFundingRate"]), 6)
        open_interest = float(response.get("openInterest", 0))

        print(f"📊 Funding Rate: {funding_rate:.4f} | Open Interest: {open_interest:.2f}")
        return funding_rate, open_interest

    except Exception as e:
        print(f"❌ Error Fetching Funding Rate: {e}")
        return 0, 0

def get_fibonacci_levels(symbol="BTCUSDT"):
    """Calculate Fibonacci retracement levels based on recent high and low."""
    binance_symbol = symbol.replace("/", "")
    endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr"

    try:
        response = requests.get(endpoint, params={"symbol": binance_symbol}).json()
        low_price = float(response["lowPrice"])   # 24-hour lowest price
        high_price = float(response["highPrice"])  # 24-hour highest price

        # Calculate Fibonacci levels
        diff = high_price - low_price
        fib_levels = {
            "0.236": high_price - diff * 0.236,
            "0.382": high_price - diff * 0.382,
            "0.500": high_price - diff * 0.500,
            "0.618": high_price - diff * 0.618,
            "0.786": high_price - diff * 0.786
        }

        return fib_levels
    except Exception as e:
        print(f"❌ Error Fetching Fibonacci Levels: {e}")
        return {"0.236": None, "0.382": None, "0.500": None, "0.618": None, "0.786": None}

def get_sma_levels(symbol="BTCUSDT"):
    """Fetch SMA 50 & SMA 200 from Binance OHLCV data."""
    try:
        endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
        params = {"symbol": symbol.replace("/", ""), "interval": "1h", "limit": 200}  # Fetch 200 hours

        response = requests.get(endpoint, params=params).json()
        closes = np.array([float(entry[4]) for entry in response])  # Closing prices

        # Calculate SMA 50 and SMA 200
        sma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else np.nan
        sma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else np.nan

        return sma_50, sma_200
    except Exception as e:
        print(f"❌ Error Fetching SMA Levels: {e}")
        return np.nan, np.nan

def get_rsi(symbol="BTCUSDT"):
    """Fetch RSI (Relative Strength Index) from OHLCV data."""
    try:
        endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
        params = {"symbol": symbol.replace("/", ""), "interval": "1h", "limit": 100}  # Fetch 100 hours

        response = requests.get(endpoint, params=params).json()
        closes = pd.Series([float(entry[4]) for entry in response])  # Closing prices

        # Compute RSI (14-period)
        rsi = ta.momentum.RSIIndicator(closes, window=14).rsi().iloc[-1]
        return round(rsi, 2)
    except Exception as e:
        print(f"❌ Error Fetching RSI: {e}")
        return np.nan

def get_optimal_entry(symbol="BTCUSDT", side="BUY"):
    """Determine the best entry price using order book bid/ask prices, support/resistance, SMA, and RSI."""

    # ✅ Fetch Order Book Data (Top Bid/Ask)
    response = requests.get(f"{BINANCE_FUTURES_URL}/fapi/v1/depth", params={"symbol": symbol.replace("/", ""), "limit": 10}).json()
    top_bid = float(response["bids"][0][0])  # Highest bid price (buyers)
    top_ask = float(response["asks"][0][0])  # Lowest ask price (sellers)

    # ✅ Fetch Support & Resistance Levels
    support, resistance = get_support_resistance(symbol)

    # ✅ Fetch SMA 50/200
    sma_50, sma_200 = get_sma_levels(symbol)

    # ✅ Fetch RSI
    rsi = get_rsi(symbol)

    # ✅ Base entry on bid/ask price
    entry_price = top_bid if side == "BUY" else top_ask

    # ✅ Prevent Buying at Resistance & Selling at Support
    if side == "BUY" and entry_price >= resistance * 0.99:
        print(f"⚠️ Skipping Trade: {symbol} is near resistance at {resistance:.2f}")
        return None
    if side == "SELL" and entry_price <= support * 1.01:
        print(f"⚠️ Skipping Trade: {symbol} is near support at {support:.2f}")
        return None

    # ✅ Adjust Entry Based on SMA Trend
    if side == "BUY" and sma_50 > sma_200:  # Uptrend
        entry_price = min(entry_price, sma_50 * 0.995)  # Slight discount
    elif side == "SELL" and sma_50 < sma_200:  # Downtrend
        entry_price = max(entry_price, sma_50 * 1.005)  # Slight premium

    # ✅ Adjust for RSI (Avoid Overbought/Oversold)
    if side == "BUY" and rsi > 70:  # Overbought
        print(f"⚠️ RSI Overbought ({rsi:.2f}), Avoiding BUY Entry")
        return None
    if side == "SELL" and rsi < 30:  # Oversold
        print(f"⚠️ RSI Oversold ({rsi:.2f}), Avoiding SELL Entry")
        return None

    print(f"""
    📊 Optimal Entry Calculation:
    - 🏦 Order Book: Top Bid = {top_bid:.2f}, Top Ask = {top_ask:.2f}
    - 🛑 Support Level: {support:.2f}, Resistance Level: {resistance:.2f}
    - 📊 SMA 50: {sma_50:.2f}, SMA 200: {sma_200:.2f}
    - 💡 RSI: {rsi:.2f}
    - ✅ Final Suggested Entry Price: {entry_price:.2f} (Side: {side})
    """)

    return entry_price



def get_order_status(symbol, order_id):
    """Fetches the status of an existing order on Binance Futures."""
    try:
        headers = {"X-MBX-APIKEY": API_KEY}
        params = {
            "symbol": symbol.replace("/", ""),
            "orderId": order_id,
            "timestamp": int(time.time() * 1000)
        }
        params["signature"] = generate_signature(params)

        response = requests.get(f"{BINANCE_FUTURES_URL}/fapi/v1/order", headers=headers, params=params)
        order_data = response.json()

        if response.status_code == 200:
            return order_data["status"]  # Returns `NEW`, `FILLED`, `CANCELED`, etc.
        else:
            print(f"❌ Error Fetching Order Status: {order_data}")
            return None
    except Exception as e:
        print(f"❌ Error Fetching Order Status for {symbol}: {e}")
        return None

def get_order_book_liquidity(symbol="BTCUSDT", depth=10):
    """Fetches order book liquidity from Binance Futures and returns total volume."""
    try:
        response = requests.get(f"{BINANCE_FUTURES_URL}/fapi/v1/depth", params={"symbol": symbol.replace("/", ""), "limit": depth}).json()

        # Sum up total volume for top `depth` levels
        total_bid_volume = sum([float(entry[1]) for entry in response["bids"]])  # Buy side volume
        total_ask_volume = sum([float(entry[1]) for entry in response["asks"]])  # Sell side volume
        total_liquidity = total_bid_volume + total_ask_volume  # Overall market liquidity

        print(f"📊 Market Liquidity | Bids: {total_bid_volume:.2f} | Asks: {total_ask_volume:.2f} | Total: {total_liquidity:.2f}")
        return total_liquidity
    except Exception as e:
        print(f"❌ Error Fetching Order Book Liquidity: {e}")
        return 0

def check_unfilled_orders():
    if not open_trades:
        return

    for symbol, trade in list(open_trades.items()):
        if trade.get("trade_type") == "LIMIT":
            order_status = get_order_status(symbol, trade["order_id"])

            if order_status == "NEW":
                new_predicted_price = predict_price_movement(symbol)
                price_deviation = abs((new_predicted_price - trade["entry_price"]) / trade["entry_price"]) * 100

                print(f"🔍 Checking {symbol} Limit Order | Entry: {trade['entry_price']} | New Prediction: {new_predicted_price:.2f} | Deviation: {price_deviation:.2f}%")

                if price_deviation < 0.2:
                    print(f"✅ Deviation {price_deviation:.2f}% is small, keeping existing order.")
                    continue

                elif 0.2 <= price_deviation <= 0.5:
                    order_book_depth = get_order_book_liquidity(symbol)
                    if order_book_depth > 500:
                        print(f"📊 Liquidity High ({order_book_depth}), keeping order.")
                        continue
                    else:
                        print(f"⚠️ Liquidity Low, adjusting order.")

                print(f"⚠️ High Deviation {price_deviation:.2f}% detected, canceling old order.")
                cancel_order(symbol, trade["order_id"])
                del open_trades[symbol]  # ✅ Remove canceled trade
                place_ai_trade(symbol)  # ✅ Recalculate & place new order

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

def cancel_order(symbol, order_id):
    """Cancels an open limit order and removes it from tracking."""
    try:
        headers = {"X-MBX-APIKEY": API_KEY}
        cancel_params = {
            "symbol": symbol.replace("/", ""),
            "orderId": order_id,
            "timestamp": int(time.time() * 1000)
        }
        cancel_params["signature"] = generate_signature(cancel_params)

        response = requests.delete(f"{BINANCE_FUTURES_URL}/fapi/v1/order", headers=headers, params=cancel_params)
        cancel_result = response.json()

        if response.status_code == 200:
            print(f"✅ Order {order_id} for {symbol} successfully canceled.")
            if symbol in open_trades:
                del open_trades[symbol]  # ✅ Remove from tracking
        else:
            print(f"❌ Failed to cancel order {order_id}: {cancel_result}")

    except Exception as e:
        print(f"❌ Error canceling order {order_id}: {e}")

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
def place_ai_trade(symbol="BTC/USDT"):
    predicted_price, current_price = predict_price_movement(symbol)
    sentiment_score = get_combined_news_sentiment()
    support, resistance = get_support_resistance(symbol)

    if sentiment_score < 0:
        print("⚠️ FA Signals Bearish Market, Skipping Trade")
        return

    if predicted_price > current_price:
        side = "BUY"
        position_side = "LONG"
    else:
        side = "SELL"
        position_side = "SHORT"

    if side == "BUY" and current_price >= resistance * 0.99:
        print(f"⚠️ Skipping Trade: {symbol} is near resistance at {resistance:.2f}")
        return
    if side == "SELL" and current_price <= support * 1.01:
        print(f"⚠️ Skipping Trade: {symbol} is near support at {support:.2f}")
        return

    if not is_profitable_setup(symbol, side):
        print(f"⚠️ Skipping Trade: Similar setups had <60% win rate")
        return

    confidence = get_trade_confidence(symbol)
    if confidence < 75:
        print(f"⚠️ Skipping Trade: Confidence too low ({confidence}%)")
        return

    should_dca_now, dca_size = should_dca(symbol)

    if open_trades and not should_dca_now:
        print(f"⚠️ Skipping Trade: Already an Open Position for {symbol}")
        return False, 0

    inflow, buy_volume = get_whale_exchange_flows()

    print(f"""
    📊 Market Analysis:
    - Trade Confidence {confidence:,.2f}
    - 🔹 Whale Activity: {"HIGH" if buy_volume > 10_000 else "LOW"}
    """)

    position_size = dca_size if should_dca_now else get_trade_size(symbol)
    stop_loss, take_profit = get_dynamic_sl_tp(symbol, current_price, side)

    if stop_loss is None or take_profit is None:
        print(f"❌ Error: Stop-Loss or Take-Profit calculation failed. Skipping trade.")
        return

    # ✅ Get Optimal Entry Price Using Order Book Data
    entry_price = get_optimal_entry(symbol)

    order_type = "LIMIT"

    print(f"🚀 AI Trading Signal: {side} | Order Amount: {position_size:.6f} BTC | Entry Price: {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f} | Mode: Hedge")

    headers = {"X-MBX-APIKEY": API_KEY}

    # ✅ Step 1: Place Limit Order
    limit_order_params = {
        "symbol": symbol.replace("/", ""),
        "side": side.upper(),
        "type": order_type,
        "price": entry_price,
        "quantity": position_size,
        "timeInForce": "GTC", 
        "positionSide": position_side,
        "timestamp": int(time.time() * 1000)
    }

    limit_order_params["signature"] = generate_signature(limit_order_params)
    limit_response = requests.post(f"{BINANCE_FUTURES_URL}/fapi/v1/order", headers=headers, params=limit_order_params)
   
    print(f"✅ Limit Order Sent: {limit_response.json()}")

    if limit_response.status_code == 200:
        # ✅ Step 2: Place Stop-Loss Order
        sl_order_params = {
            "symbol": symbol.replace("/", ""),
            "side": "SELL" if side == "BUY" else "BUY",
            "type": "STOP_MARKET",
            "stopPrice": stop_loss,
            "timeInForce": "GTC", 
            "positionSide": position_side,
            "closePosition": "true",
            "timestamp": int(time.time() * 1000)
        }

        sl_order_params["signature"] = generate_signature(sl_order_params)
        sl_response = requests.post(f"{BINANCE_FUTURES_URL}/fapi/v1/order", headers=headers, params=sl_order_params)
        print(f"✅ Stop-Loss Order Executed: {sl_response.json()}")

        # ✅ Step 3: Place Take-Profit Order
        tp_order_params = {
            "symbol": symbol.replace("/", ""),
            "side": "SELL" if side == "BUY" else "BUY",
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": take_profit,
            "positionSide": position_side,
            "timeInForce": "GTC", 
            "closePosition": "true",
            "timestamp": int(time.time() * 1000)
        }

        tp_order_params["signature"] = generate_signature(tp_order_params)
        tp_response = requests.post(f"{BINANCE_FUTURES_URL}/fapi/v1/order", headers=headers, params=tp_order_params)
        print(f"✅ Take-Profit Order Executed: {tp_response.json()}")

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
    """Periodically checks for unfilled limit orders and adjusts them if necessary."""
    check_unfilled_orders()
    time.sleep(get_market_volatility())
