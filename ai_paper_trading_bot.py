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
from nltk.sentiment import SentimentIntensityAnalyzer
import requests
import nltk
import csv
import logging
import datetime
import pandas as pd
from websocket import WebSocketApp
import ta
from tensorflow.keras.losses import MeanSquaredError
import subprocess



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

# Binance Futures API Base URL
BINANCE_FUTURES_URL = "https://fapi.binance.com"

custom_objects = {"mse": MeanSquaredError()}

SOUND_FILE = '/System/Library/Sounds/Glass.aiff'


# Load Trained LSTM Model
model5m = tf.keras.models.load_model("optimized_5m_lstm_model.h5", custom_objects=custom_objects)
model1h = tf.keras.models.load_model("optimized_1h_lstm_model.h5", custom_objects=custom_objects)
model4h = tf.keras.models.load_model("optimized_4h_lstm_model.h5", custom_objects=custom_objects)




# Initialize Binance Futures Client
binance = ccxt.binance({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "options": {"defaultType": "future", "adjustForTimeDifference": True},
    "rateLimit": 1200,
})

binance.load_markets()

paper_trades = {}
paper_balance = 10000
canceled_paper_trades = []
closed_paper_trades = []
CANCELED_TRADES_CSV = "canceled_paper_trades.csv"
PAPER_TRADES_CSV = "paper_trades_results.csv" 

# ✅ Ensure canceled trades CSV exists
if not os.path.exists(CANCELED_TRADES_CSV):
    with open(CANCELED_TRADES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "side", "entry_price", "reason"])  # ✅ CSV Headers

def log_paper_trade(trade):
    """Logs a completed paper trade to CSV."""
    trade_duration = (trade["exit_time"] - trade["entry_time"]).total_seconds() if "exit_time" in trade else None

    with open(PAPER_TRADES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            trade["entry_time"], trade["symbol"], trade["side"], trade["entry_price"], 
            trade["exit_price"], trade["position_size"], trade["pnl"], trade_duration, trade["status"]
        ])

INITIAL_BALANCE = float(binance.fetch_balance({"type": "future"})["info"]["totalWalletBalance"])
consecutive_losses = 0  # Track losing streak

def final_prediction(symbol):
    input_data_5m = get_real_time_data(symbol, '5m')
    input_data_1h = get_real_time_data(symbol, '1h')
    input_data_4h = get_real_time_data(symbol, '4h')
    
    pred_5m = model5m.predict(input_data_5m)[0][0]
    pred_1h = model1h.predict(input_data_1h)[0][0]
    pred_4h = model4h.predict(input_data_4h)[0][0]
    
    return (0.25 * pred_5m) + (0.35 * pred_1h) + (0.4 * pred_4h)

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


def play_sound():
    subprocess.run(["afplay", SOUND_FILE])
    
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

def fetch_btc_price_momentum(symbol="BTCUSDT"):
    """
    Fetches the last 7 days of BTC prices to calculate:
    - Price Momentum (current price vs. 7-day average)
    - ATR (Volatility) as a percentage of price
    """
    try:
        # ✅ Fetch last 7 days of BTC price data
        history_endpoint = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": "1d", "limit": 8}  # Fetch 8 days to compute 7-day ATR
        response = requests.get(history_endpoint, params=params).json()

        # ✅ Extract closing prices & high-low ranges
        closing_prices = np.array([float(day[4]) for day in response])  # Closing prices
        high_prices = np.array([float(day[2]) for day in response])  # High prices
        low_prices = np.array([float(day[3]) for day in response])  # Low prices

        # ✅ Compute 7-day average closing price (momentum baseline)
        avg_price = np.mean(closing_prices[:-1])  # Exclude last day (current price)
        current_price = closing_prices[-1]

        # ✅ Compute ATR (Normalized)
        tr = np.maximum(high_prices[1:] - low_prices[1:],  # High - Low
                        np.abs(high_prices[1:] - closing_prices[:-1]),  # High - Prev Close
                        np.abs(low_prices[1:] - closing_prices[:-1]))  # Low - Prev Close
        atr = np.mean(tr)  # Average True Range (ATR)

        # ✅ Normalize ATR as a percentage of average price
        atr_pct = (atr / avg_price) * 100  # ATR as percentage of price

        # ✅ Compute Momentum (relative change vs. 7-day avg)
        price_momentum = ((current_price - avg_price) / avg_price) * 100  # Percentage change

        return price_momentum, atr_pct  # ATR is now a percentage

    except Exception as e:
        print(f"❌ Error Fetching BTC Price Momentum: {e}")
        return 0, 0  # Default values if API fails

logging.basicConfig(filename="fundamental_score_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

def get_fundamental_score():
    """
    Computes the fundamental score based on:
    - News Sentiment
    - Whale Activity (Exchange Inflow)
    - Buy Volume (Whale Buying Activity)
    - Funding Rates
    - Price Momentum & Volatility (Replaces Open Interest)
    """

    # ✅ Fetch real-time market data
    news_sentiment = get_combined_news_sentiment() or 0  # Default to neutral if None
    exchange_inflow, buy_volume = get_whale_exchange_flows()
    funding_rate = get_funding_rate()[0] or 0  # Only funding rate
    whale_activity = get_whale_activity()

    # ✅ Fetch 7-day historical averages
    history = fetch_last_7_days_market_data()
    avg_exchange_inflow = history.get("exchange_inflow", exchange_inflow)  # Use current if history missing
    avg_buy_volume = history.get("buy_volume", buy_volume)
    avg_funding_rate = history.get("funding_rate", funding_rate)

    # ✅ Fetch BTC Price Momentum & Volatility
    price_momentum, volatility = fetch_btc_price_momentum()

    # ✅ Initialize score (Neutral 50%)
    fundamental_score = 50  

    # ✅ Log Raw Values Before Adjustments
    logging.info(f"Initial Fundamental Score: {fundamental_score}")
    logging.info(f"News Sentiment: {news_sentiment}")
    logging.info(f"Exchange Inflow: {exchange_inflow}, Avg Exchange Inflow: {avg_exchange_inflow}")
    logging.info(f"Buy Volume: {buy_volume}, Avg Buy Volume: {avg_buy_volume}")
    logging.info(f"Funding Rate: {funding_rate}, Avg Funding Rate: {avg_funding_rate}")
    logging.info(f"Price Momentum: {price_momentum}")
    logging.info(f"Volatility (ATR %): {volatility}")
    logging.info(f"Whale Activity: {whale_activity}")

    # ✅ News Sentiment Weight (40%)
    sentiment_adjustment = min(max(news_sentiment * 40, -20), 20)
    fundamental_score += sentiment_adjustment
    logging.info(f"News Sentiment Adjustment: {sentiment_adjustment}, New Score: {fundamental_score}")

    # ✅ Whale Activity (Exchange Inflow) Weight (30%)
    if whale_activity == 'HIGH':
        fundamental_score += 15
        logging.info("Whale Activity HIGH: +15 Score")
    elif exchange_inflow < 0.8 * avg_exchange_inflow:
        fundamental_score -= 10
        logging.info("Low Exchange Inflow: -10 Score")

    # ✅ Buy Volume Weight (Whale Buying Activity)
    if buy_volume > 1.5 * avg_buy_volume:
        fundamental_score += 10
        logging.info("High Buy Volume: +10 Score")
    elif buy_volume < 0.7 * avg_buy_volume:
        fundamental_score -= 5
        logging.info("Low Buy Volume: -5 Score")

    # ✅ Funding Rate Weight (20%)
    if abs(funding_rate) > 1.5 * abs(avg_funding_rate):
        fundamental_score -= 15
        logging.info("Extreme Funding Rate: -15 Score")

    # ✅ Price Momentum Weight (NEW - Replaces Open Interest)
    if price_momentum > 2:
        fundamental_score += 10
        logging.info("Positive Price Momentum: +10 Score")
    elif price_momentum < -2:
        fundamental_score -= 10
        logging.info("Negative Price Momentum: -10 Score")

    # ✅ Volatility (ATR) Weight (NEW)
    if volatility > max(0.03 * abs(price_momentum), 1.5):
        fundamental_score -= 10
        logging.info("High Volatility: -10 Score")

    # ✅ Prevent Fundamental Score from Reaching 0 Too Often
    fundamental_score = max(10, fundamental_score)  # Ensure minimum score of 10 to prevent 0

    # ✅ Log Final Score
    logging.info(f"Final Fundamental Score: {fundamental_score}\n")

    return min(100, fundamental_score)  # Keep


def get_funding_rate(symbol="BTCUSDT"):
    try:
        # Fetch latest funding rate
        response = requests.get(f"{BINANCE_FUTURES_URL}/fapi/v1/fundingRate", params={"symbol": symbol, "limit": 1}).json()

        if isinstance(response, list) and len(response) > 0:
            # Extract funding rate and open interest while preserving precision
            funding_rate = float(response[0].get("fundingRate", "0"))  # Avoid premature rounding
            open_interest = float(response[0].get("openInterest", "0"))  # Ensure valid float
            
            # Display accurate funding rate (6 decimal places)
            print(f"📊 Funding Rate: {funding_rate:.6f} | Open Interest: {open_interest:.2f}")
            return funding_rate, open_interest

        else:
            return 0.0, 0.0  # Handle unexpected API response

    except Exception as e:
        print(f"❌ Error Fetching Funding Rate: {e}")
        return 0.0, 0.0

def fetch_binance_data(endpoint, params=None, max_retries=3):
    """
    Fetch data from Binance API with retry logic.
    - Retries up to `max_retries` times if the request fails.
    - Uses exponential backoff for rate limits.
    """
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(f"{BINANCE_FUTURES_URL}{endpoint}", params=params, timeout=5)
            
            # ✅ Ensure request is successful
            if response.status_code == 200 and response.text.strip():
                return response.json()

            # ✅ Handle rate limit response
            elif response.status_code == 429:
                wait_time = (2 ** retries)  # Exponential backoff
                print(f"🚨 Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

        except requests.exceptions.RequestException as e:
            print(f"⚠️ API Request Error ({endpoint}): {e}")

        retries += 1
        time.sleep(2 ** retries)  # Exponential backoff delay

    print(f"❌ Failed to fetch data from {endpoint} after {max_retries} retries.")
    return None  # Ensure we return None if all retries fail

def fetch_last_7_days_market_data(symbol="BTCUSDT"):
    """
    Fetches the last 7 days of BTC market data, ensuring accurate values:
    - Buy Volume (BTC)
    - Exchange Inflow (Quote Volume)
    - Funding Rates
    - Open Interest
    """

    try:
        # ✅ Fetch last 7 days of BTC volume & exchange inflow
        history_data = fetch_binance_data("/fapi/v1/klines", {"symbol": symbol, "interval": "1d", "limit": 7})
        if history_data:
            historical_volumes = [float(day[5]) for day in history_data]  # Buy Volume (BTC)
            historical_exchange_inflow = [float(day[7]) for day in history_data]  # Quote Volume (USDT)
        else:
            raise ValueError("⚠️ Missing data from klines API")

        # ✅ Fetch real-time funding rates from Binance
        funding_data = fetch_binance_data("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 7})
        if funding_data:
            historical_funding_rates = [float(entry.get("fundingRate", 0)) for entry in funding_data]
        else:
            raise ValueError("⚠️ Missing data from fundingRate API")


        # ✅ Compute and return 7-day averages
        return {
            "buy_volume": np.mean(historical_volumes),
            "funding_rate": np.mean(historical_funding_rates),
            "exchange_inflow": np.mean(historical_exchange_inflow),
        }

    except Exception as e:
        print(f"❌ Critical Error Fetching Market History: {e}")
        raise RuntimeError("🚨 Market data retrieval failed.")  # Ensure program stops if data is missing

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


def check_unfilled_paper_orders():
    """Checks unfilled paper trades & adjusts based on real market conditions."""
    if not paper_trades:
        return

    for symbol, trade in list(paper_trades.items()):
        if trade.get("trade_type") == "LIMIT":
            # ✅ Get real-time order book prices
            top_bid, top_ask = get_optimal_entry(symbol, trade('side'))
            
            if trade["side"] == "BUY" and top_ask <= trade["entry_price"]:
                print(f"✅ Paper Trade {symbol} BUY order filled at {top_ask}")
                trade["status"] = "FILLED"
                trade["entry_price"] = top_ask  # ✅ Use actual filled price
                continue  
            elif trade["side"] == "SELL" and top_bid >= trade["entry_price"]:
                print(f"✅ Paper Trade {symbol} SELL order filled at {top_bid}")
                trade["status"] = "FILLED"
                trade["entry_price"] = top_bid  # ✅ Use actual filled price
                continue  

            # ✅ Price Deviation Check
            new_predicted_price = predict_price_movement(symbol)
            price_deviation = abs((new_predicted_price - trade["entry_price"]) / trade["entry_price"]) * 100

            print(f"🔍 Checking {symbol} Limit Order | Entry: {trade['entry_price']} | New Prediction: {new_predicted_price:.2f} | Deviation: {price_deviation:.2f}%")

            if price_deviation < 0.2:
                print(f"✅ Deviation {price_deviation:.2f}% is small, keeping order open.")
                continue

            elif 0.2 <= price_deviation <= 0.5:
                order_book_depth = get_order_book_liquidity(symbol)
                if order_book_depth > 500:
                    print(f"📊 Market Liquidity is high ({order_book_depth}), keeping order.")
                    continue
                else:
                    print(f"⚠️ Market Liquidity is low, order might need adjustment.")

            # ✅ Cancel Order if Market Has Moved Too Much
            print(f"⚠️ High Deviation {price_deviation:.2f}% detected, canceling order.")

            # ✅ Log to CSV
            with open(CANCELED_TRADES_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([trade["timestamp"], symbol, trade["side"], trade["entry_price"], "High Price Deviation"])

            log_paper_trade(trade)
            # ✅ Store canceled order in memory (optional)
            canceled_paper_trades.append(trade)

            # ✅ Remove from active trades
            del paper_trades[symbol]  
            place_paper_trade(symbol)  # ✅ Recalculate & place a new order


# Function to determine market volatility and adjust trading frequency
def get_market_volatility(symbol="BTC/USDT"):
    binance_symbol = symbol.replace("/", "")
    endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/24hr"
    price_change_pct = None
    
    try:
        response = requests.get(endpoint, params={"symbol": binance_symbol}).json()
        price_change_pct = abs(float(response["priceChangePercent"]))  # 24h price change percentage
        
        # Adjust trading frequency based on volatility
        if price_change_pct > 5:  # High volatility
            return 60, price_change_pct  # Check every 1 minute
        elif price_change_pct > 2:  # Medium volatility
            return 300, price_change_pct  # Check every 5 minutes
        else:  # Low volatility
            return 900, price_change_pct  # Check every 15 minutes
        
    except Exception as e:
        print(f"❌ Error Fetching Market Volatility: {e}")
        return 900 , price_change_pct # Default to 15 minutes if API fails
    
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

def monitor_paper_trades():
    """Continuously check active paper trades for SL/TP triggers."""
    if not paper_trades:
        return

    for symbol, trade in list(paper_trades.items()):
        if trade["status"] != "FILLED":
            continue  # ✅ Skip orders that are still unfilled

        current_price = get_real_time_price(symbol)

        if trade["side"] == "BUY":
            if current_price >= trade["take_profit"]:
                print(f"✅ TP Hit! Closing BUY {symbol} at {trade['take_profit']}")
                trade["status"] = "CLOSED"
                trade["exit_price"] = trade["take_profit"]
                play_sound()
            elif current_price <= trade["stop_loss"]:
                print(f"❌ SL Hit! Closing BUY {symbol} at {trade['stop_loss']}")
                trade["status"] = "CLOSED"
                trade["exit_price"] = trade["stop_loss"]
                play_sound()

        elif trade["side"] == "SELL":
            if current_price <= trade["take_profit"]:
                print(f"✅ TP Hit! Closing SELL {symbol} at {trade['take_profit']}")
                trade["status"] = "CLOSED"
                trade["exit_price"] = trade["take_profit"]
                play_sound()
            elif current_price >= trade["stop_loss"]:
                print(f"❌ SL Hit! Closing SELL {symbol} at {trade['stop_loss']}")
                trade["status"] = "CLOSED"
                trade["exit_price"] = trade["stop_loss"]
                play_sound()

        # ✅ Remove closed trade from tracking
        if trade["status"] == "CLOSED":
            trade["pnl"] = (trade["exit_price"] - trade["entry_price"]) * trade["position_size"]
            print(f"📊 Trade Closed: {symbol} | PnL: {trade['pnl']:.6f} BTC")
            closed_paper_trades.append(trade)
            del paper_trades[symbol]
            play_sound()


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
def get_real_time_data(symbol="BTC/USDT", interval="1h", X=50):
    binance_symbol = symbol.replace("/", "")
    endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
    params = {"symbol": binance_symbol, "interval": interval, "limit": 250}  # Fetch more rows

    response = requests.get(endpoint, params=params)
    data = response.json()
    
    if not isinstance(data, list) or len(data) < X:
        raise ValueError(f"🚨 Error: Binance API returned only {len(data)} rows, but {X} are required!")

    # Extract OHLCV data
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', '_1', '_2', '_3', '_4', '_5', '_6'])
    df = df[['close', 'volume']].astype(float)

    # ✅ Compute Technical Indicators
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['close'], low=df['close'], close=df['close'], window=14).average_true_range()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['EMA_10'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()

    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()

    bollinger = ta.volatility.BollingerBands(df['close'], window=20)
    df['bollinger_h'] = bollinger.bollinger_hband()
    df['bollinger_l'] = bollinger.bollinger_lband()

    df['SMA_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
    df['SMA_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()

    # ✅ Drop rows with NaNs caused by initial indicator calculations
    df.dropna(inplace=True)

    # ✅ Ensure at least `X` rows available after dropping NaNs
    if len(df) < X:
        raise ValueError(f"🚨 Error: After dropping NaNs, only {len(df)} rows remain, but {X} are required!")

    # ✅ Normalize Data
    min_vals, max_vals = df.min(), df.max()
    
    # Handle cases where min = max (causes division by zero in normalization)
    for col in df.columns:
        if min_vals[col] == max_vals[col]:
            df[col] = 0  # If all values are the same, set to zero
        else:
            df[col] = (df[col] - min_vals[col]) / (max_vals[col] - min_vals[col])

    # ✅ Final check for NaNs
    if df.isnull().values.any():
        raise ValueError("🚨 Error: Normalized Data Still Contains NaNs!")

    # ✅ Ensure shape is (1, X, 10)
    final_input = np.expand_dims(df.values[-X:], axis=0)
    print(f"✅ Final Input Shape: {final_input.shape}")  # Debugging
    return final_input


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
            if symbol in paper_trades:
                del paper_trades[symbol]  # ✅ Remove from tracking
        else:
            print(f"❌ Failed to cancel order {order_id}: {cancel_result}")

    except Exception as e:
        print(f"❌ Error canceling order {order_id}: {e}")

# Function to predict price movement using LSTM model
def predict_price_movement(symbol="BTC/USDT"):
    prediction = final_prediction(symbol)
    
    # Get real-time price range from Binance
    min_price, max_price = get_price_min_max(symbol)

    # Proper denormalization formula
    predicted_price = prediction * (max_price - min_price) + min_price  

    # Get real-time futures price
    futures_price = float(requests.get(f"{BINANCE_FUTURES_URL}/fapi/v1/ticker/price", params={"symbol": symbol.replace("/", "")}).json()["price"])

    print(f"📙 Predicted Price: {predicted_price:.2f} | Current Price: {futures_price:.2f}")
    
    return predicted_price, futures_price


def get_dynamic_sl_tp(symbol="BTC/USDT", entry_price=0, side="BUY", predicted_price=0):
    """Calculate dynamic Stop-Loss (SL) & Take-Profit (TP) with ATR, Predicted Price, and Support/Resistance."""
    try:
        binance_symbol = symbol.replace("/", "")
        endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
        params = {"symbol": binance_symbol, "interval": "1h", "limit": 15}  # ✅ Fetch last 15 candles

        response = requests.get(endpoint, params=params).json()
        highs = np.array([float(entry[2]) for entry in response])
        lows = np.array([float(entry[3]) for entry in response])
        closes = np.array([float(entry[4]) for entry in response])

        # ✅ Ensure arrays have the same length (trim to 14 elements)
        highs, lows, closes = highs[-14:], lows[-14:], closes[-14:]

        # ✅ Use np.roll() to get previous closes properly
        prev_closes = np.roll(closes, shift=1)
        prev_closes[0] = closes[0]  # Prevents NaN issues

        # ✅ Calculate ATR
        tr1 = highs - lows
        tr2 = np.abs(highs - prev_closes)
        tr3 = np.abs(lows - prev_closes)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))  # True Range
        atr = np.mean(tr)  # Average True Range

        # ✅ Get Support/Resistance Levels
        support, resistance = get_support_resistance(symbol)

        # ✅ Factor in Predicted Price to Improve SL & TP Calculation
        if predicted_price == 0:
            predicted_price = entry_price  # Default if prediction is missing

        # ✅ Define Multiplier Based on Volatility
        volatility_factor = max(1.5, min(3.0, atr / (entry_price * 0.01)))  # Scale dynamically
        
        if side == "BUY":
            stop_loss = max(entry_price - (1.5 * atr * volatility_factor), support * 0.99)
            take_profit = min(entry_price + (3 * atr * volatility_factor), resistance * 1.01)

            # ✅ Ensure SL < Entry < TP
            if stop_loss >= entry_price:
                stop_loss = entry_price - (1.5 * atr * volatility_factor)
            if take_profit <= entry_price:
                take_profit = entry_price + (3 * atr * volatility_factor)

            # ✅ Adjust TP if the Predicted Price is Higher than Entry
            if predicted_price > entry_price:
                take_profit = max(take_profit, predicted_price * 1.02)  # Ensure TP is at least 2% above predicted price

        else:  # `SELL`
            stop_loss = min(entry_price + (1.5 * atr * volatility_factor), resistance * 1.01)
            take_profit = max(entry_price - (3 * atr * volatility_factor), support * 0.99)

            # ✅ Ensure TP < Entry < SL
            if stop_loss <= entry_price:
                stop_loss = entry_price + (1.5 * atr * volatility_factor)
            if take_profit >= entry_price:
                take_profit = entry_price - (3 * atr * volatility_factor)

            # ✅ Adjust TP if the Predicted Price is Lower than Entry
            if predicted_price < entry_price:
                take_profit = min(take_profit, predicted_price * 0.98)  # Ensure TP is at least 2% below predicted price

        # ✅ Enforce Minimum 2:1 Risk-Reward Ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rrr = reward / risk if risk > 0 else 0

        if rrr < 2.0:  # If RRR < 2:1, widen TP
            take_profit = entry_price + (risk * 2.5) if side == "BUY" else entry_price - (risk * 2.5)

        print(f"📊 SL/TP Calculation: SL = {stop_loss:.2f}, Entry = {entry_price:.2f}, TP = {take_profit:.2f}, ATR = {atr:.2f}, RRR = {rrr:.2f}")

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
    
    if result == "LOSS":
        consecutive_losses += 1
    else:
        consecutive_losses = 0  # Reset on win

    print(f"📊 Trade Closed & Logged: {trade_type.upper()} {side} {symbol} | PnL: {pnl}")

def get_macd_crossover(symbol="BTCUSDT"):
    """Detects MACD crossovers using Binance OHLCV data"""
    try:
        endpoint = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": "1h", "limit": 50}  # Fetch 50 recent candles
        response = requests.get(endpoint, params=params)

        # ✅ Handle Empty or Invalid Response
        if response.status_code != 200 or not response.text.strip():
            raise ValueError("Invalid response from Binance API")

        data = response.json()

        # ✅ Ensure We Have Enough Candles (50)
        if len(data) < 50:
            raise ValueError(f"Not enough data received: {len(data)} candles")

        # ✅ Extract Closing Prices
        closes = np.array([float(entry[4]) for entry in data])

        # ✅ Ensure `closes` has enough data to compute MACD
        if len(closes) < 26:  # MACD needs at least 26 periods
            raise ValueError(f"Not enough closing price data: {len(closes)}")

        # ✅ MACD Calculation
        short_ema = pd.Series(closes).ewm(span=12, adjust=False).mean()
        long_ema = pd.Series(closes).ewm(span=26, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # ✅ Ensure MACD Series Has Enough Data
        if len(macd_line) < 2 or len(signal_line) < 2:
            raise ValueError("MACD calculation returned insufficient data")

        # ✅ Detect Crossover
        if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            return "BULLISH"  # Bullish crossover
        elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            return "BEARISH"  # Bearish crossover
        return "NEUTRAL"

    except Exception as e:
        print(f"❌ Error Fetching MACD Crossover: {e}")
        return "NEUTRAL"

def is_valid_trade(symbol, side):
    sma_50, sma_200 = get_sma_levels(symbol)
    rsi = get_rsi(symbol)
    sentiment_score = get_combined_news_sentiment()
    
    if side == "BUY" and sma_50 < sma_200:
        return False  # Avoid buying in a downtrend
    if side == "SELL" and sma_50 > sma_200:
        return False  # Avoid shorting in an uptrend
    if rsi > 70 and side == "BUY":
        return False  # Overbought condition
    if rsi < 30 and side == "SELL":
        return False  # Oversold condition
    if sentiment_score < 0:
        return False  # Bearish sentiment
    
    return True

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

def get_historical_win_rate(symbol="BTCUSDT"):
    """
    Computes the historical win rate for a symbol based on past trade logs.
    If insufficient trades exist, decays the confidence contribution.
    """
    try:
        df = pd.read_csv("trade_logs.csv")  # Load past trade logs
        df = df[df["symbol"] == symbol]

        if len(df) < 10:
            # Penalize win rate contribution if trade count is too low
            return 0.25  # Default to 25% if not enough data

        # Compute win rate based on last 50 trades (if available)
        df = df.tail(50)
        win_rate = df["result"].value_counts(normalize=True).get("WIN", 0)

        return round(min(max(win_rate, 0.25), 0.75), 2)  # Cap win rate between 25% and 75%

    except Exception as e:
        print(f"❌ Error Fetching Historical Win Rate: {e}")
        return 0.25  # Default to low confidence if error occurs

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
 
def get_trend_strength(symbol="BTCUSDT"):
    """
    Computes trend strength based on 10-period vs. 50-period moving average.
    """
    try:
        binance_symbol = symbol.replace("/", "")
        endpoint = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
        params = {"symbol": binance_symbol, "interval": "1h", "limit": 50}  # Fetch last 50 candles

        response = requests.get(endpoint, params=params).json()
        closes = np.array([float(entry[4]) for entry in response])  # Extract closing prices

        # Compute moving averages
        short_ma = np.mean(closes[-10:])  # Short-term trend (10 periods)
        long_ma = np.mean(closes)  # Long-term trend (50 periods)

        # Compute trend strength as a capped percentage difference
        trend_strength = ((short_ma - long_ma) / long_ma) * 100

        return round(min(max(trend_strength, -5), 5), 2)  # Cap trend strength between [-5%, +5%]

    except Exception as e:
        print(f"❌ Error Fetching Trend Strength: {e}")
        return 0  # Default to neutral trend if API fails

   
def get_trade_confidence(symbol="BTCUSDT"):
    """
    Computes trade confidence based on:
    - Trend Strength (0.4 weight)
    - ATR / Price Ratio (0.2 weight)
    - Fundamental Score (0.3 weight)
    - Historical Win Rate (0.1 weight, penalized if low trade count)
    """

    weights = {"trend_strength": 0.4, "atr": 0.2, "fundamental": 0.3, "historical_win_rate": 0.1}
    
    # Fetch individual components
    trend_strength = get_trend_strength(symbol) * weights["trend_strength"]
    atr = (get_atr(symbol) / get_real_time_price(symbol)) * weights["atr"]
    fundamental_score = (get_fundamental_score() / 100) * weights["fundamental"]  # Normalize to [0,1]
    historical_win_rate = get_historical_win_rate(symbol) * weights["historical_win_rate"]

    # Normalize ATR so it doesn't dominate confidence
    atr = min(max(atr, 0), 0.05)  # Cap ATR contribution to avoid over-scaling

    # Compute overall confidence
    confidence = (abs(trend_strength) + atr + fundamental_score + historical_win_rate) * 100
    
    return round(min(100, max(0, confidence)), 2)  # Ensure confidence is between [0,100]

def check_circuit_breaker():
    global consecutive_losses
    current_balance = float(binance.fetch_balance({"type": "future"})["info"]["totalWalletBalance"])

    # **Circuit Breaker 1: Stop if balance drops by 20%** ✅
    if current_balance < INITIAL_BALANCE * 0.8:
        print(f"🚨 Circuit Breaker Activated! Balance Dropped by 20% ({current_balance}). Stopping Trading!")
        return True

    # **Circuit Breaker 2: Stop if 3 consecutive losses** ✅
    if consecutive_losses >= 3:
        print(f"🚨 Circuit Breaker Activated! 3 Consecutive Losses Detected. Stopping Trading!")
        return True
    return False

def get_whale_activity(symbol="BTCUSDT", spike_threshold=1.5):
    try:
        # Fetch current 24H trading volume from Binance
        endpoint = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(endpoint, params={"symbol": symbol}).json()
        current_btc_volume = float(response["volume"])  # Current 24H BTC volume

        # Fetch last 7 days of daily BTC volumes from Binance historical data API
        history_endpoint = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": "1d",  # Daily data
            "limit": 7  # Fetch last 7 days
        }
        history_response = requests.get(history_endpoint, params=params).json()

        # Extract the daily trading volumes (BTC) for the last 7 days
        historical_volumes = [float(day[5]) for day in history_response]  # day[5] = daily BTC volume

        # Calculate the historical average volume
        historical_avg_btc_volume = np.mean(historical_volumes) if historical_volumes else current_btc_volume

        # Compute Whale Activity %
        whale_activity_percent = (current_btc_volume / historical_avg_btc_volume) * 100

        # Determine Whale Activity Level
        whale_activity = "HIGH" if whale_activity_percent >= spike_threshold * 100 else "LOW"

        return whale_activity

    except Exception as e:
        print(f"❌ Error Fetching Whale Activity: {e}")
        return "LOW"
    
def get_paper_paper_trades(symbol="BTC/USDT"):
    """Simulate fetching open trades (since we don’t call Binance API)."""
    return paper_trades.get(symbol, None)

def get_paper_trade_size(symbol="BTC/USDT", risk_per_trade=0.02):
    """Simulated trade size calculation with risk management (returns BTC)."""
    global paper_balance

    # Simulate existing positions
    active_positions = [trade for trade in paper_trades.values()]
    num_active_trades = len(active_positions)
    unrealized_pnl = sum(trade["unrealized_pnl"] for trade in active_positions)
    _ , price_change_pct = get_market_volatility()
    
    volatility = price_change_pct
    
    # ✅ Fetch the latest BTC price to convert USDT → BTC
    current_price = get_real_time_price(symbol)

    # ✅ Base position size in USDT based on risk percentage
    position_size_usdt = (risk_per_trade * paper_balance) / volatility if volatility > 0 else (risk_per_trade * paper_balance)

    # ✅ Convert USDT position size to BTC
    position_size_btc = position_size_usdt / current_price

    # ✅ Adjust for Unrealized PnL
    if unrealized_pnl < -0.03 * paper_balance:  # Large unrealized losses
        position_size_btc *= 0.5  # Reduce position size

    # ✅ Adjust for multiple active trades
    if num_active_trades > 3:
        position_size_btc *= 0.7  # Reduce size if too many trades are open

    # ✅ Ensure minimum trade size (Binance minimum is 0.005 BTC)
    min_size = 0.005
    position_size_btc = max(min_size, position_size_btc)

    print(f"📊 Paper Trade Size: {position_size_btc:.6f} BTC | Active Trades: {num_active_trades} | Unrealized PnL: {unrealized_pnl:.2f}")
    
    return position_size_btc


def should_paper_dca(symbol="BTC/USDT"):
    """Simulate deciding when to apply DCA (Dollar-Cost Averaging)."""
    open_trade = get_paper_paper_trades(symbol)
    if not open_trade:
        return False, 0

    predicted_price, current_price = predict_price_movement(symbol)
    rsi = get_rsi(symbol)
    macd_crossover = get_macd_crossover(symbol)
    
    if open_trade["unrealized_pnl"] < -0.02 * paper_balance and predicted_price > current_price and rsi < 30 and macd_crossover == "BULLISH":
        atr = get_atr()
        atr_dca_multiplier = 0.3 if atr < 20 else 0.5
        dca_size = abs(open_trade["position_size"]) * atr_dca_multiplier
        return True, round(dca_size, 6)

    return False, 0

def place_paper_trade(symbol="BTC/USDT"):
    """Simulated AI trading for paper trading."""
    global paper_balance

    predicted_price, current_price = predict_price_movement(symbol)
    sentiment_score = get_combined_news_sentiment()
    support, resistance = get_support_resistance(symbol)

    if predicted_price > current_price:
        side = "BUY"
    else:
        side = "SELL"

    should_dca_now, dca_size = should_paper_dca(symbol)
    confidence = get_trade_confidence(symbol)

    print(f"""
    📊 Market Analysis:
    - Trade Confidence {confidence:,.2f}
    - 🔹 Whale Activity: {get_whale_activity()}
    """)

    position_size = dca_size if should_dca_now else get_paper_trade_size(symbol)
    # ✅ Get Optimal Entry Price Using Order Book Data
    entry_price = get_optimal_entry(symbol, side)
    stop_loss, take_profit = get_dynamic_sl_tp(symbol, entry_price, side, predicted_price)

    
    
    print(f"🚀 Paper Trading: {side} | Amount: {position_size or 0:.6f} BTC | Entry: {entry_price or 0:.2f} | SL: {stop_loss or 0:.2f} | TP: {take_profit or 0:.2f}")

    if sentiment_score < 0 and confidence < 95:
        print("⚠️ FA Signals Bearish Market, Skipping Trade")
        return

    if side == "BUY" and current_price >= resistance * 0.99:
        print(f"⚠️ Skipping Trade: {symbol} is near resistance at {resistance:.2f}")
        return
    if side == "SELL" and current_price <= support * 1.01:
        print(f"⚠️ Skipping Trade: {symbol} is near support at {support:.2f}")
        return

    if not is_profitable_setup(symbol, side):
        print(f"⚠️ Skipping Trade: Similar setups had <60% win rate")
        return

    if confidence < 70:
        print(f"⚠️ Skipping Trade: Confidence too low ({confidence}%)")
        return

    if symbol in paper_trades and not should_dca_now:
        print(f"⚠️ Skipping Trade: Already an Open Position for {symbol}")
        return

    if stop_loss is None or take_profit is None:
        print(f"❌ Error: Stop-Loss or Take-Profit calculation failed. Skipping trade.")
        return
    
    # ✅ **Calculate Risk-to-Reward Ratio (RRR)**
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    rrr = reward / risk if risk > 0 else 0

    print(f"📊 Paper Trade RRR Calculation: Risk = {risk:.2f}, Reward = {reward:.2f}, RRR = {rrr:.2f}")

    if rrr < 2.0:
        print(f"❌ Paper Trade Rejected: RRR {rrr:.2f} is too low (needs at least 2:1).")
        return

    # **Simulated Trade Execution**
    paper_trades[symbol] = {
        "side": side,
        "position_size": position_size,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "trade_type": "DCA" if should_dca_now else "New",
        "unrealized_pnl": 0.0,  # Will update dynamically
        "status": 'pending'
    }

    # Deduct trade cost from virtual balance
    paper_balance -= entry_price * position_size
    play_sound()
    print(f"💰 New Paper Balance: ${paper_balance:,.2f}")

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


while True:
    print("🔄 Checking Market for AI Trade Signal...")
    if not check_circuit_breaker():
        place_paper_trade()
        monitor_paper_trades()
        print("======trades", paper_trades)
        print("======cancel orders ", len(canceled_paper_trades))
        print("======closed orders ", len(closed_paper_trades))

    volatility, price_change_pct = get_market_volatility()
    time.sleep(volatility)
