import pandas as pd
import requests
import ta

# Binance Futures API Base URL
BINANCE_FUTURES_URL = "https://fapi.binance.com"

def fetch_binance_ohlcv(symbol="BTCUSDT", interval="4h", total_candles=20000):
    limit_per_request = 1000  # Binance max per request
    all_data = []
    end_time = None

    while len(all_data) < total_candles:
        url = f"{BINANCE_FUTURES_URL}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit_per_request}
        if end_time:
            params["endTime"] = end_time  

        response = requests.get(url, params=params).json()
        if not response or "code" in response:
            print(f"❌ API Error: {response}")
            break  

        df = pd.DataFrame(response, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_base_volume", "taker_quote_volume", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)

        # ✅ Fix: Extend instead of append
        all_data.extend(df[["timestamp", "open", "high", "low", "close", "volume"]].values.tolist())

        print(f"📊 Fetched {len(df)} candles, Total: {len(all_data)}/{total_candles}")
        print("=========All data size", len(all_data), interval)

        # ✅ Update end_time correctly for next request
        end_time = int(df["timestamp"].iloc[0].timestamp() * 1000)

    # Convert final list back to DataFrame
    full_df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    full_df.sort_values("timestamp", inplace=True)

    return full_df[:total_candles]

def compute_technical_indicators(df):
    df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["RSI"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["EMA_10"] = ta.trend.EMAIndicator(df["close"], window=10).ema_indicator()
    df["MACD"] = ta.trend.MACD(df["close"]).macd()

    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bollinger_h"] = bb.bollinger_hband()
    df["bollinger_l"] = bb.bollinger_lband()

    df["SMA_50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
    df["SMA_200"] = ta.trend.SMAIndicator(df["close"], window=200).sma_indicator()

    return df 

# ✅ Fetch OHLCV data for multiple timeframes
df_4h = fetch_binance_ohlcv("BTCUSDT", interval="1h")

df_4h = compute_technical_indicators(df_4h)


# ✅ Save to CSV
csv_filename = "btc_usdt_4h_technical_large.csv"
df_4h.to_csv(csv_filename, index=False)
print(f"✅ Dataset saved as {csv_filename}")
