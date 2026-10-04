import os
import sys
import logging
import pandas as pd
from datetime import datetime, timezone
import roostoo_client as api
from data import DataManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s")
logger = logging.getLogger("backfill")

# Copy the binance fetch logic from backtest.py
import requests

def fetch_binance_klines(symbol: str, limit: int = 75) -> pd.DataFrame:
    """Fetch recent hourly klines from Binance."""
    url = "https://api.binance.com/api/v3/klines"
    b_symbol = symbol.split("/")[0].upper() + "USDT"
    
    params = {
        "symbol": b_symbol,
        "interval": "1h",
        "limit": limit
    }
    
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        logger.warning(f"Failed to fetch {b_symbol}: {resp.text}")
        return pd.DataFrame()
        
    data = resp.json()
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    # Roostoo expects timestamp as index
    # Binance open_time/close_time are ms. Let's use close_time rounded to nearest hour.
    df["timestamp"] = pd.to_datetime(df["close_time"], unit="ms", utc=True).dt.round("h")
    df.set_index("timestamp", inplace=True)
    df["close"] = df["close"].astype(float)
    
    # For volume, use quote asset volume (USD equivalent)
    df["quote_asset_volume"] = df["quote_asset_volume"].astype(float)
    
    return df[["close", "quote_asset_volume"]]

def run():
    dm = DataManager()
    dm.refresh_universe()
    logger.info(f"Loaded universe with {len(dm.universe)} pairs")
    
    prices = {}
    volumes = {}
    
    for pair in dm.universe.keys():
        logger.info(f"Fetching {pair}...")
        df = fetch_binance_klines(pair, limit=75)
        if not df.empty:
            prices[pair] = df["close"]
            volumes[pair] = df["quote_asset_volume"]
            
    if not prices:
        logger.error("No data fetched")
        return
        
    price_df = pd.DataFrame(prices)
    vol_df = pd.DataFrame(volumes)
    
    # Ensure DatetimeIndex with UTC
    price_df.index = pd.to_datetime(price_df.index, utc=True, format="mixed")
    vol_df.index = pd.to_datetime(vol_df.index, utc=True, format="mixed")

    # Sort index
    price_df.sort_index(inplace=True)
    vol_df.sort_index(inplace=True)
    
    # Combine with any existing data in dm
    if not dm.price_history.empty:
        price_df = pd.concat([dm.price_history, price_df])
        price_df.index = pd.to_datetime(price_df.index, utc=True, format="mixed")
        price_df = price_df[~price_df.index.duplicated(keep="last")]
        price_df.sort_index(inplace=True)
        
    if not dm.volume_history.empty:
        vol_df = pd.concat([dm.volume_history, vol_df])
        vol_df.index = pd.to_datetime(vol_df.index, utc=True, format="mixed")
        vol_df = vol_df[~vol_df.index.duplicated(keep="last")]
        vol_df.sort_index(inplace=True)
    
    dm.price_history = price_df
    dm.volume_history = vol_df
    
    dm._persist_history()
    logger.info(f"Saved {len(price_df)} rows to price_history.csv and volume_history.csv")
    
if __name__ == "__main__":
    run()
