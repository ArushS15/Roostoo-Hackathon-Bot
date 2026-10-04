"""
data.py — Ticker polling, local price + volume history buffer, and universe management.

Responsibilities:
  - Pull the tradable universe from exchangeInfo (dynamic, not hardcoded).
  - Poll tickers periodically and maintain in-memory price AND volume history DataFrames.
  - Compute per-horizon returns, per-horizon volatility, and volume surge.
  - Backfill historical hourly OHLCV from Binance at startup for immediate signal readiness.
  - Persist all history to disk so restarts don't lose accumulated data.
"""

import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np
import requests

import roostoo_client as api
from config import (
    MIN_TRADE_VALUE, PRICE_HISTORY_MAX_HOURS, STATE_DIR,
    TICKER_POLL_SECONDS, VOL_EPSILON,
    VOLUME_SURGE_WINDOW_HOURS, VOLUME_SURGE_AVG_DAYS, VOLUME_SURGE_CAP,
)

logger = logging.getLogger("data")


def _roostoo_to_binance_symbol(pair: str) -> Optional[str]:
    """
    Convert a Roostoo pair like 'BTC/USD' to a Binance symbol like 'BTCUSDT'.
    Returns None for pairs that are unlikely to exist on Binance (e.g. stock
    tokens like TSLAB, NVDAB, custom competition tokens like BMT, COINB).
    """
    # Tokens known to NOT exist on Binance (stock tokens, custom competition tokens)
    # We skip these rather than hammering Binance with 404s
    NON_BINANCE_PREFIXES = {
        "TSLAB", "NVDAB", "MSFTB", "GOOGLB", "AMDB", "PLTRB", "MSTRB",
        "CBRSB", "SPCXB", "NBISB", "INTCB", "SNDKB", "SKHYB", "CRCLB",
        "LITEB", "QCOMB", "GLWB",
        "BMT", "COINB", "WDCB", "STO", "METAB", "WLFI", "HEMI",
        "SOMI", "XPL", "AVNT", "MUB", "EDEN", "MIRA", "TUT",
        "1000CHEEMS",
    }
    coin = pair.split("/")[0].upper()
    if coin in NON_BINANCE_PREFIXES:
        return None
    return coin + "USDT"


class DataManager:
    """Manages the tradable universe and rolling price + volume history."""

    def __init__(self):
        self.universe: dict = {}          # pair -> exchange info metadata
        self.price_history: pd.DataFrame = pd.DataFrame()   # index=timestamp, columns=pairs
        self.volume_history: pd.DataFrame = pd.DataFrame()  # index=timestamp, columns=pairs
        self._history_file = os.path.join(STATE_DIR, "price_history.csv")
        self._volume_file = os.path.join(STATE_DIR, "volume_history.csv")
        self._universe_file = os.path.join(STATE_DIR, "universe.json")
        os.makedirs(STATE_DIR, exist_ok=True)
        self._load_persisted()

    # ─── Universe ────────────────────────────────────────────────────────

    def refresh_universe(self) -> dict:
        """
        Pull exchangeInfo, filter to CanTrade pairs, optionally apply a
        liquidity filter using UnitTradeValue from the ticker endpoint.
        Returns the universe dict {pair: metadata}.
        """
        info = api.exchange_info()
        if not info or "TradePairs" not in info:
            logger.error("Failed to fetch exchangeInfo -- keeping stale universe")
            return self.universe

        tradable = {}
        for pair, meta in info["TradePairs"].items():
            if meta.get("CanTrade", False):
                tradable[pair] = meta
        logger.info("exchangeInfo: %d tradable pairs found", len(tradable))

        # Liquidity filter via ticker
        ticker_data = api.ticker()
        if ticker_data and ticker_data.get("Success") and ticker_data.get("Data"):
            for pair in list(tradable.keys()):
                tv = ticker_data["Data"].get(pair, {}).get("UnitTradeValue", 0)
                if tv < MIN_TRADE_VALUE:
                    logger.info("Dropping %s -- UnitTradeValue %.0f < %d",
                                pair, tv, MIN_TRADE_VALUE)
                    del tradable[pair]

        self.universe = tradable
        self._persist_universe()
        logger.info("Universe set: %d pairs", len(self.universe))
        return self.universe

    # ─── Binance backfill ────────────────────────────────────────────────

    def backfill_from_binance(self, limit: int = 100):
        """
        Fetch the last `limit` hourly candles from Binance for every pair
        in the universe that has a Binance equivalent. This seeds the price
        and volume history so the bot can compute momentum scores immediately
        on the very first rebalance, instead of waiting 72+ hours.

        Only fills in data for timestamps we don't already have (idempotent).
        """
        if not self.universe:
            logger.warning("backfill_from_binance: universe is empty, skipping")
            return

        price_frames = {}
        volume_frames = {}
        fetched = 0
        skipped = 0

        for pair in self.universe:
            b_sym = _roostoo_to_binance_symbol(pair)
            if b_sym is None:
                skipped += 1
                continue

            try:
                resp = requests.get(
                    "https://api.binance.com/api/v3/klines",
                    params={"symbol": b_sym, "interval": "1h", "limit": limit},
                    timeout=10,
                )
                if resp.status_code != 200:
                    logger.debug("Binance skip %s (%s): HTTP %d", pair, b_sym, resp.status_code)
                    skipped += 1
                    continue

                data = resp.json()
                if not data:
                    skipped += 1
                    continue

                closes = []
                volumes = []
                timestamps = []
                for k in data:
                    ts = pd.Timestamp(k[6], unit="ms", tz="UTC").round("h")
                    timestamps.append(ts)
                    closes.append(float(k[4]))       # close price
                    volumes.append(float(k[7]))       # quote asset volume (USD)

                price_frames[pair] = pd.Series(closes, index=timestamps, name=pair)
                volume_frames[pair] = pd.Series(volumes, index=timestamps, name=pair)
                fetched += 1

                # Rate-limit: Binance allows 1200 req/min, stay well under
                if fetched % 10 == 0:
                    time.sleep(1)

            except Exception as e:
                logger.debug("Binance backfill error for %s: %s", pair, e)
                skipped += 1

        if not price_frames:
            logger.warning("Binance backfill: 0 pairs fetched (skipped %d)", skipped)
            return

        # Build DataFrames
        new_prices = pd.DataFrame(price_frames)
        new_volumes = pd.DataFrame(volume_frames)

        # Merge with existing persisted data (don't overwrite newer live data)
        if not self.price_history.empty:
            combined = pd.concat([new_prices, self.price_history])
            combined.index = pd.to_datetime(combined.index, utc=True, format="mixed")
            # Keep the last occurrence at each timestamp (live data wins)
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            self.price_history = combined
        else:
            new_prices.index = pd.to_datetime(new_prices.index, utc=True, format="mixed")
            self.price_history = new_prices.sort_index()

        if not self.volume_history.empty:
            combined = pd.concat([new_volumes, self.volume_history])
            combined.index = pd.to_datetime(combined.index, utc=True, format="mixed")
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            self.volume_history = combined
        else:
            new_volumes.index = pd.to_datetime(new_volumes.index, utc=True, format="mixed")
            self.volume_history = new_volumes.sort_index()

        # Trim to max history window
        now = pd.Timestamp.now(tz="UTC")
        cutoff = now - pd.Timedelta(hours=PRICE_HISTORY_MAX_HOURS)
        self.price_history = self.price_history[self.price_history.index >= cutoff]
        self.volume_history = self.volume_history[self.volume_history.index >= cutoff]

        self._persist_history()
        logger.info(
            "Binance backfill complete: %d pairs fetched, %d skipped, "
            "history now has %d rows x %d columns",
            fetched, skipped, len(self.price_history), len(self.price_history.columns),
        )

    # ─── Price + Volume history ──────────────────────────────────────────

    def poll_ticker(self) -> Optional[dict]:
        """
        Fetch all tickers, append LastPrice and UnitTradeValue for each pair
        in the universe to the history DataFrames.
        Returns the raw ticker data dict.
        """
        ticker_data = api.ticker()
        if not ticker_data or not ticker_data.get("Success"):
            logger.warning("Ticker poll failed")
            return None

        now = pd.Timestamp.now(tz="UTC")
        price_row = {}
        volume_row = {}
        for pair in self.universe:
            pdata = ticker_data.get("Data", {}).get(pair)
            if pdata:
                price_row[pair] = pdata.get("LastPrice", np.nan)
                # Use UnitTradeValue as our volume proxy (consistent USD terms)
                volume_row[pair] = pdata.get("UnitTradeValue", 0)

        if price_row:
            new_price = pd.DataFrame([price_row], index=pd.to_datetime([now], utc=True))
            new_vol = pd.DataFrame([volume_row], index=pd.to_datetime([now], utc=True))

            if self.price_history.empty:
                self.price_history = new_price
            else:
                self.price_history = pd.concat([self.price_history, new_price])
                self.price_history.index = pd.to_datetime(self.price_history.index, utc=True, format="mixed")
                self.price_history = self.price_history[~self.price_history.index.duplicated(keep="last")]
                self.price_history.sort_index(inplace=True)

            if self.volume_history.empty:
                self.volume_history = new_vol
            else:
                self.volume_history = pd.concat([self.volume_history, new_vol])
                self.volume_history.index = pd.to_datetime(self.volume_history.index, utc=True, format="mixed")
                self.volume_history = self.volume_history[~self.volume_history.index.duplicated(keep="last")]
                self.volume_history.sort_index(inplace=True)

            # Trim old data
            cutoff = now - pd.Timedelta(hours=PRICE_HISTORY_MAX_HOURS)
            self.price_history = self.price_history[self.price_history.index >= cutoff]
            self.volume_history = self.volume_history[self.volume_history.index >= cutoff]

            self._persist_history()
            logger.debug("Ticker polled: %d pairs, history length %d",
                         len(price_row), len(self.price_history))

        return ticker_data

    def get_returns(self, pair: str, lookback_hours: int) -> Optional[float]:
        """
        Compute the simple percentage return over the last `lookback_hours`
        for a given pair, using the price history buffer.
        Returns None if insufficient data.
        """
        if pair not in self.price_history.columns:
            return None

        series = self.price_history[pair].dropna()
        if len(series) < 2:
            return None

        now = series.index[-1]
        cutoff = now - pd.Timedelta(hours=lookback_hours)
        past = series[series.index <= cutoff]
        if past.empty:
            return None

        price_then = past.iloc[-1]
        price_now = series.iloc[-1]
        if price_then == 0:
            return None
        return (price_now - price_then) / price_then

    def get_horizon_volatility(self, pair: str, lookback_hours: int) -> Optional[float]:
        """
        Compute realized volatility (stdev of sub-interval returns) within
        the given lookback window — this is the PER-HORIZON sigma used in
        the signal formula (sigma_{i,k}).

        Returns None if insufficient data, never returns 0 (floors at VOL_EPSILON).
        """
        if pair not in self.price_history.columns:
            return None

        series = self.price_history[pair].dropna()
        if len(series) < 5:
            return None

        now = series.index[-1]
        cutoff = now - pd.Timedelta(hours=lookback_hours)
        window = series[series.index >= cutoff]

        if len(window) < 5:
            return None

        # Compute returns at the polling interval (raw, not resampled)
        rets = window.pct_change().dropna()
        if len(rets) < 3:
            return None

        vol = rets.std()
        if vol is None or np.isnan(vol) or vol < VOL_EPSILON:
            return VOL_EPSILON
        return vol

    def get_global_volatility(self, pair: str, window_hours: int = 168) -> Optional[float]:
        """
        Global realized volatility for vol-parity position sizing
        (stdev of hourly returns over the window).
        """
        if pair not in self.price_history.columns:
            return None

        series = self.price_history[pair].dropna()
        now = series.index[-1]
        cutoff = now - pd.Timedelta(hours=window_hours)
        window = series[series.index >= cutoff]

        if len(window) < 10:
            return None

        hourly = window.resample("1h").last().dropna()
        if len(hourly) < 3:
            return None

        returns = hourly.pct_change().dropna()
        vol = returns.std()
        return vol if vol and vol > 0 else None

    def get_volume_surge(self, pair: str) -> float:
        """
        VolumeSurge_i = Volume_{6h} / avg(Volume_{6h})_{3d}

        Ratio of current rolling 6h volume to the average 6h volume
        over the trailing 3 days.  Capped at VOLUME_SURGE_CAP.
        Returns 1.0 if insufficient data.
        """
        if pair not in self.volume_history.columns:
            return 1.0

        series = self.volume_history[pair].dropna()
        if len(series) < 5:
            return 1.0

        now = series.index[-1]

        # Current 6h volume (sum of readings in last 6h)
        cutoff_6h = now - pd.Timedelta(hours=VOLUME_SURGE_WINDOW_HOURS)
        recent = series[series.index >= cutoff_6h]
        if recent.empty:
            return 1.0
        current_vol = recent.mean()  # average trade value in the window

        # 3-day average of 6h volumes
        cutoff_3d = now - pd.Timedelta(days=VOLUME_SURGE_AVG_DAYS)
        historical = series[series.index >= cutoff_3d]
        if historical.empty or len(historical) < 10:
            return 1.0
        avg_vol = historical.mean()

        if avg_vol <= 0:
            return 1.0

        surge = current_vol / avg_vol
        # Clamp to [0.1, VOLUME_SURGE_CAP] — don't let it go negative or crazy
        surge = max(0.1, min(surge, VOLUME_SURGE_CAP))
        return surge

    def get_latest_price(self, pair: str) -> Optional[float]:
        """Return the most recent price for a pair from history."""
        if pair not in self.price_history.columns:
            return None
        series = self.price_history[pair].dropna()
        return series.iloc[-1] if len(series) > 0 else None

    def get_latest_prices_dict(self) -> dict:
        """Return {pair: latest_price} for all pairs with data."""
        result = {}
        for pair in self.universe:
            p = self.get_latest_price(pair)
            if p is not None:
                result[pair] = p
        return result

    # ─── Persistence ─────────────────────────────────────────────────────

    def _persist_history(self):
        try:
            self.price_history.to_csv(self._history_file)
            self.volume_history.to_csv(self._volume_file)
        except Exception as e:
            logger.error("Failed to persist history: %s", e)

    def _persist_universe(self):
        try:
            with open(self._universe_file, "w") as f:
                json.dump(self.universe, f, indent=2)
        except Exception as e:
            logger.error("Failed to persist universe: %s", e)

    def _load_persisted(self):
        # Load price history
        if os.path.exists(self._history_file):
            try:
                df = pd.read_csv(self._history_file, index_col=0)
                df.index = pd.to_datetime(df.index, utc=True, format="mixed")
                df = df[~df.index.duplicated(keep="last")]
                df.sort_index(inplace=True)
                self.price_history = df
                logger.info("Loaded %d rows of price history from disk",
                            len(self.price_history))
            except Exception as e:
                logger.warning("Could not load price history: %s", e)

        # Load volume history
        if os.path.exists(self._volume_file):
            try:
                df = pd.read_csv(self._volume_file, index_col=0)
                df.index = pd.to_datetime(df.index, utc=True, format="mixed")
                df = df[~df.index.duplicated(keep="last")]
                df.sort_index(inplace=True)
                self.volume_history = df
                logger.info("Loaded %d rows of volume history from disk",
                            len(self.volume_history))
            except Exception as e:
                logger.warning("Could not load volume history: %s", e)

        # Load universe
        if os.path.exists(self._universe_file):
            try:
                with open(self._universe_file) as f:
                    self.universe = json.load(f)
                logger.info("Loaded universe from disk: %d pairs",
                            len(self.universe))
            except Exception as e:
                logger.warning("Could not load universe: %s", e)

