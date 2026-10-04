"""
backtest.py -- Offline backtester implementing the cross-sectional factor
momentum strategy (long-only) with:
  - Per-horizon vol normalization (R_{k}/sigma_{k} per lookback)
  - Volume surge multiplier (capped)
  - Rank-buffer turnover control (enter top 3, hold to top 6)
  - Vol-parity position sizing
  - BTC regime filter
  - Drawdown circuit breaker

Fetches historical OHLCV from Binance. Reports Sharpe, Sortino, Calmar,
max drawdown, turnover, and PnL.
"""

import os
import logging
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from config import (
    LOOKBACK_SHORT, LOOKBACK_MED, LOOKBACK_LONG, LOOKBACK_WEIGHTS,
    VOL_EPSILON, VOLUME_SURGE_CAP,
    VOLUME_SURGE_WINDOW_HOURS, VOLUME_SURGE_AVG_DAYS,
    REGIME_MA_PERIOD, REGIME_PAIR,
    TOP_N_ENTER, TOP_N_EXIT,
    MAX_SINGLE_POSITION_PCT,
    RISK_ON_GROSS_EXPOSURE, RISK_OFF_GROSS_EXPOSURE,
    DRAWDOWN_THRESHOLD, DRAWDOWN_RECOVERY, CIRCUIT_BREAKER_EXPOSURE,
    TAKER_FEE, INITIAL_CAPITAL, REBALANCE_INTERVAL_HOURS,
)

logger = logging.getLogger("backtest")

_bot_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_bot_dir)
RESULTS_DIR = os.path.join(_bot_dir, "backtest_results")


# =====================================================================
#  Data fetching (Binance public API -- no auth needed)
# =====================================================================

BINANCE_BASE = "https://api.binance.com"

PAIR_MAP = {
    "BTC/USD": "BTCUSDT",
    "ETH/USD": "ETHUSDT",
    "BNB/USD": "BNBUSDT",
    "SOL/USD": "SOLUSDT",
    "XRP/USD": "XRPUSDT",
    "ADA/USD": "ADAUSDT",
    "DOGE/USD": "DOGEUSDT",
    "AVAX/USD": "AVAXUSDT",
    "LINK/USD": "LINKUSDT",
    "DOT/USD": "DOTUSDT",
    "TRX/USD": "TRXUSDT",
    "LTC/USD": "LTCUSDT",
    "UNI/USD": "UNIUSDT",
    "NEAR/USD": "NEARUSDT",
    "SUI/USD": "SUIUSDT",
    "ENA/USD": "ENAUSDT",
    "FIL/USD": "FILUSDT",
    "HBAR/USD": "HBARUSDT",
    "SEI/USD": "SEIUSDT",
    "ARB/USD": "ARBUSDT",
    "FET/USD": "FETUSDT",
    "PEPE/USD": "PEPEUSDT",
    "WLD/USD": "WLDUSDT",
    "BONK/USD": "BONKUSDT",
    "CRV/USD": "CRVUSDT",
    "AAVE/USD": "AAVEUSDT",
    "TAO/USD": "TAOUSDT",
    "ICP/USD": "ICPUSDT",
    "PENDLE/USD": "PENDLEUSDT",
    "ONDO/USD": "ONDOUSDT",
}


def fetch_binance_klines(symbol: str, interval: str = "1h",
                         days: int = 90) -> pd.DataFrame:
    """Fetch hourly klines from Binance for a single symbol."""
    url = f"{BINANCE_BASE}/api/v3/klines"
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_data = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            current_start = data[-1][0] + 1
        except Exception as e:
            logger.error("Binance fetch error for %s: %s", symbol, e)
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    for col in ["open", "high", "low", "close", "volume", "quote_vol"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume", "quote_vol"]]


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache_klines")


def fetch_all_pairs(pairs: list[str], days: int = 90) -> dict[str, pd.DataFrame]:
    """Fetch klines for all pairs with disk caching."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    result = {}
    for pair in pairs:
        binance_sym = PAIR_MAP.get(pair)
        if not binance_sym:
            continue
        cache_file = os.path.join(CACHE_DIR, f"{binance_sym}_{days}d.csv")
        if os.path.exists(cache_file):
            mtime = os.path.getmtime(cache_file)
            if (datetime.now().timestamp() - mtime) < 12 * 3600:
                try:
                    df = pd.read_csv(cache_file, index_col=0)
                    df.index = pd.to_datetime(df.index, utc=True, format="mixed")
                    result[pair] = df
                    continue
                except Exception:
                    pass

        logger.info("Fetching %s (%s) -- %d days...", pair, binance_sym, days)
        df = fetch_binance_klines(binance_sym, days=days)
        if not df.empty:
            result[pair] = df
            try:
                df.to_csv(cache_file)
            except Exception:
                pass
            logger.info("  -> %d bars for %s", len(df), pair)
    return result


# =====================================================================
#  Signal computation (mirrors signals.py logic for backtest)
# =====================================================================

def _compute_scores_at_bar(
    close_panel: pd.DataFrame,
    volume_panel: pd.DataFrame,
    returns: pd.DataFrame,
    bar_idx: int,
    pairs: list[str],
) -> dict[str, float]:
    """
    Compute cross-sectional factor momentum scores at a specific bar index.

    Score_i = [w1 * R_{6h}/sigma_{6h} + w2 * R_{24h}/sigma_{24h}
               + w3 * R_{72h}/sigma_{72h}] * VolumeSurge_i
    """
    lookbacks = [LOOKBACK_SHORT, LOOKBACK_MED, LOOKBACK_LONG]
    scores = {}

    for pair in pairs:
        if pair not in close_panel.columns:
            continue

        series = close_panel[pair].iloc[:bar_idx + 1].dropna()
        if len(series) < LOOKBACK_LONG + 1:
            continue

        # Per-horizon return and sigma
        terms = []
        weights_used = []

        for lb, w in zip(lookbacks, LOOKBACK_WEIGHTS):
            if len(series) <= lb:
                continue

            # Return over lookback
            ret = series.iloc[-1] / series.iloc[-lb - 1] - 1

            # Sigma = stdev of sub-interval returns within the lookback window
            rets_window = returns[pair].iloc[max(0, bar_idx - lb):bar_idx + 1].dropna()
            if len(rets_window) < 3:
                continue
            sigma = rets_window.std()
            sigma = max(sigma, VOL_EPSILON)

            terms.append(ret / sigma)
            weights_used.append(w)

        if len(terms) < 2:
            continue

        # Normalize weights
        w_sum = sum(weights_used)
        norm_weights = [w / w_sum for w in weights_used]
        raw_score = sum(t * w for t, w in zip(terms, norm_weights))

        # Volume surge: 6h volume / 3-day avg 6h volume
        vol_surge = 1.0
        if pair in volume_panel.columns:
            vol_series = volume_panel[pair].iloc[:bar_idx + 1].dropna()
            if len(vol_series) > VOLUME_SURGE_AVG_DAYS * 24:
                recent_vol = vol_series.iloc[-VOLUME_SURGE_WINDOW_HOURS:].mean()
                avg_vol = vol_series.iloc[-(VOLUME_SURGE_AVG_DAYS * 24):].mean()
                if avg_vol > 0:
                    vol_surge = recent_vol / avg_vol
                    vol_surge = max(0.1, min(vol_surge, VOLUME_SURGE_CAP))

        scores[pair] = raw_score * vol_surge

    return scores


def _apply_rank_buffer_bt(
    scores: dict[str, float],
    current_longs: set[str],
) -> set[str]:
    """Rank-buffer logic for backtest (mirrors signals.apply_rank_buffer)."""
    if not scores:
        return set()

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranks = {pair: i + 1 for i, (pair, _) in enumerate(ranked)}

    new_longs = set()
    for pair, rank in ranks.items():
        if rank <= TOP_N_ENTER:
            new_longs.add(pair)
        elif pair in current_longs and rank <= TOP_N_EXIT:
            new_longs.add(pair)

    return new_longs


# =====================================================================
#  Backtest engine (long-only)
# =====================================================================

def run_backtest(
    klines: dict[str, pd.DataFrame],
    fee_rate: float = TAKER_FEE,
    initial_capital: float = INITIAL_CAPITAL,
    rebalance_hours: int = 4,
) -> tuple[pd.DataFrame, int, dict]:
    """
    Long-only backtest with: per-horizon vol-normalized momentum, volume surge,
    rank buffer, vol-parity sizing, regime filter, circuit breaker.

    Signal on bar t, trade executes on bar t+1 (no lookahead).

    Returns (equity_curve_df, trade_count, diagnostics).
    """
    pairs = list(klines.keys())

    # Build unified panels
    close_panel = pd.DataFrame({p: klines[p]["close"] for p in pairs}).dropna(how="all")
    volume_panel = pd.DataFrame({p: klines[p]["quote_vol"] for p in pairs}).dropna(how="all")
    returns = close_panel.pct_change()

    # Align volume panel to close panel index
    volume_panel = volume_panel.reindex(close_panel.index)

    # Warmup: need enough bars for the longest lookback
    warmup = max(LOOKBACK_LONG, REGIME_MA_PERIOD, VOLUME_SURGE_AVG_DAYS * 24) + 10
    rebal_indices = list(range(warmup, len(close_panel), rebalance_hours))

    equity = initial_capital
    peak = initial_capital
    trough = initial_capital
    breaker_active = False
    current_longs = set()
    holdings = {}  # pair -> weight (all positive)
    trade_count = 0
    total_turnover = 0.0
    long_pnl = 0.0

    equity_curve = []
    regime_pair = REGIME_PAIR if REGIME_PAIR in pairs else pairs[0]

    # Fill equity curve for warmup period
    for i in range(min(warmup, len(close_panel))):
        equity_curve.append({"time": close_panel.index[i], "equity": equity})

    for idx_pos, bar_idx in enumerate(rebal_indices):
        # Compute scores
        scores = _compute_scores_at_bar(close_panel, volume_panel, returns, bar_idx, pairs)
        if not scores:
            next_rebal = rebal_indices[idx_pos + 1] if idx_pos + 1 < len(rebal_indices) else len(close_panel)
            for h in range(bar_idx, min(next_rebal, len(close_panel))):
                period_ret = sum(
                    holdings.get(p, 0) * (returns.iloc[h].get(p, 0) if not np.isnan(returns.iloc[h].get(p, 0)) else 0)
                    for p in holdings
                )
                equity *= (1 + period_ret)
                equity_curve.append({"time": close_panel.index[h], "equity": equity})
            continue

        # Rank buffer (long only)
        new_longs = _apply_rank_buffer_bt(scores, current_longs)
        current_longs = new_longs

        # Regime
        regime_series = close_panel[regime_pair].iloc[:bar_idx + 1].dropna()
        if len(regime_series) >= REGIME_MA_PERIOD:
            ma = regime_series.rolling(REGIME_MA_PERIOD).mean().iloc[-1]
            regime = "risk_on" if regime_series.iloc[-1] > ma else "risk_off"
        else:
            regime = "risk_off"

        # Circuit breaker
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if not breaker_active and dd >= DRAWDOWN_THRESHOLD:
            breaker_active = True
            trough = equity
        if breaker_active:
            if equity < trough:
                trough = equity
            recovery = (equity - trough) / trough if trough > 0 else 0
            if recovery >= DRAWDOWN_RECOVERY:
                breaker_active = False

        # Gross exposure
        if breaker_active:
            gross = CIRCUIT_BREAKER_EXPOSURE
        elif regime == "risk_on":
            gross = RISK_ON_GROSS_EXPOSURE
        else:
            gross = RISK_OFF_GROSS_EXPOSURE

        # Vol-parity long weights
        target = {}
        if new_longs:
            inv_vols = {}
            for p in new_longs:
                vol_w = returns[p].iloc[max(0, bar_idx - 168):bar_idx + 1].std()
                inv_vols[p] = 1.0 / max(vol_w, VOL_EPSILON) if vol_w and not np.isnan(vol_w) else 1.0
            total_iv = sum(inv_vols.values())
            for p in new_longs:
                w = (inv_vols[p] / total_iv) * gross if total_iv > 0 else gross / len(new_longs)
                target[p] = min(w, MAX_SINGLE_POSITION_PCT)

        # Execute with fees
        old_weights = dict(holdings)
        new_weights = target
        turnover = 0.0
        all_p = set(list(old_weights.keys()) + list(new_weights.keys()))
        for p in all_p:
            old_w = old_weights.get(p, 0)
            new_w = new_weights.get(p, 0)
            turnover += abs(new_w - old_w)

        fee_cost = turnover * equity * fee_rate
        equity -= fee_cost
        total_turnover += turnover
        if turnover > 0.01:
            trade_count += 1

        holdings = new_weights

        # Apply returns until next rebalance
        next_rebal = rebal_indices[idx_pos + 1] if idx_pos + 1 < len(rebal_indices) else len(close_panel)
        for h in range(bar_idx + 1, min(next_rebal, len(close_panel))):
            period_ret = 0.0
            for p, w in holdings.items():
                r = returns.iloc[h].get(p, 0)
                if np.isnan(r):
                    r = 0
                pnl = w * r
                period_ret += pnl
                long_pnl += pnl * equity

            equity *= (1 + period_ret)
            equity_curve.append({"time": close_panel.index[h], "equity": equity})

    diagnostics = {
        "total_turnover": round(total_turnover, 2),
        "avg_turnover_per_rebal": round(total_turnover / max(trade_count, 1), 3),
        "long_pnl": round(long_pnl, 2),
    }

    return pd.DataFrame(equity_curve), trade_count, diagnostics


# =====================================================================
#  Metrics
# =====================================================================

def compute_metrics(equity_curve: pd.DataFrame) -> dict:
    """Compute Sharpe, Sortino, Calmar, max drawdown, and total return."""
    if len(equity_curve) < 2:
        return {}

    eq = equity_curve["equity"].values
    rets = np.diff(eq) / eq[:-1]

    ann_factor = np.sqrt(8760)  # hourly bars

    mean_ret = np.mean(rets)
    std_ret = np.std(rets)
    downside = rets[rets < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-10

    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0
    sortino = (mean_ret / downside_std * ann_factor) if downside_std > 0 else 0

    peak = np.maximum.accumulate(eq)
    drawdowns = (peak - eq) / peak
    max_dd = np.max(drawdowns)

    total_hours = len(rets)
    total_return = (eq[-1] / eq[0]) - 1
    ann_return = (1 + total_return) ** (8760 / max(total_hours, 1)) - 1
    calmar = (ann_return / max_dd) if max_dd > 0 else 0

    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    return {
        "total_return": f"{total_return * 100:.2f}%",
        "annualized_return": f"{ann_return * 100:.2f}%",
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown": f"{max_dd * 100:.2f}%",
        "composite_score": round(composite, 3),
        "total_bars": len(equity_curve),
    }


# =====================================================================
#  Entry point
# =====================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    parser = argparse.ArgumentParser(description="Backtest cross-sectional factor momentum (long-only)")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--fee", type=float, default=TAKER_FEE, help="Fee rate per trade")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Initial capital")
    parser.add_argument("--rebalance-hours", type=int, default=REBALANCE_INTERVAL_HOURS, help="Rebalance interval (hours)")
    args = parser.parse_args()

    # Fetch data
    pairs = list(PAIR_MAP.keys())
    klines = fetch_all_pairs(pairs, days=args.days)
    if not klines:
        logger.error("No data fetched -- exiting")
        return

    logger.info("=" * 60)
    logger.info("Running backtest: %d pairs, %d days, rebal every %dh, capital $%dk",
                len(klines), args.days, args.rebalance_hours, args.capital / 1000)
    logger.info("Config: TOP_N_ENTER=%d, TOP_N_EXIT=%d, exposure=%.0f/%.0f",
                TOP_N_ENTER, TOP_N_EXIT,
                RISK_ON_GROSS_EXPOSURE * 100, RISK_OFF_GROSS_EXPOSURE * 100)
    logger.info("=" * 60)

    # Run backtest
    eq_df, tc, diag = run_backtest(
        klines, fee_rate=args.fee, initial_capital=args.capital,
        rebalance_hours=args.rebalance_hours,
    )

    metrics = compute_metrics(eq_df)

    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS (Long-Only)")
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info("  %-22s : %s", k, v)
    logger.info("  %-22s : %d", "trade_count", tc)
    logger.info("  %-22s : %.2f", "total_turnover", diag["total_turnover"])
    logger.info("  %-22s : %.3f", "avg_turnover/rebal", diag["avg_turnover_per_rebal"])
    logger.info("  %-22s : $%.2f", "long_pnl", diag["long_pnl"])
    logger.info("=" * 60)

    # Stress test: 2x fees
    logger.info("\n--- STRESS TEST: 2x fees ---")
    eq2, tc2, _ = run_backtest(
        klines, fee_rate=args.fee * 2, initial_capital=args.capital,
        rebalance_hours=args.rebalance_hours,
    )
    m2 = compute_metrics(eq2)
    for k, v in m2.items():
        logger.info("  %-22s : %s", k, v)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"equity_curve_{ts}.csv"
    out_path = os.path.join(RESULTS_DIR, filename)
    eq_df.to_csv(out_path, index=False)
    parent_results = os.path.join(_parent_dir, "backtest_results")
    if os.path.exists(parent_results) and parent_results != RESULTS_DIR:
        eq_df.to_csv(os.path.join(parent_results, filename), index=False)
    logger.info("Equity curve saved to %s", out_path)


if __name__ == "__main__":
    main()
