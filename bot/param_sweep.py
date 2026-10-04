"""
param_sweep.py — Quick parameter sweep to find optimal TOP_N_ENTER + exposure.
Tests combinations and reports the best composite score.
"""
import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)

# Import backtest components
from backtest import fetch_all_pairs, PAIR_MAP, _compute_scores_at_bar, compute_metrics
from config import (
    LOOKBACK_SHORT, LOOKBACK_MED, LOOKBACK_LONG, LOOKBACK_WEIGHTS,
    VOL_EPSILON, VOLUME_SURGE_CAP,
    VOLUME_SURGE_WINDOW_HOURS, VOLUME_SURGE_AVG_DAYS,
    REGIME_MA_PERIOD, REGIME_PAIR,
    MAX_SINGLE_POSITION_PCT,
    DRAWDOWN_THRESHOLD, DRAWDOWN_RECOVERY, CIRCUIT_BREAKER_EXPOSURE,
    TAKER_FEE,
)


def run_sweep_backtest(klines, top_n_enter, top_n_exit, gross_on, gross_off,
                       fee_rate=TAKER_FEE, initial_capital=50_000, rebalance_hours=4):
    """Stripped-down backtest for parameter sweep."""
    pairs = list(klines.keys())
    close_panel = pd.DataFrame({p: klines[p]["close"] for p in pairs}).dropna(how="all")
    volume_panel = pd.DataFrame({p: klines[p]["quote_vol"] for p in pairs}).dropna(how="all")
    returns = close_panel.pct_change()
    volume_panel = volume_panel.reindex(close_panel.index)

    warmup = max(LOOKBACK_LONG, REGIME_MA_PERIOD, VOLUME_SURGE_AVG_DAYS * 24) + 10
    rebal_indices = list(range(warmup, len(close_panel), rebalance_hours))

    equity = initial_capital
    peak = initial_capital
    trough = initial_capital
    breaker_active = False
    current_longs = set()
    holdings = {}
    trade_count = 0

    equity_curve = []
    regime_pair = REGIME_PAIR if REGIME_PAIR in pairs else pairs[0]

    for i in range(min(warmup, len(close_panel))):
        equity_curve.append({"time": close_panel.index[i], "equity": equity})

    for idx_pos, bar_idx in enumerate(rebal_indices):
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

        # Rank buffer
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranks = {pair: i + 1 for i, (pair, _) in enumerate(ranked)}

        new_longs = set()
        for pair, rank in ranks.items():
            if rank <= top_n_enter:
                new_longs.add(pair)
            elif pair in current_longs and rank <= top_n_exit:
                new_longs.add(pair)
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
            gross = gross_on
        else:
            gross = gross_off

        # Vol-parity weights
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

        # Execute
        old_weights = dict(holdings)
        turnover = sum(abs(target.get(p, 0) - old_weights.get(p, 0))
                       for p in set(list(old_weights) + list(target)))
        fee_cost = turnover * equity * fee_rate
        equity -= fee_cost
        if turnover > 0.01:
            trade_count += 1
        holdings = target

        # Apply returns
        next_rebal = rebal_indices[idx_pos + 1] if idx_pos + 1 < len(rebal_indices) else len(close_panel)
        for h in range(bar_idx + 1, min(next_rebal, len(close_panel))):
            period_ret = 0.0
            for p, w in holdings.items():
                r = returns.iloc[h].get(p, 0)
                if np.isnan(r):
                    r = 0
                period_ret += w * r
            equity *= (1 + period_ret)
            equity_curve.append({"time": close_panel.index[h], "equity": equity})

    return pd.DataFrame(equity_curve), trade_count


def main():
    print("Fetching data...")
    pairs = list(PAIR_MAP.keys())
    klines = fetch_all_pairs(pairs, days=90)
    if not klines:
        print("No data!")
        return

    print(f"Got {len(klines)} pairs. Running parameter sweep...\n")

    # Parameter grid
    configs = [
        # (top_n_enter, top_n_exit, gross_on, gross_off, label)
        (3, 6,  0.80, 0.40, "N=3, exit=6,  80/40"),
        (3, 8,  0.80, 0.40, "N=3, exit=8,  80/40"),
        (3, 8,  0.90, 0.50, "N=3, exit=8,  90/50"),
        (5, 10, 0.80, 0.40, "N=5, exit=10, 80/40"),
        (5, 10, 0.95, 0.50, "N=5, exit=10, 95/50"),
        (5, 8,  0.80, 0.40, "N=5, exit=8,  80/40"),
        (7, 12, 0.80, 0.40, "N=7, exit=12, 80/40"),
        (7, 12, 0.90, 0.50, "N=7, exit=12, 90/50"),
        (10, 15, 0.80, 0.40, "N=10, exit=15, 80/40"),
        (3, 6,  0.90, 0.50, "N=3, exit=6,  90/50"),
    ]

    results = []
    for top_n, exit_n, g_on, g_off, label in configs:
        eq, tc = run_sweep_backtest(klines, top_n, exit_n, g_on, g_off)
        m = compute_metrics(eq)
        results.append({
            "config": label,
            "return": m.get("total_return", "?"),
            "sharpe": m.get("sharpe", 0),
            "sortino": m.get("sortino", 0),
            "calmar": m.get("calmar", 0),
            "max_dd": m.get("max_drawdown", "?"),
            "composite": m.get("composite_score", 0),
            "trades": tc,
        })
        print(f"  {label:30s} | ret={m.get('total_return','?'):>8s} | "
              f"sharpe={m.get('sharpe',0):>7.3f} | sortino={m.get('sortino',0):>7.3f} | "
              f"calmar={m.get('calmar',0):>8.3f} | dd={m.get('max_drawdown','?'):>7s} | "
              f"comp={m.get('composite_score',0):>7.3f} | trades={tc}")

    # Sort by composite
    results.sort(key=lambda x: x["composite"], reverse=True)
    print("\n" + "=" * 80)
    print("BEST CONFIGS (by composite score):")
    print("=" * 80)
    for i, r in enumerate(results[:5]):
        print(f"  #{i+1}: {r['config']:30s} | ret={r['return']:>8s} | "
              f"composite={r['composite']:>7.3f} | dd={r['max_dd']:>7s}")


if __name__ == "__main__":
    main()
