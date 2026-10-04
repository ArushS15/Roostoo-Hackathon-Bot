# Roostoo Momentum Strategy Bot

**Cross-sectional momentum strategy** for the Roostoo HK vs AU vs IN Quant Trading Hackathon.

Ranks all tradable coins by risk-adjusted momentum, goes long the strongest, optionally shorts the weakest, filters through a BTC-based market regime check, and sizes positions by inverse volatility with a drawdown circuit breaker.

## Architecture

```
bot/
├── config.py          # All tunable parameters — no magic numbers elsewhere
├── roostoo_client.py  # Signed HTTP wrapper (HMAC-SHA256), retries, Success-flag checking
├── data.py            # Universe management, ticker polling, price history buffer
├── signals.py         # Pure functions: momentum scores, regime filter (no API calls)
├── risk.py            # Pure functions: vol-parity sizing, caps, circuit breaker
├── execution.py       # Weight deltas → buy/sell/short orders, precision enforcement
├── state.py           # JSON persistence for holdings, equity curve, peak/trough
├── main.py            # APScheduler loop — ALL trades originate here
├── backtest.py        # Offline backtester using Binance historical data
├── requirements.txt
└── .env.example
```

## Quick Start

```bash
cd bot

# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env with your RST_API_KEY and RST_SECRET_KEY

# 3. Run the backtester first to validate signals
python backtest.py --days 90

# 4. Run the live bot
python main.py
```

## Strategy Overview

| Component | Description |
|-----------|-------------|
| **Signal** | Blended momentum (6h / 24h / 72h returns) ÷ per-horizon volatility × volume surge multiplier |
| **Regime** | BTC price vs 50-period hourly MA → risk-on (90% exposure) / risk-off (50% exposure) |
| **Sizing** | Inverse-volatility parity, capped at 30% per position |
| **Risk** | 8% drawdown circuit breaker (cuts to 20% gross exposure) |
| **Rebalance** | Every 12 hours with rank buffer hysteresis (enter top 5, hold to top 10) to minimize turnover |

## Performance (90-Day Backtest)

- **Total Return**: **+11.07%** ($100k capital)
- **Annualized Return**: **+58.93%**
- **Sharpe Ratio**: **1.372**
- **Sortino Ratio**: **1.482**
- **Calmar Ratio**: **4.059**
- **Max Drawdown**: **14.52%**

## Key Safety Features

1. **Long-Only Alpha Execution** — eliminates borrow fee drag and short squeeze risks
2. **Dynamic universe** — pulled live from `exchangeInfo`, never hardcoded
3. **Binance Seed Backfill** — seeds price and volume history on startup for immediate signal readiness
4. **Rank Buffer Hysteresis** — stops churn and excessive transaction fees
5. **State persistence** — survives restarts without losing equity peak or active holdings
6. **Full logging** — every signal, order, and API response is logged to disk
