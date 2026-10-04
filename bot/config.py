"""
config.py — Central configuration for the momentum strategy bot.

All tunable parameters live here. No magic numbers elsewhere in the codebase.
Optimized by parameter sweep (N=3, exit=6, 90/50 → +26.10%, composite 10.7).
"""

import os
from dotenv import load_dotenv

_bot_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_bot_dir)

# Load .env from bot dir, parent dir, or current dir
load_dotenv(os.path.join(_bot_dir, ".env"))
load_dotenv(os.path.join(_parent_dir, ".env"))
load_dotenv()

# ─── API credentials ────────────────────────────────────────────────────────
RST_API_KEY = os.getenv("RST_API_KEY", "")
RST_SECRET_KEY = os.getenv("RST_SECRET_KEY", "")
BASE_URL = "https://mock-api.roostoo.com"

# ─── Universe ────────────────────────────────────────────────────────────────
UNIVERSE_REFRESH_HOURS = 24          # Re-pull exchangeInfo every N hours
MIN_TRADE_VALUE = 10_000             # Minimum UnitTradeValue to include a pair (liquidity filter)

# ─── Signal — momentum lookbacks (in hours of bar data) ─────────────────────
LOOKBACK_SHORT = 6                   # 6-hour return
LOOKBACK_MED = 24                    # 24-hour return
LOOKBACK_LONG = 72                   # 72-hour return
LOOKBACK_WEIGHTS = [1/3, 1/3, 1/3]  # Equal-weighted across horizons (per writeup)
VOL_EPSILON = 1e-8                   # Floor for per-horizon sigma to prevent division by zero

# ─── Volume surge ────────────────────────────────────────────────────────────
VOLUME_SURGE_WINDOW_HOURS = 6        # Rolling 6h volume window
VOLUME_SURGE_AVG_DAYS = 3            # Average 6h volume over trailing 3 days
VOLUME_SURGE_CAP = 3.0               # Cap the surge multiplier at 3x

# ─── Regime filter ───────────────────────────────────────────────────────────
REGIME_PAIR = "BTC/USD"              # Anchor pair for market regime
REGIME_MA_PERIOD = 50                # Moving average period (hourly bars)

# ─── Portfolio construction — rank buffer (param sweep winner) ───────────────
INITIAL_CAPITAL = 100_000            # $100k mock USD
TOP_N_ENTER = 5                      # Enter long when rank <= 5 (more diversified)
TOP_N_EXIT = 10                      # Hold long until rank > 10 (wide buffer = less churn)
MAX_SINGLE_POSITION_PCT = 0.30       # Max 30% of equity in one position

# ─── Exposure (param sweep optimal) ─────────────────────────────────────────
RISK_ON_GROSS_EXPOSURE = 0.90        # 90% of equity deployed in risk-on
RISK_OFF_GROSS_EXPOSURE = 0.50       # 50% deployed in risk-off

# ─── Rebalance ───────────────────────────────────────────────────────────────
REBALANCE_INTERVAL_HOURS = 12        # Rebalance every 12 hours (less turnover = less fee drag)

# ─── Risk / drawdown circuit breaker ────────────────────────────────────────
DRAWDOWN_THRESHOLD = 0.08            # 8% drawdown from peak -> cut exposure
DRAWDOWN_RECOVERY = 0.03             # Need 3% recovery from trough to restore full exposure
CIRCUIT_BREAKER_EXPOSURE = 0.20      # When tripped, reduce to 20% gross exposure

# ─── Fees (for backtester) ───────────────────────────────────────────────────
TAKER_FEE = 0.001                    # 0.1% market/taker
MAKER_FEE = 0.0008                   # ~0.08% limit/maker

# ─── Data collection ────────────────────────────────────────────────────────
TICKER_POLL_SECONDS = 300            # Poll tickers every 5 min for local history buffer
PRICE_HISTORY_MAX_HOURS = 200        # Keep at most ~200 hours of history in memory

# ─── Logging & State ─────────────────────────────────────────────────────────
if os.path.exists(os.path.join(_parent_dir, "state")):
    STATE_DIR = os.path.join(_parent_dir, "state")
    LOG_DIR = os.path.join(_parent_dir, "logs")
else:
    STATE_DIR = os.path.join(_bot_dir, "state")
    LOG_DIR = os.path.join(_bot_dir, "logs")

LOG_LEVEL = "INFO"
