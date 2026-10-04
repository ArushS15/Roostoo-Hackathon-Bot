"""
state.py — Persist portfolio state, equity curve, and peak so restarts
don't lose context.

State is saved as JSON after every rebalance cycle.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from config import STATE_DIR

logger = logging.getLogger("state")

STATE_FILE = os.path.join(STATE_DIR, "bot_state.json")
EQUITY_CURVE_FILE = os.path.join(STATE_DIR, "equity_curve.json")


def _default_state() -> dict:
    return {
        "current_holdings": {},     # pair -> weight (fraction of equity, signed)
        "equity": 0.0,              # Set from real balance at startup
        "peak_equity": 0.0,         # Set from real balance at startup
        "trough_equity": 0.0,       # Set from real balance at startup
        "breaker_active": False,
        "shorting_enabled": True,
        "last_rebalance_ts": None,
        "total_trades": 0,
        "active_days": set(),
    }


def load_state() -> dict:
    """Load persisted state from disk, or return defaults."""
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            # Convert active_days back to set
            state["active_days"] = set(state.get("active_days", []))
            logger.info("State loaded from disk (equity=%.0f, trades=%d, active_days=%d)",
                        state.get("equity", 0), state.get("total_trades", 0),
                        len(state.get("active_days", set())))
            return state
        except Exception as e:
            logger.warning("Failed to load state: %s — using defaults", e)
    return _default_state()


def save_state(state: dict):
    """Persist state to disk."""
    os.makedirs(STATE_DIR, exist_ok=True)
    try:
        # Convert set to list for JSON serialization
        state_copy = dict(state)
        state_copy["active_days"] = list(state.get("active_days", set()))
        with open(STATE_FILE, "w") as f:
            json.dump(state_copy, f, indent=2)
        logger.debug("State saved to disk")
    except Exception as e:
        logger.error("Failed to save state: %s", e)


def record_equity(equity: float):
    """Append a timestamped equity reading to the equity curve file."""
    os.makedirs(STATE_DIR, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "equity": round(equity, 2),
    }
    try:
        curve = []
        if os.path.exists(EQUITY_CURVE_FILE):
            with open(EQUITY_CURVE_FILE) as f:
                curve = json.load(f)
        curve.append(entry)
        with open(EQUITY_CURVE_FILE, "w") as f:
            json.dump(curve, f, indent=1)
    except Exception as e:
        logger.error("Failed to record equity: %s", e)


def load_equity_curve() -> list[dict]:
    """Load the equity curve from disk."""
    if os.path.exists(EQUITY_CURVE_FILE):
        try:
            with open(EQUITY_CURVE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []
