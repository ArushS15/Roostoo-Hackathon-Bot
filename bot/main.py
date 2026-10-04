"""
main.py -- Entry point and scheduling loop for the momentum strategy bot.

Pure long-only cross-sectional factor momentum.
Every trading decision originates from the scheduled rebalance cycle here.
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.blocking import BlockingScheduler

from config import (
    REBALANCE_INTERVAL_HOURS, TICKER_POLL_SECONDS,
    UNIVERSE_REFRESH_HOURS, LOG_DIR, LOG_LEVEL,
)
from data import DataManager
from signals import compute_momentum_scores, compute_regime, apply_rank_buffer
from risk import compute_target_weights, check_circuit_breaker
from execution import execute_rebalance, get_portfolio_value, cancel_all_pending
from state import load_state, save_state, record_equity


# --- Logging setup ---

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    root.addHandler(ch)
    root.addHandler(fh)

    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    return logging.getLogger("main")


# --- Global state ---

data_mgr: DataManager = None
bot_state: dict = None
logger = None


# =====================================================================
#  Scheduled jobs
# =====================================================================

def job_poll_ticker():
    """Periodic ticker poll -- builds up price + volume history."""
    try:
        data_mgr.poll_ticker()
    except Exception as e:
        logger.error("Ticker poll error: %s", e, exc_info=True)


def job_refresh_universe():
    """Daily universe refresh from exchangeInfo."""
    try:
        data_mgr.refresh_universe()
    except Exception as e:
        logger.error("Universe refresh error: %s", e, exc_info=True)


def job_rebalance():
    """
    The core rebalance cycle -- this is where ALL trading decisions are made.

    Steps:
      1. Poll latest ticker data
      2. Compute portfolio value and check circuit breaker
      3. Compute momentum scores (per-horizon vol norm + volume surge)
      4. Apply rank buffer to determine long set
      5. Determine market regime
      6. Compute target weights (vol-parity + caps)
      7. Execute trades to move from current to target
      8. Persist state
    """
    global bot_state

    try:
        logger.info("=" * 60)
        logger.info("REBALANCE CYCLE START")
        logger.info("=" * 60)

        # 1. Fresh ticker data
        data_mgr.poll_ticker()

        # 2. Portfolio value + circuit breaker
        equity = get_portfolio_value(data_mgr.universe)
        if equity is None:
            logger.error("Cannot determine portfolio value -- skipping rebalance")
            return

        bot_state["equity"] = equity
        record_equity(equity)
        logger.info("Portfolio value: $%.2f", equity)

        breaker_active, peak, trough = check_circuit_breaker(
            equity,
            bot_state["peak_equity"],
            bot_state["trough_equity"],
            bot_state["breaker_active"],
        )
        bot_state["breaker_active"] = breaker_active
        bot_state["peak_equity"] = peak
        bot_state["trough_equity"] = trough

        # 3. Momentum scores
        scores = compute_momentum_scores(data_mgr)
        if not scores:
            logger.warning("No momentum scores computed -- need more price history")
            save_state(bot_state)
            return

        # 4. Apply rank buffer
        current_longs = set(bot_state.get("current_longs", []))
        new_longs, ranks = apply_rank_buffer(scores, current_longs)
        bot_state["current_longs"] = list(new_longs)

        # 5. Regime
        regime = compute_regime(data_mgr)

        # 6. Compute target weights (vol-parity + caps)
        volatilities = {}
        for pair in data_mgr.universe:
            vol = data_mgr.get_global_volatility(pair)
            if vol:
                volatilities[pair] = vol

        target_weights = compute_target_weights(
            long_pairs=new_longs,
            scores=scores,
            volatilities=volatilities,
            regime=regime,
            breaker_active=breaker_active,
        )

        # 7. Execute
        prices = data_mgr.get_latest_prices_dict()
        new_holdings, trade_count = execute_rebalance(
            target_weights=target_weights,
            current_holdings=bot_state["current_holdings"],
            equity=equity,
            prices=prices,
            universe=data_mgr.universe,
        )

        bot_state["current_holdings"] = new_holdings
        bot_state["total_trades"] += trade_count

        # Track active days
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if trade_count > 0:
            bot_state["active_days"].add(today)

        bot_state["last_rebalance_ts"] = datetime.now(timezone.utc).isoformat()

        # 8. Persist
        save_state(bot_state)

        logger.info("REBALANCE COMPLETE -- %d trades, %d total, %d active days",
                     trade_count, bot_state["total_trades"],
                     len(bot_state["active_days"]))
        logger.info("Longs: %s", list(new_longs))
        logger.info("Holdings: %s", new_holdings)
        logger.info("=" * 60)

    except Exception as e:
        logger.error("REBALANCE ERROR: %s", e, exc_info=True)
        save_state(bot_state)


# =====================================================================
#  Startup
# =====================================================================

def startup_checks():
    """Run read-only smoke tests at boot."""
    global bot_state

    logger.info("--- Startup smoke tests ---")

    # Server time
    st = __import__("roostoo_client").server_time()
    if st:
        logger.info("Server time: %s", st.get("ServerTime"))
    else:
        logger.error("FATAL: Cannot reach Roostoo API")
        sys.exit(1)

    # Exchange info -> build universe
    data_mgr.refresh_universe()
    if not data_mgr.universe:
        logger.error("FATAL: No tradable pairs found")
        sys.exit(1)

    # Balance
    bal = __import__("roostoo_client").balance()
    if bal and bal.get("Success"):
        logger.info("Balance: %s", bal.get("SpotWallet", {}))
    else:
        logger.error("FATAL: Cannot fetch balance -- check API keys")
        sys.exit(1)

    # Cancel any stale pending orders
    cancel_all_pending()

    # Initial portfolio value
    equity = get_portfolio_value(data_mgr.universe)
    if equity:
        bot_state["equity"] = equity
        if bot_state["peak_equity"] < equity:
            bot_state["peak_equity"] = equity
        if bot_state["trough_equity"] == 0:
            bot_state["trough_equity"] = equity
        record_equity(equity)
        logger.info("Starting equity: $%.2f (peak: $%.2f)", equity, bot_state["peak_equity"])

    save_state(bot_state)
    logger.info("--- Startup complete ---")


def main():
    global data_mgr, bot_state, logger

    logger = setup_logging()
    logger.info("Roostoo Cross-Sectional Factor Momentum Bot starting...")
    logger.info("MODE: Long-only, TOP_N=3, exit=6, exposure 90/50")

    # Load persisted state
    bot_state = load_state()
    bot_state.setdefault("current_longs", [])

    data_mgr = DataManager()

    # Startup checks
    startup_checks()

    # Backfill price history from Binance (100 hourly candles per pair)
    # so the bot can compute momentum scores on the very first rebalance.
    logger.info("Backfilling price history from Binance...")
    data_mgr.backfill_from_binance(limit=100)

    # One live ticker poll to get the current instant
    data_mgr.poll_ticker()

    # --- Schedule jobs ---
    scheduler = BlockingScheduler()

    # Ticker polling -- every 5 minutes
    scheduler.add_job(
        job_poll_ticker,
        "interval",
        seconds=TICKER_POLL_SECONDS,
        id="ticker_poll",
        max_instances=1,
    )

    # Universe refresh -- daily
    scheduler.add_job(
        job_refresh_universe,
        "interval",
        hours=UNIVERSE_REFRESH_HOURS,
        id="universe_refresh",
        max_instances=1,
    )

    # Rebalance -- every N hours
    scheduler.add_job(
        job_rebalance,
        "interval",
        hours=REBALANCE_INTERVAL_HOURS,
        id="rebalance",
        max_instances=1,
        next_run_time=datetime.now(),  # run immediately on first start
    )

    logger.info(
        "Scheduler started: ticker every %ds, rebalance every %dh, "
        "universe refresh every %dh",
        TICKER_POLL_SECONDS, REBALANCE_INTERVAL_HOURS, UNIVERSE_REFRESH_HOURS,
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot shutting down...")
        save_state(bot_state)


if __name__ == "__main__":
    main()
