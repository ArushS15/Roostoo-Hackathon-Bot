"""
execution.py — Translate target weights into concrete Roostoo orders.

Handles:
  - Weight deltas → buy/sell decisions (long-only)
  - Precision and MiniOrder enforcement from exchangeInfo
  - Order placement and basic reconciliation
  - Logging of every order attempt and response
"""

import math
import logging
from typing import Optional

import roostoo_client as api
from config import TAKER_FEE

logger = logging.getLogger("execution")


# ═══════════════════════════════════════════════════════════════════════════
#  Precision helpers
# ═══════════════════════════════════════════════════════════════════════════

def _floor_to_precision(value: float, precision: int) -> float:
    """Floor a value to the given decimal precision (never round up)."""
    factor = 10 ** precision
    return math.floor(value * factor) / factor


def _meets_min_order(pair: str, quantity: float, price: float,
                     universe: dict) -> bool:
    """Check if price * quantity >= MiniOrder for this pair."""
    meta = universe.get(pair, {})
    mini = meta.get("MiniOrder", 1.0)
    return (quantity * price) >= mini


# ═══════════════════════════════════════════════════════════════════════════
#  Core execution (long-only)
# ═══════════════════════════════════════════════════════════════════════════

def execute_rebalance(
    target_weights: dict[str, float],
    current_holdings: dict[str, float],
    equity: float,
    prices: dict[str, float],
    universe: dict,
) -> tuple[dict[str, float], int]:
    """
    Compare target weights vs current holdings, compute deltas, and
    execute the necessary buy/sell orders.

    Returns (new_holdings, trade_count).
    """
    trade_count = 0
    new_holdings = dict(current_holdings)

    # --- Close positions we no longer want ---
    for pair in list(current_holdings.keys()):
        if pair not in target_weights:
            if current_holdings[pair] > 0:
                # Sell long position
                ok = _close_long(pair, equity, current_holdings[pair], prices, universe)
                if ok:
                    trade_count += 1
                    del new_holdings[pair]

    # --- Adjust or open positions ---
    for pair, target_w in target_weights.items():
        current_w = current_holdings.get(pair, 0)
        delta_w = target_w - current_w

        if abs(delta_w) < 0.02:
            # Skip small adjustments (less than 2% of equity) to reduce fee drag
            continue

        price = prices.get(pair)
        if not price or price <= 0:
            logger.warning("No price for %s — skipping", pair)
            continue

        if delta_w > 0:
            # Need to buy more
            ok = _open_or_add_long(pair, delta_w, equity, price, universe)
        else:
            # Need to reduce long
            ok = _reduce_long(pair, abs(delta_w), equity, price, universe)

        if ok:
            trade_count += 1
            new_holdings[pair] = target_w

    return new_holdings, trade_count


# ═══════════════════════════════════════════════════════════════════════════
#  Long order helpers
# ═══════════════════════════════════════════════════════════════════════════

def _open_or_add_long(pair: str, weight_delta: float, equity: float,
                      price: float, universe: dict) -> bool:
    """Buy to open or add to a long position."""
    dollar_amount = weight_delta * equity
    meta = universe.get(pair, {})
    amt_precision = meta.get("AmountPrecision", 2)
    quantity = _floor_to_precision(dollar_amount / price, amt_precision)

    if quantity <= 0 or not _meets_min_order(pair, quantity, price, universe):
        logger.info("Skip BUY %s — qty=%.6f below minimum", pair, quantity)
        return False

    logger.info("→ BUY %s qty=%.6f (~$%.0f)", pair, quantity, dollar_amount)
    result = api.place_order(pair, "BUY", quantity, order_type="MARKET")
    if result and result.get("Success"):
        logger.info("✓ BUY %s filled: %s", pair, result.get("OrderDetail", {}))
        return True
    else:
        logger.error("✗ BUY %s failed: %s", pair, result)
        return False


def _reduce_long(pair: str, weight_delta: float, equity: float,
                 price: float, universe: dict) -> bool:
    """Sell to reduce a long position."""
    dollar_amount = weight_delta * equity
    meta = universe.get(pair, {})
    amt_precision = meta.get("AmountPrecision", 2)
    quantity = _floor_to_precision(dollar_amount / price, amt_precision)

    if quantity <= 0 or not _meets_min_order(pair, quantity, price, universe):
        logger.info("Skip SELL %s — qty=%.6f below minimum", pair, quantity)
        return False

    logger.info("→ SELL %s qty=%.6f (~$%.0f)", pair, quantity, dollar_amount)
    result = api.place_order(pair, "SELL", quantity, order_type="MARKET")
    if result and result.get("Success"):
        logger.info("✓ SELL %s filled: %s", pair, result.get("OrderDetail", {}))
        return True
    else:
        logger.error("✗ SELL %s failed: %s", pair, result)
        return False


def _close_long(pair: str, equity: float, weight: float,
                prices: dict, universe: dict) -> bool:
    """Fully sell out of a long position."""
    price = prices.get(pair, 0)
    if price <= 0:
        return False
    return _reduce_long(pair, weight, equity, price, universe)


# ═══════════════════════════════════════════════════════════════════════════
#  Utility
# ═══════════════════════════════════════════════════════════════════════════

def get_portfolio_value(universe: dict) -> Optional[float]:
    """
    Compute total portfolio value: USD balance + sum(coin_balance * price).
    """
    bal_data = api.balance()
    if not bal_data or not bal_data.get("Success"):
        return None

    wallet = bal_data.get("SpotWallet", {})
    usd = wallet.get("USD", {}).get("Free", 0) + wallet.get("USD", {}).get("Lock", 0)

    # Get ticker for all coin valuations
    ticker_data = api.ticker()
    if not ticker_data or not ticker_data.get("Success"):
        return usd  # fallback to just USD

    total = usd
    for pair, meta in universe.items():
        coin = meta.get("Coin", pair.split("/")[0])
        coin_bal = wallet.get(coin, {})
        coin_total = coin_bal.get("Free", 0) + coin_bal.get("Lock", 0)
        if coin_total > 0:
            price = ticker_data.get("Data", {}).get(pair, {}).get("LastPrice", 0)
            total += coin_total * price

    return total


def cancel_all_pending():
    """Cancel all pending orders to start clean."""
    result = api.cancel_order()
    if result and result.get("Success"):
        canceled = result.get("CanceledList", [])
        if canceled:
            logger.info("Canceled %d pending orders: %s", len(canceled), canceled)
    # "no pending order" returns Success=false — that's fine
