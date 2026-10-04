"""
risk.py -- Position sizing, exposure management, and drawdown circuit breaker.

Pure functions -- no API calls.  Takes portfolio state + rank-buffer outputs
as input, returns target weights as output.  Long-only.
"""

import logging

import numpy as np

from config import (
    MAX_SINGLE_POSITION_PCT,
    RISK_ON_GROSS_EXPOSURE, RISK_OFF_GROSS_EXPOSURE,
    DRAWDOWN_THRESHOLD, DRAWDOWN_RECOVERY, CIRCUIT_BREAKER_EXPOSURE,
)

logger = logging.getLogger("risk")


# =====================================================================
#  Drawdown circuit breaker
# =====================================================================

def check_circuit_breaker(equity: float, peak_equity: float,
                          trough_equity: float, breaker_active: bool
                          ) -> tuple[bool, float, float]:
    """
    Check and manage the drawdown circuit breaker.

    Returns (breaker_active, updated_peak, updated_trough).
    """
    # Update peak
    if equity > peak_equity:
        peak_equity = equity

    drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

    if not breaker_active:
        if drawdown >= DRAWDOWN_THRESHOLD:
            logger.warning("!! CIRCUIT BREAKER TRIPPED -- drawdown %.2f%% "
                           "(equity=%.0f, peak=%.0f)",
                           drawdown * 100, equity, peak_equity)
            breaker_active = True
            trough_equity = equity
    else:
        # Update trough
        if equity < trough_equity:
            trough_equity = equity

        # Check for recovery
        if trough_equity > 0:
            recovery = (equity - trough_equity) / trough_equity
            if recovery >= DRAWDOWN_RECOVERY:
                logger.info("OK Circuit breaker RELEASED -- recovered %.2f%% from trough",
                            recovery * 100)
                breaker_active = False

    return breaker_active, peak_equity, trough_equity


# =====================================================================
#  Target weight computation (long-only, vol-parity)
# =====================================================================

def compute_target_weights(
    long_pairs: set[str],
    scores: dict[str, float],
    volatilities: dict[str, float],
    regime: str,
    breaker_active: bool,
) -> dict[str, float]:
    """
    Compute target portfolio weights (as fraction of equity, all positive).

    Uses inverse-volatility weighting to size each long position,
    capped at MAX_SINGLE_POSITION_PCT per position.

    Returns {pair: target_weight}.
    """
    # Determine gross exposure
    if breaker_active:
        gross = CIRCUIT_BREAKER_EXPOSURE
        logger.info("Breaker active -> gross exposure capped at %.0f%%", gross * 100)
    elif regime == "risk_on":
        gross = RISK_ON_GROSS_EXPOSURE
    else:
        gross = RISK_OFF_GROSS_EXPOSURE

    # Vol-parity weighting for longs
    long_weights = _vol_parity_weights(list(long_pairs), volatilities, gross)

    # Apply position caps
    target = {}
    for pair, w in long_weights.items():
        target[pair] = min(w, MAX_SINGLE_POSITION_PCT)

    # Logging
    total_long = sum(target.values())
    logger.info("Target weights: %s (regime=%s, gross=%.0f%%, total_long=%.2f, breaker=%s)",
                {k: f"{v:.3f}" for k, v in target.items()},
                regime, gross * 100, total_long, breaker_active)
    return target


def _vol_parity_weights(pairs: list[str], volatilities: dict[str, float],
                        budget: float) -> dict[str, float]:
    """
    Inverse-volatility weighting: weight each position inversely proportional
    to its recent volatility, normalized to the budget.
    """
    if not pairs or budget <= 0:
        return {}

    inv_vols = {}
    for pair in pairs:
        vol = volatilities.get(pair)
        if vol and vol > 0:
            inv_vols[pair] = 1.0 / vol
        else:
            inv_vols[pair] = 1.0  # fallback: equal weight

    total_inv_vol = sum(inv_vols.values())
    if total_inv_vol == 0:
        return {}

    weights = {}
    for pair in pairs:
        weights[pair] = (inv_vols[pair] / total_inv_vol) * budget

    return weights
