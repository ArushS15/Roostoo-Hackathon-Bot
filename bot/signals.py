"""
signals.py — Pure signal computation: cross-sectional factor momentum with
              per-horizon vol normalization, volume surge weighting,
              and rank-buffer turnover control.

NO API calls in this module -- it takes data as input and returns scores.
This makes it reusable by both the live bot and the backtester.

Signal formula:
  Score_i = [w1 * R_{i,6h}/sigma_{i,6h} + w2 * R_{i,24h}/sigma_{i,24h}
             + w3 * R_{i,72h}/sigma_{i,72h}] * VolumeSurge_i

Rank buffer:
  Enter long when rank <= TOP_N_ENTER (3), hold until rank > TOP_N_EXIT (6).
"""

import logging

import numpy as np
import pandas as pd

from config import (
    LOOKBACK_SHORT, LOOKBACK_MED, LOOKBACK_LONG, LOOKBACK_WEIGHTS,
    VOL_EPSILON, VOLUME_SURGE_CAP,
    REGIME_PAIR, REGIME_MA_PERIOD,
    TOP_N_ENTER, TOP_N_EXIT,
)

logger = logging.getLogger("signals")


# =====================================================================
#  Momentum score -- per-horizon vol normalization + volume surge
# =====================================================================

def compute_momentum_scores(data_manager) -> dict[str, float]:
    """
    For each pair in the universe, compute:
      Score_i = [w1 * R_{6h}/sigma_{6h} + w2 * R_{24h}/sigma_{24h}
                 + w3 * R_{72h}/sigma_{72h}] * VolumeSurge_i

    Returns {pair: score}.  Pairs with insufficient data are omitted.
    """
    scores = {}
    lookbacks = [LOOKBACK_SHORT, LOOKBACK_MED, LOOKBACK_LONG]

    for pair in data_manager.universe:
        # Compute per-horizon return and sigma
        terms = []
        weights_used = []

        for lb, w in zip(lookbacks, LOOKBACK_WEIGHTS):
            ret = data_manager.get_returns(pair, lb)
            sigma = data_manager.get_horizon_volatility(pair, lb)

            if ret is None or sigma is None:
                continue

            # Floor sigma to prevent blowup on stagnant pairs
            sigma = max(sigma, VOL_EPSILON)
            terms.append(ret / sigma)
            weights_used.append(w)

        # Need at least 2 horizons to produce a meaningful score
        if len(terms) < 2:
            continue

        # Normalize weights to sum to 1
        w_sum = sum(weights_used)
        normalized_weights = [w / w_sum for w in weights_used]
        raw_score = sum(t * w for t, w in zip(terms, normalized_weights))

        # Volume surge multiplier
        vol_surge = data_manager.get_volume_surge(pair)
        raw_score *= vol_surge

        scores[pair] = raw_score

        logger.debug("Score %s: terms=%s surge=%.2f -> score=%.4f",
                      pair, [f"{t:.3f}" for t in terms], vol_surge, raw_score)

    logger.info("Momentum scores computed for %d/%d pairs",
                len(scores), len(data_manager.universe))
    return scores


# =====================================================================
#  Regime filter
# =====================================================================

def compute_regime(data_manager) -> str:
    """
    Determine market regime using the anchor pair's price vs. its
    N-period moving average on hourly bars.

    Returns "risk_on" or "risk_off".
    """
    pair = REGIME_PAIR
    if pair not in data_manager.price_history.columns:
        logger.warning("Regime pair %s not in history -- defaulting to risk_off", pair)
        return "risk_off"

    series = data_manager.price_history[pair].dropna()
    if len(series) < 5:
        logger.warning("Not enough data for regime filter -- defaulting to risk_off")
        return "risk_off"

    # Resample to hourly
    hourly = series.resample("1h").last().dropna()
    if len(hourly) < REGIME_MA_PERIOD:
        logger.info("Only %d hourly bars (need %d for MA) -- defaulting to risk_off",
                     len(hourly), REGIME_MA_PERIOD)
        return "risk_off"

    ma = hourly.rolling(REGIME_MA_PERIOD).mean()
    current_price = hourly.iloc[-1]
    current_ma = ma.iloc[-1]

    if np.isnan(current_ma):
        return "risk_off"

    regime = "risk_on" if current_price > current_ma else "risk_off"
    logger.info("Regime: %s (price=%.2f, MA%d=%.2f)",
                regime, current_price, REGIME_MA_PERIOD, current_ma)
    return regime


# =====================================================================
#  Rank buffer -- the core turnover control mechanism
# =====================================================================

def apply_rank_buffer(
    scores: dict[str, float],
    current_longs: set[str],
) -> tuple[set[str], dict[str, int]]:
    """
    Apply rank-based hysteresis buffer to control turnover.

    Rule:
      - Enter long when rank <= TOP_N_ENTER, hold until rank > TOP_N_EXIT.

    Args:
        scores: {pair: score} from compute_momentum_scores
        current_longs: set of currently held long pairs

    Returns:
        (new_longs, ranks) where ranks is {pair: rank} for logging
    """
    if not scores:
        return set(), {}

    # Sort descending by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranks = {pair: i + 1 for i, (pair, _) in enumerate(ranked)}

    # ── Long book ──
    new_longs = set()
    for pair, rank in ranks.items():
        if rank <= TOP_N_ENTER:
            # Qualifies for entry
            new_longs.add(pair)
        elif pair in current_longs and rank <= TOP_N_EXIT:
            # Currently held, hasn't dropped out of the buffer zone
            new_longs.add(pair)
        # else: either not held, or held but rank > TOP_N_EXIT -> drop

    # Log buffer actions
    entered_longs = new_longs - current_longs
    exited_longs = current_longs - new_longs

    if entered_longs:
        logger.info("LONG ENTER: %s", {p: f"rank={ranks.get(p)}" for p in entered_longs})
    if exited_longs:
        logger.info("LONG EXIT:  %s", {p: f"rank={ranks.get(p)}" for p in exited_longs})

    # Log rank of every held position
    if new_longs:
        held_ranks = {p: ranks.get(p, "?") for p in sorted(new_longs)}
        logger.info("Held ranks: %s", held_ranks)

    return new_longs, ranks
