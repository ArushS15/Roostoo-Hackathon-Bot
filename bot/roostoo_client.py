"""
roostoo_client.py — Thin, signed HTTP wrapper for the Roostoo mock exchange API.

Every request is logged. Every response checks the `Success` flag.
Retries transient failures. Never throws on API errors — returns None or
a dict with Success=False so callers can branch gracefully.
"""

import time
import hmac
import hashlib
import logging
import requests
from typing import Optional

from config import BASE_URL, RST_API_KEY, RST_SECRET_KEY

logger = logging.getLogger("roostoo_client")

MAX_RETRIES = 3
RETRY_BACKOFF_S = 2


# ─── Signing helpers ────────────────────────────────────────────────────────

def _timestamp_ms() -> str:
    """13-digit millisecond timestamp as a string."""
    return str(int(time.time() * 1000))


def _sign(params: dict) -> tuple[dict, str]:
    """
    Build sorted param string, compute HMAC-SHA256, return (headers, body_str).
    """
    params["timestamp"] = _timestamp_ms()
    sorted_keys = sorted(params.keys())
    total_params = "&".join(f"{k}={params[k]}" for k in sorted_keys)

    signature = hmac.new(
        RST_SECRET_KEY.encode("utf-8"),
        total_params.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "RST-API-KEY": RST_API_KEY,
        "MSG-SIGNATURE": signature,
    }
    return headers, total_params


# ─── Generic request layer ──────────────────────────────────────────────────

def _request(method: str, path: str, params: Optional[dict] = None,
             signed: bool = False) -> Optional[dict]:
    """
    Fire an HTTP request with optional signing.  Retries on network errors.
    Returns the parsed JSON dict, or None on total failure.
    """
    url = f"{BASE_URL}{path}"
    params = dict(params or {})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if signed:
                headers, body_str = _sign(params)
                if method == "GET":
                    # For signed GETs, params go as query string
                    resp = requests.get(url, headers=headers, params=params, timeout=15)
                else:
                    headers["Content-Type"] = "application/x-www-form-urlencoded"
                    resp = requests.post(url, headers=headers, data=body_str, timeout=15)
            else:
                if method == "GET":
                    resp = requests.get(url, params=params, timeout=15)
                else:
                    resp = requests.post(url, data=params, timeout=15)

            resp.raise_for_status()
            data = resp.json()
            logger.debug("API %s %s → %s", method, path, data)

            # Check the Success flag (most endpoints have it)
            if "Success" in data and not data["Success"]:
                logger.warning("API %s %s returned Success=false: %s",
                               method, path, data.get("ErrMsg", ""))
            return data

        except requests.exceptions.RequestException as exc:
            logger.error("API %s %s attempt %d/%d failed: %s",
                         method, path, attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_S * attempt)

    logger.error("API %s %s — all %d retries exhausted", method, path, MAX_RETRIES)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC (unsigned) endpoints
# ═══════════════════════════════════════════════════════════════════════════

def server_time() -> Optional[dict]:
    """GET /v3/serverTime"""
    return _request("GET", "/v3/serverTime")


def exchange_info() -> Optional[dict]:
    """GET /v3/exchangeInfo — returns tradable pairs, precision rules, etc."""
    return _request("GET", "/v3/exchangeInfo")


def ticker(pair: Optional[str] = None) -> Optional[dict]:
    """GET /v3/ticker — needs timestamp (RCL_TSCheck)."""
    params = {"timestamp": _timestamp_ms()}
    if pair:
        params["pair"] = pair
    return _request("GET", "/v3/ticker", params=params)


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNED endpoints — balance / orders
# ═══════════════════════════════════════════════════════════════════════════

def balance() -> Optional[dict]:
    """GET /v3/balance"""
    return _request("GET", "/v3/balance", signed=True)


def pending_count() -> Optional[dict]:
    """GET /v3/pending_count"""
    return _request("GET", "/v3/pending_count", signed=True)


def place_order(pair: str, side: str, quantity: float,
                order_type: str = "MARKET", price: Optional[float] = None) -> Optional[dict]:
    """
    POST /v3/place_order
    side: "BUY" | "SELL"
    order_type: "MARKET" | "LIMIT"
    """
    params = {
        "pair": pair,
        "side": side.upper(),
        "type": order_type.upper(),
        "quantity": str(quantity),
    }
    if order_type.upper() == "LIMIT" and price is not None:
        params["price"] = str(price)
    return _request("POST", "/v3/place_order", params=params, signed=True)


def query_order(order_id: Optional[int] = None, pair: Optional[str] = None,
                pending_only: Optional[bool] = None) -> Optional[dict]:
    """POST /v3/query_order"""
    params = {}
    if order_id is not None:
        params["order_id"] = str(order_id)
    else:
        if pair:
            params["pair"] = pair
        if pending_only is not None:
            params["pending_only"] = "TRUE" if pending_only else "FALSE"
    return _request("POST", "/v3/query_order", params=params, signed=True)


def cancel_order(order_id: Optional[int] = None, pair: Optional[str] = None) -> Optional[dict]:
    """POST /v3/cancel_order"""
    params = {}
    if order_id is not None:
        params["order_id"] = str(order_id)
    elif pair:
        params["pair"] = pair
    return _request("POST", "/v3/cancel_order", params=params, signed=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SHORT endpoints (v6)
# ═══════════════════════════════════════════════════════════════════════════

def short_open(pair: str, collateral: float,
               order_type: str = "MARKET", price: Optional[float] = None) -> Optional[dict]:
    """POST /v6/short_open — sized by collateral, not quantity."""
    params = {
        "pair": pair,
        "collateral": str(collateral),
    }
    if order_type.upper() == "LIMIT" and price is not None:
        params["order_type"] = "LIMIT"
        params["price"] = str(price)
    return _request("POST", "/v6/short_open", params=params, signed=True)


def short_close(pair: str, close_qty: Optional[float] = None,
                close_pct: Optional[float] = None) -> Optional[dict]:
    """POST /v6/short_close"""
    params = {"pair": pair}
    if close_qty is not None:
        params["close_qty"] = str(close_qty)
    elif close_pct is not None:
        params["close_pct"] = str(close_pct)
    return _request("POST", "/v6/short_close", params=params, signed=True)


def short_positions() -> Optional[dict]:
    """GET /v6/short_positions"""
    return _request("GET", "/v6/short_positions", signed=True)
