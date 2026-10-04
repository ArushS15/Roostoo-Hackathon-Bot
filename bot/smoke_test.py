"""Quick read-only smoke test — verifies API connectivity and keys."""
import json
from config import RST_API_KEY
import roostoo_client as api

print("=" * 50)
print("SMOKE TEST")
print("=" * 50)

# 1. Server time
print("\n1. Server Time")
st = api.server_time()
print(f"   -> {st}")

# 2. Exchange Info
print("\n2. Exchange Info")
info = api.exchange_info()
if info and "TradePairs" in info:
    pairs = list(info["TradePairs"].keys())
    print(f"   -> {len(pairs)} pairs: {pairs}")
    for p, meta in info["TradePairs"].items():
        print(f"     {p}: CanTrade={meta.get('CanTrade')}, "
              f"PricePrecision={meta.get('PricePrecision')}, "
              f"AmountPrecision={meta.get('AmountPrecision')}, "
              f"MiniOrder={meta.get('MiniOrder')}")
else:
    print(f"   -> FAILED: {info}")

# 3. Ticker (all pairs)
print("\n3. Ticker (all pairs)")
ticker = api.ticker()
if ticker and ticker.get("Success"):
    for pair, data in ticker.get("Data", {}).items():
        print(f"   {pair}: Last={data.get('LastPrice')}, "
              f"Change={data.get('Change')}, "
              f"UnitTradeValue={data.get('UnitTradeValue')}")
else:
    print(f"   -> FAILED: {ticker}")

# 4. Balance
print("\n4. Balance")
bal = api.balance()
if bal and bal.get("Success"):
    for coin, amounts in bal.get("Wallet", {}).items():
        free = amounts.get("Free", 0)
        lock = amounts.get("Lock", 0)
        if free > 0 or lock > 0:
            print(f"   {coin}: Free={free}, Lock={lock}")
else:
    print(f"   -> FAILED: {bal}")

# 5. Pending orders
print("\n5. Pending Orders")
pending = api.pending_count()
print(f"   -> {pending}")

# 6. Short positions
print("\n6. Short Positions")
shorts = api.short_positions()
print(f"   -> {shorts}")

print("\n" + "=" * 50)
print("SMOKE TEST COMPLETE")
print("=" * 50)
