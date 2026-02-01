import os
import time
import json
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

# -----------------------------
# Config (Environment Variables)
# -----------------------------
def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default)).strip()

TRACKED_WALLET_RAW = env_str("TRACKED_WALLET", "")
WEBHOOK_SECRET = env_str("WEBHOOK_SECRET", "")

# Paper trading parameters
SOL_PRICE_USD = env_float("SOL_PRICE_USD", 100.0)
START_CASH_USD = env_float("START_CASH_USD", 500.0)
MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 100.0)

# Reserve/tradable logic:
# PROFIT_TO_CASH_PCT = 0.40 means 40% of PROFITS go to reserve (cash), 60% stay tradable.
PROFIT_TO_CASH_PCT = env_float("PROFIT_TO_CASH_PCT", 0.40)

# Forced exit (dead coin time stop)
HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)  # 15 minutes default
FORCED_EXIT_FALLBACK_MULT = env_float("FORCED_EXIT_FALLBACK_MULT", 0.50)  # if no last price, assume 50% of entry

# Background loop cadence
FORCED_EXIT_CHECK_EVERY_SECONDS = env_int("FORCED_EXIT_CHECK_EVERY_SECONDS", 15)

SERVICE_NAME = env_str("SERVICE_NAME", "sol-paper-bot")


def parse_tracked_wallets(raw: str) -> Set[str]:
    """
    Accepts:
      - single address
      - comma-separated addresses
      - newline-separated addresses
    """
    if not raw:
        return set()
    parts = []
    for chunk in raw.replace("\n", ",").split(","):
        w = chunk.strip()
        if w:
            parts.append(w)
    return set(parts)

TRACKED_WALLETS: Set[str] = parse_tracked_wallets(TRACKED_WALLET_RAW)

# -----------------------------
# In-memory state (paper trading)
# -----------------------------
@dataclass
class Position:
    mint: str
    qty: float
    avg_entry_usd: float
    opened_at: int
    last_update_at: int = field(default_factory=lambda: int(time.time()))

@dataclass
class Trade:
    ts: int
    side: str  # "BUY" or "SELL"
    mint: str
    qty: float
    price_usd: float
    notional_usd: float
    reason: str

state = {
    "started_at": int(time.time()),
    "cash_usd": START_CASH_USD,          # tradable cash
    "reserve_cash_usd": 0.0,             # reserve pool (from profits)
    "positions": {},                     # mint -> Position
    "trades": [],                        # List[Trade]
    "events": [],                        # raw events (trimmed)
    "counters": {
        "webhooks_received": 0,
        "webhooks_unauthorized": 0,
        "buys": 0,
        "sells": 0,
        "forced_exits": 0,
        "skipped_low_cash": 0,
        "skipped_no_price": 0,
    },
    "last_price_usd": {},                # mint -> float
}

EVENTS_KEEP = 200  # keep last N events


# -----------------------------
# Helpers
# -----------------------------
LAMPORTS_PER_SOL = 1_000_000_000

def now() -> int:
    return int(time.time())

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def add_event(source: str, payload: Any):
    state["events"].append({
        "received_at": now(),
        "source": source,
        "payload": payload
    })
    if len(state["events"]) > EVENTS_KEEP:
        state["events"] = state["events"][-EVENTS_KEEP:]

def record_trade(tr: Trade):
    state["trades"].append(tr.__dict__)
    # keep trades manageable
    if len(state["trades"]) > 500:
        state["trades"] = state["trades"][-500:]


def estimate_price_from_swap(tx: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Attempts to compute mint price USD from a Helius enhanced SWAP event:
      - Detects tracked wallet's nativeBalanceChange in accountData
      - Detects tokenTransfers involving tracked wallet
      - price_usd = abs(native_change_sol * SOL_PRICE_USD) / token_amount
    Returns dict: {mint: price_usd} possibly for one mint.
    """
    try:
        account_data = tx.get("accountData", [])
        token_transfers = tx.get("tokenTransfers", [])
        tx_type = tx.get("type", "")

        if tx_type != "SWAP":
            return None

        # Find native balance change for any tracked wallet
        native_change_lamports = None
        native_wallet = None
        for row in account_data:
            acct = row.get("account")
            if acct in TRACKED_WALLETS:
                native_change_lamports = row.get("nativeBalanceChange")
                native_wallet = acct
                break

        if native_change_lamports is None:
            return None

        native_change_sol = native_change_lamports / LAMPORTS_PER_SOL
        native_usd = abs(native_change_sol) * SOL_PRICE_USD
        if native_usd <= 0:
            return None

        # Find one token transfer that involves tracked wallet
        for tt in token_transfers:
            mint = tt.get("mint")
            amt = tt.get("tokenAmount")
            from_user = tt.get("fromUserAccount")
            to_user = tt.get("toUserAccount")

            if not mint or amt is None:
                continue

            # Only consider if tracked wallet is sender or receiver
            if (from_user in TRACKED_WALLETS) or (to_user in TRACKED_WALLETS):
                token_amt = abs(float(amt))
                if token_amt <= 0:
                    continue
                price_usd = native_usd / token_amt
                if price_usd > 0:
                    return {mint: price_usd}

        return None
    except Exception:
        return None


def detect_signal_from_swap(tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Detects BUY/SELL for tracked wallets from tokenTransfers direction.
    BUY  = tracked wallet receives token mint
    SELL = tracked wallet sends token mint
    Returns: {"side": "BUY"/"SELL", "mint": str, "price_usd": float}
    """
    if tx.get("type") != "SWAP":
        return None

    # Update last price if possible
    price_map = estimate_price_from_swap(tx)
    if price_map:
        for m, p in price_map.items():
            state["last_price_usd"][m] = p

    token_transfers = tx.get("tokenTransfers", [])
    if not token_transfers:
        return None

    # Prefer transfers that involve tracked wallet
    for tt in token_transfers:
        mint = tt.get("mint")
        amt = tt.get("tokenAmount")
        from_user = tt.get("fromUserAccount")
        to_user = tt.get("toUserAccount")

        if not mint or amt is None:
            continue

        if to_user in TRACKED_WALLETS:
            # tracked wallet received token => BUY signal
            price = state["last_price_usd"].get(mint)
            return {"side": "BUY", "mint": mint, "price_usd": price}

        if from_user in TRACKED_WALLETS:
            # tracked wallet sent token => SELL signal
            price = state["last_price_usd"].get(mint)
            return {"side": "SELL", "mint": mint, "price_usd": price}

    return None


def paper_buy(mint: str, price_usd: Optional[float], reason: str):
    if price_usd is None or price_usd <= 0:
        state["counters"]["skipped_no_price"] += 1
        return

    cash = float(state["cash_usd"])

    # Always leave MIN_CASH_LEFT_USD untouched
    max_spend = cash - MIN_CASH_LEFT_USD
    if max_spend <= 0:
        state["counters"]["skipped_low_cash"] += 1
        return

    spend = min(MAX_BUY_USD, max_spend)
    spend = clamp(spend, 0.0, max_spend)
    if spend <= 0:
        state["counters"]["skipped_low_cash"] += 1
        return

    qty = spend / price_usd
    ts = now()

    pos: Optional[Position] = state["positions"].get(mint)
    if pos is None:
        state["positions"][mint] = Position(
            mint=mint,
            qty=qty,
            avg_entry_usd=price_usd,
            opened_at=ts,
            last_update_at=ts
        )
    else:
        # weighted average entry
        new_qty = pos.qty + qty
        new_avg = (pos.qty * pos.avg_entry_usd + qty * price_usd) / new_qty
        pos.qty = new_qty
        pos.avg_entry_usd = new_avg
        pos.last_update_at = ts

    state["cash_usd"] = cash - spend
    state["counters"]["buys"] += 1

    record_trade(Trade(
        ts=ts,
        side="BUY",
        mint=mint,
        qty=qty,
        price_usd=price_usd,
        notional_usd=spend,
        reason=reason
    ))


def paper_sell(mint: str, price_usd: Optional[float], reason: str, sell_all: bool = True):
    pos: Optional[Position] = state["positions"].get(mint)
    if pos is None:
        return

    # If we don't have a price, use fallback based on entry (conservative)
    if price_usd is None or price_usd <= 0:
        price_usd = max(0.0000000001, pos.avg_entry_usd * FORCED_EXIT_FALLBACK_MULT)

    ts = now()

    qty = pos.qty if sell_all else pos.qty
    proceeds = qty * price_usd
    cost = qty * pos.avg_entry_usd
    pnl = proceeds - cost

    # Return cost basis to tradable cash
    cash = float(state["cash_usd"])
    cash += cost

    # Split profits into reserve/tradable (losses stay in tradable)
    if pnl > 0:
        to_reserve = pnl * PROFIT_TO_CASH_PCT
        to_tradable = pnl - to_reserve
        state["reserve_cash_usd"] = float(state["reserve_cash_usd"]) + to_reserve
        cash += to_tradable
    else:
        cash += pnl  # pnl is negative

    state["cash_usd"] = cash

    # Remove position
    del state["positions"][mint]

    state["counters"]["sells"] += 1

    record_trade(Trade(
        ts=ts,
        side="SELL",
        mint=mint,
        qty=qty,
        price_usd=price_usd,
        notional_usd=proceeds,
        reason=reason
    ))


# -----------------------------
# Forced exit background loop
# -----------------------------
async def forced_exit_loop():
    while True:
        try:
            ts = now()
            mints = list(state["positions"].keys())
            for mint in mints:
                pos: Position = state["positions"].get(mint)
                if not pos:
                    continue
                age = ts - int(pos.opened_at)
                if age >= HOLD_MAX_SECONDS:
                    # time-based forced exit
                    last_price = state["last_price_usd"].get(mint)
                    paper_sell(mint, last_price, reason=f"FORCED_EXIT_TIME>{HOLD_MAX_SECONDS}s")
                    state["counters"]["forced_exits"] += 1
        except Exception:
            # swallow to keep loop alive
            pass

        await asyncio.sleep(FORCED_EXIT_CHECK_EVERY_SECONDS)


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title=SERVICE_NAME)

@app.on_event("startup")
async def on_startup():
    # Start forced exit loop
    asyncio.create_task(forced_exit_loop())


@app.get("/health")
def health():
    return {"ok": True, "service": SERVICE_NAME}


@app.get("/paper/state")
def paper_state():
    # Convert positions to JSON-safe dict
    pos_out = {}
    for mint, p in state["positions"].items():
        pos_out[mint] = {
            "mint": p.mint,
            "qty": p.qty,
            "avg_entry_usd": p.avg_entry_usd,
            "opened_at": p.opened_at,
            "age_seconds": now() - p.opened_at,
        }

    return {
        "cash_usd": round(float(state["cash_usd"]), 6),
        "reserve_cash_usd": round(float(state["reserve_cash_usd"]), 6),
        "positions": pos_out,
        "trades_count": len(state["trades"]),
        "counters": state["counters"],
        "started_at": state["started_at"],
        "config": {
            "SOL_PRICE_USD": SOL_PRICE_USD,
            "START_CASH_USD": START_CASH_USD,
            "MAX_BUY_USD": MAX_BUY_USD,
            "MIN_CASH_LEFT_USD": MIN_CASH_LEFT_USD,
            "PROFIT_TO_CASH_PCT": PROFIT_TO_CASH_PCT,
            "HOLD_MAX_SECONDS": HOLD_MAX_SECONDS,
            "FORCED_EXIT_FALLBACK_MULT": FORCED_EXIT_FALLBACK_MULT,
            "TRACKED_WALLETS_COUNT": len(TRACKED_WALLETS),
        }
    }


@app.get("/events")
def get_events():
    return {"count": len(state["events"]), "events": state["events"]}


@app.get("/paper/trades")
def paper_trades():
    return {"count": len(state["trades"]), "trades": state["trades"]}


@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(default=None),
):
    state["counters"]["webhooks_received"] += 1

    # Auth check (Helius "Authentication Header")
    if WEBHOOK_SECRET:
        if not x_webhook_secret or x_webhook_secret != WEBHOOK_SECRET:
            state["counters"]["webhooks_unauthorized"] += 1
            raise HTTPException(status_code=401, detail="Unauthorized")

    body = await request.json()
    # Helius can send list of transactions or dict; normalize to list
    txs: List[Dict[str, Any]]
    if isinstance(body, list):
        txs = body
    elif isinstance(body, dict) and "data" in body and isinstance(body["data"], list):
        txs = body["data"]
    elif isinstance(body, dict):
        txs = [body]
    else:
        txs = []

    add_event("helius", txs)

    # Process signals
    for tx in txs:
        sig = detect_signal_from_swap(tx)
        if not sig:
            continue

        side = sig["side"]
        mint = sig["mint"]
        price = sig.get("price_usd")

        if side == "BUY":
            paper_buy(mint, price, reason="WHALE_BUY_SIGNAL")
        elif side == "SELL":
            # If whale sells, we exit our position (if we have it)
            paper_sell(mint, price, reason="WHALE_SELL_SIGNAL")

    return JSONResponse({"ok": True})
