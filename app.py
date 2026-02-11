import os
import time
import hmac
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# -----------------------------
# Helpers
# -----------------------------
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v)

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else float(default)
    except Exception:
        return float(default)

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(float(v)) if v is not None else int(default)
    except Exception:
        return int(default)

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def mask_secret(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    if len(s) <= 6:
        return "*" * len(s)
    return s[:2] + "*" * (len(s) - 4) + s[-2:]

def constant_time_equals(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None:
        return False
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))

def extract_secret_from_headers(headers) -> Optional[str]:
    """
    Accept:
      - x-webhook-secret: <secret>
      - authorization: Bearer <secret>
      - authorization: <secret>
    """
    x = headers.get("x-webhook-secret")
    if x:
        return x.strip()

    auth = headers.get("authorization")
    if not auth:
        return None

    auth = auth.strip()
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return auth

def now_ts() -> int:
    return int(time.time())

# -----------------------------
# Config / Env
# -----------------------------
DEBUG_WEBHOOK = env_bool("DEBUG_WEBHOOK", True)

WEBHOOK_SECRET = env_str("WEBHOOK_SECRET", "")
WEBHOOK_PATH_TOKEN = env_str("WEBHOOK_PATH_TOKEN", "")

SOL_PRICE_USD = env_float("SOL_PRICE_USD", 115.0)
START_CASH_USD = env_float("START_CASH_USD", 500.0)
MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 100.0)

RESERVE_PCT = env_float("RESERVE_PCT", 0.6)
TRADABLE_PCT = env_float("TRADABLE_PCT", 0.4)

HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.5)

tracked_raw = env_str("TRACKED_WALLETS", "").strip()
if not tracked_raw:
    tracked_raw = env_str("TRACKED_WALLET", "").strip()

TRACKED_WALLETS: List[str] = [
    w.strip()
    for w in tracked_raw.replace("\n", ",").replace(" ", ",").split(",")
    if w.strip()
]

# -----------------------------
# State
# -----------------------------
def initial_split(start_cash: float):
    reserve = round(start_cash * float(RESERVE_PCT), 2)
    cash = round(start_cash - reserve, 2)
    return cash, reserve

cash0, reserve0 = initial_split(START_CASH_USD)

state: Dict[str, Any] = {
    "cash_usd": cash0,
    "reserve_cash_usd": reserve0,
    "positions": {},
    "trades_count": 0,
    "counters": {
        "webhooks_received": 0,
        "webhooks_unauthorized": 0,
        "skipped_no_secret": 0,
        "skipped_bad_payload": 0,
        "skipped_bad_path": 0,
        "skipped_low_cash": 0,
        "buys": 0,
        "sells": 0,
        "forced_exits": 0,
    },
    "started_at": now_ts(),
    "recent_trades": [],
}

events_log: List[Dict[str, Any]] = []

def push_event(e: Dict[str, Any]) -> None:
    events_log.append(e)
    if len(events_log) > 500:
        del events_log[:200]

def push_trade(t: Dict[str, Any]) -> None:
    state["recent_trades"].append(t)
    if len(state["recent_trades"]) > 50:
        state["recent_trades"] = state["recent_trades"][-50:]
    state["trades_count"] += 1

def config_snapshot():
    return {
        "SOL_PRICE_USD": SOL_PRICE_USD,
        "START_CASH_USD": START_CASH_USD,
        "MAX_BUY_USD": MAX_BUY_USD,
        "MIN_CASH_LEFT_USD": MIN_CASH_LEFT_USD,
        "RESERVE_PCT": RESERVE_PCT,
        "TRADABLE_PCT": TRADABLE_PCT,
        "HOLD_MAX_SECONDS": HOLD_MAX_SECONDS,
        "FORCED_EXIT_FALLBACK_MULTI": FORCED_EXIT_FALLBACK_MULTI,
        "TRACKED_WALLETS_COUNT": len(TRACKED_WALLETS),
        "DEBUG_WEBHOOK": DEBUG_WEBHOOK,
        "WEBHOOK_PATH_TOKEN_SET": bool(WEBHOOK_PATH_TOKEN),
        "WEBHOOK_SECRET_SET": bool(WEBHOOK_SECRET),
    }

# -----------------------------
# Paper Trading Logic
# -----------------------------
def paper_buy(symbol: str, usd: float, reason: str, meta: Dict[str, Any]):
    usd = round(float(usd), 2)
    if usd <= 0:
        return False

    if state["cash_usd"] - usd < MIN_CASH_LEFT_USD:
        state["counters"]["skipped_low_cash"] += 1
        push_event({"ts": now_ts(), "kind": "skipped_low_cash"})
        return False

    qty = usd / SOL_PRICE_USD

    pos = state["positions"].get(symbol)
    if not pos:
        pos = {
            "symbol": symbol,
            "qty": 0.0,
            "cost_usd": 0.0,
            "entry_ts": now_ts(),
            "avg_price": SOL_PRICE_USD,
        }

    pos["qty"] += qty
    pos["cost_usd"] += usd
    pos["avg_price"] = pos["cost_usd"] / pos["qty"]
    state["positions"][symbol] = pos

    state["cash_usd"] = round(state["cash_usd"] - usd, 2)
    state["counters"]["buys"] += 1

    push_trade({
        "ts": now_ts(),
        "side": "BUY",
        "symbol": symbol,
        "usd": usd,
        "qty": qty,
        "price": SOL_PRICE_USD,
        "reason": reason,
        "meta": meta,
    })

    return True

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/events")
def events():
    return {"count": len(events_log), "events": events_log}

@app.get("/paper/state")
def paper_state():
    return {
        "cash_usd": state["cash_usd"],
        "reserve_cash_usd": state["reserve_cash_usd"],
        "positions": state["positions"],
        "trades_count": state["trades_count"],
        "counters": state["counters"],
        "config": config_snapshot(),
        "recent_trades": state["recent_trades"],
    }

@app.post("/webhook/{token}")
async def webhook(token: str, req: Request):

    if token != WEBHOOK_PATH_TOKEN:
        state["counters"]["skipped_bad_path"] += 1
        return JSONResponse({"error": "not found"}, status_code=404)

    got = extract_secret_from_headers(req.headers)

    if not constant_time_equals(got, WEBHOOK_SECRET):
        state["counters"]["webhooks_unauthorized"] += 1
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    try:
        payload = await req.json()
    except Exception:
        state["counters"]["skipped_bad_payload"] += 1
        return JSONResponse({"error": "bad json"}, status_code=400)

    state["counters"]["webhooks_received"] += 1

    txs = payload if isinstance(payload, list) else [payload]

    for tx in txs:
        tx_type = str(tx.get("type", "")).upper()
        wallet = tx.get("account")

        if wallet in TRACKED_WALLETS and tx_type in ("SWAP", "TRANSFER"):
            paper_buy(
                symbol="SOL",
                usd=min(MAX_BUY_USD, state["cash_usd"]),
                reason="matched_helius_event",
                meta={"wallet": wallet}
            )

    return {"ok": True}
