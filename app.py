import os
import time
import hmac
from typing import Any, Dict, List, Union, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# ============================
# Helpers
# ============================

def getenv_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def getenv_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default

def getenv_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(float(v))
    except Exception:
        return default

def mask_secret(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    if len(s) <= 4:
        return "*" * len(s)
    return s[:2] + ("*" * (len(s) - 4)) + s[-2:]

def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a or "", b or "")

def extract_secret_from_headers(headers) -> Optional[str]:
    # Prefer x-webhook-secret
    x = headers.get("x-webhook-secret")
    if x:
        return x.strip()

    # Also accept Authorization header
    auth = headers.get("authorization")
    if auth:
        auth = auth.strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return auth
    return None

# ============================
# ENV CONFIG
# ============================

DEBUG_WEBHOOK = getenv_bool("DEBUG_WEBHOOK", False)

START_CASH_USD = getenv_float("START_CASH_USD", 500.0)
MAX_BUY_USD = getenv_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = getenv_float("MIN_CASH_LEFT_USD", 100.0)
RESERVE_PCT = getenv_float("RESERVE_PCT", 0.60)
TRADABLE_PCT = getenv_float("TRADABLE_PCT", 0.40)
HOLD_MAX_SECONDS = getenv_int("HOLD_MAX_SECONDS", 900)
FORCED_EXIT_FALLBACK_MULTI = getenv_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)
SOL_PRICE_USD = getenv_float("SOL_PRICE_USD", 115.0)

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
WEBHOOK_PATH_TOKEN = os.getenv("WEBHOOK_PATH_TOKEN", "")

_raw_wallets = os.getenv("TRACKED_WALLET", "")
TRACKED_WALLETS = [
    w.strip()
    for w in _raw_wallets.replace("\n", ",").replace(" ", ",").split(",")
    if w.strip()
]

# ============================
# In-Memory State
# ============================

MAX_EVENTS = 500

state: Dict[str, Any] = {
    "cash_usd": START_CASH_USD * TRADABLE_PCT,
    "reserve_cash_usd": START_CASH_USD * RESERVE_PCT,
    "positions": {},
    "trades_count": 0,
    "counters": {
        "webhooks_received": 0,
        "webhooks_unauthorized": 0,
        "skipped_no_secret": 0,
        "skipped_bad_payload": 0,
        "skipped_bad_path": 0,
        "buys": 0,
        "sells": 0,
        "forced_exits": 0,
        "skipped_low_cash": 0,
    },
    "started_at": int(time.time()),
    "events": [],
    "recent_trades": [],
}

def push_event(evt: Dict[str, Any]) -> None:
    state["events"].append(evt)
    if len(state["events"]) > MAX_EVENTS:
        state["events"] = state["events"][-MAX_EVENTS:]

# ============================
# Routes
# ============================

@app.get("/")
def root():
    return {"ok": True, "service": "sol-paper-bot"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/events")
def events():
    return {"count": len(state["events"]), "events": state["events"]}

@app.get("/paper/state")
def paper_state():
    return {
        "cash_usd": state["cash_usd"],
        "reserve_cash_usd": state["reserve_cash_usd"],
        "positions": state["positions"],
        "trades_count": state["trades_count"],
        "counters": state["counters"],
        "started_at": state["started_at"],
        "config": {
            "TRACKED_WALLETS_COUNT": len(TRACKED_WALLETS),
            "DEBUG_WEBHOOK": DEBUG_WEBHOOK,
            "WEBHOOK_PATH_TOKEN_SET": bool(WEBHOOK_PATH_TOKEN),
        },
        "recent_trades": state["recent_trades"],
    }

# ============================
# WEBHOOK ROUTE
# ============================

@app.post("/webhook/{token}")
async def webhook(token: str, req: Request):

    # ---- 1) PATH TOKEN GATE ----
    if not WEBHOOK_PATH_TOKEN:
        state["counters"]["skipped_bad_path"] += 1
        return JSONResponse({"error": "WEBHOOK_PATH_TOKEN missing on server"}, status_code=500)

    if token != WEBHOOK_PATH_TOKEN:
        state["counters"]["skipped_bad_path"] += 1
        return JSONResponse({"error": "not found"}, status_code=404)

    # ---- 2) SECRET HEADER CHECK ----
    if not WEBHOOK_SECRET:
        state["counters"]["skipped_no_secret"] += 1
        return JSONResponse({"error": "WEBHOOK_SECRET missing on server"}, status_code=500)

    got = extract_secret_from_headers(req.headers)

    if (got is None) or (not constant_time_equals(got, WEBHOOK_SECRET)):
        state["counters"]["webhooks_unauthorized"] += 1
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    # ---- 3) PARSE JSON ----
    try:
        payload: Union[Dict[str, Any], List[Any]] = await req.json()
    except Exception:
        state["counters"]["skipped_bad_payload"] += 1
        return JSONResponse({"error": "bad json"}, status_code=400)

    state["counters"]["webhooks_received"] += 1

    transactions = payload if isinstance(payload, list) else [payload]

    matched = 0

    for tx in transactions:
        if not isinstance(tx, dict):
            continue

        tx_type = tx.get("type")
        wallet = tx.get("feePayer") or tx.get("signer")

        if wallet and wallet in TRACKED_WALLETS and tx_type in ["SWAP", "TRANSFER"]:
            matched += 1
            state["counters"]["buys"] += 1

            push_event({
                "ts": int(time.time()),
                "kind": "paper_buy_simulated",
                "wallet": wallet,
                "type": tx_type,
                "signature": tx.get("signature"),
            })

    push_event({
        "ts": int(time.time()),
        "kind": "webhook_ok",
        "matched": matched,
        "tracked_wallets_count": len(TRACKED_WALLETS),
    })

    return {"ok": True}
