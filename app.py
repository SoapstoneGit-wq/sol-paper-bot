import os
import time
import hmac
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

app = FastAPI()

# ---------------------------
# Config helpers
# ---------------------------

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def parse_tracked_wallets(raw: str) -> List[str]:
    # Accept comma-separated list, with or without spaces/newlines
    wallets = []
    for part in raw.replace("\n", ",").split(","):
        p = part.strip()
        if p:
            wallets.append(p)
    # de-dupe while preserving order
    seen = set()
    out = []
    for w in wallets:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out

def mask_secret(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = str(s)
    if len(s) <= 4:
        return "*" * len(s)
    return s[:2] + "*" * (len(s) - 4) + s[-2:]


# ---------------------------
# Runtime config (env)
# ---------------------------

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()

TRACKED_WALLET_RAW = os.getenv("TRACKED_WALLET", "").strip()
TRACKED_WALLETS = parse_tracked_wallets(TRACKED_WALLET_RAW)

START_CASH_USD = env_float("START_CASH_USD", 500.0)
RESERVE_PCT = env_float("RESERVE_PCT", 0.60)   # 60% reserve
TRADABLE_PCT = env_float("TRADABLE_PCT", 0.40) # 40% tradable

MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 100.0)

SOL_PRICE_USD = env_float("SOL_PRICE_USD", 115.0)

# Forced exit
HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)  # 900 seconds = 15 minutes
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)

DEBUG_WEBHOOK = env_bool("DEBUG_WEBHOOK", False)

# ---------------------------
# In-memory paper state
# ---------------------------

STATE: Dict[str, Any] = {
    "cash_usd": round(START_CASH_USD * TRADABLE_PCT, 2),
    "reserve_cash_usd": round(START_CASH_USD * RESERVE_PCT, 2),
    "positions": {},  # mint -> { "usd_cost": float, "opened_at": int, "source": str }
    "trades_count": 0,
    "counters": {
        "webhooks_received": 0,
        "webhooks_unauthorized": 0,
        "skipped_no_secret": 0,
        "skipped_bad_payload": 0,
        "buys": 0,
        "sells": 0,
        "forced_exits": 0,
        "skipped_low_cash": 0,
    },
    "events": [],          # rolling debug / info events
    "recent_trades": [],   # rolling trades
    "started_at": int(time.time()),
}

MAX_EVENTS = 250
MAX_TRADES = 100

def push_event(evt: Dict[str, Any]) -> None:
    STATE["events"].append(evt)
    if len(STATE["events"]) > MAX_EVENTS:
        STATE["events"] = STATE["events"][-MAX_EVENTS:]

def push_trade(tr: Dict[str, Any]) -> None:
    STATE["recent_trades"].append(tr)
    if len(STATE["recent_trades"]) > MAX_TRADES:
        STATE["recent_trades"] = STATE["recent_trades"][-MAX_TRADES:]

def now_ts() -> int:
    return int(time.time())

# ---------------------------
# Auth (DROP-IN PATCH)
# Accept secret in either:
#   - x-webhook-secret: <secret>
#   - authorization: <secret>
#   - authorization: Bearer <secret>
#   - authorization: Token <secret>
# ---------------------------

def extract_secret_from_headers(headers: Dict[str, str]) -> Tuple[Optional[str], Dict[str, Any]]:
    # FastAPI headers are case-insensitive, but we normalize access anyway.
    x = headers.get("x-webhook-secret") or headers.get("X-Webhook-Secret")
    auth = headers.get("authorization") or headers.get("Authorization")

    got = None
    used = None

    if x and x.strip():
        got = x.strip()
        used = "x-webhook-secret"
    elif auth and auth.strip():
        a = auth.strip()
        # Handle "Bearer xxx" or "Token xxx"
        parts = a.split()
        if len(parts) == 2 and parts[0].lower() in ("bearer", "token"):
            got = parts[1].strip()
            used = "authorization_scheme"
        else:
            got = a
            used = "authorization_raw"

    debug = {
        "x_present": bool(x),
        "auth_present": bool(auth),
        "x_len": len(x.strip()) if x else 0,
        "auth_len": len(auth.strip()) if auth else 0,
        "used": used,
        "got_masked": mask_secret(got),
    }
    return got, debug

def webhook_authorized(request_headers: Dict[str, str]) -> Tuple[bool, Dict[str, Any]]:
    server_secret = WEBHOOK_SECRET.strip()
    got, hdr_debug = extract_secret_from_headers(request_headers)

    ok = False
    reason = None

    if not server_secret:
        # If you forgot to set WEBHOOK_SECRET in Render, that's a hard fail
        ok = False
        reason = "server_missing_secret"
    elif not got:
        ok = False
        reason = "missing_header"
    else:
        # constant-time compare
        ok = hmac.compare_digest(got, server_secret)
        reason = "match" if ok else "mismatch"

    debug = {
        "reason": reason,
        "server_secret_len": len(server_secret),
        "server_secret_masked": mask_secret(server_secret),
        **hdr_debug,
    }
    return ok, debug


# ---------------------------
# Trading logic (simple paper model)
# ---------------------------

def can_buy() -> bool:
    cash = float(STATE["cash_usd"])
    # keep MIN_CASH_LEFT_USD available
    return (cash - MIN_CASH_LEFT_USD) >= 1.0

def paper_buy(mint: str, source: str) -> None:
    cash = float(STATE["cash_usd"])
    spendable = max(0.0, cash - MIN_CASH_LEFT_USD)
    usd_to_spend = min(MAX_BUY_USD, spendable)

    if usd_to_spend <= 0.0:
        STATE["counters"]["skipped_low_cash"] += 1
        return

    STATE["cash_usd"] = round(cash - usd_to_spend, 2)
    STATE["positions"][mint] = {
        "usd_cost": round(usd_to_spend, 2),
        "opened_at": now_ts(),
        "source": source,
    }
    STATE["trades_count"] += 1
    STATE["counters"]["buys"] += 1

    push_trade({
        "ts": now_ts(),
        "side": "BUY",
        "mint": mint,
        "usd": round(usd_to_spend, 2),
        "source": source,
    })

def paper_sell(mint: str, reason: str, price_mult: float = 1.0) -> None:
    pos = STATE["positions"].get(mint)
    if not pos:
        return

    usd_cost = float(pos.get("usd_cost", 0.0))
    usd_return = round(usd_cost * float(price_mult), 2)

    cash = float(STATE["cash_usd"])
    STATE["cash_usd"] = round(cash + usd_return, 2)
    del STATE["positions"][mint]

    STATE["trades_count"] += 1
    STATE["counters"]["sells"] += 1

    push_trade({
        "ts": now_ts(),
        "side": "SELL",
        "mint": mint,
        "usd": usd_return,
        "reason": reason,
    })

def apply_forced_exits() -> None:
    if HOLD_MAX_SECONDS <= 0:
        return

    ts = now_ts()
    to_close = []
    for mint, pos in list(STATE["positions"].items()):
        opened = int(pos.get("opened_at", ts))
        age = ts - opened
        if age >= HOLD_MAX_SECONDS:
            to_close.append(mint)

    for mint in to_close:
        paper_sell(mint, reason="FORCED_EXIT_TIME", price_mult=FORCED_EXIT_FALLBACK_MULTI)
        STATE["counters"]["forced_exits"] += 1

def extract_signal_mints(payload_items: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Very simple heuristic:
      - If tracked wallet appears as a signer / account in tx, and there are tokenTransfers,
        we treat any non-null mint seen as a "signal mint".
    Returns (buy_mints, sell_mints) — currently we only produce buy signals.
    """
    buy_mints: List[str] = []
    sell_mints: List[str] = []

    tracked_set = set(TRACKED_WALLETS)

    for item in payload_items:
        # Try common enhanced fields
        accounts = set()

        # Some payloads include "feePayer" or "signer" or "signers"
        for k in ("feePayer", "signer"):
            v = item.get(k)
            if isinstance(v, str):
                accounts.add(v)
        v = item.get("signers")
        if isinstance(v, list):
            for s in v:
                if isinstance(s, str):
                    accounts.add(s)

        # Some include "accounts" list
        v = item.get("accounts")
        if isinstance(v, list):
            for a in v:
                if isinstance(a, str):
                    accounts.add(a)

        is_tracked = any(a in tracked_set for a in accounts) if accounts else True  # if unknown, still accept

        token_transfers = item.get("tokenTransfers")
        if not is_tracked or not isinstance(token_transfers, list):
            continue

        for t in token_transfers:
            if not isinstance(t, dict):
                continue
            mint = t.get("mint")
            if isinstance(mint, str) and mint and mint.lower() != "so11111111111111111111111111111111111111112":
                buy_mints.append(mint)

    # de-dupe
    def dedupe(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return dedupe(buy_mints), dedupe(sell_mints)


# ---------------------------
# Routes
# ---------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "sol-paper-bot"}

@app.get("/events")
def events() -> Dict[str, Any]:
    return {"count": len(STATE["events"]), "events": STATE["events"]}

@app.get("/paper/state")
def paper_state() -> Dict[str, Any]:
    cfg = {
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
    }
    return {
        "cash_usd": STATE["cash_usd"],
        "reserve_cash_usd": STATE["reserve_cash_usd"],
        "positions": STATE["positions"],
        "trades_count": STATE["trades_count"],
        "counters": STATE["counters"],
        "started_at": STATE["started_at"],
        "config": cfg,
        "recent_trades": STATE["recent_trades"],
    }

@app.post("/webhook")
async def webhook(req: Request) -> Response:
    STATE["counters"]["webhooks_received"] += 1

    # Auth
    headers = {k.lower(): v for k, v in req.headers.items()}
    ok, auth_debug = webhook_authorized(headers)

    if not ok:
        STATE["counters"]["webhooks_unauthorized"] += 1
        if auth_debug.get("reason") == "missing_header":
            STATE["counters"]["skipped_no_secret"] += 1

        if DEBUG_WEBHOOK:
            push_event({
                "ts": now_ts(),
                "kind": "webhook_unauthorized_debug",
                **auth_debug,
                "headers_keys_sample": sorted(list(headers.keys()))[:30],
            })
        return JSONResponse(status_code=401, content={"ok": False, "error": "unauthorized"})

    # Parse JSON
    try:
        body = await req.json()
    except Exception:
        STATE["counters"]["skipped_bad_payload"] += 1
        if DEBUG_WEBHOOK:
            push_event({"ts": now_ts(), "kind": "bad_json"})
        return JSONResponse(status_code=200, content={"ok": True, "ignored": "bad_json"})

    # Normalize payload into list-of-items
    payload_items: List[Dict[str, Any]] = []
    payload_type = type(body).__name__

    if isinstance(body, list):
        payload_items = [x for x in body if isinstance(x, dict)]
    elif isinstance(body, dict):
        # Some webhooks wrap content under a key
        for key in ("data", "events", "transactions"):
            v = body.get(key)
            if isinstance(v, list):
                payload_items = [x for x in v if isinstance(x, dict)]
                break
        if not payload_items:
            # treat dict itself as single item
            payload_items = [body]
    else:
        STATE["counters"]["skipped_bad_payload"] += 1
        if DEBUG_WEBHOOK:
            push_event({"ts": now_ts(), "kind": "bad_payload_type", "payload_type": payload_type})
        return JSONResponse(status_code=200, content={"ok": True, "ignored": "bad_payload_type"})

    # Record success event
    push_event({
        "ts": now_ts(),
        "kind": "webhook_ok",
        "payload_type": "list" if isinstance(body, list) else "dict",
        "keys": None if not isinstance(body, dict) else sorted(list(body.keys()))[:25],
        "tracked_wallets_count": len(TRACKED_WALLETS),
    })

    # Apply forced exits on every webhook tick
    apply_forced_exits()

    # Trading signals (simple heuristic)
    buy_mints, sell_mints = extract_signal_mints(payload_items)

    # Execute paper actions
    for mint in buy_mints:
        if mint in STATE["positions"]:
            continue
        if not can_buy():
            STATE["counters"]["skipped_low_cash"] += 1
            break
        paper_buy(mint, source="webhook_signal")

    for mint in sell_mints:
        if mint in STATE["positions"]:
            paper_sell(mint, reason="webhook_signal_sell", price_mult=1.0)

    return JSONResponse(status_code=200, content={"ok": True})

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "routes": ["/health", "/webhook (POST)", "/paper/state", "/events"]
    }
