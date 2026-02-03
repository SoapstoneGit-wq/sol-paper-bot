import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="sol-paper-bot")

# ----------------------------
# Helpers
# ----------------------------

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    return float(v)

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(v)

def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v)

def now_ts() -> int:
    return int(time.time())

def mask(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = str(s)
    if len(s) <= 6:
        return "*" * len(s)
    return s[:2] + "*" * (len(s) - 4) + s[-2:]

def normalize_auth_value(v: Optional[str]) -> Optional[str]:
    """Helius sends the configured auth value in the `Authorization` header.
    Sometimes people use 'Bearer <token>' — we normalize by stripping 'Bearer '.
    """
    if not v:
        return None
    v = v.strip()
    if v.lower().startswith("bearer "):
        v = v[7:].strip()
    return v

def parse_tracked_wallets(raw: str) -> List[str]:
    # Accept comma-separated list in one env var, no spaces required.
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]

# ----------------------------
# Config
# ----------------------------

WEBHOOK_SECRET = env_str("WEBHOOK_SECRET", "").strip()

# Wallet list can be a comma-separated list (your 6 wallets in one env var is fine)
TRACKED_WALLETS_RAW = env_str("TRACKED_WALLET", "")
TRACKED_WALLETS = parse_tracked_wallets(TRACKED_WALLETS_RAW)

# Paper trading configuration
SOL_PRICE_USD = env_float("SOL_PRICE_USD", 115.0)          # only used for display/estimates; not required for auth
START_CASH_USD = env_float("START_CASH_USD", 500.0)
MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 100.0)

# 60% reserve / 40% tradable (what you asked for)
RESERVE_PCT = env_float("RESERVE_PCT", 0.60)
TRADABLE_PCT = env_float("TRADABLE_PCT", 0.40)

# Forced exit for dead meme coins (seconds)
HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)        # 900 = 15 minutes
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)

DEBUG_WEBHOOK = env_str("DEBUG_WEBHOOK", "false").lower() in ("1", "true", "yes", "y")

# ----------------------------
# In-memory state (paper bot)
# ----------------------------

STATE: Dict[str, Any] = {
    "started_at": now_ts(),
    "cash_usd": float(START_CASH_USD) * float(TRADABLE_PCT),
    "reserve_cash_usd": float(START_CASH_USD) * float(RESERVE_PCT),
    "positions": {},   # token -> { "qty": float, "entry_usd": float, "opened_at": int }
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
    "recent_trades": [],
    "events": [],  # debug/event log
}

def add_event(kind: str, data: Dict[str, Any]) -> None:
    evt = {"ts": now_ts(), "kind": kind, **data}
    STATE["events"].append(evt)
    # keep last 200 only
    if len(STATE["events"]) > 200:
        STATE["events"] = STATE["events"][-200:]

def record_trade(action: str, token: str, details: Dict[str, Any]) -> None:
    STATE["trades_count"] += 1
    STATE["recent_trades"].append(
        {"ts": now_ts(), "action": action, "token": token, **details}
    )
    if len(STATE["recent_trades"]) > 50:
        STATE["recent_trades"] = STATE["recent_trades"][-50:]

def force_exit_sweep() -> None:
    """If a position is older than HOLD_MAX_SECONDS, close it using a fallback value."""
    now = now_ts()
    to_close = []
    for token, pos in STATE["positions"].items():
        opened_at = int(pos.get("opened_at", now))
        age = now - opened_at
        if HOLD_MAX_SECONDS > 0 and age >= HOLD_MAX_SECONDS:
            to_close.append(token)

    for token in to_close:
        pos = STATE["positions"].get(token)
        if not pos:
            continue
        qty = float(pos.get("qty", 0.0))
        entry_usd = float(pos.get("entry_usd", 0.0))

        # fallback exit value (very conservative)
        exit_usd = max(0.0, entry_usd * float(FORCED_EXIT_FALLBACK_MULTI))
        proceeds = qty * exit_usd

        STATE["cash_usd"] += proceeds
        del STATE["positions"][token]

        STATE["counters"]["forced_exits"] += 1
        STATE["counters"]["sells"] += 1
        record_trade("FORCED_EXIT", token, {"qty": qty, "exit_usd": exit_usd, "proceeds_usd": proceeds})
        add_event("forced_exit", {"token": token, "age_seconds": HOLD_MAX_SECONDS, "exit_usd": exit_usd})

# ----------------------------
# Auth (THIS is the fix)
# ----------------------------

def is_authorized(req: Request) -> bool:
    """Helius sends your configured auth value in the `Authorization` header."""
    if not WEBHOOK_SECRET:
        # If you forgot to set WEBHOOK_SECRET, we reject to be safe.
        return False

    # Primary: Authorization header (Helius)
    got_auth = normalize_auth_value(req.headers.get("authorization"))

    # Secondary fallback: x-webhook-secret (if you ever use a sender that sets it)
    got_x = normalize_auth_value(req.headers.get("x-webhook-secret"))

    expected = normalize_auth_value(WEBHOOK_SECRET)

    return (got_auth == expected) or (got_x == expected)

# ----------------------------
# Routes
# ----------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "sol-paper-bot"}

@app.get("/events")
def events() -> Dict[str, Any]:
    # Return the event list (debug)
    return {"count": len(STATE["events"]), "events": STATE["events"]}

@app.get("/paper/state")
def paper_state() -> Dict[str, Any]:
    # Run forced exit sweep on every state read to keep it simple
    force_exit_sweep()
    return {
        "cash_usd": round(float(STATE["cash_usd"]), 4),
        "reserve_cash_usd": round(float(STATE["reserve_cash_usd"]), 4),
        "positions": STATE["positions"],
        "trades_count": STATE["trades_count"],
        "counters": STATE["counters"],
        "started_at": STATE["started_at"],
        "config": {
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
        },
    }

@app.post("/webhook")
async def webhook(req: Request) -> JSONResponse:
    # Verify auth
    if not is_authorized(req):
        STATE["counters"]["webhooks_unauthorized"] += 1

        # debug why (without leaking secrets)
        got_auth = req.headers.get("authorization")
        got_x = req.headers.get("x-webhook-secret")

        if not got_auth and not got_x:
            STATE["counters"]["skipped_no_secret"] += 1
            reason = "missing_header"
        else:
            reason = "mismatch"

        if DEBUG_WEBHOOK:
            add_event(
                "webhook_unauthorized_debug",
                {
                    "reason": reason,
                    "auth_present": bool(got_auth),
                    "x_present": bool(got_x),
                    "got_auth_masked": mask(got_auth),
                    "got_x_masked": mask(got_x),
                    "server_secret_masked": mask(WEBHOOK_SECRET),
                    "headers_keys_sample": sorted(list(req.headers.keys()))[:40],
                },
            )

        raise HTTPException(status_code=401, detail="Unauthorized")

    # Authorized: parse JSON (best effort)
    STATE["counters"]["webhooks_received"] += 1
    force_exit_sweep()

    try:
        payload = await req.json()
    except Exception:
        STATE["counters"]["skipped_bad_payload"] += 1
        add_event("webhook_bad_payload", {"note": "request body was not valid JSON"})
        return JSONResponse({"ok": True, "note": "bad payload (not JSON) accepted"}, status_code=200)

    # Always log a lightweight event so you can see it working
    # (don’t store full payload; can be huge)
    add_event(
        "webhook_ok",
        {
            "payload_type": type(payload).__name__,
            "keys": list(payload.keys())[:30] if isinstance(payload, dict) else None,
            "tracked_wallets_count": len(TRACKED_WALLETS),
        },
    )

    # Paper-trade logic placeholder:
    # Your Helius payload schema depends on webhook type (raw vs enhanced).
    # For now we just confirm receipt. Once auth is fixed and events are flowing,
    # we can map Helius fields -> BUY/SELL simulation cleanly.
    return JSONResponse({"ok": True}, status_code=200)

@app.get("/")
def root() -> Dict[str, Any]:
    # Optional: stop Render/visitors from seeing 404s at /
    return {"ok": True, "service": "sol-paper-bot", "endpoints": ["/health", "/paper/state", "/events", "/webhook"]}

# The app object is `app` for: uvicorn app:app
