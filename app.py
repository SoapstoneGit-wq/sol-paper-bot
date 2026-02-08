import os
import time
import hmac
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(float(v))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if not v:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def safe_strip(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = v.strip()
    # remove accidental wrapping quotes
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        v = v[1:-1].strip()
    return v


def mask_secret(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = str(s)
    if len(s) <= 4:
        return "*" * len(s)
    return s[:2] + ("*" * (len(s) - 4)) + s[-2:]


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def extract_secret_from_headers(headers) -> Optional[str]:
    """
    Accept secret from either:
      - x-webhook-secret: <secret>
      - authorization: <secret>
      - authorization: Bearer <secret>
      - authorization: x-webhook-secret: <secret>
    """
    x = headers.get("x-webhook-secret")
    if x:
        return safe_strip(x)

    auth = headers.get("authorization")
    if not auth:
        return None

    auth = auth.strip()

    if auth.lower().startswith("bearer "):
        return safe_strip(auth[7:].strip())

    if ":" in auth:
        left, right = auth.split(":", 1)
        if left.strip().lower() in ("x-webhook-secret", "x_webhook_secret"):
            return safe_strip(right.strip())

    return safe_strip(auth)


# -----------------------------
# Config
# -----------------------------
WEBHOOK_SECRET = safe_strip(os.getenv("WEBHOOK_SECRET") or "")
WEBHOOK_PATH_TOKEN = safe_strip(os.getenv("WEBHOOK_PATH_TOKEN") or "")
DEBUG_WEBHOOK = env_bool("DEBUG_WEBHOOK", False)

TRACKED_WALLET_RAW = os.getenv("TRACKED_WALLET") or ""
TRACKED_WALLETS = [p.strip() for p in TRACKED_WALLET_RAW.split(",") if p.strip()]

SOL_PRICE_USD = env_float("SOL_PRICE_USD", 115.0)
START_CASH_USD = env_float("START_CASH_USD", 500.0)
MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 100.0)
RESERVE_PCT = env_float("RESERVE_PCT", 0.60)
TRADABLE_PCT = env_float("TRADABLE_PCT", 0.40)
HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)

STARTED_AT = int(time.time())

state: Dict[str, Any] = {
    "cash_usd": round(START_CASH_USD * TRADABLE_PCT, 2),
    "reserve_cash_usd": round(START_CASH_USD * RESERVE_PCT, 2),
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
    "started_at": STARTED_AT,
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
        "WEBHOOK_PATH_TOKEN_SET": bool(WEBHOOK_PATH_TOKEN),
    },
    "recent_trades": [],
}

events: List[Dict[str, Any]] = []


def push_event(evt: Dict[str, Any], limit: int = 300) -> None:
    events.append(evt)
    if len(events) > limit:
        del events[: len(events) - limit]


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "sol-paper-bot",
        "routes": ["/health", "/events", "/paper/state", "/webhook/{token}"],
    }


@app.get("/health")
def health():
    return {"ok": True, "service": "sol-paper-bot"}


@app.get("/events")
def get_events():
    return {"count": len(events), "events": events}


@app.get("/paper/state")
def paper_state():
    return state


@app.post("/webhook/{token}")
async def webhook(token: str, req: Request):
    # 0) Path token gate (stops scanners)
    if not WEBHOOK_PATH_TOKEN:
        state["counters"]["skipped_bad_path"] += 1
        push_event({"ts": int(time.time()), "kind": "server_misconfig", "reason": "WEBHOOK_PATH_TOKEN missing"})
        return JSONResponse({"error": "WEBHOOK_PATH_TOKEN missing on server"}, status_code=500)

    if token != WEBHOOK_PATH_TOKEN:
        state["counters"]["skipped_bad_path"] += 1
        if DEBUG_WEBHOOK:
            push_event({"ts": int(time.time()), "kind": "bad_path", "got": token})
        return JSONResponse({"error": "not found"}, status_code=404)

    # 1) Secret header check
    if not WEBHOOK_SECRET:
        state["counters"]["skipped_no_secret"] += 1
        push_event({"ts": int(time.time()), "kind": "server_misconfig", "reason": "WEBHOOK_SECRET missing"})
        return JSONResponse({"error": "WEBHOOK_SECRET missing on server"}, status_code=500)

    got = extract_secret_from_headers(req.headers)

    if (got is None) or (not constant_time_equals(got, WEBHOOK_SECRET)):
        state["counters"]["webhooks_unauthorized"] += 1
        if DEBUG_WEBHOOK:
            push_event(
                {
                    "ts": int(time.time()),
                    "kind": "webhook_unauthorized_debug",
                    "reason": "missing_or_mismatch_header",
                    "x_present": req.headers.get("x-webhook-secret") is not None,
                    "auth_present": req.headers.get("authorization") is not None,
                    "x_len": len(req.headers.get("x-webhook-secret") or ""),
                    "auth_len": len(req.headers.get("authorization") or ""),
                    "server_secret_len": len(WEBHOOK_SECRET),
                    "server_secret_masked": mask_secret(WEBHOOK_SECRET),
                    "got_masked": mask_secret(got),
                }
            )
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    # 2) Parse payload
    try:
        payload: Union[Dict[str, Any], List[Any]] = await req.json()
    except Exception:
        state["counters"]["skipped_bad_payload"] += 1
        push_event({"ts": int(time.time()), "kind": "bad_payload"})
        return JSONResponse({"ok": False, "error": "bad json"}, status_code=400)

    state["counters"]["webhooks_received"] += 1

    payload_type = "dict" if isinstance(payload, dict) else ("list" if isinstance(payload, list) else str(type(payload)))

    push_event(
        {
            "ts": int(time.time()),
            "kind": "webhook_ok",
            "payload_type": payload_type,
            "tracked_wallets_count": len(TRACKED_WALLETS),
        }
    )
    return {"ok": True}
