import os
import time
import json
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse


# ----------------------------
# Config helpers
# ----------------------------

def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None and v != "" else default
    except Exception:
        return default

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None and v != "" else default
    except Exception:
        return default

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def parse_tracked_wallets(raw: str) -> List[str]:
    """
    TRACKED_WALLET can be:
      - single wallet
      - comma-separated list
      - newline-separated list
      - mixed
    """
    if not raw:
        return []
    # split on commas and newlines
    parts = []
    for chunk in raw.replace("\n", ",").split(","):
        w = chunk.strip()
        if w:
            parts.append(w)
    # de-dup preserving order
    seen = set()
    out = []
    for w in parts:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


# ----------------------------
# Bot state (paper trading)
# ----------------------------

START_CASH_USD = env_float("START_CASH_USD", 500.0)
MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 100.0)

RESERVE_PCT = env_float("RESERVE_PCT", 0.60)
TRADABLE_PCT = env_float("TRADABLE_PCT", 0.40)

HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)  # 900 = 15 minutes
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)

# If you don’t want to guess SOL->USD, you can keep this; it’s only for *paper* math.
SOL_PRICE_USD = env_float("SOL_PRICE_USD", 115.0)

WEBHOOK_SECRET = env_str("WEBHOOK_SECRET", "")

DEBUG_WEBHOOK = env_bool("DEBUG_WEBHOOK", False)


# Paper balances
reserve_cash_usd = round(START_CASH_USD * RESERVE_PCT, 2)
cash_usd = round(START_CASH_USD - reserve_cash_usd, 2)

# positions keyed by mint or token id (depends on payload)
positions: Dict[str, Dict[str, Any]] = {}

# counters + event log
started_at = int(time.time())
counters = {
    "webhooks_received": 0,
    "webhooks_unauthorized": 0,
    "skipped_no_secret": 0,
    "skipped_bad_payload": 0,
    "buys": 0,
    "sells": 0,
    "forced_exits": 0,
    "skipped_low_cash": 0,
}

events: List[Dict[str, Any]] = []
EVENTS_MAX = 200  # keep last N

last_webhook_debug: Dict[str, Any] = {}


def push_event(e: Dict[str, Any]) -> None:
    events.append(e)
    if len(events) > EVENTS_MAX:
        del events[: len(events) - EVENTS_MAX]


def mask_secret(s: str) -> str:
    if not s:
        return ""
    if len(s) <= 6:
        return "*" * len(s)
    return s[:2] + "*" * (len(s) - 4) + s[-2:]


def header_lookup(headers: Dict[str, str], name: str) -> Optional[str]:
    """
    Case-insensitive header lookup.
    """
    name_l = name.lower()
    for k, v in headers.items():
        if k.lower() == name_l:
            return v
    return None


app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "service": "sol-paper-bot"}


@app.get("/events")
def get_events():
    return {"count": len(events), "events": events}


@app.get("/paper/state")
def paper_state():
    return {
        "cash_usd": cash_usd,
        "reserve_cash_usd": reserve_cash_usd,
        "positions": positions,
        "trades_count": (counters["buys"] + counters["sells"]),
        "counters": counters,
        "started_at": started_at,
        "config": {
            "SOL_PRICE_USD": SOL_PRICE_USD,
            "START_CASH_USD": START_CASH_USD,
            "MAX_BUY_USD": MAX_BUY_USD,
            "MIN_CASH_LEFT_USD": MIN_CASH_LEFT_USD,
            "RESERVE_PCT": RESERVE_PCT,
            "TRADABLE_PCT": TRADABLE_PCT,
            "HOLD_MAX_SECONDS": HOLD_MAX_SECONDS,
            "FORCED_EXIT_FALLBACK_MULTI": FORCED_EXIT_FALLBACK_MULTI,
            "TRACKED_WALLETS_COUNT": len(parse_tracked_wallets(env_str("TRACKED_WALLET", ""))),
            "DEBUG_WEBHOOK": DEBUG_WEBHOOK,
        },
    }


@app.get("/debug/last")
def debug_last():
    if not DEBUG_WEBHOOK:
        return {"debug": False, "message": "DEBUG_WEBHOOK is not enabled"}
    return {"debug": True, "last_webhook_debug": last_webhook_debug}


@app.post("/webhook")
async def webhook(request: Request):
    global last_webhook_debug

    # --- auth: expect header x-webhook-secret to match WEBHOOK_SECRET
    counters["webhooks_received"] += 1

    if not WEBHOOK_SECRET:
        counters["skipped_no_secret"] += 1
        push_event({"ts": int(time.time()), "kind": "webhook_skipped", "reason": "no_server_secret_set"})
        return JSONResponse({"ok": True}, status_code=200)

    got = header_lookup(dict(request.headers), "x-webhook-secret")
    if got is None or got != WEBHOOK_SECRET:
        counters["webhooks_unauthorized"] += 1

        # ultra-safe debug info (no secret leakage)
        if DEBUG_WEBHOOK:
            last_webhook_debug = {
                "ts": int(time.time()),
                "kind": "webhook_unauthorized_debug",
                "reason": "missing_or_mismatch_header",
                "header_present": got is not None,
                "got_len": 0 if got is None else len(got),
                "server_secret_len": len(WEBHOOK_SECRET),
                "server_secret_masked": mask_secret(WEBHOOK_SECRET),
                "got_masked": None if got is None else mask_secret(got),
                "headers_keys_sample": sorted(list(dict(request.headers).keys()))[:25],
            }
            push_event(last_webhook_debug)
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    # --- read body
    raw_bytes = await request.body()

    payload: Any = None
    payload_type = None
    list_len: Optional[int] = None
    keys: Optional[List[str]] = None
    preview: Optional[Dict[str, Any]] = None

    try:
        payload = json.loads(raw_bytes.decode("utf-8")) if raw_bytes else None
        payload_type = type(payload).__name__
    except Exception:
        counters["skipped_bad_payload"] += 1
        push_event({"ts": int(time.time()), "kind": "webhook_bad_payload", "reason": "json_decode_failed"})
        return JSONResponse({"ok": True}, status_code=200)

    # --- summarize payload shape (THIS is what we need next)
    try:
        if isinstance(payload, list):
            list_len = len(payload)
            if list_len > 0 and isinstance(payload[0], dict):
                keys = sorted(list(payload[0].keys()))
                # small preview, truncated values to keep it safe/light
                preview = {}
                for k in keys[:20]:
                    v = payload[0].get(k)
                    # keep preview small
                    if isinstance(v, (dict, list)):
                        pv = json.dumps(v)[:300]
                    else:
                        pv = str(v)[:300]
                    preview[k] = pv
            else:
                keys = None
        elif isinstance(payload, dict):
            keys = sorted(list(payload.keys()))
            preview = {}
            for k in keys[:20]:
                v = payload.get(k)
                if isinstance(v, (dict, list)):
                    pv = json.dumps(v)[:300]
                else:
                    pv = str(v)[:300]
                preview[k] = pv
        else:
            keys = None
    except Exception:
        keys = None

    event_ok = {
        "ts": int(time.time()),
        "kind": "webhook_ok",
        "payload_type": payload_type,
        "list_len": list_len,
        "keys": keys,
        "preview": preview,
        "tracked_wallets_count": len(parse_tracked_wallets(env_str("TRACKED_WALLET", ""))),
    }
    push_event(event_ok)

    if DEBUG_WEBHOOK:
        last_webhook_debug = event_ok

    # --- TODO: trading logic goes here once we see real fields from Helius
    # For now, just ack.
    return JSONResponse({"ok": True}, status_code=200)
