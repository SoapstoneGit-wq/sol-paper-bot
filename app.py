import os
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


# ============================================================
# Helpers
# ============================================================

def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v)


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    return float(v)


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(float(v))


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def mask_secret(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    if len(s) <= 4:
        return "*" * len(s)
    return s[:2] + ("*" * (len(s) - 4)) + s[-2:]


def header_lookup(headers: Dict[str, str], key: str) -> Optional[str]:
    """
    Case-insensitive header lookup.
    FastAPI request.headers is already case-insensitive, but we normalize anyway.
    """
    key_l = key.lower()
    for k, v in headers.items():
        if k.lower() == key_l:
            return v
    return None


def normalize_secret_header(v: Optional[str]) -> Optional[str]:
    """
    Accept either raw secret or "Bearer <secret>" style.
    """
    if v is None:
        return None
    s = str(v).strip()
    if s.lower().startswith("bearer "):
        return s[7:].strip()
    return s


def parse_tracked_wallets(raw: str) -> List[str]:
    """
    Accept:
      - single wallet
      - comma-separated wallets
      - whitespace/newline separated wallets
    """
    if not raw:
        return []
    raw = raw.replace("\n", ",").replace("\r", ",").replace(" ", ",").strip()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    # de-dupe while preserving order
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def now_ts() -> int:
    return int(time.time())


# ============================================================
# Config (env)
# ============================================================

TRACKED_WALLET_RAW = env_str("TRACKED_WALLET", "")
TRACKED_WALLETS = parse_tracked_wallets(TRACKED_WALLET_RAW)

WEBHOOK_SECRET = env_str("WEBHOOK_SECRET", "").strip()

START_CASH_USD = env_float("START_CASH_USD", 500.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 100.0)
MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)

# 60/40 split
RESERVE_PCT = env_float("RESERVE_PCT", 0.60)
TRADABLE_PCT = env_float("TRADABLE_PCT", 0.40)

# Forced exit
HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)  # 900 seconds = 15 minutes
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)

# Optional: used only for paper value estimates
SOL_PRICE_USD = env_float("SOL_PRICE_USD", 115.0)

DEBUG_WEBHOOK = env_bool("DEBUG_WEBHOOK", False)


# ============================================================
# State (in-memory)
# ============================================================

started_at = now_ts()

cash_usd = round(START_CASH_USD * TRADABLE_PCT, 2)
reserve_cash_usd = round(START_CASH_USD * RESERVE_PCT, 2)

# positions[token] = {"qty": float, "avg_price": float, "opened_at": int}
positions: Dict[str, Dict[str, Any]] = {}

trades_count = 0
recent_trades: List[Dict[str, Any]] = []

counters: Dict[str, int] = {
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
MAX_EVENTS = 200


def push_event(e: Dict[str, Any]) -> None:
    events.append(e)
    if len(events) > MAX_EVENTS:
        del events[0:len(events) - MAX_EVENTS]


def add_trade(t: Dict[str, Any]) -> None:
    global trades_count
    trades_count += 1
    recent_trades.append(t)
    if len(recent_trades) > 25:
        del recent_trades[0:len(recent_trades) - 25]


def maybe_forced_exit() -> None:
    """
    If a position is held longer than HOLD_MAX_SECONDS, force-exit it.
    Since we don't have reliable market pricing in this simple paper bot,
    we "exit" at avg_price * FORCED_EXIT_FALLBACK_MULTI (defaults to 0.5).
    """
    global cash_usd
    now = now_ts()
    to_close: List[Tuple[str, Dict[str, Any]]] = []
    for token, pos in positions.items():
        opened_at = int(pos.get("opened_at", now))
        if (now - opened_at) >= HOLD_MAX_SECONDS:
            to_close.append((token, pos))

    for token, pos in to_close:
        qty = float(pos.get("qty", 0.0))
        avg_price = float(pos.get("avg_price", 0.0))
        exit_price = avg_price * float(FORCED_EXIT_FALLBACK_MULTI)
        proceeds = qty * exit_price
        cash_usd = round(cash_usd + proceeds, 6)

        counters["forced_exits"] += 1
        counters["sells"] += 1

        add_trade({
            "ts": now_ts(),
            "type": "FORCED_EXIT",
            "token": token,
            "qty": qty,
            "price": exit_price,
            "proceeds_usd": proceeds,
            "reason": f"held>{HOLD_MAX_SECONDS}s",
        })

        push_event({
            "ts": now_ts(),
            "kind": "forced_exit",
            "token": token,
            "qty": qty,
            "exit_price": exit_price,
            "proceeds_usd": proceeds,
        })

        positions.pop(token, None)


# ============================================================
# Routes
# ============================================================

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "sol-paper-bot"}


@app.get("/events")
def get_events() -> Dict[str, Any]:
    return {"count": len(events), "events": events[-MAX_EVENTS:]}


@app.get("/paper/state")
def paper_state() -> Dict[str, Any]:
    # Try forced exits on every state read, so dead positions don't linger
    maybe_forced_exit()

    return {
        "cash_usd": float(round(cash_usd, 6)),
        "reserve_cash_usd": float(round(reserve_cash_usd, 6)),
        "positions": positions,
        "trades_count": trades_count,
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
            "TRACKED_WALLETS_COUNT": len(TRACKED_WALLETS),
            "DEBUG_WEBHOOK": DEBUG_WEBHOOK,
        },
        "recent_trades": recent_trades,
    }


@app.post("/webhook")
async def webhook(request: Request) -> JSONResponse:
    """
    Webhook receiver. Auth patch included:
      Accepts secret from either:
        - x-webhook-secret
        - authorization (supports Bearer)
    """
    global cash_usd

    counters["webhooks_received"] += 1

    # ------------------------------------------------------------
    # AUTH PATCH (x-webhook-secret OR authorization)
    # ------------------------------------------------------------
    if not WEBHOOK_SECRET:
        counters["skipped_no_secret"] += 1
        push_event({"ts": now_ts(), "kind": "webhook_skipped", "reason": "no_server_secret_set"})
        return JSONResponse({"ok": True}, status_code=200)

    headers_dict = dict(request.headers)

    got_x = header_lookup(headers_dict, "x-webhook-secret")
    got_auth = header_lookup(headers_dict, "authorization")

    got_x_n = normalize_secret_header(got_x)
    got_auth_n = normalize_secret_header(got_auth)

    if (got_x_n != WEBHOOK_SECRET) and (got_auth_n != WEBHOOK_SECRET):
        counters["webhooks_unauthorized"] += 1

        if DEBUG_WEBHOOK:
            push_event({
                "ts": now_ts(),
                "kind": "webhook_unauthorized_debug",
                "reason": "missing_or_mismatch_header",
                "x_present": got_x is not None,
                "auth_present": got_auth is not None,
                "x_len": 0 if got_x is None else len(str(got_x)),
                "auth_len": 0 if got_auth is None else len(str(got_auth)),
                "server_secret_len": len(WEBHOOK_SECRET),
                "server_secret_masked": mask_secret(WEBHOOK_SECRET),
                "x_masked": None if got_x is None else mask_secret(str(got_x)),
                "auth_masked": None if got_auth is None else mask_secret(str(got_auth)),
                "headers_keys_sample": sorted(list(headers_dict.keys()))[:25],
            })

        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    # ------------------------------------------------------------
    # Parse payload (we keep it flexible; Helius can send different shapes)
    # ------------------------------------------------------------
    try:
        payload = await request.json()
    except Exception:
        counters["skipped_bad_payload"] += 1
        push_event({"ts": now_ts(), "kind": "webhook_bad_payload"})
        return JSONResponse({"ok": True}, status_code=200)

    # Optional: forced exits check on every webhook tick
    maybe_forced_exit()

    # ------------------------------------------------------------
    # Minimal webhook "OK" event (so you can confirm auth works)
    # ------------------------------------------------------------
    payload_type = "dict"
    keys = None
    tracked_wallets_count = len(TRACKED_WALLETS)

    if isinstance(payload, list):
        payload_type = "list"
        keys = None
    elif isinstance(payload, dict):
        payload_type = "dict"
        keys = list(payload.keys())[:25]

    push_event({
        "ts": now_ts(),
        "kind": "webhook_ok",
        "payload_type": payload_type,
        "keys": keys,
        "tracked_wallets_count": tracked_wallets_count,
    })

    # ------------------------------------------------------------
    # NOTE:
    # We are NOT executing buys/sells yet because we haven't locked down
    # the exact event schema you want to trade on.
    # This keeps the bot safe while you verify webhooks reliably arrive.
    # ------------------------------------------------------------

    return JSONResponse({"ok": True}, status_code=200)
