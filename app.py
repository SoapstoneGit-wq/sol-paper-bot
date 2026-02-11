import os
import time
import json
import hmac
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# -------------------------
# Config (env vars)
# -------------------------
def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)

DEBUG_WEBHOOK = env_bool("DEBUG_WEBHOOK", True)

WEBHOOK_SECRET = (os.getenv("WEBHOOK_SECRET") or "").strip()
WEBHOOK_PATH_TOKEN = (os.getenv("WEBHOOK_PATH_TOKEN") or "").strip()

START_CASH_USD = env_float("START_CASH_USD", 500.0)
MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 100.0)

RESERVE_PCT = env_float("RESERVE_PCT", 0.60)
TRADABLE_PCT = env_float("TRADABLE_PCT", 0.40)

SOL_PRICE_USD = env_float("SOL_PRICE_USD", 115.0)

HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)  # 15 min
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.5)  # sell at 50% of entry as placeholder

TRACKED_WALLET = (os.getenv("TRACKED_WALLET") or "").strip()
TRACKED_WALLETS: List[str] = []
if TRACKED_WALLET:
    # allow comma or newline separated
    raw = TRACKED_WALLET.replace("\n", ",")
    TRACKED_WALLETS = [w.strip() for w in raw.split(",") if w.strip()]

# What tx types we consider actionable (Helius enhanced has "type": "SWAP", "TRANSFER", etc.)
ALLOWED_TYPES = set(["SWAP", "TRANSFER"])

# -------------------------
# In-memory state
# -------------------------
EVENTS_MAX = 500

state: Dict[str, Any] = {
    "cash_usd": round(START_CASH_USD * TRADABLE_PCT, 6),
    "reserve_cash_usd": round(START_CASH_USD * RESERVE_PCT, 6),
    "positions": {},  # symbol -> {qty, cost_usd, entry_ts, entry_price}
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
    "started_at": int(time.time()),
    "recent_trades": [],
}

events: List[Dict[str, Any]] = []

def push_event(evt: Dict[str, Any]) -> None:
    events.append(evt)
    if len(events) > EVENTS_MAX:
        del events[0: len(events) - EVENTS_MAX]

def mask_secret(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    if len(s) <= 6:
        return "*" * len(s)
    return s[:2] + "*" * (len(s) - 4) + s[-2:]

def constant_time_equals(a: str, b: str) -> bool:
    # Use hmac.compare_digest for timing-safe compare
    try:
        return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
    except Exception:
        return False

def extract_secret_from_headers(headers) -> Optional[str]:
    # Accept either x-webhook-secret: <secret>
    x = headers.get("x-webhook-secret")
    if x:
        return x.strip()
    # Or Authorization: Bearer <secret>
    auth = headers.get("authorization")
    if auth:
        auth = auth.strip()
        if auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1].strip()
        # If someone pasted raw secret into Authorization
        return auth
    return None

def config_snapshot() -> Dict[str, Any]:
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

def add_recent_trade(tr: Dict[str, Any]) -> None:
    state["recent_trades"].append(tr)
    if len(state["recent_trades"]) > 50:
        state["recent_trades"] = state["recent_trades"][-50:]

# -------------------------
# Paper trading core
# -------------------------
def paper_buy_sol(usd: float, reason: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    try:
        usd = float(usd)
    except Exception:
        return False

    if usd <= 0:
        return False

    cash = float(state.get("cash_usd", 0.0))
    if cash - usd < float(MIN_CASH_LEFT_USD):
        state["counters"]["skipped_low_cash"] += 1
        if DEBUG_WEBHOOK:
            push_event({"ts": int(time.time()), "kind": "skip_buy_low_cash", "cash": cash, "usd": usd, "min_left": MIN_CASH_LEFT_USD})
        return False

    px = float(SOL_PRICE_USD)
    if px <= 0:
        return False
    qty = usd / px

    # Apply state mutation
    state["cash_usd"] = round(cash - usd, 6)

    pos = state["positions"].get("SOL", {"qty": 0.0, "cost_usd": 0.0, "entry_ts": int(time.time()), "entry_price": px})
    pos["qty"] = round(float(pos["qty"]) + float(qty), 12)
    pos["cost_usd"] = round(float(pos["cost_usd"]) + float(usd), 6)
    # Keep the earliest entry_ts (for hold timer) if already holding
    if "entry_ts" not in pos:
        pos["entry_ts"] = int(time.time())
    if "entry_price" not in pos:
        pos["entry_price"] = px
    state["positions"]["SOL"] = pos

    state["trades_count"] = int(state.get("trades_count", 0)) + 1
    state["counters"]["buys"] = int(state["counters"].get("buys", 0)) + 1

    evt = {
        "ts": int(time.time()),
        "kind": "paper_buy_filled",
        "symbol": "SOL",
        "usd": usd,
        "qty": qty,
        "price": px,
        "cash_after": state["cash_usd"],
        "reason": reason,
    }
    if meta:
        evt["meta"] = meta
    push_event(evt)
    add_recent_trade(evt)
    return True

def paper_sell_sol_all(reason: str, forced: bool = False) -> bool:
    pos = state["positions"].get("SOL")
    if not pos:
        return False

    qty = float(pos.get("qty", 0.0))
    if qty <= 0:
        return False

    entry_price = float(pos.get("entry_price", SOL_PRICE_USD))
    # placeholder sell price: either current SOL_PRICE_USD or forced fallback multiple
    if forced:
        sell_price = max(0.0, entry_price * float(FORCED_EXIT_FALLBACK_MULTI))
    else:
        sell_price = float(SOL_PRICE_USD)

    usd_out = qty * sell_price

    cash = float(state.get("cash_usd", 0.0))
    state["cash_usd"] = round(cash + usd_out, 6)

    # clear position
    del state["positions"]["SOL"]

    state["trades_count"] = int(state.get("trades_count", 0)) + 1
    state["counters"]["sells"] = int(state["counters"].get("sells", 0)) + 1
    if forced:
        state["counters"]["forced_exits"] = int(state["counters"].get("forced_exits", 0)) + 1

    evt = {
        "ts": int(time.time()),
        "kind": "paper_sell_filled" if not forced else "paper_forced_exit",
        "symbol": "SOL",
        "qty": qty,
        "price": sell_price,
        "usd_out": usd_out,
        "cash_after": state["cash_usd"],
        "reason": reason,
    }
    push_event(evt)
    add_recent_trade(evt)
    return True

def maybe_force_exits(now_ts: int) -> None:
    # If holding SOL longer than HOLD_MAX_SECONDS, force exit.
    pos = state["positions"].get("SOL")
    if not pos:
        return
    entry_ts = int(pos.get("entry_ts", now_ts))
    held = now_ts - entry_ts
    if held >= int(HOLD_MAX_SECONDS):
        paper_sell_sol_all(reason=f"held>{HOLD_MAX_SECONDS}s", forced=True)

# -------------------------
# Matching logic (Helius enhanced payload)
# -------------------------
def tx_type_of(obj: Dict[str, Any]) -> str:
    t = obj.get("type")
    if isinstance(t, str):
        return t.upper()
    return ""

def tx_signature_of(obj: Dict[str, Any]) -> str:
    s = obj.get("signature")
    return s if isinstance(s, str) else ""

def object_contains_any_wallet(obj: Any, wallets: List[str]) -> bool:
    """
    Very forgiving match: checks common fields AND falls back to string search.
    """
    if not wallets:
        return False

    # common top-level fields
    if isinstance(obj, dict):
        for k in ("feePayer", "feePayerAccount", "source", "destination"):
            v = obj.get(k)
            if isinstance(v, str) and v in wallets:
                return True

        # common Helius transfer arrays
        for arr_name in ("nativeTransfers", "tokenTransfers"):
            arr = obj.get(arr_name)
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict):
                        for k in ("fromUserAccount", "toUserAccount", "fromAccount", "toAccount", "owner"):
                            v = item.get(k)
                            if isinstance(v, str) and v in wallets:
                                return True

    # fallback: string search (works even if schema changes)
    try:
        blob = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        return any(w in blob for w in wallets)
    except Exception:
        return False

def should_buy_from_tx(tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Returns a meta dict if actionable, else None.
    """
    t = tx_type_of(tx)
    if t not in ALLOWED_TYPES:
        return None
    if not object_contains_any_wallet(tx, TRACKED_WALLETS):
        return None

    return {
        "type": t,
        "signature": tx_signature_of(tx),
    }

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "sol-paper-bot", "hint": "use /health /paper/state /events"}

@app.get("/health")
def health():
    return {"ok": True, "service": "sol-paper-bot"}

@app.get("/events")
def get_events():
    return {"count": len(events), "events": events}

@app.get("/paper/state")
def paper_state():
    # run force exit check here too so you can see sells even if webhooks are quiet
    maybe_force_exits(int(time.time()))
    return {
        "cash_usd": state["cash_usd"],
        "reserve_cash_usd": state["reserve_cash_usd"],
        "positions": state["positions"],
        "trades_count": state["trades_count"],
        "counters": state["counters"],
        "started_at": state["started_at"],
        "config": config_snapshot(),
        "recent_trades": state["recent_trades"],
    }

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

    # 3) Always run forced exit checks on each webhook tick
    now_ts = int(time.time())
    maybe_force_exits(now_ts)

    # 4) Decide if this webhook should trigger a paper buy
    # Helius enhanced usually posts a LIST of transactions
    txs: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        txs = [x for x in payload if isinstance(x, dict)]
    elif isinstance(payload, dict):
        # sometimes wrapped
        inner = payload.get("events") or payload.get("transactions") or payload.get("data")
        if isinstance(inner, list):
            txs = [x for x in inner if isinstance(x, dict)]
        else:
            txs = [payload]

    matched = 0
    last_meta = None

    for tx in txs:
        meta = should_buy_from_tx(tx)
        if not meta:
            continue
        matched += 1
        last_meta = meta
        # Execute a REAL paper buy (this is the critical part that changes cash/positions)
        usd_to_spend = min(float(MAX_BUY_USD), float(state["cash_usd"]))  # safety cap
        paper_buy_sol(
            usd=usd_to_spend,
            reason="matched_helius_event",
            meta={
                "type": meta.get("type"),
                "signature": meta.get("signature"),
            },
        )
        # only 1 buy per webhook burst to keep it sane
        break

    push_event(
        {
            "ts": int(time.time()),
            "kind": "webhook_match_summary",
            "matched": matched,
            "last": last_meta,
        }
    )

    return {"ok": True, "matched": matched}
