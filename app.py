import os
import time
import json
import hmac
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


# -----------------------------
# Helpers: env parsing
# -----------------------------
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v.strip()


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = v.strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def parse_wallets(raw: str) -> List[str]:
    if not raw:
        return []
    # allow comma / space / newline separated
    parts = []
    for chunk in raw.replace("\n", ",").replace(" ", ",").split(","):
        c = chunk.strip()
        if c:
            parts.append(c)
    # de-dupe, preserve order
    seen = set()
    out = []
    for w in parts:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def mask_secret(s: Optional[str]) -> str:
    if not s:
        return ""
    if len(s) <= 4:
        return "*" * len(s)
    return s[:2] + ("*" * (len(s) - 4)) + s[-2:]


def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def extract_secret_from_headers(headers) -> Optional[str]:
    # Prefer explicit x-webhook-secret header
    x = headers.get("x-webhook-secret")
    if x:
        return x.strip()

    # Some systems send Authorization: Bearer <secret>
    auth = headers.get("authorization")
    if auth:
        a = auth.strip()
        if a.lower().startswith("bearer "):
            return a.split(" ", 1)[1].strip()
        return a
    return None


# -----------------------------
# Config (Render env vars)
# -----------------------------
DEBUG_WEBHOOK = env_bool("DEBUG_WEBHOOK", True)

WEBHOOK_SECRET = env_str("WEBHOOK_SECRET", "")
WEBHOOK_PATH_TOKEN = env_str("WEBHOOK_PATH_TOKEN", "")

TRACKED_WALLETS = parse_wallets(env_str("TRACKED_WALLETS", ""))

# Paper trading knobs
SOL_PRICE_USD = env_float("SOL_PRICE_USD", 100.0)

START_CASH_USD = env_float("START_CASH_USD", 800.0)
MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 25.0)

# Profit split: reserve grows from sells (only). Cash also grows.
RESERVE_PCT = env_float("RESERVE_PCT", 0.60)

# Time-based forced exit
HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)

EVENTS_MAX = 500


# -----------------------------
# State (in-memory)
# -----------------------------
state: Dict[str, Any] = {
    "cash_usd": float(START_CASH_USD),
    "reserve_cash_usd": 0.0,
    "positions": {},  # symbol -> position dict
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
        del events[: len(events) - EVENTS_MAX]


def add_recent_trade(t: Dict[str, Any]) -> None:
    state["recent_trades"].append(t)
    if len(state["recent_trades"]) > 50:
        del state["recent_trades"][: len(state["recent_trades"]) - 50]


# -----------------------------
# Paper trading engine
# -----------------------------
def get_position(symbol: str) -> Optional[Dict[str, Any]]:
    return state["positions"].get(symbol)


def paper_buy(symbol: str, usd: float, reason: str, meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    usd = float(max(0.0, usd))
    if usd <= 0:
        return None

    cash = float(state["cash_usd"])
    if (cash - usd) < float(MIN_CASH_LEFT_USD):
        state["counters"]["skipped_low_cash"] += 1
        push_event(
            {
                "ts": int(time.time()),
                "kind": "skip_low_cash",
                "cash": round(cash, 6),
                "want": round(usd, 6),
            }
        )
        return None

    price = float(SOL_PRICE_USD)  # placeholder (real token pricing comes next)
    qty = 0.0 if price <= 0 else usd / price

    pos = get_position(symbol)
    now = int(time.time())

    if not pos:
        pos = {
            "symbol": symbol,
            "qty": 0.0,
            "cost_usd": 0.0,
            "avg_px": 0.0,
            "opened_ts": now,
            "last_buy_ts": now,
        }
        state["positions"][symbol] = pos

    # Update position avg cost
    new_cost = float(pos["cost_usd"]) + usd
    new_qty = float(pos["qty"]) + qty
    new_avg = 0.0 if new_qty <= 0 else new_cost / new_qty

    pos["qty"] = round(new_qty, 12)
    pos["cost_usd"] = round(new_cost, 6)
    pos["avg_px"] = round(new_avg, 12)
    pos["last_buy_ts"] = now

    state["cash_usd"] = round(cash - usd, 6)
    state["counters"]["buys"] += 1
    state["trades_count"] += 1

    evt = {
        "ts": now,
        "kind": "paper_buy",
        "symbol": symbol,
        "usd": round(usd, 6),
        "qty": round(qty, 12),
        "cash_after": round(float(state["cash_usd"]), 6),
        "reason": reason,
        "meta": meta or {},
    }
    push_event(evt)
    add_recent_trade(evt)
    return evt


def paper_sell(symbol: str, qty: float, reason: str, meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    pos = get_position(symbol)
    if not pos:
        return None

    qty = float(max(0.0, qty))
    if qty <= 0:
        return None

    held = float(pos["qty"])
    if held <= 0:
        return None

    sell_qty = min(qty, held)

    # Placeholder pricing (real token pricing comes next)
    price = float(SOL_PRICE_USD)
    proceeds = sell_qty * price

    # Cost basis proportionally
    cost_usd = float(pos["cost_usd"])
    if held > 0:
        cost_for_sold = cost_usd * (sell_qty / held)
    else:
        cost_for_sold = 0.0

    pnl = proceeds - cost_for_sold

    # Update position
    new_qty = held - sell_qty
    new_cost = max(0.0, cost_usd - cost_for_sold)

    pos["qty"] = round(new_qty, 12)
    pos["cost_usd"] = round(new_cost, 6)
    if new_qty <= 0:
        pos["avg_px"] = 0.0

    # Split *profit* only into reserve; principal always returns to cash.
    cash = float(state["cash_usd"])
    reserve = float(state["reserve_cash_usd"])

    reserve_add = 0.0
    cash_add = proceeds

    if pnl > 0:
        reserve_add = pnl * float(RESERVE_PCT)
        cash_add = proceeds - reserve_add

    state["reserve_cash_usd"] = round(reserve + reserve_add, 6)
    state["cash_usd"] = round(cash + cash_add, 6)

    state["counters"]["sells"] += 1
    state["trades_count"] += 1

    now = int(time.time())
    evt = {
        "ts": now,
        "kind": "paper_sell",
        "symbol": symbol,
        "qty": round(sell_qty, 12),
        "proceeds_usd": round(proceeds, 6),
        "pnl_usd": round(pnl, 6),
        "reserve_add": round(reserve_add, 6),
        "cash_after": round(float(state["cash_usd"]), 6),
        "reserve_after": round(float(state["reserve_cash_usd"]), 6),
        "reason": reason,
        "meta": meta or {},
    }
    push_event(evt)
    add_recent_trade(evt)

    # Clean up empty positions
    if float(pos["qty"]) <= 0:
        try:
            del state["positions"][symbol]
        except Exception:
            pass

    return evt


def maybe_force_exit() -> None:
    # Sell positions if held longer than HOLD_MAX_SECONDS
    now = int(time.time())
    to_sell = []
    for sym, pos in list(state["positions"].items()):
        opened = int(pos.get("opened_ts") or now)
        age = now - opened
        if age >= int(HOLD_MAX_SECONDS) and float(pos.get("qty") or 0) > 0:
            to_sell.append((sym, float(pos["qty"]), age))

    for sym, qty, age in to_sell:
        # Use fallback multiple of entry price (approx)
        # Here: we simulate by setting SOL_PRICE_USD * multi for this sale.
        global SOL_PRICE_USD
        entry_avg = float(state["positions"][sym].get("avg_px") or SOL_PRICE_USD)
        old_price = SOL_PRICE_USD
        SOL_PRICE_USD = max(0.000001, entry_avg * float(FORCED_EXIT_FALLBACK_MULTI))
        paper_sell(
            sym,
            qty=qty,
            reason="time_exit",
            meta={"hold_max_seconds": int(HOLD_MAX_SECONDS), "age_seconds": int(age)},
        )
        SOL_PRICE_USD = old_price
        state["counters"]["forced_exits"] += 1


# -----------------------------
# Matching (simple heuristic)
# -----------------------------
def extract_fee_payer(tx: Dict[str, Any]) -> Optional[str]:
    # Helius enhanced tx often has "feePayer"
    fp = tx.get("feePayer")
    if isinstance(fp, str) and fp:
        return fp
    # sometimes nested:
    ap = tx.get("accountData")
    if isinstance(ap, list) and ap:
        # no perfect guess; return None
        return None
    return None


def is_matched_tx(tx: Dict[str, Any]) -> bool:
    if not TRACKED_WALLETS:
        return False
    fp = extract_fee_payer(tx)
    if fp and fp in TRACKED_WALLETS:
        return True
    return False


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/events")
def get_events():
    return {"count": len(events), "events": events}


@app.get("/paper/state")
def paper_state():
    return {
        "cash_usd": float(state["cash_usd"]),
        "reserve_cash_usd": float(state["reserve_cash_usd"]),
        "positions": state["positions"],
        "trades_count": int(state["trades_count"]),
        "counters": state["counters"],
        "started_at": int(state["started_at"]),
        "config": {
            "SOL_PRICE_USD": float(SOL_PRICE_USD),
            "START_CASH_USD": float(START_CASH_USD),
            "MAX_BUY_USD": float(MAX_BUY_USD),
            "MIN_CASH_LEFT_USD": float(MIN_CASH_LEFT_USD),
            "RESERVE_PCT": float(RESERVE_PCT),
            "HOLD_MAX_SECONDS": int(HOLD_MAX_SECONDS),
            "FORCED_EXIT_FALLBACK_MULTI": float(FORCED_EXIT_FALLBACK_MULTI),
            "TRACKED_WALLETS_COUNT": len(TRACKED_WALLETS),
            "DEBUG_WEBHOOK": bool(DEBUG_WEBHOOK),
            "WEBHOOK_PATH_TOKEN_SET": bool(WEBHOOK_PATH_TOKEN),
            "WEBHOOK_SECRET_SET": bool(WEBHOOK_SECRET),
        },
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

    # ✅ NEW PATCH: log ONE raw payload sample (so we can see what Helius sends)
    # This is intentionally lightweight: dict payload logs itself; list payload logs only first element.
    try:
        sample = payload if isinstance(payload, dict) else (payload[:1] if isinstance(payload, list) else str(type(payload)))
        push_event({"ts": int(time.time()), "kind": "raw_payload_sample", "payload_type": payload_type, "sample": sample})
    except Exception:
        push_event({"ts": int(time.time()), "kind": "raw_payload_sample_failed"})

    # 3) Basic webhook OK event
    matched = 0
    tx_type = None
    signature = None

    # For Helius "enhanced" webhooks, payload is typically a list of tx objects
    if isinstance(payload, list) and payload:
        first = payload[0] if isinstance(payload[0], dict) else None
        if first:
            tx_type = first.get("type")
            signature = first.get("signature")

        for item in payload:
            if not isinstance(item, dict):
                continue
            if is_matched_tx(item):
                matched = 1
                tx_type = item.get("type") or tx_type
                signature = item.get("signature") or signature

                # On match, do a paper buy (placeholder until we extract real swap pricing)
                paper_buy(
                    symbol="SOL",
                    usd=min(float(MAX_BUY_USD), 25.0),
                    reason="matched_helius_event",
                    meta={"signature": signature, "type": tx_type},
                )

    push_event(
        {
            "ts": int(time.time()),
            "kind": "webhook_ok",
            "payload_type": payload_type,
            "matched": matched,
            "tracked_wallets_count": len(TRACKED_WALLETS),
            "signature": signature,
            "type": tx_type,
        }
    )

    # 4) Forced exits happen opportunistically on incoming webhook traffic
    maybe_force_exit()

    return {"ok": True}
