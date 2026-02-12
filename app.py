import os
import time
import hmac
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# ----------------------------
# Config helpers
# ----------------------------
def env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()

def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, str(default)))
    except Exception:
        return default

def env_int(name: str, default: int) -> int:
    try:
        return int(env_str(name, str(default)))
    except Exception:
        return default

def env_bool(name: str, default: bool = False) -> bool:
    val = env_str(name, "true" if default else "false").lower()
    return val in ("1", "true", "yes", "y", "on")

def parse_csv_set(s: str) -> set:
    if not s:
        return set()
    return set([x.strip() for x in s.split(",") if x.strip()])

def mask_secret(s: Optional[str]) -> str:
    if not s:
        return ""
    if len(s) <= 6:
        return "*" * len(s)
    return s[:2] + "*" * (len(s) - 4) + s[-2:]


# ----------------------------
# ENV / CONFIG
# ----------------------------
WEBHOOK_PATH_TOKEN = env_str("WEBHOOK_PATH_TOKEN", "")
WEBHOOK_SECRET = env_str("WEBHOOK_SECRET", "")

DEBUG_WEBHOOK = env_bool("DEBUG_WEBHOOK", True)

START_CASH_USD = env_float("START_CASH_USD", 500.0)
RESERVE_PCT = env_float("RESERVE_PCT", 0.60)
TRADABLE_PCT = env_float("TRADABLE_PCT", 0.40)

MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 100.0)

SOL_PRICE_USD = env_float("SOL_PRICE_USD", 100.0)

HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)

TRACKED_WALLETS = list(parse_csv_set(env_str("TRACKED_WALLETS", "")))

MAX_EVENTS = env_int("MAX_EVENTS", 500)


# ----------------------------
# State
# ----------------------------
def init_state() -> Dict[str, Any]:
    reserve_cash = round(START_CASH_USD * RESERVE_PCT, 2)
    tradable_cash = round(START_CASH_USD * TRADABLE_PCT, 2)
    return {
        "cash_usd": tradable_cash,
        "reserve_cash_usd": reserve_cash,
        "positions": {},  # symbol -> position dict
        "trades_count": 0,
        "seen_signatures": set(),  # to avoid double-counting
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
        "events": [],
        "started_at": int(time.time()),
    }

state = init_state()


def push_event(evt: Dict[str, Any]) -> None:
    state["events"].append(evt)
    if len(state["events"]) > MAX_EVENTS:
        state["events"] = state["events"][-MAX_EVENTS:]


def constant_time_equals(a: str, b: str) -> bool:
    try:
        return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))
    except Exception:
        return False


def extract_secret_from_headers(headers) -> Optional[str]:
    # Primary: x-webhook-secret
    x = headers.get("x-webhook-secret")
    if x:
        return x.strip()

    # Fallback: Authorization (Helius sometimes uses this)
    auth = headers.get("authorization")
    if not auth:
        return None
    auth = auth.strip()

    # Accept "Bearer <token>" or raw token
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return auth


# ----------------------------
# Paper trading functions
# ----------------------------
def paper_buy(symbol: str, usd: float, reason: str, meta: Dict[str, Any]) -> bool:
    usd = round(float(usd), 2)
    if usd <= 0:
        return False

    if state["cash_usd"] - usd < MIN_CASH_LEFT_USD:
        state["counters"]["skipped_low_cash"] += 1
        push_event({"ts": int(time.time()), "kind": "skip_low_cash", "cash": state["cash_usd"], "want": usd})
        return False

    px = SOL_PRICE_USD if symbol == "SOL" else SOL_PRICE_USD
    qty = round(usd / px, 8)

    pos = state["positions"].get(symbol)
    now = int(time.time())

    if pos:
        # add to existing position (weighted avg)
        new_qty = pos["qty"] + qty
        new_cost = pos["cost_usd"] + usd
        pos["qty"] = new_qty
        pos["cost_usd"] = new_cost
        pos["avg_px"] = round(new_cost / new_qty, 6)
        pos["last_buy_ts"] = now
    else:
        state["positions"][symbol] = {
            "symbol": symbol,
            "qty": qty,
            "cost_usd": usd,
            "avg_px": round(usd / qty, 6) if qty > 0 else px,
            "opened_ts": now,
            "last_buy_ts": now,
        }

    state["cash_usd"] = round(state["cash_usd"] - usd, 2)
    state["counters"]["buys"] += 1
    state["trades_count"] += 1

    push_event(
        {
            "ts": now,
            "kind": "paper_buy",
            "symbol": symbol,
            "usd": usd,
            "qty": qty,
            "cash_after": state["cash_usd"],
            "reason": reason,
            "meta": meta,
        }
    )
    return True


def paper_sell(symbol: str, reason: str, meta: Dict[str, Any]) -> bool:
    pos = state["positions"].get(symbol)
    if not pos:
        return False

    now = int(time.time())
    px = SOL_PRICE_USD if symbol == "SOL" else SOL_PRICE_USD

    proceeds = round(pos["qty"] * px, 2)
    pnl = round(proceeds - pos["cost_usd"], 2)

    state["cash_usd"] = round(state["cash_usd"] + proceeds, 2)
    state["counters"]["sells"] += 1
    state["trades_count"] += 1

    push_event(
        {
            "ts": now,
            "kind": "paper_sell",
            "symbol": symbol,
            "qty": pos["qty"],
            "proceeds_usd": proceeds,
            "pnl_usd": pnl,
            "cash_after": state["cash_usd"],
            "reason": reason,
            "meta": meta,
        }
    )

    del state["positions"][symbol]
    return True


def maybe_time_exit_positions() -> None:
    now = int(time.time())
    to_sell = []
    for sym, pos in state["positions"].items():
        age = now - int(pos.get("opened_ts", now))
        if age >= HOLD_MAX_SECONDS:
            to_sell.append(sym)

    for sym in to_sell:
        state["counters"]["forced_exits"] += 1
        paper_sell(sym, reason="time_exit", meta={"hold_max_seconds": HOLD_MAX_SECONDS})


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "sol-paper-bot"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/events")
def events():
    # show newest first
    return {"count": len(state["events"]), "events": list(state["events"])}

@app.get("/paper/state")
def paper_state():
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
        "WEBHOOK_PATH_TOKEN_SET": bool(WEBHOOK_PATH_TOKEN),
        "WEBHOOK_SECRET_SET": bool(WEBHOOK_SECRET),
    }
    return {
        "cash_usd": state["cash_usd"],
        "reserve_cash_usd": state["reserve_cash_usd"],
        "positions": state["positions"],
        "trades_count": state["trades_count"],
        "counters": state["counters"],
        "started_at": state["started_at"],
        "config": cfg,
        "recent_trades": [],
    }

@app.post("/webhook/{token}")
async def webhook(token: str, req: Request):
    # 0) Path token gate
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

    # 3) Basic trade logic: buy when any tracked wallet appears in an incoming SWAP/TRANSFER
    txs: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        txs = [x for x in payload if isinstance(x, dict)]
        payload_type = "list"
    elif isinstance(payload, dict):
        # some webhooks wrap txs
        if isinstance(payload.get("data"), list):
            txs = [x for x in payload["data"] if isinstance(x, dict)]
        else:
            txs = [payload]
        payload_type = "dict"
    else:
        payload_type = str(type(payload))

    matched = 0
    for tx in txs:
        sig = tx.get("signature") or tx.get("transactionSignature") or ""
        tx_type = (tx.get("type") or tx.get("transactionType") or "").upper()

        # Pull possible wallet fields from enhanced payloads
        wallet_candidates = []
        for k in ("feePayer", "fee_payer", "wallet", "account", "source", "destination"):
            v = tx.get(k)
            if isinstance(v, str) and v:
                wallet_candidates.append(v)

        # If tx contains accounts list, add them
        accounts = tx.get("accounts")
        if isinstance(accounts, list):
            for a in accounts:
                if isinstance(a, str):
                    wallet_candidates.append(a)

        is_tracked = any(w in TRACKED_WALLETS for w in wallet_candidates)
        if not is_tracked:
            continue

        # de-dupe by signature
        if sig and sig in state["seen_signatures"]:
            continue
        if sig:
            state["seen_signatures"].add(sig)

        matched += 1

        # execute a real paper buy (this is what changes cash)
        paper_buy(
            symbol="SOL",
            usd=min(float(MAX_BUY_USD), float(state["cash_usd"])),
            reason="matched_helius_event",
            meta={"signature": sig, "type": tx_type},
        )

        push_event(
            {
                "ts": int(time.time()),
                "kind": "webhook_ok",
                "payload_type": payload_type,
                "matched": 1,
                "tracked_wallets_count": len(TRACKED_WALLETS),
            }
        )

    if DEBUG_WEBHOOK and matched == 0:
        push_event(
            {
                "ts": int(time.time()),
                "kind": "webhook_ok",
                "payload_type": payload_type,
                "matched": 0,
                "tracked_wallets_count": len(TRACKED_WALLETS),
            }
        )

    # 4) time-based exits
    maybe_time_exit_positions()

    return {"ok": True, "matched": matched}
