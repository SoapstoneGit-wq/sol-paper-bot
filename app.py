import os
import time
import hmac
from typing import Any, Dict, List, Optional, Union, Set, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# -------------------------
# Env helpers
# -------------------------
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v)

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
        return int(v)
    except Exception:
        return int(default)

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

# -------------------------
# Config
# -------------------------
DEBUG_WEBHOOK = env_bool("DEBUG_WEBHOOK", True)

WEBHOOK_SECRET = env_str("WEBHOOK_SECRET", "")
WEBHOOK_PATH_TOKEN = env_str("WEBHOOK_PATH_TOKEN", "")

START_CASH_USD = env_float("START_CASH_USD", 500.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 100.0)
MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)

SOL_PRICE_USD = env_float("SOL_PRICE_USD", 100.0)

RESERVE_PCT = env_float("RESERVE_PCT", 0.6)
TRADABLE_PCT = env_float("TRADABLE_PCT", 0.4)

HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.5)

# TRACKED_WALLETS: comma-separated list
TRACKED_WALLETS_RAW = env_str("TRACKED_WALLETS", "").strip()
TRACKED_WALLETS: List[str] = []
if TRACKED_WALLETS_RAW:
    TRACKED_WALLETS = [w.strip() for w in TRACKED_WALLETS_RAW.split(",") if w.strip()]

TRACKED_SET: Set[str] = set(TRACKED_WALLETS)

# -------------------------
# State + event log
# -------------------------
EVENTS: List[Dict[str, Any]] = []
MAX_EVENTS = 500

state: Dict[str, Any] = {
    "cash_usd": float(START_CASH_USD * TRADABLE_PCT),
    "reserve_cash_usd": float(START_CASH_USD * RESERVE_PCT),
    "positions": {},  # symbol -> dict
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
}

def push_event(e: Dict[str, Any]) -> None:
    EVENTS.append(e)
    if len(EVENTS) > MAX_EVENTS:
        del EVENTS[: len(EVENTS) - MAX_EVENTS]

def mask_secret(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s)
    if len(s) <= 6:
        return "*" * len(s)
    return s[:2] + "*" * (len(s) - 4) + s[-2:]

def constant_time_equals(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))

def extract_secret_from_headers(headers) -> Optional[str]:
    """
    Handles:
      - x-webhook-secret: <value>
      - authorization: <value>
      - authorization: Bearer <value>
    """
    x = headers.get("x-webhook-secret")
    if x:
        return x.strip()

    auth = headers.get("authorization")
    if not auth:
        return None
    auth = auth.strip()

    # Common webhook provider behavior:
    # Authorization: Bearer <token>
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()

    return auth

# -------------------------
# Paper trading primitives
# -------------------------
def paper_buy(symbol: str, usd: float, reason: str, meta: Dict[str, Any]) -> bool:
    usd = float(max(0.0, usd))
    if usd <= 0:
        return False

    # Enforce cash floor
    if state["cash_usd"] - usd < MIN_CASH_LEFT_USD:
        state["counters"]["skipped_low_cash"] += 1
        push_event({"ts": int(time.time()), "kind": "paper_buy_skipped_low_cash", "usd": usd})
        return False

    px = float(SOL_PRICE_USD) if symbol.upper() == "SOL" else 1.0
    qty = usd / px if px > 0 else 0.0
    if qty <= 0:
        return False

    # Mutate state (THIS is what makes cash decrease)
    state["cash_usd"] = float(state["cash_usd"] - usd)
    pos = state["positions"].get(symbol, {"qty": 0.0, "avg_px": 0.0, "cost_usd": 0.0})
    old_qty = float(pos["qty"])
    old_cost = float(pos["cost_usd"])

    new_qty = old_qty + qty
    new_cost = old_cost + usd
    avg_px = (new_cost / new_qty) if new_qty > 0 else px

    pos.update({"qty": new_qty, "avg_px": avg_px, "cost_usd": new_cost, "opened_ts": int(time.time())})
    state["positions"][symbol] = pos

    state["counters"]["buys"] += 1
    state["trades_count"] += 1

    push_event(
        {
            "ts": int(time.time()),
            "kind": "paper_buy_executed",
            "symbol": symbol,
            "usd": usd,
            "px": px,
            "qty": qty,
            "cash_usd_after": state["cash_usd"],
            "reason": reason,
            "meta": meta,
        }
    )
    return True

# -------------------------
# Webhook matching helpers
# -------------------------
def pull_wallet_candidates(tx: Dict[str, Any]) -> Set[str]:
    """
    Extracts likely wallet fields from Helius enhanced payload objects.
    This isn't perfect, but it reliably catches feePayer + transfer participants.
    """
    cands: Set[str] = set()

    for k in ("feePayer", "source", "destination", "owner", "wallet", "account", "authority"):
        v = tx.get(k)
        if isinstance(v, str) and v:
            cands.add(v)

    for list_key in ("nativeTransfers", "tokenTransfers"):
        items = tx.get(list_key)
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                for k in ("fromUserAccount", "toUserAccount", "from", "to", "source", "destination", "owner"):
                    v = it.get(k)
                    if isinstance(v, str) and v:
                        cands.add(v)

    return cands

def get_sig_and_type(tx: Dict[str, Any]) -> Tuple[str, str]:
    sig = (
        tx.get("signature")
        or tx.get("transactionSignature")
        or tx.get("txSignature")
        or ""
    )
    t = tx.get("type") or tx.get("transactionType") or ""
    return str(sig), str(t)

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"ok": True, "service": "sol-paper-bot"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/events")
def events():
    return {"count": len(EVENTS), "events": EVENTS}

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
    # 0) Path token gate (stops scanners)
    if not WEBHOOK_PATH_TOKEN:
        state["counters"]["skipped_bad_path"] += 1
        push_event({"ts": int(time.time()), "kind": "server_misconfig", "reason": "WEBHOOK_PATH_TOKEN missing"})
        return JSONResponse({"error": "WEBHOOK_PATH_TOKEN missing on server"}, status_code=500)

    if token != WEBHOOK_PATH_TOKEN:
        state["counters"]["skipped_bad_path"] += 1
        if DEBUG_WEBHOOK:
            push_event({"ts": int(time.time()), "kind": "bad_path", "got": token})
        # pretend not found to avoid leaking endpoint
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
    matched = 0

    # 3) Attempt to simulate buys from matched tracked wallets
    tx_list: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        tx_list = [x for x in payload if isinstance(x, dict)]
    elif isinstance(payload, dict):
        # Some providers wrap transactions; try common keys
        for k in ("transactions", "data", "events"):
            if isinstance(payload.get(k), list):
                tx_list = [x for x in payload[k] if isinstance(x, dict)]
                break

    for tx in tx_list:
        sig, tx_type = get_sig_and_type(tx)
        wallet_cands = pull_wallet_candidates(tx)
        hit = next((w for w in wallet_cands if w in TRACKED_SET), None)
        if not hit:
            continue

        matched += 1

        # Simulate a buy (you can later replace this with smarter logic)
        paper_buy(
            symbol="SOL",
            usd=min(float(MAX_BUY_USD), 25.0),
            reason="matched_helius_event",
            meta={"wallet": hit, "signature": sig, "type": tx_type},
        )

    push_event(
        {
            "ts": int(time.time()),
            "kind": "webhook_ok",
            "payload_type": payload_type,
            "tracked_wallets_count": len(TRACKED_WALLETS),
            "matched": matched,
        }
    )
    return {"ok": True}
