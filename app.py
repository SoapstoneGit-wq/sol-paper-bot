import os
import time
import hmac
from typing import Any, Dict, List, Optional, Set
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

# -----------------------------
# Config (Environment Variables)
# -----------------------------
def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return float(default)
    return float(raw)

def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return int(default)
    return int(raw)

def _parse_wallets(raw: str) -> Set[str]:
    # Accept comma-separated. Trim whitespace. Ignore empties.
    # Example: "addr1,addr2, addr3"
    wallets = set()
    for part in (raw or "").split(","):
        w = part.strip()
        if w:
            wallets.add(w)
    return wallets

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
TRACKED_WALLETS_RAW = os.getenv("TRACKED_WALLET", "")
TRACKED_WALLETS: Set[str] = _parse_wallets(TRACKED_WALLETS_RAW)

START_CASH_USD = _get_float("START_CASH_USD", 500.0)

# 60% reserve / 40% tradable:
# Reserve is for forced exits / safety.
RESERVE_PCT = _get_float("RESERVE_PCT", 0.60)  # 0.60 means 60% reserve
TRADABLE_PCT = max(0.0, min(1.0, 1.0 - RESERVE_PCT))

MIN_CASH_LEFT_USD = _get_float("MIN_CASH_LEFT_USD", 100.0)
MAX_BUY_USD = _get_float("MAX_BUY_USD", 25.0)

# Forced exit (dead coin) timer
HOLD_MAX_SECONDS = _get_int("HOLD_MAX_SECONDS", 900)  # 900 seconds = 15 minutes

# If we don't know price, use a conservative fallback:
# 0.5 means we assume you get 50% back on forced exit (bad exit).
FORCED_EXIT_FALLBACK_MULTI = _get_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)

# Optional; only used if you later want to estimate USD from SOL amounts
SOL_PRICE_USD = _get_float("SOL_PRICE_USD", 115.0)

EVENTS_LIMIT = 200

# -----------------------------
# In-memory state
# -----------------------------
state: Dict[str, Any] = {
    "started_at": int(time.time()),
    "counters": {
        "webhooks_received": 0,
        "webhooks_unauthorized": 0,
        "buys": 0,
        "sells": 0,
        "forced_exits": 0,
        "skipped_low_cash": 0,
        "skipped_no_secret": 0,
        "skipped_no_signal": 0,
    },
    "events": [],  # last N authorized webhook payload summaries
    "positions": {},  # mint -> position
    "trades": [],  # trade log
}

# Split starting cash into reserve + tradable
# Tradable is what you can buy with; reserve is protected cash.
state["reserve_cash_usd"] = round(START_CASH_USD * RESERVE_PCT, 4)
state["cash_usd"] = round(START_CASH_USD * TRADABLE_PCT, 4)


# -----------------------------
# Helpers
# -----------------------------
def _constant_time_equal(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))

def _auth_ok(x_webhook_secret: Optional[str], authorization: Optional[str]) -> bool:
    """
    Accept either:
      - Header: x-webhook-secret: <secret>
      - Header: Authorization: Bearer <secret>
    """
    if not WEBHOOK_SECRET:
        # If you forgot to set WEBHOOK_SECRET, block everything.
        state["counters"]["skipped_no_secret"] += 1
        return False

    if x_webhook_secret and _constant_time_equal(x_webhook_secret.strip(), WEBHOOK_SECRET.strip()):
        return True

    if authorization:
        # allow "Bearer <secret>"
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
            if _constant_time_equal(token, WEBHOOK_SECRET.strip()):
                return True

    return False

def _now() -> int:
    return int(time.time())

def _add_event(item: Dict[str, Any]) -> None:
    ev = state["events"]
    ev.append(item)
    if len(ev) > EVENTS_LIMIT:
        del ev[: len(ev) - EVENTS_LIMIT]

def _trade_log(entry: Dict[str, Any]) -> None:
    state["trades"].append(entry)
    # keep it from growing forever
    if len(state["trades"]) > 500:
        del state["trades"][:200]

def _force_exit_if_needed() -> None:
    """Close any position older than HOLD_MAX_SECONDS using fallback."""
    now = _now()
    to_close = []
    for mint, pos in state["positions"].items():
        age = now - pos["opened_at"]
        if age >= HOLD_MAX_SECONDS:
            to_close.append(mint)

    for mint in to_close:
        pos = state["positions"].pop(mint, None)
        if not pos:
            continue

        invested = float(pos["invested_usd"])
        returned = round(invested * FORCED_EXIT_FALLBACK_MULTI, 4)

        # Returned funds go to tradable cash (but you still keep reserve intact)
        state["cash_usd"] = round(state["cash_usd"] + returned, 4)

        state["counters"]["forced_exits"] += 1
        _trade_log({
            "ts": _now(),
            "type": "FORCED_EXIT",
            "mint": mint,
            "invested_usd": invested,
            "returned_usd": returned,
            "held_seconds": now - pos["opened_at"],
        })

def _available_to_buy() -> float:
    # Always keep MIN_CASH_LEFT_USD in tradable cash untouched
    return max(0.0, float(state["cash_usd"]) - float(MIN_CASH_LEFT_USD))

def _paper_buy(mint: str, reason: str) -> None:
    _force_exit_if_needed()

    if mint in state["positions"]:
        # Already holding
        return

    avail = _available_to_buy()
    if avail <= 0:
        state["counters"]["skipped_low_cash"] += 1
        return

    spend = round(min(MAX_BUY_USD, avail), 4)
    if spend <= 0:
        state["counters"]["skipped_low_cash"] += 1
        return

    state["cash_usd"] = round(float(state["cash_usd"]) - spend, 4)
    state["positions"][mint] = {
        "mint": mint,
        "invested_usd": spend,
        "opened_at": _now(),
        "reason": reason,
    }
    state["counters"]["buys"] += 1
    _trade_log({
        "ts": _now(),
        "type": "BUY",
        "mint": mint,
        "usd": spend,
        "cash_after": state["cash_usd"],
        "reserve_cash": state["reserve_cash_usd"],
        "reason": reason,
    })

def _paper_sell(mint: str, reason: str) -> None:
    _force_exit_if_needed()

    pos = state["positions"].pop(mint, None)
    if not pos:
        return

    invested = float(pos["invested_usd"])

    # For now, we assume break-even on normal sell signal (because we don't have price feed here).
    # You can later upgrade to compute from on-chain SOL flows or a price API.
    returned = invested

    # Returned funds go to tradable cash
    state["cash_usd"] = round(float(state["cash_usd"]) + returned, 4)

    state["counters"]["sells"] += 1
    _trade_log({
        "ts": _now(),
        "type": "SELL",
        "mint": mint,
        "invested_usd": invested,
        "returned_usd": returned,
        "cash_after": state["cash_usd"],
        "reserve_cash": state["reserve_cash_usd"],
        "reason": reason,
    })

def _extract_signals(payload: Any) -> List[Dict[str, str]]:
    """
    Helius "enhanced" webhooks often send a list of tx objects.
    We look for tokenTransfers where tracked wallet is sender/receiver.
    BUY signal: toUserAccount is tracked (mint not SOL)
    SELL signal: fromUserAccount is tracked (mint not SOL)
    """
    signals: List[Dict[str, str]] = []

    if not isinstance(payload, list):
        return signals

    for tx in payload:
        if not isinstance(tx, dict):
            continue

        token_transfers = tx.get("tokenTransfers") or []
        if not isinstance(token_transfers, list):
            continue

        tx_source = tx.get("source") or ""
        tx_type = tx.get("type") or ""

        for tt in token_transfers:
            if not isinstance(tt, dict):
                continue
            mint = tt.get("mint") or ""
            if not mint:
                continue

            # Ignore SOL mint
            if mint == "So11111111111111111111111111111111111111112":
                continue

            to_user = (tt.get("toUserAccount") or "").strip()
            from_user = (tt.get("fromUserAccount") or "").strip()

            if to_user in TRACKED_WALLETS:
                signals.append({"side": "BUY", "mint": mint, "why": f"{tx_source}:{tx_type}:to_tracked"})
            elif from_user in TRACKED_WALLETS:
                signals.append({"side": "SELL", "mint": mint, "why": f"{tx_source}:{tx_type}:from_tracked"})

    return signals


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "service": "sol-paper-bot"}

@app.get("/events")
def events():
    return {"count": len(state["events"]), "events": state["events"]}

@app.get("/paper/state")
def paper_state():
    _force_exit_if_needed()
    return {
        "cash_usd": state["cash_usd"],
        "reserve_cash_usd": state["reserve_cash_usd"],
        "positions": state["positions"],
        "trades_count": len(state["trades"]),
        "counters": state["counters"],
        "started_at": state["started_at"],
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
        },
        "recent_trades": state["trades"][-20:],
    }

@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
):
    # Auth
    if not _auth_ok(x_webhook_secret, authorization):
        state["counters"]["webhooks_unauthorized"] += 1

        # Helpful debug (does NOT reveal secret)
        # Shows which headers we saw (presence only).
        return JSONResponse(
            status_code=401,
            content={
                "ok": False,
                "error": "Unauthorized",
                "saw_x_webhook_secret": x_webhook_secret is not None,
                "saw_authorization": authorization is not None,
            },
        )

    state["counters"]["webhooks_received"] += 1

    body = await request.json()

    # Helius sometimes wraps as {"payload":[...]} or directly [...]
    payload = body.get("payload") if isinstance(body, dict) else body

    signals = _extract_signals(payload)

    # Store event summary (authorized only)
    _add_event({
        "received_at": _now(),
        "source": "helius",
        "signals": signals[:10],  # keep it small
    })

    if not signals:
        state["counters"]["skipped_no_signal"] += 1
        _force_exit_if_needed()
        return {"ok": True, "signals": 0}

    # Apply signals
    for s in signals:
        side = s["side"]
        mint = s["mint"]
        why = s["why"]

        if side == "BUY":
            _paper_buy(mint, reason=why)
        elif side == "SELL":
            _paper_sell(mint, reason=why)

    _force_exit_if_needed()
    return {"ok": True, "signals": len(signals)}
