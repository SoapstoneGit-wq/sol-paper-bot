import os
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse


# ----------------------------
# Config helpers
# ----------------------------
def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def get_tracked_wallets() -> List[str]:
    # Accept comma-separated list (no spaces required; spaces are fine too)
    raw = os.getenv("TRACKED_WALLET", "") or ""
    wallets = [w.strip() for w in raw.split(",") if w.strip()]
    return wallets


def now_ts() -> int:
    return int(time.time())


# ----------------------------
# App + State
# ----------------------------
app = FastAPI()

START_CASH_USD = env_float("START_CASH_USD", 500.0)
MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 100.0)

# 60% reserve / 40% tradable
RESERVE_PCT = env_float("RESERVE_PCT", 0.60)
TRADABLE_PCT = 1.0 - RESERVE_PCT

# Forced exit for dead coins (seconds)
HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)  # 900 seconds = 15 minutes

# If we force-exit and don't know price, sell at this multiple of cost basis
# 0.5 = assume we get back 50% of what we spent (tunable)
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)

# Optional — only used for display/debug; paper trading logic here is USD-based
SOL_PRICE_USD = env_float("SOL_PRICE_USD", 115.0)

# Auth secret
WEBHOOK_SECRET = (os.getenv("WEBHOOK_SECRET", "") or "").strip()

# In-memory state (resets on deploy/restart)
STATE: Dict[str, Any] = {
    "started_at": now_ts(),
    "cash_usd": round(START_CASH_USD * TRADABLE_PCT, 2),
    "reserve_cash_usd": round(START_CASH_USD * RESERVE_PCT, 2),
    "positions": {},  # mint -> {qty, cost_usd, opened_at, last_update}
    "events": [],     # recent webhook payload summaries (sanitized-ish)
    "trades": [],     # recent trades
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
}


def clamp_recent(list_obj: List[Any], max_len: int = 200) -> None:
    if len(list_obj) > max_len:
        del list_obj[:-max_len]


def auth_ok(request: Request) -> Tuple[bool, str]:
    """
    Helius "Authentication Header" should be:
      x-webhook-secret: <YOUR_SECRET>

    We check header value against WEBHOOK_SECRET.
    We do NOT log the actual secret; only lengths.
    """
    expected = WEBHOOK_SECRET.strip()
    if not expected:
        # If secret not set in env, fail closed (safer)
        return False, "WEBHOOK_SECRET not set"

    provided = request.headers.get("x-webhook-secret", "")
    if provided is None:
        provided = ""
    provided = provided.strip()

    if not provided:
        return False, "missing header x-webhook-secret"

    if provided != expected:
        # Safe debug info (no secret leakage)
        return False, f"secret mismatch (provided_len={len(provided)} expected_len={len(expected)})"

    return True, "ok"


def extract_swaps(payload: Any) -> List[Dict[str, Any]]:
    """
    Helius enhanced webhook commonly sends payload as a list of tx objects.
    We look for tx with type == "SWAP".
    """
    swaps: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        for tx in payload:
            if isinstance(tx, dict) and tx.get("type") == "SWAP":
                swaps.append(tx)
    elif isinstance(payload, dict):
        # Sometimes payload is wrapped
        maybe = payload.get("payload")
        if isinstance(maybe, list):
            for tx in maybe:
                if isinstance(tx, dict) and tx.get("type") == "SWAP":
                    swaps.append(tx)
    return swaps


def classify_swap_for_tracked_wallets(tx: Dict[str, Any], tracked: List[str]) -> Optional[Dict[str, Any]]:
    """
    Very lightweight classification:
      - If tokenTransfers shows a mint transferred TO a tracked wallet => treat as BUY signal for that mint
      - If tokenTransfers shows a mint transferred FROM a tracked wallet => treat as SELL signal for that mint

    We ignore SOL mint (So111...) for "coin" mint detection.
    """
    transfers = tx.get("tokenTransfers") or []
    if not isinstance(transfers, list) or not transfers:
        return None

    # Find first non-SOL mint transfer involving tracked wallets
    for t in transfers:
        if not isinstance(t, dict):
            continue
        mint = t.get("mint")
        if not mint or not isinstance(mint, str):
            continue
        if mint.startswith("So11111111111111111111111111111111111111112"):
            continue

        to_user = t.get("toUserAccount")
        from_user = t.get("fromUserAccount")

        # BUY: tokens to tracked wallet
        if isinstance(to_user, str) and to_user in tracked:
            return {"side": "BUY", "mint": mint, "tracked_wallet": to_user}

        # SELL: tokens from tracked wallet
        if isinstance(from_user, str) and from_user in tracked:
            return {"side": "SELL", "mint": mint, "tracked_wallet": from_user}

    return None


def forced_exit_if_needed() -> None:
    """
    Force-exit positions older than HOLD_MAX_SECONDS.
    Since this is paper-trading without live prices, we sell at cost * FORCED_EXIT_FALLBACK_MULTI.
    """
    if HOLD_MAX_SECONDS <= 0:
        return

    ts = now_ts()
    positions: Dict[str, Any] = STATE["positions"]
    to_close: List[str] = []

    for mint, pos in positions.items():
        opened = int(pos.get("opened_at", ts))
        age = ts - opened
        if age >= HOLD_MAX_SECONDS:
            to_close.append(mint)

    for mint in to_close:
        pos = positions.get(mint)
        if not pos:
            continue
        cost = float(pos.get("cost_usd", 0.0))
        qty = float(pos.get("qty", 0.0))
        proceeds = round(cost * FORCED_EXIT_FALLBACK_MULTI, 2)

        # Close
        del positions[mint]

        # Realize PnL
        pnl = round(proceeds - cost, 2)

        # Cash handling: reserve only takes a share of positive profit
        if pnl > 0:
            reserve_add = round(pnl * RESERVE_PCT, 2)
            cash_add = round(pnl - reserve_add, 2)
            STATE["reserve_cash_usd"] = round(float(STATE["reserve_cash_usd"]) + reserve_add, 2)
            STATE["cash_usd"] = round(float(STATE["cash_usd"]) + cost + cash_add, 2)
        else:
            # Loss stays in tradable cash
            STATE["cash_usd"] = round(float(STATE["cash_usd"]) + proceeds, 2)

        STATE["trades_count"] += 1
        STATE["counters"]["forced_exits"] += 1
        STATE["counters"]["sells"] += 1
        STATE["trades"].append({
            "ts": ts,
            "side": "FORCED_SELL",
            "mint": mint,
            "qty": qty,
            "cost_usd": cost,
            "proceeds_usd": proceeds,
            "pnl_usd": pnl,
            "reason": f"hold>{HOLD_MAX_SECONDS}s",
        })
        clamp_recent(STATE["trades"], 200)


def paper_buy(mint: str) -> bool:
    cash = float(STATE["cash_usd"])
    # keep MIN_CASH_LEFT_USD available
    if cash - MAX_BUY_USD < MIN_CASH_LEFT_USD:
        STATE["counters"]["skipped_low_cash"] += 1
        return False

    spend = round(min(MAX_BUY_USD, cash - MIN_CASH_LEFT_USD), 2)
    if spend <= 0:
        STATE["counters"]["skipped_low_cash"] += 1
        return False

    # For paper: qty is "spend units" (since we don't have real price)
    qty = spend

    STATE["cash_usd"] = round(cash - spend, 2)
    STATE["positions"][mint] = {
        "qty": qty,
        "cost_usd": spend,
        "opened_at": now_ts(),
        "last_update": now_ts(),
    }

    STATE["trades_count"] += 1
    STATE["counters"]["buys"] += 1
    STATE["trades"].append({
        "ts": now_ts(),
        "side": "BUY",
        "mint": mint,
        "qty": qty,
        "cost_usd": spend,
    })
    clamp_recent(STATE["trades"], 200)
    return True


def paper_sell(mint: str) -> bool:
    pos = STATE["positions"].get(mint)
    if not pos:
        return False

    cost = float(pos.get("cost_usd", 0.0))
    qty = float(pos.get("qty", 0.0))

    # For now, assume flat exit at cost (breakeven) unless forced exit triggers.
    proceeds = round(cost, 2)
    pnl = round(proceeds - cost, 2)

    del STATE["positions"][mint]

    if pnl > 0:
        reserve_add = round(pnl * RESERVE_PCT, 2)
        cash_add = round(pnl - reserve_add, 2)
        STATE["reserve_cash_usd"] = round(float(STATE["reserve_cash_usd"]) + reserve_add, 2)
        STATE["cash_usd"] = round(float(STATE["cash_usd"]) + cost + cash_add, 2)
    else:
        STATE["cash_usd"] = round(float(STATE["cash_usd"]) + proceeds, 2)

    STATE["trades_count"] += 1
    STATE["counters"]["sells"] += 1
    STATE["trades"].append({
        "ts": now_ts(),
        "side": "SELL",
        "mint": mint,
        "qty": qty,
        "cost_usd": cost,
        "proceeds_usd": proceeds,
        "pnl_usd": pnl,
        "reason": "signal",
    })
    clamp_recent(STATE["trades"], 200)
    return True


@app.get("/health")
def health():
    return {"ok": True, "service": "sol-paper-bot"}


@app.get("/events")
def events():
    return {"count": len(STATE["events"]), "events": STATE["events"]}


@app.get("/paper/state")
def paper_state():
    tracked = get_tracked_wallets()
    return {
        "cash_usd": STATE["cash_usd"],
        "reserve_cash_usd": STATE["reserve_cash_usd"],
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
            "TRACKED_WALLETS_COUNT": len(tracked),
        },
        "recent_trades": STATE["trades"][-25:],
    }


@app.post("/webhook")
async def webhook(request: Request):
    ok, reason = auth_ok(request)
    if not ok:
        STATE["counters"]["webhooks_unauthorized"] += 1
        # IMPORTANT: do not leak secrets
        # But include safe reason in the response so you can diagnose
        return JSONResponse({"ok": False, "error": "unauthorized", "detail": reason}, status_code=401)

    # Authorized
    STATE["counters"]["webhooks_received"] += 1

    # Forced exits happen on any incoming webhook tick
    forced_exit_if_needed()

    try:
        body = await request.json()
    except Exception:
        STATE["counters"]["skipped_bad_payload"] += 1
        return JSONResponse({"ok": False, "error": "bad_json"}, status_code=400)

    # Helius often sends a list; you previously had payload wrapped in [{"..."}]
    # Your /events screenshot earlier showed: {"payload":[{...}]}
    payload = body.get("payload") if isinstance(body, dict) else body

    tracked = get_tracked_wallets()
    swaps = extract_swaps(payload)

    # Record a light event summary
    STATE["events"].append({
        "received_at": now_ts(),
        "source": "helius",
        "swaps_seen": len(swaps),
    })
    clamp_recent(STATE["events"], 200)

    # Process swaps
    for tx in swaps:
        sig = tx.get("signature")
        info = classify_swap_for_tracked_wallets(tx, tracked)
        if not info:
            continue

        mint = info["mint"]
        side = info["side"]

        if side == "BUY":
            # Only open a new position if we don't already have one in that mint
            if mint not in STATE["positions"]:
                did = paper_buy(mint)
                if did:
                    # Add trace trade info (light)
                    STATE["events"][-1]["last_trade"] = {"side": "BUY", "mint": mint, "sig": sig}
        else:
            did = paper_sell(mint)
            if did:
                STATE["events"][-1]["last_trade"] = {"side": "SELL", "mint": mint, "sig": sig}

    return {"ok": True}
