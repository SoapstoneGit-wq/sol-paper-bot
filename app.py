import os
import json
import time
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# ----------------------------
# Config
# ----------------------------
APP_NAME = "sol-paper-bot"
STATE_FILE = "paper_state.json"

WEBHOOK_SECRET_ENV = "WEBHOOK_SECRET"
START_CASH_ENV = "START_CASH_USD"

# Profit split rule:
# If a completed trade (buy->sell) had >= 50% return, split profit:
#   - 50% of PROFIT goes to reserve_cash_usd (protected, not used for buying)
#   - 50% of PROFIT remains in cash_usd (compounds trading bankroll)
PROFIT_SPLIT_MIN_RETURN = 0.50  # 50%
PROFIT_SPLIT_MAX_RETURN = 100.0  # 10,000% = 100x return as decimal (100.0)
PROFIT_SPLIT_RATIO_TO_RESERVE = 0.50  # 50% to reserve


# ----------------------------
# App
# ----------------------------
app = FastAPI(title=APP_NAME)

# ----------------------------
# In-memory state
# ----------------------------
DEFAULT_STATE = {
    "started_at": None,
    "cash_usd": None,
    "reserve_cash_usd": 0.0,  # protected profits
    "positions": {},          # token_mint -> position object
    "events": [],             # last N webhook events
    "trades": [],             # closed trades history
    "counters": {
        "webhooks_received": 0,
        "webhooks_unauthorized": 0,
        "buys": 0,
        "sells": 0,
    },
}

state: Dict[str, Any] = {}


# ----------------------------
# Helpers: state persistence
# ----------------------------
def load_state() -> Dict[str, Any]:
    # If no file, initialize fresh
    if not os.path.exists(STATE_FILE):
        s = json.loads(json.dumps(DEFAULT_STATE))
        s["started_at"] = int(time.time())
        start_cash = float(os.getenv(START_CASH_ENV, "500"))
        s["cash_usd"] = start_cash
        return s

    try:
        with open(STATE_FILE, "r") as f:
            s = json.load(f)
        # Backfill any missing keys
        for k, v in DEFAULT_STATE.items():
            if k not in s:
                s[k] = json.loads(json.dumps(v))
        if s["started_at"] is None:
            s["started_at"] = int(time.time())
        if s["cash_usd"] is None:
            s["cash_usd"] = float(os.getenv(START_CASH_ENV, "500"))
        if "reserve_cash_usd" not in s:
            s["reserve_cash_usd"] = 0.0
        return s
    except Exception:
        # If corrupted, reset safely
        s = json.loads(json.dumps(DEFAULT_STATE))
        s["started_at"] = int(time.time())
        s["cash_usd"] = float(os.getenv(START_CASH_ENV, "500"))
        return s


def save_state(s: Dict[str, Any]) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(s, f, indent=2)
    except Exception as e:
        # Don’t crash the server if saving fails
        print("WARN: failed to save state:", str(e))


def trim_events(s: Dict[str, Any], keep: int = 200) -> None:
    if len(s["events"]) > keep:
        s["events"] = s["events"][-keep:]


def trim_trades(s: Dict[str, Any], keep: int = 500) -> None:
    if len(s["trades"]) > keep:
        s["trades"] = s["trades"][-keep:]


# ----------------------------
# Helpers: webhook auth (robust)
# ----------------------------
def get_webhook_secret_from_request(request: Request) -> str:
    """
    Accepts multiple formats:
    1) x-webhook-secret: <secret>
    2) Authorization: <secret>   (fallback)
    3) If provider accidentally sends "x-webhook-secret: <secret>" as the VALUE, parse it.
    """
    val = request.headers.get("x-webhook-secret") or request.headers.get("X-Webhook-Secret")

    if not val:
        val = request.headers.get("authorization") or request.headers.get("Authorization")

    if not val:
        return ""

    val = val.strip()

    # If value looks like: "x-webhook-secret: the_secret"
    if ":" in val and val.lower().startswith("x-webhook-secret"):
        _, maybe_secret = val.split(":", 1)
        val = maybe_secret.strip()

    return val


def require_valid_webhook(request: Request) -> None:
    expected = (os.getenv(WEBHOOK_SECRET_ENV) or "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail=f"Server misconfigured: {WEBHOOK_SECRET_ENV} missing")

    got = get_webhook_secret_from_request(request)

    if not got or got != expected:
        state["counters"]["webhooks_unauthorized"] += 1
        raise HTTPException(status_code=401, detail="Unauthorized")


# ----------------------------
# Helpers: paper trading bookkeeping
# ----------------------------
def open_or_add_position(token_mint: str, usd_spent: float, token_qty: float, price_usd: float) -> None:
    """
    Simple average-cost position.
    """
    pos = state["positions"].get(token_mint)
    if not pos:
        pos = {
            "token_mint": token_mint,
            "qty": 0.0,
            "avg_cost_usd_per_token": 0.0,
            "total_cost_usd": 0.0,
            "opened_at": int(time.time()),
            "last_price_usd": price_usd,
            "peak_price_usd": price_usd,
        }

    new_total_cost = pos["total_cost_usd"] + usd_spent
    new_qty = pos["qty"] + token_qty

    # Avoid divide-by-zero
    if new_qty > 0:
        pos["avg_cost_usd_per_token"] = new_total_cost / new_qty

    pos["qty"] = new_qty
    pos["total_cost_usd"] = new_total_cost
    pos["last_price_usd"] = price_usd
    pos["peak_price_usd"] = max(pos.get("peak_price_usd", price_usd), price_usd)

    state["positions"][token_mint] = pos


def close_position(token_mint: str, price_usd: float, reason: str = "sell") -> Optional[Dict[str, Any]]:
    """
    Sells full position at price_usd and returns closed trade record.
    Applies profit split rule if return >= 50% and <= 10,000%.
    """
    pos = state["positions"].get(token_mint)
    if not pos or pos["qty"] <= 0:
        return None

    qty = float(pos["qty"])
    total_cost = float(pos["total_cost_usd"])
    proceeds = qty * price_usd
    pnl = proceeds - total_cost

    # Return as decimal: (proceeds / cost) - 1
    ret = (proceeds / total_cost - 1.0) if total_cost > 0 else 0.0

    # Put proceeds back into cash (paper)
    state["cash_usd"] += proceeds

    # Profit split: only applies if pnl > 0 and return in range
    reserved = 0.0
    if pnl > 0 and (ret >= PROFIT_SPLIT_MIN_RETURN) and (ret <= PROFIT_SPLIT_MAX_RETURN):
        reserved = pnl * PROFIT_SPLIT_RATIO_TO_RESERVE
        # Move from cash -> reserve (protected)
        state["cash_usd"] -= reserved
        state["reserve_cash_usd"] += reserved

    trade = {
        "token_mint": token_mint,
        "qty": qty,
        "avg_cost_usd_per_token": pos["avg_cost_usd_per_token"],
        "sell_price_usd": price_usd,
        "total_cost_usd": total_cost,
        "proceeds_usd": proceeds,
        "pnl_usd": pnl,
        "return_decimal": ret,
        "reserved_profit_usd": reserved,
        "reason": reason,
        "opened_at": pos.get("opened_at"),
        "closed_at": int(time.time()),
        "peak_price_usd": pos.get("peak_price_usd", price_usd),
    }

    # Remove position
    del state["positions"][token_mint]
    state["trades"].append(trade)
    trim_trades(state)

    return trade


# ----------------------------
# Minimal “decision” logic (safe defaults)
# ----------------------------
def maybe_paper_buy(token_mint: str, price_usd: float) -> Optional[Dict[str, Any]]:
    """
    Placeholder buy logic.
    You can replace this later with your real strategy.
    For now:
      - Buys a small fixed amount ONLY if we do not already hold the token.
    """
    if token_mint in state["positions"]:
        return None

    # Only use trading cash (NOT reserve)
    cash = float(state["cash_usd"])
    buy_usd = min(10.0, cash)  # tiny test buy
    if buy_usd <= 0.0 or price_usd <= 0.0:
        return None

    token_qty = buy_usd / price_usd
    state["cash_usd"] -= buy_usd
    open_or_add_position(token_mint, usd_spent=buy_usd, token_qty=token_qty, price_usd=price_usd)
    state["counters"]["buys"] += 1

    return {"action": "buy", "token_mint": token_mint, "usd": buy_usd, "qty": token_qty, "price_usd": price_usd}


def maybe_paper_sell(token_mint: str, price_usd: float) -> Optional[Dict[str, Any]]:
    """
    Placeholder sell logic.
    For now:
      - If we have a position and return >= 20%, sell.
    """
    pos = state["positions"].get(token_mint)
    if not pos:
        return None

    avg = float(pos["avg_cost_usd_per_token"])
    if avg <= 0 or price_usd <= 0:
        return None

    ret = (price_usd / avg) - 1.0
    if ret >= 0.20:
        trade = close_position(token_mint, price_usd, reason="take_profit_20pct")
        state["counters"]["sells"] += 1
        return {"action": "sell", "trade": trade}

    # Update last/peak
    pos["last_price_usd"] = price_usd
    pos["peak_price_usd"] = max(pos.get("peak_price_usd", price_usd), price_usd)
    state["positions"][token_mint] = pos
    return None


# ----------------------------
# Parse Helius Enhanced payload (best-effort)
# ----------------------------
def extract_token_mint_and_price(payload: Any) -> Optional[Dict[str, Any]]:
    """
    Helius enhanced webhooks vary. We do best-effort extraction.
    If we can't find anything, we just store the event.

    This tries:
      - payload may be list[events]
      - token mint could appear in tokenBalanceChanges[].mint
      - price is not always included; if missing, we skip buy/sell decisions
    """
    # Many Helius webhooks send a list of tx objects
    events = payload if isinstance(payload, list) else [payload]

    # We’ll just scan for the first mint we can find
    for ev in events:
        if not isinstance(ev, dict):
            continue

        tbc = ev.get("tokenBalanceChanges") or []
        if isinstance(tbc, list) and len(tbc) > 0:
            first = tbc[0]
            if isinstance(first, dict):
                mint = first.get("mint")
                if mint:
                    # price_usd is often NOT included; leave None
                    return {"token_mint": mint, "price_usd": None}

    return None


# ----------------------------
# Routes
# ----------------------------
@app.on_event("startup")
async def on_startup():
    global state
    state = load_state()
    save_state(state)
    print(f"{APP_NAME} started. cash_usd={state['cash_usd']} reserve_cash_usd={state['reserve_cash_usd']}")


@app.get("/health")
async def health():
    return {"ok": True, "service": APP_NAME}


@app.get("/events")
async def get_events():
    return {"count": len(state["events"]), "events": state["events"]}


@app.get("/paper/state")
async def paper_state():
    return {
        "cash_usd": state["cash_usd"],
        "reserve_cash_usd": state["reserve_cash_usd"],
        "positions": state["positions"],
        "trades_count": len(state["trades"]),
        "counters": state["counters"],
        "started_at": state["started_at"],
    }


@app.get("/paper/trades")
async def paper_trades():
    return {"count": len(state["trades"]), "trades": state["trades"]}


@app.post("/webhook")
async def webhook(request: Request):
    # 1) Auth
    require_valid_webhook(request)

    # 2) Read payload
    payload = await request.json()

    # 3) Store event (trimmed)
    state["counters"]["webhooks_received"] += 1
    state["events"].append({
        "received_at": int(time.time()),
        "source": "helius",
        "payload": payload,   # NOTE: can be big; kept trimmed to last 200
    })
    trim_events(state)

    # 4) Best-effort extract token mint and price
    extracted = extract_token_mint_and_price(payload)

    actions: List[Dict[str, Any]] = []

    # If we can’t extract price, we can’t do buy/sell decisions reliably.
    # But webhook + auth + storage will still work (no more 401 if headers match).
    if extracted and extracted.get("token_mint"):
        token_mint = extracted["token_mint"]
        price_usd = extracted.get("price_usd")

        # If you later add price extraction, these actions will trigger.
        if isinstance(price_usd, (int, float)) and price_usd > 0:
            a1 = maybe_paper_sell(token_mint, float(price_usd))
            if a1:
                actions.append(a1)

            a2 = maybe_paper_buy(token_mint, float(price_usd))
            if a2:
                actions.append(a2)

    save_state(state)
    return JSONResponse({"ok": True, "actions": actions})
