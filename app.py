import os
import json
import time
import hmac
import hashlib
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

APP_NAME = "sol-paper-bot"

# -----------------------------
# Config (ENV VARS)
# -----------------------------
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")  # if you use one
STATE_PATH = os.getenv("STATE_PATH", "paper_state.json")

# Starting cash (default $500)
START_CASH_USD = float(os.getenv("START_CASH_USD", "500"))

# Profit banking rules
BANKING_ENABLED = os.getenv("BANKING_ENABLED", "true").lower() == "true"
BANK_ROI_MIN = float(os.getenv("BANK_ROI_MIN", "0.50"))       # +50%
BANK_ROI_MAX = float(os.getenv("BANK_ROI_MAX", "100.0"))      # +10,000% = 100x
BANK_PROFIT_SPLIT = float(os.getenv("BANK_PROFIT_SPLIT", "0.50"))  # bank 50% of profit

# Simple event buffer size
MAX_EVENTS = int(os.getenv("MAX_EVENTS", "200"))

app = FastAPI(title=APP_NAME)

# -----------------------------
# State helpers
# -----------------------------
def _default_state() -> Dict[str, Any]:
    return {
        "cash": START_CASH_USD,         # spendable cash
        "banked_cash": 0.0,             # "compounding reserve"
        "positions": {},                # mint -> position object
        "events": [],                   # recent webhook events
        "trades": []                    # executed trades (paper)
    }

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        st = _default_state()
        save_state(st)
        return st
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        st = _default_state()
        save_state(st)
        return st

def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)

def push_event(state: Dict[str, Any], evt: Dict[str, Any]) -> None:
    state["events"].append(evt)
    if len(state["events"]) > MAX_EVENTS:
        state["events"] = state["events"][-MAX_EVENTS:]


# -----------------------------
# Security (optional)
# -----------------------------
def verify_webhook(request_body: bytes, secret: str, header_sig: Optional[str]) -> None:
    """
    If you use a webhook signature header, verify it here.
    If you’re NOT using signatures, just leave WEBHOOK_SECRET blank.
    """
    if not secret:
        return  # no verification configured
    if not header_sig:
        raise HTTPException(status_code=401, detail="Missing signature header")

    # Typical pattern: HMAC SHA256 hex digest
    mac = hmac.new(secret.encode("utf-8"), msg=request_body, digestmod=hashlib.sha256)
    expected = mac.hexdigest()

    # Allow either exact hex or "sha256=<hex>"
    provided = header_sig.replace("sha256=", "").strip()
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid signature")


# -----------------------------
# Trading / accounting helpers
# -----------------------------
def record_trade(state: Dict[str, Any], trade: Dict[str, Any]) -> None:
    state["trades"].append(trade)

def bank_profit_if_applicable(state: Dict[str, Any], realized_profit: float, roi: float) -> None:
    """
    If ROI is in [BANK_ROI_MIN, BANK_ROI_MAX], bank BANK_PROFIT_SPLIT of profit.
    """
    if not BANKING_ENABLED:
        return
    if realized_profit <= 0:
        return
    if roi < BANK_ROI_MIN or roi > BANK_ROI_MAX:
        return

    bank_amt = realized_profit * BANK_PROFIT_SPLIT
    # Remove from spendable cash, add to banked
    state["cash"] -= bank_amt
    state["banked_cash"] += bank_amt

def open_position(state: Dict[str, Any], mint: str, usd_spent: float, qty: float, price: float, source: str) -> None:
    # Spend cash
    if usd_spent > state["cash"]:
        usd_spent = state["cash"]
    if usd_spent <= 0:
        return

    state["cash"] -= usd_spent
    state["positions"][mint] = {
        "qty": qty,
        "avg_cost_usd_per_token": (usd_spent / qty) if qty else 0.0,
        "last_price_usd": price,
        "peak_price_usd": price,
        "opened_at": int(time.time()),
        "source": source,
    }
    record_trade(state, {
        "ts": int(time.time()),
        "type": "BUY",
        "mint": mint,
        "usd_spent": usd_spent,
        "qty": qty,
        "price": price,
        "source": source,
    })

def close_position(state: Dict[str, Any], mint: str, price: float, source: str) -> None:
    pos = state["positions"].get(mint)
    if not pos:
        return

    qty = float(pos.get("qty", 0.0))
    if qty <= 0:
        state["positions"].pop(mint, None)
        return

    cost_per = float(pos.get("avg_cost_usd_per_token", 0.0))
    cost_basis = qty * cost_per
    proceeds = qty * price
    profit = proceeds - cost_basis

    # ROI = profit / cost_basis
    roi = (profit / cost_basis) if cost_basis > 0 else 0.0

    # Add proceeds back to cash first
    state["cash"] += proceeds

    # Bank part of profit if it meets your big-win rule
    bank_profit_if_applicable(state, realized_profit=profit, roi=roi)

    record_trade(state, {
        "ts": int(time.time()),
        "type": "SELL",
        "mint": mint,
        "qty": qty,
        "price": price,
        "proceeds": proceeds,
        "cost_basis": cost_basis,
        "profit": profit,
        "roi": roi,
        "source": source,
        "banked_profit_split": BANK_PROFIT_SPLIT if (BANKING_ENABLED and profit > 0 and BANK_ROI_MIN <= roi <= BANK_ROI_MAX) else 0.0
    })

    state["positions"].pop(mint, None)


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "app": APP_NAME}

@app.get("/events")
def events():
    st = load_state()
    return {"count": len(st.get("events", [])), "events": st.get("events", [])}

@app.get("/paper/state")
def paper_state():
    st = load_state()
    return {
        "cash": st.get("cash", 0.0),
        "banked_cash": st.get("banked_cash", 0.0),
        "positions": st.get("positions", {}),
        "trades_count": len(st.get("trades", [])),
        "events_count": len(st.get("events", [])),
        "banking": {
            "enabled": BANKING_ENABLED,
            "roi_min": BANK_ROI_MIN,
            "roi_max": BANK_ROI_MAX,
            "profit_split": BANK_PROFIT_SPLIT,
        }
    }

@app.get("/paper/trades")
def paper_trades(limit: int = 100):
    st = load_state()
    trades = st.get("trades", [])
    return {"count": len(trades), "trades": trades[-limit:]}

@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_signature: Optional[str] = Header(default=None),
):
    raw = await request.body()

    # Optional signature verification
    verify_webhook(raw, WEBHOOK_SECRET, x_webhook_signature)

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    st = load_state()

    # Store event (so /events shows it)
    push_event(st, {
        "received_at": time.time(),
        "source": "webhook",
        "payload": payload
    })

    # -----------------------------
    # Minimal parsing for your Helius-like structure
    # -----------------------------
    # Your example looks like: {"events":[{"payload":[{... tx ...}]}]}
    # but sometimes you POST a single tx object. We handle both.

    tx_objects: List[Dict[str, Any]] = []

    if isinstance(payload, dict) and "events" in payload and isinstance(payload["events"], list):
        # list of events
        for e in payload["events"]:
            if isinstance(e, dict) and "payload" in e and isinstance(e["payload"], list):
                for tx in e["payload"]:
                    if isinstance(tx, dict):
                        tx_objects.append(tx)
    elif isinstance(payload, list):
        # list of txs
        for tx in payload:
            if isinstance(tx, dict):
                tx_objects.append(tx)
    elif isinstance(payload, dict):
        tx_objects.append(payload)

    # -----------------------------
    # Paper-trade logic placeholder
    #
    # NOTE:
    # I’m NOT making your bot “copy trade” here.
    # This just demonstrates how the "bank profits" accounting works
    # once your existing buy/sell decision code calls open_position/close_position.
    #
    # You should plug your real decision rules in this loop.
    # -----------------------------
    for tx in tx_objects:
        tx_type = tx.get("type")  # e.g. "SWAP"
        token_transfers = tx.get("tokenTransfers") or []
        source = tx.get("source", "unknown")

        # Example: detect mint involved
        mint = None
        token_amount = None
        if isinstance(token_transfers, list) and token_transfers:
            mint = token_transfers[0].get("mint")
            token_amount = token_transfers[0].get("tokenAmount")

        # If you have price info elsewhere, set it here.
        # In your current system you appear to track last_price_usd in state.
        # We default to 0 (won’t trade).
        price = 0.0

        # Update last/peak price if we have a position and you computed price somewhere else
        if mint and mint in st["positions"] and price > 0:
            pos = st["positions"][mint]
            pos["last_price_usd"] = price
            if price > float(pos.get("peak_price_usd", 0.0)):
                pos["peak_price_usd"] = price

        # IMPORTANT:
        # This file does not guess buys/sells. Your existing bot code should decide.
        # Example hook points:
        # - if should_buy(...): open_position(...)
        # - if should_sell(...): close_position(...)
        #
        # So for now, we do nothing.

    save_state(st)
    return JSONResponse({"ok": True})
