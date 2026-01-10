import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

# =========================================================
# CONFIG
# =========================================================

APP_NAME = "sol-paper-bot"

START_CASH_USD = 500.0
FIXED_TRADE_USD = 100.0
FIXED_FEE_USD = 3.00

LOG_DIR = Path("logs")
EVENT_LOG = LOG_DIR / "events.jsonl"
STATE_FILE = LOG_DIR / "paper_state.json"

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

app = FastAPI(title=APP_NAME)

# =========================================================
# UTIL
# =========================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def ensure_logs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not EVENT_LOG.exists():
        EVENT_LOG.write_text("")

def append_event(obj):
    ensure_logs()
    with EVENT_LOG.open("a") as f:
        f.write(json.dumps(obj) + "\n")

# =========================================================
# PAPER STATE
# =========================================================

def load_state():
    if not STATE_FILE.exists():
        return {
            "cash": START_CASH_USD,
            "positions": {},
            "trades": []
        }
    return json.loads(STATE_FILE.read_text())

def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))

# =========================================================
# HELIUS PARSING
# =========================================================

def find_user_account(payload):
    accounts = payload.get("accountData", [])
    if not isinstance(accounts, list):
        return None

    for acc in accounts:
        if acc.get("tokenBalanceChanges"):
            return acc

    for acc in accounts:
        if acc.get("nativeBalanceChange", 0) != 0:
            return acc

    return None

def extract_token_delta(account):
    changes = account.get("tokenBalanceChanges", [])
    if not changes:
        return None, 0.0

    deltas = {}
    for c in changes:
        mint = c.get("mint")
        raw = c.get("rawTokenAmount", {})
        amt = raw.get("tokenAmount")
        dec = raw.get("decimals", 0)

        if mint and amt:
            qty = float(amt) / (10 ** dec)
            deltas[mint] = deltas.get(mint, 0) + qty

    if not deltas:
        return None, 0.0

    mint = max(deltas, key=lambda m: abs(deltas[m]))
    return mint, deltas[mint]

# =========================================================
# PAPER TRADE ENGINE
# =========================================================

def apply_paper_trade(payload):
    account = find_user_account(payload)
    if not account:
        return

    mint, token_delta = extract_token_delta(account)
    if not mint or token_delta == 0:
        return

    side = "BUY" if token_delta > 0 else "SELL"

    state = load_state()

    if side == "BUY":
        if state["cash"] < FIXED_TRADE_USD:
            return

        price = FIXED_TRADE_USD / abs(token_delta)

        pos = state["positions"].get(mint, {
            "qty": 0.0,
            "avg_cost": 0.0
        })

        new_qty = pos["qty"] + token_delta
        new_cost = (
            (pos["qty"] * pos["avg_cost"]) +
            FIXED_TRADE_USD
        ) / new_qty

        pos["qty"] = new_qty
        pos["avg_cost"] = new_cost
        state["positions"][mint] = pos
        state["cash"] -= FIXED_TRADE_USD + FIXED_FEE_USD

    else:  # SELL
        pos = state["positions"].get(mint)
        if not pos or pos["qty"] <= 0:
            return

        sell_qty = min(abs(token_delta), pos["qty"])
        proceeds = sell_qty * pos["avg_cost"]

        pos["qty"] -= sell_qty
        state["cash"] += proceeds - FIXED_FEE_USD

        if pos["qty"] <= 0:
            del state["positions"][mint]
        else:
            state["positions"][mint] = pos

    state["trades"].append({
        "ts": utc_now(),
        "side": side,
        "mint": mint,
        "qty": token_delta,
        "trade_usd": FIXED_TRADE_USD,
        "fee_usd": FIXED_FEE_USD,
        "sig": payload.get("signature")
    })

    state["trades"] = state["trades"][-200:]
    save_state(state)

# =========================================================
# API
# =========================================================

@app.get("/health")
def health():
    return {"ok": True, "time": utc_now()}

@app.get("/paper/state")
def paper_state():
    return load_state()

@app.get("/events")
def events():
    ensure_logs()
    lines = EVENT_LOG.read_text().splitlines()[-200:]
    return {
        "count": len(lines),
        "events": [json.loads(l) for l in lines]
    }

@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(default=None, convert_underscores=False),
    authorization: Optional[str] = Header(default=None),
):
    provided = x_webhook_secret or authorization or ""

    if provided.lower().startswith("bearer "):
        provided = provided[7:].strip()

    if WEBHOOK_SECRET and provided != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    payload = await request.json()

    append_event({
        "received_at": utc_now(),
        "source": "helius",
        "payload": payload
    })

    if isinstance(payload, list):
        for evt in payload:
            apply_paper_trade(evt)
    elif isinstance(payload, dict):
        apply_paper_trade(payload)

    return JSONResponse({"ok": True})
