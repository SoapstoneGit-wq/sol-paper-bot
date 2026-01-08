import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

APP_NAME = "sol-paper-bot"
LOG_DIR = Path("logs")
EVENT_LOG = LOG_DIR / "events.jsonl"

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

app = FastAPI(title=APP_NAME)

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_logfile() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not EVENT_LOG.exists():
        EVENT_LOG.write_text("", encoding="utf-8")

def append_jsonl(record: Dict[str, Any]) -> None:
    ensure_logfile()
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record))
        f.write("\n")
        
# =======================
# PAPER TRADING STATE
# =======================

PAPER_STATE_FILE = Path("logs/paper_state.json")
# ---- Paper trading realism knobs (USD) ----
START_CASH_USD = 500.0

MIN_TRADE_USD = 75.0          # avoid fee death
MAX_TRADE_PCT = 0.15          # max 15% per trade

FEE_FIXED_USD = 1.00          # priority / compute / bribe baseline
FEE_PCT = 0.020               # 2% total slippage + impact

def load_paper_state():
    if not PAPER_STATE_FILE.exists():
        return {
            "cash": START_CASH_USD,
            "positions": {},
            "trades": []
        }
    return json.loads(PAPER_STATE_FILE.read_text())

def save_paper_state(state):
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_STATE_FILE.write_text(json.dumps(state, indent=2))

# ===============================
# PAPER TRADE LOGIC
# ===============================

SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "100"))
FEE_FIXED_USD = float(os.getenv("FEE_FIXED_USD", "1.00"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.020"))


def _find_user_account(payload0: dict) -> dict:
    account_data = payload0.get("accountData") or []
    if not isinstance(account_data, list):
        return {}

    for a in account_data:
        if (a.get("tokenBalanceChanges") or []) != []:
            return a

    for a in account_data:
        if a.get("nativeBalanceChange", 0) != 0:
            return a

    return account_data[0] if account_data else {}


def _sum_token_delta_for_user(account_entry: dict):
    tbc = account_entry.get("tokenBalanceChanges") or []
    if not tbc:
        return (None, 0.0)

    deltas = {}
    for ch in tbc:
        mint = ch.get("mint")
        raw = (ch.get("rawTokenAmount") or {}).get("tokenAmount")
        dec = (ch.get("rawTokenAmount") or {}).get("decimals", 0)

        if mint is None or raw is None:
            continue

        qty = float(raw) / (10 ** int(dec))
        deltas[mint] = deltas.get(mint, 0.0) + qty

    mint = max(deltas, key=lambda m: abs(deltas[m]))
    return mint, deltas[mint]


def apply_paper_trade_from_helius(payload0: dict):
    evt_type = (payload0.get("type") or "").upper()

# TEMP DEBUG: don't require "SWAP" yet (Helius often labels swaps differently)
if evt_type == "":
    return


    account_entry = _find_user_account(payload0)
    if not account_entry:
        return

    sol_delta = float(account_entry.get("nativeBalanceChange", 0)) / 1e9
    mint, token_delta = _sum_token_delta_for_user(account_entry)

    if mint is None or token_delta == 0:
        return

    side = "BUY" if token_delta > 0 and sol_delta < 0 else "SELL"
    trade_usd = abs(sol_delta) * SOL_PRICE_USD
    fees_usd = FEE_FIXED_USD + trade_usd * FEE_PCT

    state = load_paper_state()

    state["cash"] += (sol_delta * SOL_PRICE_USD) - fees_usd
    state["positions"][mint] = state["positions"].get(mint, 0) + token_delta

    state["trades"].append({
        "ts": utc_now_iso(),
        "side": side,
        "mint": mint,
        "token_delta": token_delta,
        "sol_delta": sol_delta,
        "trade_usd_est": trade_usd,
        "fees_usd_est": fees_usd,
        "sig": payload0.get("signature"),
    })

    state["trades"] = state["trades"][-200:]
    save_paper_state(state)

@app.get("/paper/state")
def paper_state():
    return load_paper_state()

@app.get("/health")
def health():
    return {"ok": True, "service": APP_NAME, "time": utc_now_iso()}

@app.get("/events")
def get_events():
    ensure_logfile()
    lines = EVENT_LOG.read_text(encoding="utf-8").splitlines()[-200:]
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
    # Accept secret from multiple possible headers
    provided = x_webhook_secret or authorization or ""

    # Handle: Authorization: Bearer <secret>
    if provided.lower().startswith("bearer "):
        provided = provided[7:].strip()

    # Handle badly formatted "x-webhook-secret: value"
    if provided.lower().startswith("x-webhook-secret"):
        parts = provided.split(":", 1)
        if len(parts) == 2:
            provided = parts[1].strip()

    if WEBHOOK_SECRET and provided != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    payload = await request.json()

    record = {
        "received_at": utc_now_iso(),
        "source": "helius",
        "payload": payload,
    }
   append_jsonl(record)

# ---- PAPER TRADING ----
if isinstance(payload, list):
    for evt in payload:
        apply_paper_trade_from_helius(evt)
elif isinstance(payload, dict):
    apply_paper_trade_from_helius(payload)

return JSONResponse({"ok": True})

