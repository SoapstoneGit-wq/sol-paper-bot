import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

# ==========================================================
# BASIC APP CONFIG
# ==========================================================

APP_NAME = "sol-paper-bot"
TRACKED_WALLET = (os.getenv("TRACKED_WALLET") or "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "100"))
FEE_FIXED_USD = float(os.getenv("FEE_FIXED_USD", "1"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.02"))

LOG_DIR = Path("logs")
EVENT_LOG = LOG_DIR / "events.jsonl"
PAPER_STATE_FILE = LOG_DIR / "paper_state.json"

app = FastAPI(title=APP_NAME)

# ==========================================================
# HELPERS
# ==========================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_logfile():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not EVENT_LOG.exists():
        EVENT_LOG.write_text("", encoding="utf-8")

def append_jsonl(record: Dict[str, Any]):
    ensure_logfile()
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# ==========================================================
# PAPER STATE
# ==========================================================

def load_paper_state():
    if not PAPER_STATE_FILE.exists():
        return {"cash": 500.0, "positions": {}, "trades": []}
    return json.loads(PAPER_STATE_FILE.read_text())

def save_paper_state(state):
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_STATE_FILE.write_text(json.dumps(state, indent=2))

# ==========================================================
# Helius Parsing Logic
# ==========================================================

def _find_user_account(payload: dict) -> dict:
    accounts = payload.get("accountData") or []
    if not accounts:
        return {}

    if TRACKED_WALLET:
        for a in accounts:
            if a.get("account") == TRACKED_WALLET:
                return a

    for a in accounts:
        if a.get("nativeBalanceChange", 0) != 0:
            return a

    return accounts[0]

def _token_delta_from_transfers(payload: dict):
    transfers = payload.get("tokenTransfers") or []
    if not transfers or not TRACKED_WALLET:
        return None, 0.0

    deltas = {}
    for t in transfers:
        mint = t.get("mint")
        amt = t.get("tokenAmount")
        if mint is None or amt is None:
            continue

        amt = float(amt)
        if t.get("toUserAccount") == TRACKED_WALLET:
            deltas[mint] = deltas.get(mint, 0) + amt
        if t.get("fromUserAccount") == TRACKED_WALLET:
            deltas[mint] = deltas.get(mint, 0) - amt

    if not deltas:
        return None, 0.0

    mint = max(deltas, key=lambda m: abs(deltas[m]))
    return mint, deltas[mint]

def _token_delta_from_balance_changes(account_entry: dict):
    changes = account_entry.get("tokenBalanceChanges") or []
    if not changes:
        return None, 0.0

    deltas = {}
    for ch in changes:
        mint = ch.get("mint")
        raw = (ch.get("rawTokenAmount") or {}).get("tokenAmount")
        dec = (ch.get("rawTokenAmount") or {}).get("decimals", 0)

        if mint and raw:
            qty = float(raw) / (10 ** int(dec))
            deltas[mint] = deltas.get(mint, 0) + qty

    if not deltas:
        return None, 0.0

    mint = max(deltas, key=lambda m: abs(deltas[m]))
    return mint, deltas[mint]

# ==========================================================
# PAPER TRADE EXECUTION
# ==========================================================

def apply_paper_trade(payload: dict):
    if not isinstance(payload, dict):
        return

    account = _find_user_account(payload)
    sol_delta = float(account.get("nativeBalanceChange", 0)) / 1e9

    mint, token_delta = _token_delta_from_transfers(payload)
    if mint is None or token_delta == 0:
        mint, token_delta = _token_delta_from_balance_changes(account)

    if mint is None or token_delta == 0:
        return

    side = (
        "BUY" if token_delta > 0 and sol_delta < 0
        else "SELL" if token_delta < 0 and sol_delta > 0
        else "UNKNOWN"
    )

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
        "sig": payload.get("signature"),
        "desc": payload.get("description"),
    })

    state["trades"] = state["trades"][-200:]
    save_paper_state(state)

# ==========================================================
# ROUTES
# ==========================================================

@app.get("/health")
def health():
    return {"ok": True, "service": APP_NAME, "time": utc_now_iso()}

@app.get("/paper/state")
def paper_state():
    return load_paper_state()

@app.get("/events")
def get_events():
    ensure_logfile()
    lines = EVENT_LOG.read_text().splitlines()[-200:]
    return {"count": len(lines), "events": [json.loads(l) for l in lines]}

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

    append_jsonl({
        "received_at": utc_now_iso(),
        "source": "helius",
        "payload": payload,
    })

    if isinstance(payload, list):
        for evt in payload:
            apply_paper_trade(evt)
    elif isinstance(payload, dict):
        apply_paper_trade(payload)

    return JSONResponse({"ok": True})
