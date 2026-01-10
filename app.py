import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

APP_NAME = "sol-paper-bot"

# =========
# LOGGING
# =========
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
MAX_TRADE_PCT = 0.15          # max 15% of cash per trade

# Fee model (can be overridden by env vars)
SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "100"))
FEE_FIXED_USD = float(os.getenv("FEE_FIXED_USD", "1.00"))  # priority/compute/bribe baseline
FEE_PCT = float(os.getenv("FEE_PCT", "0.020"))             # 2% slippage+impact estimate


def load_paper_state() -> Dict[str, Any]:
    if not PAPER_STATE_FILE.exists():
        return {
            "cash": START_CASH_USD,
            "positions": {},
            "trades": []
        }
    return json.loads(PAPER_STATE_FILE.read_text(encoding="utf-8"))


def save_paper_state(state: Dict[str, Any]) -> None:
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ===============================
# HELIUS PARSING HELPERS
# ===============================

def _find_user_account(payload0: dict) -> dict:
    """
    Helius enhanced payload often includes accountData = [{account, nativeBalanceChange, tokenBalanceChanges}, ...]
    We'll pick the accountData entry that actually has balance changes.
    """
    account_data = payload0.get("accountData") or []
    if not isinstance(account_data, list) or not account_data:
        return {}

    # Prefer one with token balance changes
    for a in account_data:
        if (a.get("tokenBalanceChanges") or []) != []:
            return a

    # Otherwise prefer one with a native balance change
    for a in account_data:
        if a.get("nativeBalanceChange", 0) != 0:
            return a

    return account_data[0]


def _sum_token_delta_for_user(account_entry: dict) -> Tuple[Optional[str], float]:
    """
    Returns (mint, token_delta) for the biggest absolute token change in this event.
    """
    tbc = account_entry.get("tokenBalanceChanges") or []
    if not isinstance(tbc, list) or not tbc:
        return (None, 0.0)

    deltas: Dict[str, float] = {}
    for ch in tbc:
        mint = ch.get("mint")
        raw = (ch.get("rawTokenAmount") or {}).get("tokenAmount")
        dec = (ch.get("rawTokenAmount") or {}).get("decimals", 0)

        if mint is None or raw is None:
            continue

        try:
            qty = float(raw) / (10 ** int(dec))
        except Exception:
            continue

        deltas[mint] = deltas.get(mint, 0.0) + qty

    if not deltas:
        return (None, 0.0)

    mint = max(deltas.keys(), key=lambda m: abs(deltas[m]))
    return (mint, deltas[mint])


# ===============================
# PAPER TRADE LOGIC
# ===============================

def apply_paper_trade_from_helius(payload0: dict) -> None:
    """
    Convert ONE enhanced webhook payload object into a paper trade and update paper_state.json.

    IMPORTANT:
    - This does NOT execute real trades.
    - It "paper trades" by updating cash/positions based on webhook balance deltas.
    - It resizes the trade to OUR sizing rules:
        min $75, max 15% of cash, baseline 2% of cash (but not below min).
    """
    if not isinstance(payload0, dict):
        return

    account_entry = _find_user_account(payload0)
    if not account_entry:
        return

    # SOL change (lamports -> SOL)
    lamports = account_entry.get("nativeBalanceChange", 0) or 0
    sol_delta = float(lamports) / 1e9

    # Token change
    mint, token_delta = _sum_token_delta_for_user(account_entry)
    if mint is None or token_delta == 0:
        return

    # Decide direction based on typical swap pattern
    # BUY: token up, SOL down
    # SELL: token down, SOL up
    if token_delta > 0 and sol_delta < 0:
        side = "BUY"
    elif token_delta < 0 and sol_delta > 0:
        side = "SELL"
    else:
        # not a clean swap pattern
        return

    state = load_paper_state()
    cash = float(state.get("cash", 0.0))

    # OUR sizing: baseline 2% of cash but at least MIN_TRADE_USD
    target_usd = max(MIN_TRADE_USD, cash * 0.02)
    target_usd = min(target_usd, cash * MAX_TRADE_PCT, cash)

    # If we can't even do minimum size, skip
    if target_usd < MIN_TRADE_USD:
        return

    whale_trade_usd = abs(sol_delta) * SOL_PRICE_USD
    if whale_trade_usd <= 0:
        return

    # Scale the whale's deltas to our target_usd
    scale = target_usd / whale_trade_usd
    scaled_sol_delta = sol_delta * scale
    scaled_token_delta = token_delta * scale

    fees_usd = FEE_FIXED_USD + (target_usd * FEE_PCT)

    # Update cash (USD):
    # If scaled_sol_delta is negative (we spent SOL), cash goes down.
    state["cash"] = cash + (scaled_sol_delta * SOL_PRICE_USD) - fees_usd

    # Update positions
    positions = state.get("positions") or {}
    positions[mint] = float(positions.get(mint, 0.0)) + float(scaled_token_delta)
    state["positions"] = positions

    # Append trade record
    trades = state.get("trades") or []
    trades.append({
        "ts": utc_now_iso(),
        "side": side,
        "mint": mint,
        "token_delta": scaled_token_delta,
        "sol_delta": scaled_sol_delta,
        "trade_usd_est": target_usd,
        "fees_usd_est": fees_usd,
        "sig": payload0.get("signature"),
        "type": payload0.get("type"),
        "desc": payload0.get("description"),
    })
    state["trades"] = trades[-200:]

    save_paper_state(state)


# ===============================
# API ROUTES
# ===============================

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

    # ==== PAPER TRADING ====
    # Helius can send either a list of events or a single dict event
    if isinstance(payload, list):
        for evt in payload:
            apply_paper_trade_from_helius(evt)
    elif isinstance(payload, dict):
        apply_paper_trade_from_helius(payload)

    return JSONResponse({"ok": True})
