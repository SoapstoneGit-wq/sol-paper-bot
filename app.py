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

PAPER_STATE_FILE = LOG_DIR / "paper_state.json"

# ---- Paper trading realism knobs (USD) ----
START_CASH_USD = 500.0

MIN_TRADE_USD = 75.0   # avoid fee death
MAX_TRADE_PCT = 0.15   # max 15% of current cash per trade


def load_paper_state():
    if not PAPER_STATE_FILE.exists():
        return {"cash": START_CASH_USD, "positions": {}, "trades": []}
    state = json.loads(PAPER_STATE_FILE.read_text(encoding="utf-8"))
    # harden defaults if file exists but missing keys
    state.setdefault("cash", START_CASH_USD)
    state.setdefault("positions", {})
    state.setdefault("trades", [])
    return state


def save_paper_state(state):
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ===============================
# PAPER TRADE LOGIC
# ===============================

SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "100"))
FEE_FIXED_USD = float(os.getenv("FEE_FIXED_USD", "1.00"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.020"))


def _find_user_account(payload0: dict) -> dict:
    """
    Helius enhanced payload often includes accountData = [{account, nativeBalanceChange, tokenBalanceChanges}, ...]
    We'll pick the accountData entry that actually has balance changes.
    """
    account_data = payload0.get("accountData") or []
    if not isinstance(account_data, list):
        return {}

    # Prefer one with token balance changes
    for a in account_data:
        if (a.get("tokenBalanceChanges") or []) != []:
            return a

    # Otherwise prefer one with a native balance change
    for a in account_data:
        if a.get("nativeBalanceChange", 0) != 0:
            return a

    return account_data[0] if account_data else {}


def _sum_token_delta_for_user(account_entry: dict):
    """
    Returns (mint, token_delta) for the biggest absolute token change in this event.
    """
    tbc = account_entry.get("tokenBalanceChanges") or []
    if not isinstance(tbc, list) or not tbc:
        return (None, 0.0)

    deltas = {}
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


def apply_paper_trade_from_helius(payload0: dict):
    """
    Convert ONE enhanced webhook payload object into a paper trade and update paper_state.json.
    """
    if not isinstance(payload0, dict):
        return

    evt_type = (payload0.get("type") or "").upper()

    # TEMP DEBUG: don't require "SWAP" yet (Helius often labels swaps differently)
    # If you want to be strict later, replace this block with:
    #   if "SWAP" not in evt_type and "TOKEN_SWAP" not in evt_type: return
    if evt_type == "":
        # still continue — just means Helius didn't label type
        pass

    account_entry = _find_user_account(payload0)
    if not account_entry:
        return

    lamports = account_entry.get("nativeBalanceChange", 0) or 0
    sol_delta = float(lamports) / 1e9  # lamports -> SOL

    mint, token_delta = _sum_token_delta_for_user(account_entry)
    if mint is None or token_delta == 0:
        return

    # Determine direction
    # BUY: token up, SOL down
    # SELL: token down, SOL up
    if token_delta > 0 and sol_delta < 0:
        side = "BUY"
    elif token_delta < 0 and sol_delta > 0:
        side = "SELL"
    else:
        # Not a clean swap-like shape
        return

    trade_usd_raw = abs(sol_delta) * SOL_PRICE_USD

    state = load_paper_state()
    cash = float(state.get("cash", 0.0))

    # Realism knobs:
    # 1) skip tiny trades (fees kill you)
    if trade_usd_raw < MIN_TRADE_USD:
        return

    # 2) cap trade size to MAX_TRADE_PCT of current cash (paper sizing)
    max_allowed = cash * MAX_TRADE_PCT
    if max_allowed < MIN_TRADE_USD:
        return

    desired_trade_usd = min(trade_usd_raw, max_allowed)
    scale = desired_trade_usd / trade_usd_raw if trade_usd_raw > 0 else 0.0

    # Scale both legs so the trade "fits" your account size
    sol_delta_scaled = sol_delta * scale
    token_delta_scaled = token_delta * scale

    trade_usd = abs(sol_delta_scaled) * SOL_PRICE_USD
    fees_usd = FEE_FIXED_USD + (trade_usd * FEE_PCT)

    # Update cash (USD)
    # sol_delta is negative on BUY, positive on SELL (after scaling)
    state["cash"] = cash + (sol_delta_scaled * SOL_PRICE_USD) - fees_usd

    # Update positions
    positions = state.get("positions") or {}
    positions[mint] = float(positions.get(mint, 0.0)) + token_delta_scaled
    state["positions"] = positions

    # Append trade record
    trades = state.get("trades") or []
    trades.append(
        {
            "ts": utc_now_iso(),
            "type": evt_type,
            "side": side,
            "mint": mint,
            "token_delta": token_delta_scaled,
            "sol_delta": sol_delta_scaled,
            "trade_usd_est": trade_usd,
            "fees_usd_est": fees_usd,
            "sig": payload0.get("signature"),
            "desc": payload0.get("description"),
        }
    )
    state["trades"] = trades[-200:]
    save_paper_state(state)


# ===============================
# ROUTES
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

    if provided.lower().startswith("x-webhook-secret"):
        parts = provided.split(":", 1)
        if len(parts) == 2:
            provided = parts[1].strip()

    if WEBHOOK_SECRET and provided != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    payload = await request.json()

    record = {"received_at": utc_now_iso(), "source": "helius", "payload": payload}
    append_jsonl(record)

    # Paper trading: payload can be a list or a single object
    if isinstance(payload, list):
        for evt in payload:
            apply_paper_trade_from_helius(evt)
    elif isinstance(payload, dict):
        apply_paper_trade_from_helius(payload)

    return JSONResponse({"ok": True})
