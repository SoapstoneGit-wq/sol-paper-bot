import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

APP_NAME = "sol-paper-bot"

# =========================
# LOGGING (raw webhook events)
# =========================
LOG_DIR = Path("logs")
EVENT_LOG = LOG_DIR / "events.jsonl"
PAPER_STATE_FILE = LOG_DIR / "paper_state.json"

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()

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


# =========================
# TUNABLE SETTINGS (env vars)
# =========================

# Starting cash
START_CASH_USD = float(os.getenv("START_CASH_USD", "500"))

# How much we spend per BUY (paper)
TRADE_USD = float(os.getenv("TRADE_USD", "100"))
MIN_TRADE_USD = float(os.getenv("MIN_TRADE_USD", "25"))

# Fees / slippage model
FEE_FIXED_USD = float(os.getenv("FEE_FIXED_USD", "1.00"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.020"))  # 2%

# When whale sells, we sell this fraction of our position (0.0 - 1.0)
MIRROR_WHALE_SELL_PCT = float(os.getenv("MIRROR_WHALE_SELL_PCT", "1.0"))

# Stop loss (e.g. 0.20 = 20% down from avg cost)
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.20"))

# Trailing stop behavior
TRAIL_ACTIVATE_PCT = float(os.getenv("TRAIL_ACTIVATE_PCT", "0.35"))  # start trailing after +35%
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "0.20"))  # trail distance 20%

# Optional take-profit ladder
# Example:
# TAKE_PROFIT_LEVELS="0.40,1.00,2.00"  (40%, 100%, 200%)
# TAKE_PROFIT_SELL_PCTS="0.25,0.25,0.25" (sell 25% each time)
TP_LEVELS_STR = os.getenv("TAKE_PROFIT_LEVELS", "").strip()
TP_SELLS_STR = os.getenv("TAKE_PROFIT_SELL_PCTS", "").strip()

def _parse_csv_floats(s: str) -> List[float]:
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out

TAKE_PROFIT_LEVELS = _parse_csv_floats(TP_LEVELS_STR)
TAKE_PROFIT_SELL_PCTS = _parse_csv_floats(TP_SELLS_STR)

# Default SOL price for estimating price when we only have SOL delta
SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "100"))


# =========================
# PAPER STATE
# =========================

def load_paper_state() -> dict:
    if not PAPER_STATE_FILE.exists():
        return {
            "cash": START_CASH_USD,
            "positions": {},   # mint -> position object
            "trades": []
        }
    try:
        return json.loads(PAPER_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        # If file gets corrupted for any reason, reset cleanly
        return {
            "cash": START_CASH_USD,
            "positions": {},
            "trades": []
        }


def save_paper_state(state: dict) -> None:
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ✅ THE FIX: prune dead positions so /paper/state stays readable
def prune_zero_positions(state: dict) -> None:
    positions = state.get("positions") or {}
    to_delete = []
    for mint, pos in positions.items():
        try:
            qty = float(pos.get("qty") or 0.0)
            # prune 0 or negative, also prune NaN-ish
            if qty <= 0:
                to_delete.append(mint)
        except Exception:
            to_delete.append(mint)

    for mint in to_delete:
        del positions[mint]

    state["positions"] = positions


# =========================
# HELPERS: compute equity / pnl snapshots
# =========================

def compute_portfolio_metrics(state: dict) -> Tuple[float, float, float, float]:
    """
    returns:
      positions_value_usd, equity_usd, realized_pnl_usd, unrealized_pnl_usd
    """
    positions_value = 0.0
    realized = 0.0
    unrealized = 0.0

    positions = state.get("positions") or {}
    for mint, pos in positions.items():
        qty = float(pos.get("qty") or 0.0)
        last_price = pos.get("last_price_usd")
        avg_cost = float(pos.get("avg_cost_usd_per_token") or 0.0)

        if last_price is None:
            # no valuation if we don't have price
            continue

        last_price = float(last_price)
        positions_value += qty * last_price
        unrealized += qty * (last_price - avg_cost)

        realized += float(pos.get("realized_pnl_usd") or 0.0)

    equity = float(state.get("cash") or 0.0) + positions_value
    return positions_value, equity, realized, unrealized


# =========================
# PAPER TRADE EXECUTION
# =========================

def _fees_for_trade(trade_usd: float) -> float:
    return FEE_FIXED_USD + trade_usd * FEE_PCT


def _ensure_position(state: dict, mint: str) -> dict:
    positions = state.setdefault("positions", {})
    if mint not in positions:
        tp_done = [False for _ in TAKE_PROFIT_LEVELS]
        positions[mint] = {
            "qty": 0.0,
            "avg_cost_usd_per_token": 0.0,
            "last_price_usd": None,
            "peak_price": None,
            "trail_stop_price": None,
            "tp_done": tp_done,
            "realized_pnl_usd": 0.0,
        }
    # If TP config changed since last run, keep array aligned
    if TAKE_PROFIT_LEVELS:
        cur = positions[mint].get("tp_done")
        if not isinstance(cur, list) or len(cur) != len(TAKE_PROFIT_LEVELS):
            positions[mint]["tp_done"] = [False for _ in TAKE_PROFIT_LEVELS]
    return positions[mint]


def paper_buy(state: dict, mint: str, price_usd: float, sig: Optional[str], desc: str = "") -> None:
    cash = float(state.get("cash") or 0.0)

    trade_usd = min(TRADE_USD, cash)  # can't spend more than cash
    if trade_usd < MIN_TRADE_USD:
        return

    fees = _fees_for_trade(trade_usd)
    if trade_usd + fees > cash and cash >= MIN_TRADE_USD:
        # if fees push us over, shrink trade
        trade_usd = max(MIN_TRADE_USD, cash - fees)

    if trade_usd < MIN_TRADE_USD:
        return

    qty = trade_usd / price_usd

    pos = _ensure_position(state, mint)
    old_qty = float(pos.get("qty") or 0.0)
    old_avg = float(pos.get("avg_cost_usd_per_token") or 0.0)

    new_qty = old_qty + qty
    new_avg = ((old_qty * old_avg) + (qty * price_usd)) / new_qty if new_qty > 0 else 0.0

    pos["qty"] = new_qty
    pos["avg_cost_usd_per_token"] = new_avg
    pos["last_price_usd"] = price_usd

    # update peak/trail
    peak = pos.get("peak_price")
    if peak is None or price_usd > float(peak):
        pos["peak_price"] = price_usd
        # if trailing already active, move trail up
        if pos.get("trail_stop_price") is not None:
            pos["trail_stop_price"] = price_usd * (1.0 - TRAIL_PCT)

    # Cash update
    state["cash"] = cash - trade_usd - fees

    state.setdefault("trades", []).append({
        "ts": utc_now_iso(),
        "type": "SWAP",
        "side": "BUY",
        "mint": mint,
        "qty": qty,
        "price_usd_per_token": price_usd,
        "trade_usd_est": trade_usd,
        "fees_usd_est": fees,
        "sig": sig,
        "desc": desc or "",
    })


def paper_sell(state: dict, mint: str, price_usd: float, qty_to_sell: float, reason: str, sig: Optional[str], desc: str = "") -> None:
    pos = (state.get("positions") or {}).get(mint)
    if not pos:
        return

    held = float(pos.get("qty") or 0.0)
    if held <= 0:
        return

    qty = min(max(qty_to_sell, 0.0), held)
    if qty <= 0:
        return

    trade_usd = qty * price_usd
    fees = _fees_for_trade(trade_usd)

    avg_cost = float(pos.get("avg_cost_usd_per_token") or 0.0)
    realized = (price_usd - avg_cost) * qty - fees

    # Update position
    pos["qty"] = held - qty
    pos["last_price_usd"] = price_usd
    pos["realized_pnl_usd"] = float(pos.get("realized_pnl_usd") or 0.0) + realized

    # Cash update (we receive trade_usd - fees)
    state["cash"] = float(state.get("cash") or 0.0) + (trade_usd - fees)

    state.setdefault("trades", []).append({
        "ts": utc_now_iso(),
        "type": "SWAP",
        "side": "SELL",
        "mint": mint,
        "qty": qty,
        "price_usd_per_token": price_usd,
        "trade_usd_est": trade_usd,
        "fees_usd_est": fees,
        "realized_pnl_net": realized,
        "reason": reason,
        "sig": sig,
        "desc": desc or "",
    })


def apply_risk_checks(state: dict, mint: str, price_usd: float, sig: Optional[str]) -> None:
    pos = (state.get("positions") or {}).get(mint)
    if not pos:
        return

    qty = float(pos.get("qty") or 0.0)
    if qty <= 0:
        return

    avg = float(pos.get("avg_cost_usd_per_token") or 0.0)
    pos["last_price_usd"] = price_usd

    # Update peak
    peak = pos.get("peak_price")
    if peak is None or price_usd > float(peak):
        pos["peak_price"] = price_usd
        # If trailing is active, move it up
        if pos.get("trail_stop_price") is not None:
            pos["trail_stop_price"] = price_usd * (1.0 - TRAIL_PCT)

    # STOP LOSS (hard exit)
    if avg > 0 and price_usd <= avg * (1.0 - STOP_LOSS_PCT):
        paper_sell(state, mint, price_usd, qty, reason="STOP_LOSS", sig=sig)
        return

    # TAKE PROFITS (optional ladder)
    if avg > 0 and TAKE_PROFIT_LEVELS and TAKE_PROFIT_SELL_PCTS:
        tp_done = pos.get("tp_done") or [False for _ in TAKE_PROFIT_LEVELS]
        for i, level in enumerate(TAKE_PROFIT_LEVELS):
            if i >= len(TAKE_PROFIT_SELL_PCTS):
                break
            if i >= len(tp_done):
                break
            if tp_done[i]:
                continue
            target = avg * (1.0 + level)
            if price_usd >= target:
                sell_pct = TAKE_PROFIT_SELL_PCTS[i]
                sell_qty = qty * sell_pct
                paper_sell(state, mint, price_usd, sell_qty, reason=f"TAKE_PROFIT_{int(level*100)}PCT", sig=sig)
                # refresh qty after sell
                qty = float((state.get("positions") or {}).get(mint, {}).get("qty") or 0.0)
                tp_done[i] = True
                pos["tp_done"] = tp_done
                if qty <= 0:
                    return

    # TRAILING STOP activation
    if avg > 0:
        gain = (price_usd / avg) - 1.0
        if gain >= TRAIL_ACTIVATE_PCT and pos.get("trail_stop_price") is None:
            pos["trail_stop_price"] = price_usd * (1.0 - TRAIL_PCT)

    # TRAILING STOP trigger
    trail = pos.get("trail_stop_price")
    if trail is not None and price_usd <= float(trail):
        paper_sell(state, mint, price_usd, qty, reason="TRAILING_STOP", sig=sig)
        return


# =========================
# HELIUS EVENT PARSING
# =========================

def _find_user_account_entry(payload0: dict) -> dict:
    """
    Helius enhanced events often include accountData list with tokenBalanceChanges.
    We'll try to find the entry that has tokenBalanceChanges, else fallback.
    """
    account_data = payload0.get("accountData") or []
    if not isinstance(account_data, list) or not account_data:
        return {}

    for a in account_data:
        if (a.get("tokenBalanceChanges") or []) != []:
            return a

    for a in account_data:
        if a.get("nativeBalanceChange", 0) != 0:
            return a

    return account_data[0]


def _sum_token_delta(account_entry: dict) -> Tuple[Optional[str], float]:
    """
    Returns (mint, delta_qty). Chooses mint with largest absolute delta.
    """
    tbc = account_entry.get("tokenBalanceChanges") or []
    if not tbc:
        return None, 0.0

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
        return None, 0.0

    mint = max(deltas, key=lambda m: abs(deltas[m]))
    return mint, float(deltas[mint])


def _estimate_price_usd(sol_delta: float, token_delta: float) -> Optional[float]:
    """
    Estimate token USD price from SOL delta and token delta.
    Only valid for swaps where SOL moved opposite token.
    """
    if token_delta == 0:
        return None
    if sol_delta == 0:
        return None

    # We only trust when signs oppose (buy uses SOL, receive token; sell gives token, receive SOL)
    if (token_delta > 0 and sol_delta < 0) or (token_delta < 0 and sol_delta > 0):
        trade_usd = abs(sol_delta) * SOL_PRICE_USD
        if trade_usd <= 0:
            return None
        price = trade_usd / abs(token_delta)
        # ignore nonsense
        if price <= 0 or price > 1e9:
            return None
        return price

    return None


def apply_paper_from_helius_event(payload0: dict) -> None:
    """
    Called for each event from webhook.
    Decides buy/sell and applies risk logic.
    """
    state = load_paper_state()

    sig = payload0.get("signature")
    desc = payload0.get("description") or payload0.get("desc") or ""

    account_entry = _find_user_account_entry(payload0)
    if not account_entry:
        return

    sol_delta = float(account_entry.get("nativeBalanceChange", 0) or 0.0) / 1e9
    mint, token_delta = _sum_token_delta(account_entry)
    if mint is None or token_delta == 0:
        return

    # Try to estimate a price
    price = _estimate_price_usd(sol_delta, token_delta)

    # Decide side
    # BUY: token up, SOL down
    if token_delta > 0 and sol_delta < 0:
        if price is None:
            # No price → cannot simulate correctly → skip trade, but still log raw event in /events
            return
        paper_buy(state, mint, price_usd=price, sig=sig, desc=desc)

        # after buy, run risk checks (updates peak/trail if needed)
        apply_risk_checks(state, mint, price_usd=price, sig=sig)

    # SELL signal: token down, SOL up
    elif token_delta < 0 and sol_delta > 0:
        if price is None:
            # If no price, we can't simulate; skip
            return

        # Mirror whale sell (% of our position)
        pos = (state.get("positions") or {}).get(mint)
        if pos and float(pos.get("qty") or 0.0) > 0:
            held = float(pos.get("qty") or 0.0)
            sell_qty = held * max(0.0, min(1.0, MIRROR_WHALE_SELL_PCT))
            paper_sell(state, mint, price_usd=price, qty_to_sell=sell_qty, reason="MIRROR_WHALE_SELL", sig=sig, desc=desc)

        # risk checks still apply (trail could trigger etc.)
        apply_risk_checks(state, mint, price_usd=price, sig=sig)

    else:
        # Not a clean swap we can interpret
        return

    # keep trades bounded
    state["trades"] = (state.get("trades") or [])[-200:]

    # ✅ prune dead positions (THE FIX)
    prune_zero_positions(state)

    save_paper_state(state)


# =========================
# API ENDPOINTS
# =========================

@app.get("/health")
def health():
    return {"ok": True, "service": APP_NAME, "time": utc_now_iso()}


@app.get("/events")
def get_events():
    ensure_logfile()
    lines = EVENT_LOG.read_text(encoding="utf-8").splitlines()[-200:]
    return {"count": len(lines), "events": [json.loads(l) for l in lines]}


@app.get("/paper/state")
def paper_state():
    state = load_paper_state()
    prune_zero_positions(state)  # also prune on read, just in case

    positions_value, equity, realized, unrealized = compute_portfolio_metrics(state)

    return {
        "cash": float(state.get("cash") or 0.0),
        "positions": state.get("positions") or {},
        "positions_value_usd": positions_value,
        "equity_usd": equity,
        "realized_pnl_usd": realized,
        "unrealized_pnl_usd": unrealized,
        "trades": (state.get("trades") or [])[-200:],
    }


@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(default=None, convert_underscores=False),
    authorization: Optional[str] = Header(default=None),
):
    # Accept secret from multiple possible headers
    provided = (x_webhook_secret or authorization or "").strip()

    # Authorization: Bearer <secret>
    if provided.lower().startswith("bearer "):
        provided = provided[7:].strip()

    # If someone mistakenly sends "x-webhook-secret: <value>" as the header value
    if provided.lower().startswith("x-webhook-secret"):
        parts = provided.split(":", 1)
        if len(parts) == 2:
            provided = parts[1].strip()

    if WEBHOOK_SECRET and provided != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    payload = await request.json()

    # Log raw event
    record = {
        "received_at": utc_now_iso(),
        "source": "helius",
        "payload": payload,
    }
    append_jsonl(record)

    # Apply paper trading
    if isinstance(payload, list):
        for evt in payload:
            if isinstance(evt, dict):
                apply_paper_from_helius_event(evt)
    elif isinstance(payload, dict):
        apply_paper_from_helius_event(payload)

    return JSONResponse({"ok": True})
