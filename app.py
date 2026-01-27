import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

# =======================
# APP / LOGGING
# =======================

APP_NAME = "sol-paper-bot"
LOG_DIR = Path("logs")
EVENT_LOG = LOG_DIR / "events.jsonl"
PAPER_STATE_FILE = LOG_DIR / "paper_state.json"

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
# TUNABLE PAPER SETTINGS
# =======================
# NOTE: These are SAFE to tune. They won’t break the webhook/event plumbing.

START_CASH_USD = float(os.getenv("START_CASH_USD", "500"))

# How much YOU paper-buy on a whale buy signal (USD)
TRADE_USD = float(os.getenv("TRADE_USD", "100"))
MIN_TRADE_USD = float(os.getenv("MIN_TRADE_USD", "25"))
MAX_TRADE_PCT = float(os.getenv("MAX_TRADE_PCT", "0.15"))  # max % of cash per buy

# Fee model (paper realism)
SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "100"))
FEE_FIXED_USD = float(os.getenv("FEE_FIXED_USD", "1.00"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.020"))  # e.g., slippage+impact

# --- Exit logic knobs ---
# If price <= avg_cost * (1 - STOP_LOSS_PCT) => sell (when we next get a price update event for that mint)
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.25"))  # 25%

# Trailing stop: keep peak price, trail at (1 - TRAIL_PCT)
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "0.20"))  # 20%

# If the tracked wallet sells, do we sell too?
SELL_ON_WHALE_SELL = os.getenv("SELL_ON_WHALE_SELL", "true").lower() in ("1", "true", "yes", "y")

# If selling on whale sell: sell this fraction of our current position (1.0 = all)
WHALE_SELL_FRACTION = float(os.getenv("WHALE_SELL_FRACTION", "1.0"))

# Track multiple wallets if you want (comma-separated)
TRACKED_WALLETS_RAW = os.getenv("TRACKED_WALLETS", "").strip()
TRACKED_WALLET_SINGLE = os.getenv("TRACKED_WALLET", "").strip()  # legacy
TRACKED_WALLETS: List[str] = []
if TRACKED_WALLETS_RAW:
    TRACKED_WALLETS = [w.strip() for w in TRACKED_WALLETS_RAW.split(",") if w.strip()]
elif TRACKED_WALLET_SINGLE:
    TRACKED_WALLETS = [TRACKED_WALLET_SINGLE]


# Common mints we should ignore for “meme token position tracking”
# (Wrapped SOL)
WSOL_MINT = "So11111111111111111111111111111111111111112"


# =======================
# PAPER STATE
# =======================

def _fresh_state() -> Dict[str, Any]:
    return {
        "cash": START_CASH_USD,
        "positions": {},  # mint -> {qty, avg_cost_usd_per_token, peak_price, trail_stop_price, last_price_usd}
        "positions_value_usd": 0.0,
        "equity_usd": START_CASH_USD,
        "realized_pnl_usd": 0.0,
        "unrealized_pnl_usd": 0.0,
        "trades": [],  # most recent
    }


def load_paper_state() -> Dict[str, Any]:
    if not PAPER_STATE_FILE.exists():
        return _fresh_state()
    try:
        return json.loads(PAPER_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        # If file got corrupted somehow, reset safely
        return _fresh_state()


def save_paper_state(state: Dict[str, Any]) -> None:
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =======================
# PARSING HELPERS (HELIUS)
# =======================

def _sum_token_deltas(account_entry: dict) -> Dict[str, float]:
    """Returns mint->delta (positive received, negative sent) for the account_entry."""
    tbc = account_entry.get("tokenBalanceChanges") or []
    if not isinstance(tbc, list) or not tbc:
        return {}

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
    return deltas


def _pick_dominant_mint(deltas: Dict[str, float]) -> Tuple[Optional[str], float]:
    """Pick the mint with the largest absolute delta."""
    if not deltas:
        return None, 0.0
    mint = max(deltas.keys(), key=lambda m: abs(deltas[m]))
    return mint, deltas[mint]


def _derive_price_usd_per_token(sol_delta: float, token_delta: float) -> Optional[float]:
    """
    Use implied price from swap:
      trade_usd ~= abs(sol_delta)*SOL_PRICE_USD
      price ~= trade_usd / abs(token_delta)
    """
    if token_delta == 0:
        return None
    trade_usd = abs(sol_delta) * SOL_PRICE_USD
    if trade_usd <= 0:
        return None
    price = trade_usd / abs(token_delta)
    if price <= 0:
        return None
    return price


# =======================
# PAPER EXECUTION HELPERS
# =======================

def _ensure_position(state: Dict[str, Any], mint: str) -> Dict[str, Any]:
    pos = state["positions"].get(mint)
    if not pos:
        pos = {
            "qty": 0.0,
            "avg_cost_usd_per_token": None,
            "peak_price": None,
            "trail_stop_price": None,
            "last_price_usd": None,
        }
        state["positions"][mint] = pos
    return pos


def _recalc_portfolio(state: Dict[str, Any]) -> None:
    positions_value = 0.0
    unrealized = 0.0

    for mint, pos in list(state.get("positions", {}).items()):
        qty = float(pos.get("qty", 0.0) or 0.0)
        if qty <= 0:
            continue
        last_price = pos.get("last_price_usd")
        avg_cost = pos.get("avg_cost_usd_per_token")

        if isinstance(last_price, (int, float)) and last_price > 0:
            positions_value += qty * float(last_price)

        if isinstance(last_price, (int, float)) and isinstance(avg_cost, (int, float)) and last_price > 0 and avg_cost > 0:
            unrealized += qty * (float(last_price) - float(avg_cost))

    state["positions_value_usd"] = float(positions_value)
    state["unrealized_pnl_usd"] = float(unrealized)
    state["equity_usd"] = float(state.get("cash", 0.0) + positions_value)


def _record_trade(state: Dict[str, Any], trade: Dict[str, Any]) -> None:
    state["trades"].append(trade)
    state["trades"] = state["trades"][-300:]


def paper_buy(state: Dict[str, Any], mint: str, price: float, sig: Optional[str], reason: str) -> None:
    if mint == WSOL_MINT:
        return  # ignore WSOL “positions” for now

    cash = float(state.get("cash", 0.0))
    max_usd = cash * MAX_TRADE_PCT
    trade_usd = min(TRADE_USD, max_usd)

    if trade_usd < MIN_TRADE_USD or trade_usd <= 0:
        return
    if price <= 0:
        return

    fees = FEE_FIXED_USD + trade_usd * FEE_PCT
    total_cost = trade_usd + fees
    if total_cost > cash:
        return

    qty = trade_usd / price

    pos = _ensure_position(state, mint)
    old_qty = float(pos.get("qty", 0.0) or 0.0)
    old_avg = pos.get("avg_cost_usd_per_token")
    old_avg = float(old_avg) if isinstance(old_avg, (int, float)) and old_avg > 0 else None

    new_qty = old_qty + qty
    if old_avg is None:
        new_avg = price
    else:
        # Weighted average cost
        new_avg = (old_qty * old_avg + qty * price) / new_qty

    state["cash"] = cash - total_cost

    pos["qty"] = new_qty
    pos["avg_cost_usd_per_token"] = float(new_avg)
    pos["last_price_usd"] = float(price)

    # Peak + trailing stop init / update
    peak = pos.get("peak_price")
    peak = float(peak) if isinstance(peak, (int, float)) and peak > 0 else None
    if peak is None or price > peak:
        peak = price
    pos["peak_price"] = float(peak)
    pos["trail_stop_price"] = float(peak * (1.0 - TRAIL_PCT))

    _record_trade(state, {
        "ts": utc_now_iso(),
        "type": "SWAP",
        "side": "BUY",
        "mint": mint,
        "qty": qty,
        "price_usd_per_token": price,
        "trade_usd_est": trade_usd,
        "fees_usd_est": fees,
        "sig": sig,
        "desc": reason,
    })


def paper_sell(state: Dict[str, Any], mint: str, sell_qty: float, price: float, sig: Optional[str], reason: str) -> None:
    if mint == WSOL_MINT:
        return
    if price <= 0:
        return

    pos = state.get("positions", {}).get(mint)
    if not pos:
        return

    qty = float(pos.get("qty", 0.0) or 0.0)
    if qty <= 0:
        return

    sell_qty = float(sell_qty)
    sell_qty = _clamp(sell_qty, 0.0, qty)
    if sell_qty <= 0:
        return

    avg_cost = pos.get("avg_cost_usd_per_token")
    avg_cost = float(avg_cost) if isinstance(avg_cost, (int, float)) and avg_cost > 0 else None

    gross = sell_qty * price
    fees = FEE_FIXED_USD + gross * FEE_PCT
    net = gross - fees

    state["cash"] = float(state.get("cash", 0.0)) + net

    # realized pnl (net of fees)
    realized_net = None
    if avg_cost is not None:
        realized_net = sell_qty * (price - avg_cost) - fees
        state["realized_pnl_usd"] = float(state.get("realized_pnl_usd", 0.0)) + float(realized_net)

    remaining = qty - sell_qty
    pos["qty"] = remaining
    pos["last_price_usd"] = float(price)

    if remaining <= 0:
        # close position
        state["positions"].pop(mint, None)

    _record_trade(state, {
        "ts": utc_now_iso(),
        "type": "SWAP",
        "side": "SELL",
        "mint": mint,
        "qty": sell_qty,
        "price_usd_per_token": price,
        "trade_usd_est": gross,
        "fees_usd_est": fees,
        "realized_pnl_net": realized_net,
        "sig": sig,
        "desc": reason,
    })


def evaluate_risk_and_exit(state: Dict[str, Any], mint: str, sig: Optional[str]) -> None:
    """
    Runs after we have a fresh price for the mint.
    NOTE: With no live price feed, exits can only trigger when we receive a new event that implies a price.
    """
    pos = state.get("positions", {}).get(mint)
    if not pos:
        return

    qty = float(pos.get("qty", 0.0) or 0.0)
    if qty <= 0:
        return

    last_price = pos.get("last_price_usd")
    avg_cost = pos.get("avg_cost_usd_per_token")
    peak = pos.get("peak_price")

    if not isinstance(last_price, (int, float)) or float(last_price) <= 0:
        return
    last_price = float(last_price)

    avg_cost = float(avg_cost) if isinstance(avg_cost, (int, float)) and float(avg_cost) > 0 else None
    peak = float(peak) if isinstance(peak, (int, float)) and float(peak) > 0 else None

    # Update peak/trail on new price
    if peak is None or last_price > peak:
        peak = last_price
        pos["peak_price"] = float(peak)
        pos["trail_stop_price"] = float(peak * (1.0 - TRAIL_PCT))

    trail_stop = pos.get("trail_stop_price")
    trail_stop = float(trail_stop) if isinstance(trail_stop, (int, float)) and float(trail_stop) > 0 else None

    # A) Hard stop loss
    if avg_cost is not None:
        stop_price = avg_cost * (1.0 - STOP_LOSS_PCT)
        if last_price <= stop_price:
            paper_sell(state, mint, qty, last_price, sig, f"STOP_LOSS hit (price {last_price:.6g} <= {stop_price:.6g})")
            return

    # B) Trailing stop
    if trail_stop is not None and last_price <= trail_stop:
        paper_sell(state, mint, qty, last_price, sig, f"TRAIL_STOP hit (price {last_price:.6g} <= {trail_stop:.6g})")
        return


# =======================
# EVENT -> PAPER LOGIC
# =======================

def apply_paper_from_account_entry(state: Dict[str, Any], account_entry: dict, sig: Optional[str]) -> None:
    """
    For one tracked wallet's account_entry:
      - detect dominant mint delta + SOL delta
      - derive implied price
      - BUY: paper-buy fixed USD size
      - SELL: optionally mirror whale sells (fraction)
      - always update price + run risk checks
    """
    # nativeBalanceChange is lamports
    sol_delta = float(account_entry.get("nativeBalanceChange", 0)) / 1e9

    deltas = _sum_token_deltas(account_entry)
    mint, token_delta = _pick_dominant_mint(deltas)

    if mint is None or token_delta == 0:
        return

    # Ignore WSOL mint for now (it creates confusing “positions”)
    if mint == WSOL_MINT:
        return

    # Determine side from deltas
    # - BUY-like swap: token received (token_delta>0) and SOL spent (sol_delta<0)
    # - SELL-like swap: token sent (token_delta<0) and SOL received (sol_delta>0)
    side = None
    if token_delta > 0 and sol_delta < 0:
        side = "BUY"
    elif token_delta < 0 and sol_delta > 0:
        side = "SELL"
    else:
        # Could be transfer, fee-only, etc. Still try to infer price if possible.
        side = "UNKNOWN"

    price = _derive_price_usd_per_token(sol_delta, token_delta)
    if price is None:
        return

    # Update last price + peak/trail for this mint (even if we don't trade)
    pos = _ensure_position(state, mint)
    pos["last_price_usd"] = float(price)
    peak = pos.get("peak_price")
    peak = float(peak) if isinstance(peak, (int, float)) and float(peak) > 0 else None
    if peak is None or price > peak:
        peak = price
    pos["peak_price"] = float(peak)
    pos["trail_stop_price"] = float(peak * (1.0 - TRAIL_PCT))

    # Execute paper action based on whale signal
    if side == "BUY":
        paper_buy(state, mint, price, sig, "WHALE BUY signal")
    elif side == "SELL":
        if SELL_ON_WHALE_SELL:
            # Sell fraction of OUR current position
            our_qty = float(state.get("positions", {}).get(mint, {}).get("qty", 0.0) or 0.0)
            if our_qty > 0:
                frac = _clamp(WHALE_SELL_FRACTION, 0.0, 1.0)
                paper_sell(state, mint, our_qty * frac, price, sig, f"WHALE SELL signal (sell {frac:.2f}x)")
        # else: ignore whale sell signals

    # Risk exits (stop loss / trailing)
    evaluate_risk_and_exit(state, mint, sig)


def apply_paper_trade_from_helius_event(evt: dict) -> None:
    state = load_paper_state()

    sig = evt.get("signature") or evt.get("transactionSignature") or None
    account_data = evt.get("accountData") or []
    if not isinstance(account_data, list) or not account_data:
        return

    # If user configured tracked wallets, only process those entries.
    # If none configured, process all entries (useful for early testing).
    for entry in account_data:
        acct = (entry or {}).get("account")
        if TRACKED_WALLETS and acct not in TRACKED_WALLETS:
            continue
        apply_paper_from_account_entry(state, entry, sig)

    _recalc_portfolio(state)
    save_paper_state(state)


# =======================
# ROUTES
# =======================

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
    _recalc_portfolio(state)
    save_paper_state(state)
    return state


@app.post("/paper/reset")
def paper_reset(x_webhook_secret: Optional[str] = Header(default=None, convert_underscores=False),
                authorization: Optional[str] = Header(default=None)):
    # Optional: protect reset with webhook secret if set
    provided = x_webhook_secret or authorization or ""
    if provided.lower().startswith("bearer "):
        provided = provided[7:].strip()
    if WEBHOOK_SECRET and provided != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    state = _fresh_state()
    save_paper_state(state)
    return {"ok": True, "reset": True, "state": state}


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

    # Some UIs mistakenly paste "x-webhook-secret: value" into a single field
    if provided.lower().startswith("x-webhook-secret"):
        parts = provided.split(":", 1)
        if len(parts) == 2:
            provided = parts[1].strip()

    if WEBHOOK_SECRET and provided != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    payload = await request.json()

    record = {"received_at": utc_now_iso(), "source": "helius", "payload": payload}
    append_jsonl(record)

    # Apply to paper engine
    if isinstance(payload, list):
        for evt in payload:
            if isinstance(evt, dict):
                apply_paper_trade_from_helius_event(evt)
    elif isinstance(payload, dict):
        apply_paper_trade_from_helius_event(payload)

    return JSONResponse({"ok": True})
