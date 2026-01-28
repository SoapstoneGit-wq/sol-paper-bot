import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

APP_NAME = "sol-paper-bot"
LOG_DIR = Path("logs")
EVENT_LOG = LOG_DIR / "events.jsonl"
PAPER_STATE_FILE = LOG_DIR / "paper_state.json"

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

# -------- Paper trading knobs (env overridable) --------
START_CASH_USD = float(os.getenv("START_CASH_USD", "500"))

# If SCALE_MODE=fixed, every BUY uses TRADE_USD_FIXED (subject to cash/limits)
SCALE_MODE = os.getenv("SCALE_MODE", "fixed").lower()  # fixed | whale
TRADE_USD_FIXED = float(os.getenv("TRADE_USD_FIXED", "100"))

MIN_TRADE_USD = float(os.getenv("MIN_TRADE_USD", "25"))
MAX_TRADE_PCT = float(os.getenv("MAX_TRADE_PCT", "0.20"))  # 20% of equity max per buy

# Fees / slippage model (USD)
FEE_FIXED_USD = float(os.getenv("FEE_FIXED_USD", "1.00"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.020"))  # 2%

# SOL pricing (only used when the swap quote leg is SOL)
SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "100"))

# When whale sells a mint: sell "all" or "event_qty" (sell the same qty as whale event)
SELL_ON_WHALE_SELL = os.getenv("SELL_ON_WHALE_SELL", "all").lower()  # all | event_qty

# ---- Risk management knobs ----
# Stop loss triggers when price <= avg_cost * (1 - STOP_LOSS_PCT)
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.20"))  # 0.20 = -20%

# Trailing stop activates once price >= avg_cost * (1 + TRAIL_ACTIVATE_PCT)
TRAIL_ACTIVATE_PCT = float(os.getenv("TRAIL_ACTIVATE_PCT", "0.30"))  # +30%

# Once activated, trailing stop is peak_price * (1 - TRAIL_PCT)
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "0.20"))  # trail 20% off peak

# If true, stop loss / trailing stop can force-sell even if whale didn't sell
ENABLE_AUTO_STOPS = os.getenv("ENABLE_AUTO_STOPS", "true").lower() in ("1", "true", "yes", "y")

# Stablecoin mints (Solana mainnet)
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"

app = FastAPI(title=APP_NAME)

# =======================
# helpers / storage
# =======================

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

def load_paper_state() -> Dict[str, Any]:
    if not PAPER_STATE_FILE.exists():
        return {
            "cash": START_CASH_USD,
            "positions": {},      # mint -> {qty, avg_cost_usd_per_token, last_price_usd, peak_price_usd, trail_active}
            "realized_pnl": 0.0,
            "trades": []
        }
    return json.loads(PAPER_STATE_FILE.read_text(encoding="utf-8"))

def save_paper_state(state: Dict[str, Any]) -> None:
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def _positions_value(state: Dict[str, Any]) -> float:
    total = 0.0
    for _, pos in (state.get("positions") or {}).items():
        try:
            qty = float(pos.get("qty", 0.0))
            px = float(pos.get("last_price_usd") or 0.0)
            total += qty * px
        except Exception:
            continue
    return total

def _equity(state: Dict[str, Any]) -> float:
    return float(state.get("cash", 0.0)) + _positions_value(state)

# =======================
# Helius parsing
# =======================

def _find_user_account(payload0: dict) -> dict:
    account_data = payload0.get("accountData") or []
    if not isinstance(account_data, list) or not account_data:
        return {}

    for a in account_data:
        tbc = a.get("tokenBalanceChanges") or []
        if isinstance(tbc, list) and len(tbc) > 0:
            return a

    for a in account_data:
        if float(a.get("nativeBalanceChange", 0) or 0) != 0:
            return a

    return account_data[0]

def _token_deltas(account_entry: dict) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    tbc = account_entry.get("tokenBalanceChanges") or []
    if not isinstance(tbc, list):
        return deltas

    for ch in tbc:
        mint = ch.get("mint")
        raw = (ch.get("rawTokenAmount") or {}).get("tokenAmount")
        dec = (ch.get("rawTokenAmount") or {}).get("decimals", 0)
        if not mint or raw is None:
            continue
        try:
            qty = float(raw) / (10 ** int(dec))
        except Exception:
            continue
        deltas[mint] = deltas.get(mint, 0.0) + qty

    return deltas

def _pick_quote_and_base(account_entry: dict) -> Tuple[Optional[str], float, Optional[str], float]:
    deltas = _token_deltas(account_entry)
    sol_delta = float(account_entry.get("nativeBalanceChange", 0) or 0) / 1e9  # SOL

    usdc_delta = deltas.get(USDC_MINT, 0.0)
    usdt_delta = deltas.get(USDT_MINT, 0.0)

    quote_kind: Optional[str] = None
    quote_delta: float = 0.0

    if abs(usdc_delta) > 0:
        quote_kind, quote_delta = "USDC", usdc_delta
    elif abs(usdt_delta) > 0:
        quote_kind, quote_delta = "USDT", usdt_delta
    elif abs(sol_delta) > 0:
        quote_kind, quote_delta = "SOL", sol_delta
    else:
        quote_kind, quote_delta = None, 0.0

    candidates = {m: d for m, d in deltas.items() if m not in (USDC_MINT, USDT_MINT)}
    if not candidates:
        return quote_kind, quote_delta, None, 0.0

    base_mint = max(candidates, key=lambda m: abs(candidates[m]))
    base_delta = candidates[base_mint]
    return quote_kind, quote_delta, base_mint, base_delta

# =======================
# Paper trading execution
# =======================

def _calc_trade_usd(quote_kind: Optional[str], quote_delta: float) -> Optional[float]:
    if quote_kind == "USDC" or quote_kind == "USDT":
        return abs(quote_delta)
    if quote_kind == "SOL":
        return abs(quote_delta) * SOL_PRICE_USD
    return None

def _infer_side(base_delta: float, quote_delta: float) -> str:
    if base_delta > 0 and quote_delta < 0:
        return "BUY"
    if base_delta < 0 and quote_delta > 0:
        return "SELL"
    return "BUY" if base_delta > 0 else "SELL"

def _fees_usd(notional_usd: float) -> float:
    return float(FEE_FIXED_USD + (notional_usd * FEE_PCT))

def _force_sell(state: Dict[str, Any], mint: str, qty: float, px: float, reason: str, sig: Optional[str], desc: str) -> None:
    positions = state.setdefault("positions", {})
    pos = positions.get(mint)
    if not pos:
        return

    pos_qty = float(pos.get("qty", 0.0) or 0.0)
    pos_avg = float(pos.get("avg_cost_usd_per_token", 0.0) or 0.0)
    if qty <= 0 or pos_qty <= 0:
        return

    qty = min(qty, pos_qty)
    sell_usd = qty * px
    if sell_usd < MIN_TRADE_USD:
        return

    fees = _fees_usd(sell_usd)
    gross = qty * (px - pos_avg)
    net = gross - fees

    state["cash"] = float(state.get("cash", 0.0) or 0.0) + sell_usd - fees
    state["realized_pnl"] = float(state.get("realized_pnl", 0.0) or 0.0) + net

    new_qty = pos_qty - qty
    if new_qty <= 1e-12:
        positions.pop(mint, None)
    else:
        pos["qty"] = new_qty
        positions[mint] = pos

    state["trades"].append({
        "ts": utc_now_iso(),
        "type": "SWAP",
        "side": "SELL",
        "mint": mint,
        "qty": qty,
        "price_usd_per_token": px,
        "trade_usd_est": sell_usd,
        "fees_usd_est": fees,
        "realized_pnl_net": net,
        "reason": reason,
        "sig": sig,
        "desc": desc or "",
    })

def _apply_auto_stops_if_needed(state: Dict[str, Any], mint: str, px: float, sig: Optional[str], desc: str) -> None:
    if not ENABLE_AUTO_STOPS:
        return

    positions = state.get("positions") or {}
    pos = positions.get(mint)
    if not pos:
        return

    qty = float(pos.get("qty", 0.0) or 0.0)
    if qty <= 0:
        return

    avg = float(pos.get("avg_cost_usd_per_token", 0.0) or 0.0)
    if avg <= 0:
        return

    # Stop loss threshold
    stop_price = avg * (1.0 - STOP_LOSS_PCT)

    # Track peak + trailing activation
    peak = pos.get("peak_price_usd")
    if peak is None:
        peak = px
    try:
        peak = float(peak)
    except Exception:
        peak = px

    peak = max(peak, px)
    pos["peak_price_usd"] = peak

    trail_active = bool(pos.get("trail_active", False))
    if not trail_active and px >= avg * (1.0 + TRAIL_ACTIVATE_PCT):
        trail_active = True
    pos["trail_active"] = trail_active

    # Trailing stop price
    trail_stop = None
    if trail_active:
        trail_stop = peak * (1.0 - TRAIL_PCT)
        pos["trail_stop_price_usd"] = trail_stop
    else:
        pos["trail_stop_price_usd"] = None

    # Persist updated pos
    positions[mint] = pos
    state["positions"] = positions

    # Trigger checks (stop loss first, then trailing)
    if px <= stop_price:
        _force_sell(state, mint, qty=qty, px=px, reason="STOP_LOSS", sig=sig, desc=desc)
        return

    if trail_active and trail_stop is not None and px <= trail_stop:
        _force_sell(state, mint, qty=qty, px=px, reason="TRAILING_STOP", sig=sig, desc=desc)
        return

def apply_paper_trade_from_helius(payload0: dict) -> None:
    account_entry = _find_user_account(payload0)
    if not account_entry:
        return

    quote_kind, quote_delta, base_mint, base_delta = _pick_quote_and_base(account_entry)
    if not base_mint or base_delta == 0:
        return

    trade_usd_whale = _calc_trade_usd(quote_kind, quote_delta)
    if trade_usd_whale is None or trade_usd_whale <= 0:
        return

    side = _infer_side(base_delta, quote_delta)
    px = trade_usd_whale / max(abs(base_delta), 1e-12)

    sig = payload0.get("signature")
    desc = payload0.get("description", "") or ""

    state = load_paper_state()
    positions = state.setdefault("positions", {})

    # Ensure position structure
    pos = positions.get(base_mint) or {
        "qty": 0.0,
        "avg_cost_usd_per_token": 0.0,
        "last_price_usd": px,
        "peak_price_usd": px,
        "trail_active": False,
        "trail_stop_price_usd": None,
    }

    pos_qty = float(pos.get("qty", 0.0) or 0.0)
    pos_avg = float(pos.get("avg_cost_usd_per_token", 0.0) or 0.0)

    # Always update last price + peak tracking
    pos["last_price_usd"] = px
    positions[base_mint] = pos
    state["positions"] = positions

    # If we already hold this mint, apply auto-stops using updated px
    if pos_qty > 0:
        _apply_auto_stops_if_needed(state, base_mint, px, sig=sig, desc=desc)
        # Note: stops may have sold and removed the position, so refresh local
        positions = state.get("positions") or {}
        pos = positions.get(base_mint) or pos
        pos_qty = float(pos.get("qty", 0.0) or 0.0)
        pos_avg = float(pos.get("avg_cost_usd_per_token", 0.0) or 0.0)

    # ---------- BUY ----------
    if side == "BUY":
        equity = _equity(state)
        if equity <= 0:
            save_paper_state(state)
            return

        buy_usd = trade_usd_whale if SCALE_MODE == "whale" else TRADE_USD_FIXED
        buy_usd = min(buy_usd, equity * MAX_TRADE_PCT)
        if buy_usd < MIN_TRADE_USD:
            save_paper_state(state)
            return

        cash = float(state.get("cash", 0.0) or 0.0)
        fees = _fees_usd(buy_usd)
        total_cost = buy_usd + fees
        if cash < total_cost:
            save_paper_state(state)
            return

        our_qty = buy_usd / px

        state["cash"] = cash - buy_usd - fees

        new_qty = pos_qty + our_qty
        new_avg = ((pos_qty * pos_avg) + buy_usd) / max(new_qty, 1e-12)

        pos["qty"] = new_qty
        pos["avg_cost_usd_per_token"] = new_avg
        pos["last_price_usd"] = px
        # reset peak tracking to at least current price
        try:
            pos["peak_price_usd"] = max(float(pos.get("peak_price_usd") or px), px)
        except Exception:
            pos["peak_price_usd"] = px

        positions[base_mint] = pos
        state["positions"] = positions

        state["trades"].append({
            "ts": utc_now_iso(),
            "type": "SWAP",
            "side": "BUY",
            "mint": base_mint,
            "qty": our_qty,
            "price_usd_per_token": px,
            "trade_usd_est": buy_usd,
            "fees_usd_est": fees,
            "sig": sig,
            "desc": desc,
        })

    # ---------- SELL (whale sell mirror) ----------
    else:
        # If we don't hold it, nothing to sell
        if pos_qty <= 0:
            save_paper_state(state)
            return

        if SELL_ON_WHALE_SELL == "event_qty":
            # Whale base_delta is negative for sells.
            target_qty = min(pos_qty, abs(base_delta))
        else:
            target_qty = pos_qty

        _force_sell(state, base_mint, qty=target_qty, px=px, reason="MIRROR_WHALE_SELL", sig=sig, desc=desc)

    # keep last 500 trades
    state["trades"] = state.get("trades", [])[-500:]
    save_paper_state(state)

# =======================
# API endpoints
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

    pos_val = _positions_value(state)
    eq = float(state.get("cash", 0.0) or 0.0) + pos_val
    realized = float(state.get("realized_pnl", 0.0) or 0.0)
    unrealized = 0.0

    for _, pos in (state.get("positions") or {}).items():
        try:
            qty = float(pos.get("qty", 0.0))
            avg = float(pos.get("avg_cost_usd_per_token", 0.0))
            px = float(pos.get("last_price_usd") or 0.0)
            unrealized += qty * (px - avg)
        except Exception:
            continue

    return {
        "cash": float(state.get("cash", 0.0) or 0.0),
        "positions": state.get("positions", {}),
        "positions_value_usd": pos_val,
        "equity_usd": eq,
        "realized_pnl_usd": realized,
        "unrealized_pnl_usd": unrealized,
        "trades": state.get("trades", [])[-50:],
    }

@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(default=None, convert_underscores=False),
    authorization: Optional[str] = Header(default=None),
):
    # Accept: x-webhook-secret header OR Authorization header OR raw value in Authorization
    provided = x_webhook_secret or authorization or ""

    # Authorization: Bearer <secret>
    if isinstance(provided, str) and provided.lower().startswith("bearer "):
        provided = provided[7:].strip()

    # If someone typed "x-webhook-secret: value" into a single field, salvage it
    if isinstance(provided, str) and provided.lower().startswith("x-webhook-secret"):
        parts = provided.split(":", 1)
        if len(parts) == 2:
            provided = parts[1].strip()

    # If a secret is configured, require it
    if WEBHOOK_SECRET and provided != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    payload = await request.json()

    record = {"received_at": utc_now_iso(), "source": "helius", "payload": payload}
    append_jsonl(record)

    # Apply paper trading
    if isinstance(payload, list):
        for evt in payload:
            if isinstance(evt, dict):
                apply_paper_trade_from_helius(evt)
    elif isinstance(payload, dict):
        apply_paper_trade_from_helius(payload)

    return JSONResponse({"ok": True})
