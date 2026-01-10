import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

# =========================
# APP / LOGGING
# =========================

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

# =========================
# CONFIG (A/B/C/D)
# =========================

# --- Wallet we are "copying" ---
# IMPORTANT: set this in Render env vars so transfer parsing can work:
# TRACKED_WALLET = the wallet address you're monitoring
TRACKED_WALLET = os.getenv("TRACKED_WALLET", "").strip()

# --- Price / fee realism ---
SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "100"))        # rough SOL price for USD estimates
FEE_FIXED_USD = float(os.getenv("FEE_FIXED_USD", "1.00"))       # priority/compute/bribe baseline
FEE_PCT = float(os.getenv("FEE_PCT", "0.020"))                  # 2% slippage+impact estimate

# --- Trade sizing (C) ---
START_CASH_USD = float(os.getenv("START_CASH_USD", "500"))

# Choose how YOU size buys:
# "fixed" = always buy FIXED_TRADE_USD (if you have cash)
# "pct"   = buy min(MAX_TRADE_PCT of cash, FIXED_TRADE_USD cap if you set it)
TRADE_MODE = os.getenv("TRADE_MODE", "fixed").lower()           # fixed | pct
FIXED_TRADE_USD = float(os.getenv("FIXED_TRADE_USD", "100"))    # recommended based on your fee pain
MAX_TRADE_PCT = float(os.getenv("MAX_TRADE_PCT", "0.15"))       # max 15% of cash per buy
MIN_TRADE_USD = float(os.getenv("MIN_TRADE_USD", "75"))         # avoid fee death

# Skip trades where fees dwarf size
MAX_FEE_FRACTION = float(os.getenv("MAX_FEE_FRACTION", "0.25")) # if fees > 25% of trade, skip

# --- Risk management (B) ---
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.20"))       # 20% hard stop
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "0.20"))  # keep stop within ~20% of peak
TRAIL_ACTIVATE_PCT = float(os.getenv("TRAIL_ACTIVATE_PCT", "0.20")) # start trailing after +20%

# Partial take-profits:
# Format env like: "0.2:0.25,0.4:0.25,0.8:0.25" (profit% : fraction_to_sell)
TP_STEPS_RAW = os.getenv("TP_STEPS", "0.2:0.25,0.4:0.25,0.8:0.25")

def _parse_tp_steps(s: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            a, b = part.split(":")
            out.append((float(a), float(b)))
        except Exception:
            continue
    # sort by profit target ascending
    out.sort(key=lambda x: x[0])
    return out

TP_STEPS = _parse_tp_steps(TP_STEPS_RAW)

# --- Live trading scaffold (D) ---
# This code WILL NOT place real trades unless you implement execute_live_trade()
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "false").lower() in ("1", "true", "yes")

# =========================
# PAPER STATE
# =========================

def load_paper_state() -> Dict[str, Any]:
    if not PAPER_STATE_FILE.exists():
        return {
            "cash": START_CASH_USD,
            "positions": {},  # mint -> {qty, avg_cost_usd_per_token, peak_price, trail_stop_price, tp_done[]}
            "trades": []      # list of trade records
        }
    return json.loads(PAPER_STATE_FILE.read_text(encoding="utf-8"))

def save_paper_state(state: Dict[str, Any]) -> None:
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

# =========================
# EVENT PARSING (A)
# =========================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _infer_lamports_to_sol(amount: Any) -> float:
    """
    Enhanced Helius nativeTransfers 'amount' is typically in lamports (int).
    Convert to SOL.
    """
    try:
        v = float(amount)
    except Exception:
        return 0.0
    # If it's huge like lamports, divide; if it's already SOL, it's usually < 10^4.
    return v / 1e9 if v > 1e6 else v

def _sum_transfers_for_wallet(payload0: Dict[str, Any], wallet: str) -> Tuple[float, Dict[str, float]]:
    """
    Returns:
      sol_delta (SOL): positive means wallet gained SOL, negative means spent SOL
      token_deltas {mint: delta_tokens}: positive means wallet gained tokens, negative means spent tokens
    Uses nativeTransfers + tokenTransfers if present (best),
    falls back to accountData if not.
    """
    sol_delta = 0.0
    token_deltas: Dict[str, float] = {}

    # Preferred: nativeTransfers
    nt = payload0.get("nativeTransfers")
    if isinstance(nt, list) and wallet:
        for t in nt:
            frm = (t.get("fromUserAccount") or "").strip()
            to = (t.get("toUserAccount") or "").strip()
            amt = _infer_lamports_to_sol(t.get("amount"))
            if to == wallet:
                sol_delta += amt
            elif frm == wallet:
                sol_delta -= amt

    # Preferred: tokenTransfers
    tt = payload0.get("tokenTransfers")
    if isinstance(tt, list) and wallet:
        for t in tt:
            frm = (t.get("fromUserAccount") or "").strip()
            to = (t.get("toUserAccount") or "").strip()
            mint = (t.get("mint") or "").strip()
            amt = _safe_float(t.get("tokenAmount"), 0.0)
            if not mint:
                continue
            if to == wallet:
                token_deltas[mint] = token_deltas.get(mint, 0.0) + amt
            elif frm == wallet:
                token_deltas[mint] = token_deltas.get(mint, 0.0) - amt

    # Fallback: accountData (your older method)
    if (abs(sol_delta) < 1e-12) and (not token_deltas):
        account_data = payload0.get("accountData") or []
        if isinstance(account_data, list) and account_data:
            # pick entry with token changes, else with native change
            chosen = None
            for a in account_data:
                if (a.get("tokenBalanceChanges") or []) != []:
                    chosen = a
                    break
            if chosen is None:
                for a in account_data:
                    if (a.get("nativeBalanceChange") or 0) != 0:
                        chosen = a
                        break
            if chosen is None:
                chosen = account_data[0]

            lamports = chosen.get("nativeBalanceChange", 0) or 0
            sol_delta = _safe_float(lamports, 0.0) / 1e9

            tbc = chosen.get("tokenBalanceChanges") or []
            if isinstance(tbc, list):
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
                    token_deltas[mint] = token_deltas.get(mint, 0.0) + qty

    return sol_delta, token_deltas

def _pick_primary_mint(token_deltas: Dict[str, float]) -> Tuple[Optional[str], float]:
    if not token_deltas:
        return None, 0.0
    mint = max(token_deltas.keys(), key=lambda m: abs(token_deltas[m]))
    return mint, token_deltas[mint]

def _parse_description_side(payload0: Dict[str, Any]) -> Optional[str]:
    """
    Try to infer BUY/SELL from description text when transfers are incomplete.
    Example Helius description often like:
      "... swapped 0.992 SOL for 134,773 USDT"
    We only use this as a hint.
    """
    desc = (payload0.get("description") or "").lower()
    if "swapped" not in desc:
        return None
    # If it says "swapped X sol for Y ..." that usually means wallet spent SOL and received token => BUY
    if re.search(r"swapped\s+[0-9\.,]+\s+sol\s+for\s+", desc):
        return "BUY"
    # If it says "swapped ... for X sol" that usually means received SOL => SELL
    if re.search(r"\s+for\s+[0-9\.,]+\s+sol", desc):
        return "SELL"
    return None

def _infer_price_usd_per_token(sol_delta: float, token_delta: float) -> Optional[float]:
    """
    Price per token from swap implied by SOL spent/received.
    Needs both to be non-zero and SOL_PRICE_USD set.
    """
    if abs(sol_delta) < 1e-12 or abs(token_delta) < 1e-12:
        return None
    # value in USD / token amount
    trade_usd = abs(sol_delta) * SOL_PRICE_USD
    return trade_usd / abs(token_delta) if abs(token_delta) > 0 else None

# =========================
# RISK / SIZING (B/C)
# =========================

def _estimate_fees_usd(trade_usd: float) -> float:
    return FEE_FIXED_USD + (trade_usd * FEE_PCT)

def _choose_buy_trade_usd(state_cash: float) -> float:
    if TRADE_MODE == "pct":
        return max(0.0, min(state_cash * MAX_TRADE_PCT, FIXED_TRADE_USD if FIXED_TRADE_USD > 0 else state_cash))
    # fixed
    return max(0.0, min(FIXED_TRADE_USD, state_cash))

def _ensure_position_struct(pos: Dict[str, Any]) -> Dict[str, Any]:
    pos.setdefault("qty", 0.0)
    pos.setdefault("avg_cost_usd_per_token", 0.0)
    pos.setdefault("peak_price", 0.0)
    pos.setdefault("trail_stop_price", 0.0)
    pos.setdefault("tp_done", [False for _ in TP_STEPS])
    return pos

def _apply_take_profit_and_trailing(state: Dict[str, Any], mint: str, current_price: float) -> List[Dict[str, Any]]:
    """
    If price hits TP steps or trailing stop, create paper SELL trades.
    Returns list of generated trades (records) for logging.
    """
    trades_out: List[Dict[str, Any]] = []
    positions = state.get("positions") or {}
    pos = positions.get(mint)
    if not isinstance(pos, dict):
        return trades_out

    pos = _ensure_position_struct(pos)
    qty = float(pos.get("qty", 0.0))
    if qty <= 0:
        return trades_out

    avg = float(pos.get("avg_cost_usd_per_token", 0.0))
    if avg <= 0:
        return trades_out

    # Update peak
    peak = float(pos.get("peak_price", 0.0))
    if current_price > peak:
        peak = current_price
        pos["peak_price"] = peak

    # Hard stop-loss from avg
    hard_stop = avg * (1.0 - STOP_LOSS_PCT)

    # Trailing stop activates after +TRAIL_ACTIVATE_PCT
    trail_stop = float(pos.get("trail_stop_price", 0.0))
    if peak >= avg * (1.0 + TRAIL_ACTIVATE_PCT):
        # keep stop within TRAILING_STOP_PCT of peak
        candidate = peak * (1.0 - TRAILING_STOP_PCT)
        trail_stop = max(trail_stop, candidate)
        pos["trail_stop_price"] = trail_stop

    effective_stop = max(hard_stop, trail_stop) if trail_stop > 0 else hard_stop

    # Take profits
    tp_done: List[bool] = pos.get("tp_done") or [False for _ in TP_STEPS]
    if len(tp_done) != len(TP_STEPS):
        tp_done = [False for _ in TP_STEPS]
    pos["tp_done"] = tp_done

    for i, (tp_pct, sell_frac) in enumerate(TP_STEPS):
        if tp_done[i]:
            continue
        target_price = avg * (1.0 + tp_pct)
        if current_price >= target_price and qty > 0:
            sell_qty = qty * float(sell_frac)
            if sell_qty <= 0:
                tp_done[i] = True
                continue

            trade_usd = sell_qty * current_price
            fees = _estimate_fees_usd(trade_usd)

            # Apply sell
            state["cash"] = float(state.get("cash", 0.0)) + trade_usd - fees
            qty -= sell_qty
            pos["qty"] = qty
            tp_done[i] = True

            trades_out.append({
                "ts": utc_now_iso(),
                "type": "AUTO_TAKE_PROFIT",
                "side": "SELL",
                "mint": mint,
                "qty": sell_qty,
                "price_usd_per_token": current_price,
                "trade_usd_est": trade_usd,
                "fees_usd_est": fees,
                "reason": f"TP +{tp_pct*100:.0f}% sold {sell_frac*100:.0f}%",
            })

    # Stop-out (if after TPs we still have qty)
    if qty > 0 and current_price <= effective_stop:
        sell_qty = qty
        trade_usd = sell_qty * current_price
        fees = _estimate_fees_usd(trade_usd)

        state["cash"] = float(state.get("cash", 0.0)) + trade_usd - fees
        pos["qty"] = 0.0

        trades_out.append({
            "ts": utc_now_iso(),
            "type": "AUTO_STOP",
            "side": "SELL",
            "mint": mint,
            "qty": sell_qty,
            "price_usd_per_token": current_price,
            "trade_usd_est": trade_usd,
            "fees_usd_est": fees,
            "reason": f"Stop hit (effective_stop={effective_stop:.8f})",
        })

    positions[mint] = pos
    state["positions"] = positions
    return trades_out

# =========================
# LIVE TRADING (D scaffold)
# =========================

def execute_live_trade(side: str, mint: str, qty: float, price_hint_usd: Optional[float], payload0: Dict[str, Any]) -> None:
    """
    SAFETY: This is a scaffold. It does NOT execute real trades.
    If you later want real trading, you'd implement:
      - quoting (Jupiter, etc)
      - signing with a trading keypair
      - sending tx with priority fees
      - confirmations + retries
    """
    raise NotImplementedError("Live trading is not implemented in this build (by design).")

# =========================
# PAPER TRADING ENGINE (A/B/C)
# =========================

def apply_paper_trade_from_helius(payload0: Dict[str, Any]) -> None:
    if not isinstance(payload0, dict):
        return

    evt_type = (payload0.get("type") or "").upper()
    # We mainly care about SWAP-like activity
    if "SWAP" not in evt_type and "TOKEN_SWAP" not in evt_type:
        # Still allow description-based swap detection if it looks like a swap
        if "swapped" not in (payload0.get("description") or "").lower():
            return

    state = load_paper_state()
    cash = float(state.get("cash", 0.0))

    sol_delta, token_deltas = _sum_transfers_for_wallet(payload0, TRACKED_WALLET)
    mint, token_delta = _pick_primary_mint(token_deltas)
    if not mint or abs(token_delta) < 1e-12:
        return

    # Infer side
    side = "UNKNOWN"
    if token_delta > 0 and sol_delta < 0:
        side = "BUY"
    elif token_delta < 0 and sol_delta > 0:
        side = "SELL"
    else:
        hint = _parse_description_side(payload0)
        if hint:
            side = hint

    # Infer price
    price = _infer_price_usd_per_token(sol_delta, token_delta)

    # If we can't infer price, we still record, but we can't do risk logic well
    # We'll still update qty and cash using a conservative estimate:
    # - For buys, use chosen trade size and infer qty from implied token_delta scaling.
    # - For sells, require price; otherwise skip sell simulation.
    positions = state.get("positions") or {}
    pos = positions.get(mint, {})
    pos = _ensure_position_struct(pos)

    now = utc_now_iso()

    # ---------- BUY ----------
    if side == "BUY" or (side == "UNKNOWN" and token_delta > 0):
        buy_usd = _choose_buy_trade_usd(cash)
        if buy_usd < MIN_TRADE_USD:
            return

        fees = _estimate_fees_usd(buy_usd)
        if fees > buy_usd * MAX_FEE_FRACTION:
            # fees too high vs size
            return

        if cash < (buy_usd + fees):
            return

        # If we have an implied price, we can buy qty = buy_usd / price
        # If price unknown, we "scale" qty from wallet's token_delta vs wallet's trade_usd if possible.
        buy_qty = 0.0
        if price and price > 0:
            buy_qty = buy_usd / price
        else:
            # fallback: assume token_delta corresponds to buy_usd (rough)
            # This is imperfect but better than nothing.
            buy_qty = abs(token_delta)

        if buy_qty <= 0:
            return

        # Update cash
        state["cash"] = cash - buy_usd - fees

        # Update avg cost
        old_qty = float(pos.get("qty", 0.0))
        old_avg = float(pos.get("avg_cost_usd_per_token", 0.0))

        new_qty = old_qty + buy_qty
        new_avg = old_avg
        if price and price > 0:
            if old_qty <= 0:
                new_avg = price
            else:
                new_avg = ((old_qty * old_avg) + (buy_qty * price)) / new_qty

        pos["qty"] = new_qty
        if new_avg > 0:
            pos["avg_cost_usd_per_token"] = new_avg

        # Reset peak/trailing on fresh positions
        if float(pos.get("peak_price", 0.0)) <= 0 and price and price > 0:
            pos["peak_price"] = price
        if float(pos.get("trail_stop_price", 0.0)) <= 0 and new_avg > 0:
            pos["trail_stop_price"] = new_avg * (1.0 - STOP_LOSS_PCT)

        # Record trade
        trade_record = {
            "ts": now,
            "type": evt_type or "SWAP",
            "side": "BUY",
            "mint": mint,
            "qty": buy_qty,
            "price_usd_per_token": price,
            "trade_usd_est": buy_usd,
            "fees_usd_est": fees,
            "sig": payload0.get("signature"),
            "desc": payload0.get("description"),
        }
        state["trades"] = (state.get("trades") or []) + [trade_record]

        positions[mint] = pos
        state["positions"] = positions

        # Live trading (D) scaffold
        if ENABLE_LIVE_TRADING:
            execute_live_trade("BUY", mint, buy_qty, price, payload0)

        # Trim trades
        state["trades"] = state["trades"][-200:]
        save_paper_state(state)
        return

    # ---------- SELL ----------
    if side == "SELL" or (side == "UNKNOWN" and token_delta < 0):
        qty = float(pos.get("qty", 0.0))
        if qty <= 0:
            return

        # Need a price to simulate a sell realistically
        if not price or price <= 0:
            return

        # Sell sizing: sell up to what we have. Prefer selling proportional to wallet delta magnitude.
        sell_qty = min(qty, abs(token_delta))
        if sell_qty <= 0:
            return

        sell_usd = sell_qty * price
        if sell_usd < 1e-9:
            return

        fees = _estimate_fees_usd(sell_usd)
        if fees > sell_usd * MAX_FEE_FRACTION:
            # if fees dwarf value, skip
            return

        # Apply
        state["cash"] = float(state.get("cash", 0.0)) + sell_usd - fees
        pos["qty"] = qty - sell_qty

        # Record trade
        trade_record = {
            "ts": now,
            "type": evt_type or "SWAP",
            "side": "SELL",
            "mint": mint,
            "qty": sell_qty,
            "price_usd_per_token": price,
            "trade_usd_est": sell_usd,
            "fees_usd_est": fees,
            "sig": payload0.get("signature"),
            "desc": payload0.get("description"),
        }
        state["trades"] = (state.get("trades") or []) + [trade_record]

        positions[mint] = pos
        state["positions"] = positions

        # Live trading (D) scaffold
        if ENABLE_LIVE_TRADING:
            execute_live_trade("SELL", mint, sell_qty, price, payload0)

        state["trades"] = state["trades"][-200:]
        save_paper_state(state)
        return

    # ---------- If we have a price update, run risk logic (B) ----------
    if price and price > 0:
        generated = _apply_take_profit_and_trailing(state, mint, price)
        if generated:
            state["trades"] = (state.get("trades") or []) + generated
            state["trades"] = state["trades"][-200:]
            save_paper_state(state)

# =========================
# ROUTES
# =========================

@app.get("/paper/state")
def paper_state():
    return load_paper_state()

@app.get("/paper/trades")
def paper_trades():
    st = load_paper_state()
    return {"count": len(st.get("trades") or []), "trades": (st.get("trades") or [])}

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

    # Apply paper trading for list or single event
    if isinstance(payload, list):
        for evt in payload:
            apply_paper_trade_from_helius(evt)
    elif isinstance(payload, dict):
        apply_paper_trade_from_helius(payload)

    return JSONResponse({"ok": True})
