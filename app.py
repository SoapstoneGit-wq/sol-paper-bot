import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

APP_NAME = "sol-paper-bot"
LOG_DIR = Path("logs")
EVENT_LOG = LOG_DIR / "events.jsonl"

# =======================
# ENV / SETTINGS (easy tune)
# =======================

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

# Paper account
START_CASH_USD = float(os.getenv("START_CASH_USD", "500"))

# How much YOU buy per whale buy (fixed USD per buy) — simplest + safest to start
TRADE_USD_PER_BUY = float(os.getenv("TRADE_USD_PER_BUY", "100"))

# Safety caps
MAX_TRADE_PCT_OF_CASH = float(os.getenv("MAX_TRADE_PCT_OF_CASH", "0.25"))  # cap buy size relative to cash
MIN_TRADE_USD = float(os.getenv("MIN_TRADE_USD", "20"))  # ignore tiny buys

# Fees / realism
FEE_FIXED_USD = float(os.getenv("FEE_FIXED_USD", "1.00"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.02"))

# Price conversion (only used when payload gives SOL delta but not price directly)
SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "100"))

# Risk rules (the new stuff)
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.25"))          # 0.25 = -25%
TRAIL_STOP_PCT = float(os.getenv("TRAIL_STOP_PCT", "0.20"))        # 0.20 = 20% off peak
TRAIL_ARM_PROFIT_PCT = float(os.getenv("TRAIL_ARM_PROFIT_PCT", "0.30"))  # only arm trailing after +30%

# Whale sell mirroring
MIRROR_WHALE_SELLS = os.getenv("MIRROR_WHALE_SELLS", "true").lower() in ("1", "true", "yes")

# Take profit ladder (optional)
# Example: TAKE_PROFIT_PCTS="0.5,1.0,2.0" and TAKE_PROFIT_SELL_PCTS="0.25,0.25,0.25"
TP_PCTS = [p.strip() for p in os.getenv("TAKE_PROFIT_PCTS", "").split(",") if p.strip()]
TP_SELLS = [p.strip() for p in os.getenv("TAKE_PROFIT_SELL_PCTS", "").split(",") if p.strip()]
TAKE_PROFIT_PCTS = [float(x) for x in TP_PCTS] if TP_PCTS else []
TAKE_PROFIT_SELL_PCTS = [float(x) for x in TP_SELLS] if TP_SELLS else []

# Tracked wallet filter (optional; you can leave blank and just log everything)
TRACKED_WALLET = os.getenv("TRACKED_WALLET", "").strip()

# =======================
# APP
# =======================

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
# PAPER STATE
# =======================

PAPER_STATE_FILE = LOG_DIR / "paper_state.json"

def _default_state() -> dict:
    return {
        "cash": START_CASH_USD,
        "positions": {},  # mint -> {qty, avg_cost_usd_per_token, last_price_usd, peak_price_usd, trail_stop_price_usd, tp_done[]}
        "realized_pnl_usd": 0.0,
        "trades": [],     # latest 200
        "updated_at": utc_now_iso(),
    }

def load_paper_state() -> dict:
    if not PAPER_STATE_FILE.exists():
        return _default_state()
    try:
        return json.loads(PAPER_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return _default_state()

def save_paper_state(state: dict) -> None:
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now_iso()
    PAPER_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def _get_pos(state: dict, mint: str) -> dict:
    pos = state["positions"].get(mint)
    if not pos:
        pos = {
            "qty": 0.0,
            "avg_cost_usd_per_token": 0.0,
            "last_price_usd": None,
            "peak_price_usd": None,
            "trail_stop_price_usd": None,
            "tp_done": [False for _ in TAKE_PROFIT_PCTS],
        }
        state["positions"][mint] = pos
    # If TP config changed between deploys, fix tp_done length
    if "tp_done" not in pos or not isinstance(pos["tp_done"], list):
        pos["tp_done"] = [False for _ in TAKE_PROFIT_PCTS]
    if len(pos["tp_done"]) != len(TAKE_PROFIT_PCTS):
        pos["tp_done"] = [False for _ in TAKE_PROFIT_PCTS]
    return pos

def _round2(x: float) -> float:
    return float(f"{x:.6f}")

def _trade_fee(trade_usd: float) -> float:
    return FEE_FIXED_USD + trade_usd * FEE_PCT

# =======================
# EXTRACT “SWAP” DELTAS FROM HELIUS
# Supports: accountData tokenBalanceChanges/nativeBalanceChange
# AND: tokenTransfers/nativeTransfers formats when present.
# =======================

def _sum_token_delta_from_accountdata(payload0: dict) -> Tuple[Optional[str], float, float]:
    """
    Returns (mint, token_delta, sol_delta) for the most significant token change.
    Uses payload0["accountData"] entries.
    """
    account_data = payload0.get("accountData") or []
    if not isinstance(account_data, list) or not account_data:
        return None, 0.0, 0.0

    # If TRACKED_WALLET set, prefer matching account entry
    if TRACKED_WALLET:
        for a in account_data:
            if a.get("account") == TRACKED_WALLET:
                mint, tok = _sum_token_delta_for_entry(a)
                sol = float(a.get("nativeBalanceChange", 0.0)) / 1e9
                return mint, tok, sol

    # Otherwise pick the entry with meaningful token balance changes
    best = None
    for a in account_data:
        tbc = a.get("tokenBalanceChanges") or []
        if tbc:
            best = a
            break
    if not best:
        best = account_data[0]

    mint, tok = _sum_token_delta_for_entry(best)
    sol = float(best.get("nativeBalanceChange", 0.0)) / 1e9
    return mint, tok, sol

def _sum_token_delta_for_entry(account_entry: dict) -> Tuple[Optional[str], float]:
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
        qty = float(raw) / (10 ** int(dec))
        deltas[mint] = deltas.get(mint, 0.0) + qty

    if not deltas:
        return None, 0.0

    mint = max(deltas, key=lambda m: abs(deltas[m]))
    return mint, float(deltas[mint])

def _sum_deltas_from_transfers(payload0: dict) -> Tuple[Optional[str], float, float]:
    """
    Returns (mint, token_delta, sol_delta) using tokenTransfers/nativeTransfers.
    Only works if those arrays exist.
    """
    token_transfers = payload0.get("tokenTransfers") or []
    native_transfers = payload0.get("nativeTransfers") or []

    if not isinstance(token_transfers, list) or not isinstance(native_transfers, list):
        return None, 0.0, 0.0

    # Track deltas for the tracked wallet if set
    token_deltas: Dict[str, float] = {}
    sol_delta = 0.0

    if TRACKED_WALLET:
        for nt in native_transfers:
            try:
                amt_sol = float(nt.get("amount", 0.0)) / 1e9
                frm = nt.get("fromUserAccount")
                to = nt.get("toUserAccount")
                if frm == TRACKED_WALLET:
                    sol_delta -= amt_sol
                if to == TRACKED_WALLET:
                    sol_delta += amt_sol
            except Exception:
                pass

        for tt in token_transfers:
            try:
                mint = tt.get("mint")
                amt = tt.get("tokenAmount")
                if mint is None or amt is None:
                    continue
                amt = float(amt)
                frm = tt.get("fromUserAccount")
                to = tt.get("toUserAccount")
                if frm == TRACKED_WALLET:
                    token_deltas[mint] = token_deltas.get(mint, 0.0) - amt
                if to == TRACKED_WALLET:
                    token_deltas[mint] = token_deltas.get(mint, 0.0) + amt
            except Exception:
                pass
    else:
        # If no tracked wallet, we can't confidently sign deltas.
        return None, 0.0, 0.0

    if not token_deltas:
        return None, 0.0, sol_delta

    mint = max(token_deltas, key=lambda m: abs(token_deltas[m]))
    return mint, float(token_deltas[mint]), float(sol_delta)

def extract_trade_deltas(payload0: dict) -> Tuple[Optional[str], float, float]:
    """
    Best-effort extraction:
    - Prefer accountData if present
    - Else try transfers
    """
    mint, tok, sol = _sum_token_delta_from_accountdata(payload0)
    if mint is not None and tok != 0.0:
        return mint, tok, sol

    mint2, tok2, sol2 = _sum_deltas_from_transfers(payload0)
    if mint2 is not None and tok2 != 0.0:
        return mint2, tok2, sol2

    return None, 0.0, 0.0

# =======================
# EXECUTE PAPER TRADES
# =======================

def paper_buy(state: dict, mint: str, price_usd: float, trade_usd: float, sig: Optional[str], desc: str = "") -> None:
    if trade_usd < MIN_TRADE_USD:
        return
    # cap by cash
    max_by_cash = max(0.0, state["cash"] * MAX_TRADE_PCT_OF_CASH)
    trade_usd = min(trade_usd, max_by_cash) if max_by_cash > 0 else 0.0
    if trade_usd < MIN_TRADE_USD:
        return

    fee = _trade_fee(trade_usd)
    total_cost = trade_usd + fee
    if total_cost > state["cash"]:
        return

    qty = trade_usd / max(price_usd, 1e-12)

    pos = _get_pos(state, mint)
    old_qty = float(pos["qty"])
    old_avg = float(pos["avg_cost_usd_per_token"]) if pos["avg_cost_usd_per_token"] else 0.0

    new_qty = old_qty + qty
    new_avg = ((old_qty * old_avg) + (qty * price_usd)) / max(new_qty, 1e-12)

    pos["qty"] = _round2(new_qty)
    pos["avg_cost_usd_per_token"] = float(new_avg)
    pos["last_price_usd"] = float(price_usd)

    # peak / trail bookkeeping
    peak = pos["peak_price_usd"]
    if peak is None or price_usd > peak:
        pos["peak_price_usd"] = float(price_usd)

    state["cash"] = _round2(state["cash"] - total_cost)

    state["trades"].append({
        "ts": utc_now_iso(),
        "type": "SWAP",
        "side": "BUY",
        "mint": mint,
        "qty": qty,
        "price_usd_per_token": price_usd,
        "trade_usd_est": trade_usd,
        "fees_usd_est": fee,
        "sig": sig,
        "desc": desc,
    })

def paper_sell(state: dict, mint: str, price_usd: float, sell_qty: float, reason: str, sig: Optional[str]) -> None:
    pos = _get_pos(state, mint)
    held = float(pos["qty"])
    if held <= 0:
        return

    sell_qty = min(sell_qty, held)
    if sell_qty <= 0:
        return

    proceeds = sell_qty * price_usd
    fee = _trade_fee(proceeds)  # fee on sell proceeds (simple model)
    net = proceeds - fee

    avg = float(pos["avg_cost_usd_per_token"]) if pos["avg_cost_usd_per_token"] else 0.0
    cost_basis = sell_qty * avg
    realized = net - cost_basis

    state["cash"] = _round2(state["cash"] + net)
    state["realized_pnl_usd"] = _round2(state.get("realized_pnl_usd", 0.0) + realized)

    pos["qty"] = _round2(held - sell_qty)
    pos["last_price_usd"] = float(price_usd)

    # If fully closed, reset trailing/peak/tp flags
    if pos["qty"] <= 0:
        pos["qty"] = 0.0
        pos["peak_price_usd"] = None
        pos["trail_stop_price_usd"] = None
        pos["tp_done"] = [False for _ in TAKE_PROFIT_PCTS]

    state["trades"].append({
        "ts": utc_now_iso(),
        "type": "SWAP",
        "side": "SELL",
        "mint": mint,
        "qty": sell_qty,
        "price_usd_per_token": price_usd,
        "trade_usd_est": proceeds,
        "fees_usd_est": fee,
        "realized_pnl_net": realized,
        "reason": reason,
        "sig": sig,
        "desc": "",
    })

def update_risk_rules(state: dict, mint: str, price_usd: float, sig: Optional[str]) -> None:
    """
    Called on any price update for a mint we hold.
    Implements:
      - Stop loss
      - Trailing stop (armed after profit threshold)
      - Take profit ladder (optional)
    """
    pos = _get_pos(state, mint)
    qty = float(pos["qty"])
    if qty <= 0:
        return

    pos["last_price_usd"] = float(price_usd)

    avg = float(pos["avg_cost_usd_per_token"]) if pos["avg_cost_usd_per_token"] else 0.0
    if avg <= 0:
        return

    # peak tracking
    peak = pos["peak_price_usd"]
    if peak is None or price_usd > peak:
        peak = float(price_usd)
        pos["peak_price_usd"] = peak

    # hard stop loss
    stop_price = avg * (1.0 - STOP_LOSS_PCT)
    if price_usd <= stop_price:
        paper_sell(state, mint, price_usd, qty, reason=f"STOP_LOSS_{STOP_LOSS_PCT}", sig=sig)
        return

    # optional take-profits
    if TAKE_PROFIT_PCTS and TAKE_PROFIT_SELL_PCTS and len(TAKE_PROFIT_PCTS) == len(TAKE_PROFIT_SELL_PCTS):
        pnl_pct = (price_usd - avg) / avg
        for i, tp in enumerate(TAKE_PROFIT_PCTS):
            if i >= len(pos["tp_done"]):
                break
            if pos["tp_done"][i]:
                continue
            if pnl_pct >= tp:
                sell_part = float(TAKE_PROFIT_SELL_PCTS[i])
                sell_qty = qty * sell_part
                paper_sell(state, mint, price_usd, sell_qty, reason=f"TAKE_PROFIT_{tp}", sig=sig)
                # refresh qty after sell
                qty = float(_get_pos(state, mint)["qty"])
                pos["tp_done"][i] = True
                if qty <= 0:
                    return

    # trailing stop (only after armed)
    pnl_pct_now = (price_usd - avg) / avg
    if pnl_pct_now >= TRAIL_ARM_PROFIT_PCT:
        trail_stop = peak * (1.0 - TRAIL_STOP_PCT)
        pos["trail_stop_price_usd"] = float(trail_stop)
        if price_usd <= trail_stop:
            paper_sell(state, mint, price_usd, qty, reason=f"TRAIL_STOP_{TRAIL_STOP_PCT}", sig=sig)
            return

def apply_paper_from_helius_event(payload0: dict) -> None:
    """
    Main handler:
    - Extract mint/token_delta/sol_delta
    - Determine BUY/SELL
    - Determine an estimated price if possible
    - Execute:
        BUY: fixed TRADE_USD_PER_BUY
        SELL: if MIRROR_WHALE_SELLS -> sell proportional or full
    - Then run risk rules (stop/trail/tp)
    """
    if not isinstance(payload0, dict):
        return

    mint, token_delta, sol_delta = extract_trade_deltas(payload0)
    if mint is None or token_delta == 0.0:
        return

    sig = payload0.get("signature")
    desc = payload0.get("description") or payload0.get("desc") or ""

    # Determine a usable price estimate
    price_usd: Optional[float] = None
    if sol_delta != 0.0:
        trade_usd_from_sol = abs(sol_delta) * SOL_PRICE_USD
        price_usd = trade_usd_from_sol / max(abs(token_delta), 1e-12)

    state = load_paper_state()
    pos = _get_pos(state, mint)

    # If we don't have price from this event, fall back to last price (if any)
    if price_usd is None:
        last = pos.get("last_price_usd")
        if last is None:
            # Can't act without any price reference
            return
        price_usd = float(last)

    # Decide side
    # Typical swap:
    #   BUY  => token_delta > 0 and sol_delta < 0
    #   SELL => token_delta < 0 and sol_delta > 0
    # Some payloads may not include sol_delta — in that case we infer by token_delta sign only.
    if token_delta > 0:
        # WHALE BUY => OUR BUY
        paper_buy(
            state=state,
            mint=mint,
            price_usd=price_usd,
            trade_usd=TRADE_USD_PER_BUY,
            sig=sig,
            desc=desc,
        )
        # Update risk (in case this price becomes peak etc.)
        update_risk_rules(state, mint, price_usd, sig=sig)

    else:
        # token_delta < 0 => WHALE SOLD token
        # Update price + run risk first (price update matters)
        update_risk_rules(state, mint, price_usd, sig=sig)

        if MIRROR_WHALE_SELLS:
            held_qty = float(_get_pos(state, mint)["qty"])
            if held_qty > 0:
                # Simple mirror: sell 100% of what we hold when whale sells (safest for “copy”)
                paper_sell(state, mint, price_usd, held_qty, reason="MIRROR_WHALE_SELL", sig=sig)

    # Keep trades list bounded
    state["trades"] = state["trades"][-200:]
    save_paper_state(state)

# =======================
# API
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

    # Compute unrealized + equity for display
    positions_value = 0.0
    for mint, pos in (state.get("positions") or {}).items():
        qty = float(pos.get("qty") or 0.0)
        last = pos.get("last_price_usd")
        if qty > 0 and last is not None:
            positions_value += qty * float(last)

    cash = float(state.get("cash") or 0.0)
    realized = float(state.get("realized_pnl_usd") or 0.0)
    equity = cash + positions_value
    unrealized = equity - START_CASH_USD - realized

    state_out = dict(state)
    state_out["positions_value_usd"] = positions_value
    state_out["equity_usd"] = equity
    state_out["unrealized_pnl_usd"] = unrealized
    return state_out

@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(default=None, convert_underscores=False),
    authorization: Optional[str] = Header(default=None),
):
    # Accept secret from multiple possible headers
    provided = x_webhook_secret or authorization or ""

    # Authorization: Bearer <secret>
    if provided.lower().startswith("bearer "):
        provided = provided[7:].strip()

    # If user pasted "x-webhook-secret: value" into provider field, tolerate it
    if provided.lower().startswith("x-webhook-secret"):
        parts = provided.split(":", 1)
        if len(parts) == 2:
            provided = parts[1].strip()

    if WEBHOOK_SECRET and provided != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    payload = await request.json()

    # Log raw payload
    record = {"received_at": utc_now_iso(), "source": "helius", "payload": payload}
    append_jsonl(record)

    # Apply paper trading
    if isinstance(payload, list):
        for evt in payload:
            apply_paper_from_helius_event(evt)
    elif isinstance(payload, dict):
        apply_paper_from_helius_event(payload)

    return JSONResponse({"ok": True})
