import os
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("sol-paper-bot")

# ----------------------------
# Config (easy to tune)
# ----------------------------
SERVICE_NAME = os.getenv("SERVICE_NAME", "sol-paper-bot")

# Webhook auth (Helius sends header "x-webhook-secret" with the VALUE you set in dashboard)
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()

# Tracked whales: comma-separated wallet addresses
TRACKED_WALLETS_RAW = os.getenv("TRACKED_WALLETS", os.getenv("TRACKED_WALLET", "")).strip()
TRACKED_WALLETS = [w.strip() for w in TRACKED_WALLETS_RAW.split(",") if w.strip()]

# Paper trading parameters
START_CASH_USD = float(os.getenv("START_CASH_USD", "500"))
TRADE_USD_PER_BUY = float(os.getenv("TRADE_USD_PER_BUY", "100"))  # how much you "buy" when whale buys
FEE_USD_PER_SWAP = float(os.getenv("FEE_USD_PER_SWAP", "3.0"))

# Risk parameters
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.20"))  # 0.20 = 20% hard stop
TRAIL_ACTIVATE_PROFIT_PCT = float(os.getenv("TRAIL_ACTIVATE_PROFIT_PCT", "0.20"))  # start trailing after +20%
TRAIL_PCT = float(os.getenv("TRAIL_PCT", "0.15"))  # trail by 15%

# Optional take-profit ladder (comma-separated profit levels AND fractions)
# Example:
#   TP_LEVELS="0.25,0.50,1.00"
#   TP_FRACTIONS="0.25,0.25,0.25"
TP_LEVELS = [float(x) for x in os.getenv("TP_LEVELS", "").split(",") if x.strip()]
TP_FRACTIONS = [float(x) for x in os.getenv("TP_FRACTIONS", "").split(",") if x.strip()]

MAX_TRADES_KEEP = int(os.getenv("MAX_TRADES_KEEP", "2000"))
MAX_EVENTS_KEEP = int(os.getenv("MAX_EVENTS_KEEP", "400"))
MAX_EVENT_BYTES = int(os.getenv("MAX_EVENT_BYTES", "250000"))  # truncate huge payloads

# State persistence
STATE_PATH = os.getenv("STATE_PATH", "/var/data/state.json")  # use Render Disk mounted at /var/data
STATE_DIR = os.path.dirname(STATE_PATH) if os.path.dirname(STATE_PATH) else "."

# ----------------------------
# Helpers
# ----------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def atomic_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)

def safe_truncate(obj: Any, max_bytes: int) -> Any:
    try:
        s = json.dumps(obj, ensure_ascii=False)
        if len(s.encode("utf-8")) <= max_bytes:
            return obj
        # if too large, keep only first part
        truncated = s.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        return {"_truncated": True, "preview": truncated}
    except Exception:
        return {"_truncated": True, "preview": str(obj)[:2000]}

def is_stablecoin_mint(mint: str) -> bool:
    # USDC mint on Solana mainnet
    return mint == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

def normalize_mint(m: Optional[str]) -> str:
    return (m or "").strip()

def header_secret_value(raw: Optional[str]) -> str:
    # Some dashboards show users typing "x-webhook-secret: value"
    # We only want the value.
    if not raw:
        return ""
    s = raw.strip()
    if ":" in s and s.lower().startswith("x-webhook-secret"):
        # "x-webhook-secret: abc" => "abc"
        return s.split(":", 1)[1].strip()
    return s

# ----------------------------
# State model
# ----------------------------
def fresh_state() -> Dict[str, Any]:
    return {
        "cash": START_CASH_USD,
        "positions": {},  # mint -> position dict
        "trades": [],
        "events": [],
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }

def load_state() -> Dict[str, Any]:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
        # basic sanity
        if "cash" not in st or "positions" not in st or "trades" not in st:
            return fresh_state()
        return st
    except Exception:
        return fresh_state()

def save_state(st: Dict[str, Any]) -> None:
    st["updated_at"] = now_iso()
    atomic_write_json(STATE_PATH, st)

STATE = load_state()

# ----------------------------
# Paper engine
# ----------------------------
def get_pos(st: Dict[str, Any], mint: str) -> Dict[str, Any]:
    return st["positions"].setdefault(
        mint,
        {
            "qty": 0.0,
            "avg_cost_usd_per_token": None,
            "last_price_usd": None,
            "peak_price_usd": None,
            "trail_stop_price_usd": None,
            "tp_done": [False for _ in TP_LEVELS],
        },
    )

def record_trade(
    st: Dict[str, Any],
    side: str,
    mint: str,
    qty: float,
    price_usd: Optional[float],
    trade_usd_est: float,
    fees_usd_est: float,
    reason: str,
    sig: str = "",
    desc: str = "",
) -> None:
    t = {
        "ts": now_iso(),
        "type": "SWAP",
        "side": side,
        "mint": mint,
        "qty": qty,
        "price_usd_per_token": price_usd,
        "trade_usd_est": trade_usd_est,
        "fees_usd_est": fees_usd_est,
        "reason": reason,
        "sig": sig,
        "desc": desc,
    }
    st["trades"].append(t)
    if len(st["trades"]) > MAX_TRADES_KEEP:
        st["trades"] = st["trades"][-MAX_TRADES_KEEP:]

def mark_to_market(st: Dict[str, Any]) -> Tuple[float, float, float]:
    positions_value = 0.0
    unrealized = 0.0
    for mint, p in st["positions"].items():
        qty = float(p.get("qty") or 0.0)
        if qty <= 0:
            continue
        last = p.get("last_price_usd")
        avg = p.get("avg_cost_usd_per_token")
        if last is None or avg is None:
            continue
        positions_value += qty * float(last)
        unrealized += qty * (float(last) - float(avg))
    equity = float(st["cash"]) + positions_value
    return positions_value, equity, unrealized

def paper_buy(st: Dict[str, Any], mint: str, implied_price: Optional[float], sig: str, desc: str) -> None:
    if mint == "" or mint.startswith("So11111111111111111111111111111111111111112"):
        return
    if is_stablecoin_mint(mint):
        return

    cash = float(st["cash"])
    budget = min(TRADE_USD_PER_BUY, cash)
    if budget <= 0:
        return

    fees = FEE_USD_PER_SWAP
    total_cost = budget + fees
    if total_cost > cash:
        # if not enough for fees+budget, shrink budget
        budget = max(0.0, cash - fees)
        total_cost = budget + fees
    if budget <= 0:
        return

    # If we don't have price, we still can simulate using budget/qty later.
    # Here we create a synthetic qty from implied_price if present, otherwise qty=budget (treat price=1).
    price = float(implied_price) if implied_price and implied_price > 0 else None
    qty = (budget / price) if price else budget

    p = get_pos(st, mint)
    old_qty = float(p["qty"])
    old_avg = p["avg_cost_usd_per_token"]

    new_qty = old_qty + qty

    # compute new avg cost
    if old_qty <= 0 or old_avg is None:
        new_avg = (budget / qty) if qty > 0 else None
    else:
        old_cost = old_qty * float(old_avg)
        new_cost = old_cost + budget
        new_avg = (new_cost / new_qty) if new_qty > 0 else old_avg

    p["qty"] = new_qty
    p["avg_cost_usd_per_token"] = new_avg

    # update last/peak/trail
    if price is not None:
        p["last_price_usd"] = price
        peak = p.get("peak_price_usd")
        p["peak_price_usd"] = max(float(peak) if peak else price, price)

    st["cash"] = cash - total_cost

    record_trade(
        st=st,
        side="BUY",
        mint=mint,
        qty=qty,
        price_usd=price,
        trade_usd_est=budget,
        fees_usd_est=fees,
        reason="WHale BUY mirrored",
        sig=sig,
        desc=desc,
    )

def paper_sell_all(st: Dict[str, Any], mint: str, implied_price: Optional[float], sig: str, desc: str, reason: str) -> None:
    p = get_pos(st, mint)
    qty = float(p.get("qty") or 0.0)
    if qty <= 0:
        return

    price = float(implied_price) if implied_price and implied_price > 0 else p.get("last_price_usd")
    if price is None:
        # if we truly don't have price, assume break-even at avg cost
        avg = p.get("avg_cost_usd_per_token")
        price = float(avg) if avg else 0.0

    gross = qty * float(price)
    fees = FEE_USD_PER_SWAP
    net = gross - fees

    st["cash"] = float(st["cash"]) + max(0.0, net)

    # realized pnl (net of fees) tracked in trade record; equity endpoints will show it
    record_trade(
        st=st,
        side="SELL",
        mint=mint,
        qty=qty,
        price_usd=float(price),
        trade_usd_est=gross,
        fees_usd_est=fees,
        reason=reason,
        sig=sig,
        desc=desc,
    )

    # reset position
    p["qty"] = 0.0
    p["avg_cost_usd_per_token"] = None
    p["last_price_usd"] = float(price)
    p["peak_price_usd"] = None
    p["trail_stop_price_usd"] = None
    p["tp_done"] = [False for _ in TP_LEVELS]

def update_price_and_risk(st: Dict[str, Any], mint: str, price: Optional[float], sig: str = "", desc: str = "") -> None:
    p = get_pos(st, mint)
    qty = float(p.get("qty") or 0.0)
    if qty <= 0:
        return
    if price is None or price <= 0:
        return

    price = float(price)
    p["last_price_usd"] = price

    avg = p.get("avg_cost_usd_per_token")
    if avg is None or avg <= 0:
        return
    avg = float(avg)

    # peak tracking
    peak = p.get("peak_price_usd")
    if peak is None:
        peak = price
    else:
        peak = max(float(peak), price)
    p["peak_price_usd"] = peak

    # trailing activation
    activate_price = avg * (1.0 + TRAIL_ACTIVATE_PROFIT_PCT)
    if peak >= activate_price:
        trail_stop = peak * (1.0 - TRAIL_PCT)
        p["trail_stop_price_usd"] = trail_stop

    # Stop loss
    stop_price = avg * (1.0 - STOP_LOSS_PCT)
    if price <= stop_price:
        paper_sell_all(
            st, mint, price, sig, desc,
            reason=f"STOP_LOSS hit (price {price:.6g} <= {stop_price:.6g})"
        )
        return

    # Trailing stop
    trail_stop = p.get("trail_stop_price_usd")
    if trail_stop is not None and price <= float(trail_stop):
        paper_sell_all(
            st, mint, price, sig, desc,
            reason=f"TRAIL_STOP hit (price {price:.6g} <= {float(trail_stop):.6g})"
        )
        return

    # Optional take-profit ladder (partial sells)
    # We implement this only if TP_LEVELS and TP_FRACTIONS are valid.
    if TP_LEVELS and TP_FRACTIONS and len(TP_LEVELS) == len(TP_FRACTIONS):
        for i, lvl in enumerate(TP_LEVELS):
            if p["tp_done"][i]:
                continue
            target = avg * (1.0 + lvl)
            if price >= target:
                frac = float(TP_FRACTIONS[i])
                frac = max(0.0, min(1.0, frac))
                sell_qty = qty * frac
                if sell_qty > 0:
                    gross = sell_qty * price
                    fees = FEE_USD_PER_SWAP
                    net = gross - fees
                    st["cash"] = float(st["cash"]) + max(0.0, net)
                    record_trade(
                        st=st,
                        side="SELL",
                        mint=mint,
                        qty=sell_qty,
                        price_usd=price,
                        trade_usd_est=gross,
                        fees_usd_est=fees,
                        reason=f"TAKE_PROFIT +{lvl*100:.0f}%",
                        sig=sig,
                        desc=desc,
                    )
                    p["qty"] = max(0.0, qty - sell_qty)
                    p["tp_done"][i] = True
                break

# ----------------------------
# Whale event parsing
# ----------------------------
def extract_helius_payload(body: Any) -> List[Dict[str, Any]]:
    """
    Accepts:
      - list[txn]
      - {"payload": list[txn], ...}
      - {"events":[...]} (some wrappers)
    Returns list of txn dicts.
    """
    if isinstance(body, list):
        return [x for x in body if isinstance(x, dict)]
    if isinstance(body, dict):
        if isinstance(body.get("payload"), list):
            return [x for x in body["payload"] if isinstance(x, dict)]
        if isinstance(body.get("events"), list):
            return [x for x in body["events"] if isinstance(x, dict)]
        # fallback: single txn
        return [body]
    return []

def net_token_delta_for_wallet(txn: Dict[str, Any], wallet: str) -> List[Tuple[str, float]]:
    """
    Uses enhanced webhook 'accountData'->'tokenBalanceChanges' if present.
    Returns list of (mint, net_delta_tokens) for that wallet.
    """
    out: Dict[str, float] = {}

    acct_data = txn.get("accountData") or txn.get("account_data") or {}
    tbc = acct_data.get("tokenBalanceChanges") or []

    for ch in tbc:
        try:
            if not isinstance(ch, dict):
                continue
            user_acct = (ch.get("userAccount") or ch.get("user_account") or "").strip()
            if user_acct != wallet:
                continue
            mint = normalize_mint(ch.get("mint"))
            # token amount may come as number or string; we handle both
            amt = ch.get("tokenAmount")
            if amt is None:
                amt = ch.get("rawTokenAmount")
            if amt is None:
                continue
            delta = float(amt)
            out[mint] = out.get(mint, 0.0) + delta
        except Exception:
            continue

    return [(m, d) for (m, d) in out.items() if m and abs(d) > 0]

def implied_price_from_txn(txn: Dict[str, Any], mint: str, abs_qty: float) -> Optional[float]:
    """
    If we have nothing reliable, we keep it simple:
    - If we know we spent TRADE_USD_PER_BUY to buy abs_qty tokens -> implied price = TRADE_USD_PER_BUY / abs_qty
    This keeps your paper engine consistent and avoids 'null' prices.
    """
    if abs_qty <= 0:
        return None
    return TRADE_USD_PER_BUY / abs_qty

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "service": SERVICE_NAME, "time": now_iso()}

@app.get("/events")
def events():
    # Return most recent events wrapper
    return {"count": len(STATE.get("events", [])), "events": STATE.get("events", [])}

@app.get("/paper/state")
def paper_state():
    positions_value, equity, unrealized = mark_to_market(STATE)

    # realized pnl approximation from trades (net of fees) vs buy costs
    # Keep it simple + stable:
    realized = 0.0
    pos_cost_basis: Dict[str, float] = {}
    pos_qty: Dict[str, float] = {}

    for tr in STATE.get("trades", []):
        mint = tr.get("mint")
        side = tr.get("side")
        qty = float(tr.get("qty") or 0.0)
        price = tr.get("price_usd_per_token")
        fees = float(tr.get("fees_usd_est") or 0.0)
        if mint is None:
            continue
        if side == "BUY":
            spent = float(tr.get("trade_usd_est") or 0.0)
            pos_cost_basis[mint] = pos_cost_basis.get(mint, 0.0) + spent + fees
            pos_qty[mint] = pos_qty.get(mint, 0.0) + qty
        elif side == "SELL":
            gross = float(tr.get("trade_usd_est") or 0.0)
            received = gross - fees
            # approximate cost removed pro-rata if we tracked basis/qty
            q = pos_qty.get(mint, 0.0)
            b = pos_cost_basis.get(mint, 0.0)
            if q > 0 and qty > 0:
                frac = min(1.0, qty / q)
                cost_removed = b * frac
                pos_cost_basis[mint] = max(0.0, b - cost_removed)
                pos_qty[mint] = max(0.0, q - qty)
                realized += received - cost_removed
            else:
                # if we don't know basis, just treat as received (won't be perfect)
                realized += received

    return {
        "cash": float(STATE["cash"]),
        "positions": STATE["positions"],
        "positions_value_usd": positions_value,
        "equity_usd": equity,
        "realized_pnl_usd": realized,
        "unrealized_pnl_usd": unrealized,
        "trades": STATE["trades"][-200:],  # last 200 in response
    }

@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(default=None),
    x_webhook_secret_alt: Optional[str] = Header(default=None, convert_underscores=False),
):
    # Accept either Header parsing style; FastAPI maps "x-webhook-secret" => x_webhook_secret
    incoming = header_secret_value(x_webhook_secret or x_webhook_secret_alt)

    if WEBHOOK_SECRET:
        if not incoming or incoming != WEBHOOK_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")
    else:
        # If you didn't set WEBHOOK_SECRET, we log it but do not block (not recommended)
        log.warning("WEBHOOK_SECRET is not set; webhook requests are not authenticated.")

    body = await request.json()
    txns = extract_helius_payload(body)

    # Store event preview for debugging
    ev = {
        "received_at": now_iso(),
        "source": "helius",
        "payload": safe_truncate(body, MAX_EVENT_BYTES),
    }
    STATE.setdefault("events", []).append(ev)
    if len(STATE["events"]) > MAX_EVENTS_KEEP:
        STATE["events"] = STATE["events"][-MAX_EVENTS_KEEP:]

    # If you haven't set wallets yet, just persist event and return
    if not TRACKED_WALLETS:
        save_state(STATE)
        return JSONResponse({"ok": True, "tracked": 0, "processed": 0})

    processed = 0

    for txn in txns:
        sig = str(txn.get("signature") or txn.get("sig") or "")
        desc = str(txn.get("description") or "")

        for w in TRACKED_WALLETS:
            deltas = net_token_delta_for_wallet(txn, w)
            for mint, delta in deltas:
                # Ignore stablecoins/SOL mint
                if is_stablecoin_mint(mint) or mint.startswith("So11111111111111111111111111111111111111112"):
                    continue

                abs_qty = abs(delta)
                implied_price = implied_price_from_txn(txn, mint, abs_qty)

                # Update risk marks on any observed activity
                update_price_and_risk(STATE, mint, implied_price, sig=sig, desc=desc)

                if delta > 0:
                    # whale buy -> we buy
                    paper_buy(STATE, mint, implied_price, sig=sig, desc=desc)
                    processed += 1
                elif delta < 0:
                    # whale sell -> we sell (all)
                    paper_sell_all(
                        STATE,
                        mint,
                        implied_price,
                        sig=sig,
                        desc=desc,
                        reason="WHALE SELL mirrored",
                    )
                    processed += 1

                # After any action, re-run risk checks with latest price
                update_price_and_risk(STATE, mint, implied_price, sig=sig, desc=desc)

    # Persist state so /paper/state doesn't reset after restarts
    save_state(STATE)

    return JSONResponse({"ok": True, "tracked": len(TRACKED_WALLETS), "processed": processed})

# Optional: reset endpoint (use carefully)
@app.post("/paper/reset")
async def paper_reset(request: Request):
    # protect with same webhook secret (simple safety)
    incoming = header_secret_value(request.headers.get("x-webhook-secret", ""))
    if WEBHOOK_SECRET and incoming != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    global STATE
    STATE = fresh_state()
    save_state(STATE)
    return {"ok": True, "message": "Paper state reset", "state_path": STATE_PATH}
