import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="sol-paper-bot", version="2.0-fastapi-only")

# -----------------------------
# ENV / CONFIG
# -----------------------------
TRACKED_WALLETS = [w.strip() for w in os.getenv("TRACKED_WALLET", "").split(",") if w.strip()]
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "100"))  # used to derive token price from SOL change

# Aggressive behavior (scalping meme launches)
STARTING_CASH_USD = float(os.getenv("STARTING_CASH_USD", "500"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "12"))
MIN_TRADE_USD = float(os.getenv("MIN_TRADE_USD", "10"))
MAX_TRADE_USD = float(os.getenv("MAX_TRADE_USD", "80"))
TRADE_FRACTION_OF_TRADABLE = float(os.getenv("TRADE_FRACTION_OF_TRADABLE", "0.08"))  # 8% of tradable cash
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "150"))  # 1.5% simulated slippage
FEE_FIXED_USD = float(os.getenv("FEE_FIXED_USD", "0.25"))  # per trade (buy or sell)

# Profit locking (house money extraction)
# If a single closed trade ROI >= threshold, lock a fraction of PROFIT into locked_cash
PROFIT_LOCK_ROI = float(os.getenv("PROFIT_LOCK_ROI", "0.50"))  # 50%+
PROFIT_LOCK_FRACTION = float(os.getenv("PROFIT_LOCK_FRACTION", "0.50"))  # lock 50% of profit

# Optional: extra aggressive "big win" lock (e.g., 10x / 100x)
BIG_WIN_ROI = float(os.getenv("BIG_WIN_ROI", "10.0"))  # 10x+
BIG_WIN_LOCK_FRACTION = float(os.getenv("BIG_WIN_LOCK_FRACTION", "0.70"))  # lock 70% of profit

STATE_PATH = os.getenv("STATE_PATH", "paper_state.json")


# -----------------------------
# STATE
# -----------------------------
def now_ts() -> float:
    return time.time()


def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # default state
    return {
        "created_at": now_ts(),
        "cash": STARTING_CASH_USD,
        "locked_cash": 0.0,          # cannot be used for new buys
        "tradable_cash": STARTING_CASH_USD,  # cash available for new trades
        "realized_pnl_usd": 0.0,
        "positions": {},  # mint -> position
        "events": [],
        "count": 0,
    }


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_PATH)


STATE = load_state()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def bps_to_mult(bps: float) -> float:
    return 1.0 + (bps / 10_000.0)


# -----------------------------
# HELIUS PARSING HELPERS
# -----------------------------
def _as_list(payload: Any) -> List[Dict[str, Any]]:
    # Helius can send a list of tx objects or a single object
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        # sometimes your server wraps it
        if "events" in payload and isinstance(payload["events"], list):
            return [p for p in payload["events"] if isinstance(p, dict)]
        return [payload]
    return []


def extract_accountdata(tx: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Common places Helius enhanced payload stores account data
    # Your screenshot showed payload.accountData[...]
    p = tx.get("payload", tx)
    if isinstance(p, dict) and isinstance(p.get("accountData"), list):
        return [x for x in p["accountData"] if isinstance(x, dict)]
    # fallback
    if isinstance(tx.get("accountData"), list):
        return [x for x in tx["accountData"] if isinstance(x, dict)]
    return []


def find_wallet_account_row(account_data: List[Dict[str, Any]], wallet: str) -> Optional[Dict[str, Any]]:
    for row in account_data:
        if row.get("account") == wallet:
            return row
    return None


def token_change_for_wallet_row(wallet_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    # row.tokenBalanceChanges = [{"mint":..., "rawTokenAmount":..., "tokenAmount":..., "decimals":...}, ...]
    tbc = wallet_row.get("tokenBalanceChanges", [])
    if not isinstance(tbc, list):
        return []
    return [x for x in tbc if isinstance(x, dict) and x.get("mint")]


def native_change_sol(wallet_row: Dict[str, Any]) -> float:
    # nativeBalanceChange is in lamports in Helius enhanced
    lamports = wallet_row.get("nativeBalanceChange", 0)
    try:
        lamports = float(lamports)
    except Exception:
        lamports = 0.0
    return lamports / 1_000_000_000.0


def token_amount_change(tc: Dict[str, Any]) -> float:
    """
    tokenBalanceChanges typically includes:
      - tokenAmount (already scaled) OR rawTokenAmount + decimals
    We'll prefer tokenAmount if present, else derive.
    """
    if "tokenAmount" in tc:
        try:
            return float(tc["tokenAmount"])
        except Exception:
            pass

    raw = tc.get("rawTokenAmount", None)
    dec = tc.get("decimals", None)
    try:
        raw_f = float(raw)
        dec_i = int(dec) if dec is not None else 0
        return raw_f / (10 ** dec_i)
    except Exception:
        return 0.0


def derive_token_price_usd(sol_change: float, token_delta: float) -> Optional[float]:
    # If wallet spent SOL to get tokens: sol_change negative, token_delta positive
    # price per token = (abs(sol_change)*SOL_PRICE_USD)/token_delta
    if token_delta == 0:
        return None
    usd_value = abs(sol_change) * SOL_PRICE_USD
    price = usd_value / abs(token_delta)
    if price <= 0:
        return None
    return price


# -----------------------------
# TRADING LOGIC (PAPER)
# -----------------------------
def ensure_position(mint: str) -> Dict[str, Any]:
    pos = STATE["positions"].get(mint)
    if not pos:
        pos = {
            "mint": mint,
            "qty": 0.0,
            "avg_cost_usd_per_token": 0.0,
            "last_price_usd": 0.0,
            "peak_price_usd": 0.0,
            "opened_at": now_ts(),
            "last_update": now_ts(),
            "trade_count": 0,
        }
        STATE["positions"][mint] = pos
    return pos


def open_positions_count() -> int:
    c = 0
    for p in STATE["positions"].values():
        try:
            if float(p.get("qty", 0)) > 0:
                c += 1
        except Exception:
            pass
    return c


def compute_buy_budget_usd() -> float:
    tradable = float(STATE.get("tradable_cash", 0.0))
    budget = tradable * TRADE_FRACTION_OF_TRADABLE
    budget = clamp(budget, MIN_TRADE_USD, MAX_TRADE_USD)
    return budget


def apply_slippage(price: float, is_buy: bool) -> float:
    # buys pay worse (higher), sells receive worse (lower)
    mult = bps_to_mult(SLIPPAGE_BPS)
    return price * (mult if is_buy else (1.0 / mult))


def paper_buy(mint: str, price_usd: float, signal_token_delta: float, signal_sol_change: float) -> Dict[str, Any]:
    # keep "aggressive scalping": buy whenever tracked wallet buys, subject to cash/limits
    if open_positions_count() >= MAX_OPEN_POSITIONS and float(ensure_position(mint)["qty"]) <= 0:
        return {"action": "skip_buy", "reason": "max_open_positions"}

    budget = compute_buy_budget_usd()
    if STATE["tradable_cash"] < budget + FEE_FIXED_USD:
        return {"action": "skip_buy", "reason": "insufficient_tradable_cash", "need": budget + FEE_FIXED_USD}

    exec_price = apply_slippage(price_usd, is_buy=True)
    qty = budget / exec_price if exec_price > 0 else 0.0
    if qty <= 0:
        return {"action": "skip_buy", "reason": "bad_qty"}

    pos = ensure_position(mint)
    old_qty = float(pos["qty"])
    old_avg = float(pos["avg_cost_usd_per_token"])

    new_qty = old_qty + qty
    new_avg = ((old_qty * old_avg) + (qty * exec_price)) / new_qty if new_qty > 0 else 0.0

    pos["qty"] = new_qty
    pos["avg_cost_usd_per_token"] = new_avg
    pos["last_price_usd"] = exec_price
    pos["peak_price_usd"] = max(float(pos.get("peak_price_usd", 0.0)), exec_price)
    pos["last_update"] = now_ts()
    pos["trade_count"] = int(pos.get("trade_count", 0)) + 1

    # cash movement
    STATE["tradable_cash"] -= (budget + FEE_FIXED_USD)
    STATE["cash"] = STATE["locked_cash"] + STATE["tradable_cash"]

    return {
        "action": "buy",
        "mint": mint,
        "budget_usd": budget,
        "exec_price_usd": exec_price,
        "qty_bought": qty,
        "pos_qty": new_qty,
        "pos_avg": new_avg,
        "signal": {"token_delta": signal_token_delta, "sol_change": signal_sol_change},
    }


def maybe_lock_profit(realized_profit_usd: float, roi: float) -> float:
    """
    Returns amount locked.
    Locks only PROFIT (not principal).
    """
    if realized_profit_usd <= 0:
        return 0.0

    lock_frac = 0.0
    if roi >= BIG_WIN_ROI:
        lock_frac = BIG_WIN_LOCK_FRACTION
    elif roi >= PROFIT_LOCK_ROI:
        lock_frac = PROFIT_LOCK_FRACTION

    lock_amt = realized_profit_usd * lock_frac
    if lock_amt > 0:
        STATE["locked_cash"] += lock_amt
        STATE["tradable_cash"] += (realized_profit_usd - lock_amt)
    else:
        STATE["tradable_cash"] += realized_profit_usd

    # keep total cash consistent (principal returns elsewhere)
    STATE["cash"] = STATE["locked_cash"] + STATE["tradable_cash"]
    return lock_amt


def paper_sell_all(mint: str, price_usd: float, signal_token_delta: float, signal_sol_change: float) -> Dict[str, Any]:
    pos = ensure_position(mint)
    qty = float(pos.get("qty", 0.0))
    if qty <= 0:
        return {"action": "skip_sell", "reason": "no_position"}

    exec_price = apply_slippage(price_usd, is_buy=False)
    proceeds = qty * exec_price
    cost_basis = qty * float(pos.get("avg_cost_usd_per_token", 0.0))
    realized_profit = proceeds - cost_basis - FEE_FIXED_USD

    roi = (proceeds - FEE_FIXED_USD) / cost_basis - 1.0 if cost_basis > 0 else 0.0

    # principal always returns to tradable cash
    STATE["tradable_cash"] += cost_basis
    # now add profit and lock a portion
    locked = maybe_lock_profit(realized_profit, roi)

    STATE["realized_pnl_usd"] += realized_profit
    STATE["cash"] = STATE["locked_cash"] + STATE["tradable_cash"]

    # reset position
    pos["qty"] = 0.0
    pos["avg_cost_usd_per_token"] = 0.0
    pos["last_price_usd"] = exec_price
    pos["peak_price_usd"] = max(float(pos.get("peak_price_usd", 0.0)), exec_price)
    pos["last_update"] = now_ts()
    pos["trade_count"] = int(pos.get("trade_count", 0)) + 1

    return {
        "action": "sell_all",
        "mint": mint,
        "exec_price_usd": exec_price,
        "qty_sold": qty,
        "proceeds_usd": proceeds,
        "cost_basis_usd": cost_basis,
        "realized_profit_usd": realized_profit,
        "roi": roi,
        "locked_profit_usd": locked,
        "signal": {"token_delta": signal_token_delta, "sol_change": signal_sol_change},
        "cash": {
            "locked_cash": STATE["locked_cash"],
            "tradable_cash": STATE["tradable_cash"],
            "total_cash": STATE["cash"],
        },
    }


def handle_wallet_token_changes(wallet: str, wallet_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    sol_change = native_change_sol(wallet_row)  # + means received SOL, - means spent SOL
    tchanges = token_change_for_wallet_row(wallet_row)

    for tc in tchanges:
        mint = tc.get("mint")
        token_delta = token_amount_change(tc)

        # If we can't infer a meaningful trade price, skip trading but still log
        price = derive_token_price_usd(sol_change, token_delta)
        if price is None:
            results.append({
                "action": "skip",
                "reason": "no_price",
                "wallet": wallet,
                "mint": mint,
                "sol_change": sol_change,
                "token_delta": token_delta,
            })
            continue

        # BUY signal: spent SOL (negative), token increased (positive)
        if sol_change < 0 and token_delta > 0:
            results.append(paper_buy(mint, price, token_delta, sol_change))
            continue

        # SELL signal: received SOL (positive), token decreased (negative)
        if sol_change > 0 and token_delta < 0:
            results.append(paper_sell_all(mint, price, token_delta, sol_change))
            continue

        # Otherwise: update mark price if position exists
        pos = ensure_position(mint)
        pos["last_price_usd"] = price
        pos["peak_price_usd"] = max(float(pos.get("peak_price_usd", 0.0)), price)
        pos["last_update"] = now_ts()
        results.append({
            "action": "mark",
            "wallet": wallet,
            "mint": mint,
            "price_usd": price,
            "sol_change": sol_change,
            "token_delta": token_delta,
        })

    return results


# -----------------------------
# API
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True, "service": "sol-paper-bot", "framework": "fastapi", "ts": now_ts()}


@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(default=None, convert_underscores=False),
):
    # auth
    if WEBHOOK_SECRET:
        if not x_webhook_secret or x_webhook_secret != WEBHOOK_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized (bad x-webhook-secret)")

    payload = await request.json()
    txs = _as_list(payload)

    event_record = {
        "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "helius",
        "tx_count": len(txs),
        "raw": payload,  # keep raw for debugging
        "actions": [],
    }

    actions: List[Dict[str, Any]] = []

    for tx in txs:
        account_data = extract_accountdata(tx)
        if not account_data:
            continue

        for wallet in TRACKED_WALLETS:
            row = find_wallet_account_row(account_data, wallet)
            if not row:
                continue

            actions.extend(handle_wallet_token_changes(wallet, row))

    event_record["actions"] = actions

    # store event
    STATE["count"] = int(STATE.get("count", 0)) + 1
    STATE["events"].insert(0, event_record)
    STATE["events"] = STATE["events"][:200]  # cap memory
    save_state(STATE)

    return JSONResponse({"ok": True, "actions": actions, "cash": {
        "locked_cash": STATE["locked_cash"],
        "tradable_cash": STATE["tradable_cash"],
        "total_cash": STATE["cash"],
        "realized_pnl_usd": STATE["realized_pnl_usd"],
    }})


@app.get("/events")
def events():
    return {"count": STATE.get("count", 0), "events": STATE.get("events", [])}


@app.get("/paper/state")
def paper_state():
    # return a lighter view
    return {
        "cash": STATE.get("cash", 0.0),
        "locked_cash": STATE.get("locked_cash", 0.0),
        "tradable_cash": STATE.get("tradable_cash", 0.0),
        "realized_pnl_usd": STATE.get("realized_pnl_usd", 0.0),
        "positions": STATE.get("positions", {}),
        "count": STATE.get("count", 0),
    }
