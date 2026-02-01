import os
import json
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

# -----------------------------
# Config (ENV)
# -----------------------------
TRACKED_WALLET = os.getenv("TRACKED_WALLET", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()

# Used to convert SOL value -> USD value for swaps
SOL_PRICE_USD = float(os.getenv("SOL_PRICE_USD", "100.0"))

# Paper starting balances
START_CASH_USD = float(os.getenv("START_CASH_USD", "500.0"))
START_RESERVE_USD = float(os.getenv("START_RESERVE_USD", "0.0"))

# Risk controls
MAX_BUY_USD = float(os.getenv("MAX_BUY_USD", "25.0"))          # max buy size per trade (paper)
MIN_CASH_LEFT_USD = float(os.getenv("MIN_CASH_LEFT_USD", "100.0"))  # ALWAYS leave at least $100 tradable cash

# Profit split on SELL (PROFIT ONLY)
# 40% profit stays tradable cash, 60% profit goes to reserve cash
PROFIT_TO_CASH_PCT = float(os.getenv("PROFIT_TO_CASH_PCT", "0.40"))
PROFIT_TO_RESERVE_PCT = 1.0 - PROFIT_TO_CASH_PCT

# Optional: ignore SOL mint swaps (often noisy)
SOL_MINT = "So11111111111111111111111111111111111111112"

STATE_PATH = os.getenv("STATE_PATH", "/tmp/paper_state.json")
EVENTS_PATH = os.getenv("EVENTS_PATH", "/tmp/webhook_events.jsonl")

# -----------------------------
# Helpers
# -----------------------------
def now_ts() -> int:
    return int(time.time())

def lamports_to_sol(lamports: int) -> float:
    return lamports / 1_000_000_000.0

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "cash_usd": START_CASH_USD,
        "reserve_cash_usd": START_RESERVE_USD,
        "positions": {},  # mint -> {"qty": float, "avg_cost_usd": float, "realized_pnl_usd": float}
        "stats": {
            "webhooks_received": 0,
            "swaps_seen": 0,
            "buys": 0,
            "sells": 0,
            "ignored": 0,
            "last_event_ts": None,
        },
        "last_signatures": [],  # avoid dup processing
    }

def save_state(state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, STATE_PATH)

def append_event_log(obj: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(EVENTS_PATH), exist_ok=True)
        with open(EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")
    except Exception:
        pass

def clamp_last_signatures(state: Dict[str, Any], max_keep: int = 200) -> None:
    sigs = state.get("last_signatures", [])
    if len(sigs) > max_keep:
        state["last_signatures"] = sigs[-max_keep:]

# -----------------------------
# Parsing Helius payload
# -----------------------------
def extract_tracked_wallet_changes(tx: Dict[str, Any], tracked_wallet: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "signature": str,
        "type": str,
        "source": str,
        "timestamp": int,
        "sol_change": float (SOL),
        "token_changes": List[{"mint": str, "token_change": float}]
      }
    Uses accountData.tokenBalanceChanges and accountData.nativeBalanceChange for the tracked wallet.
    """
    signature = tx.get("signature")
    tx_type = tx.get("type")
    source = tx.get("source")
    timestamp = tx.get("timestamp")

    sol_change_sol = 0.0
    token_changes: List[Dict[str, Any]] = []

    for ad in tx.get("accountData", []) or []:
        if (ad.get("account") or "") != tracked_wallet:
            continue

        sol_change_sol = lamports_to_sol(int(ad.get("nativeBalanceChange", 0)))

        for tbc in ad.get("tokenBalanceChanges", []) or []:
            mint = tbc.get("mint")
            raw = (tbc.get("rawTokenAmount") or {})
            decimals = int(raw.get("decimals", 0))
            token_amount_str = raw.get("tokenAmount", "0")

            try:
                raw_int = int(token_amount_str)
            except Exception:
                raw_int = 0

            token_change = raw_int / (10 ** decimals) if decimals >= 0 else 0.0
            token_changes.append({"mint": mint, "token_change": token_change})

        break

    return {
        "signature": signature,
        "type": tx_type,
        "source": source,
        "timestamp": timestamp,
        "sol_change": sol_change_sol,
        "token_changes": token_changes,
    }

def pick_non_sol_token_change(token_changes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    From token changes, pick the first mint that isn't SOL mint and has non-zero change.
    Many swaps are "SOL <-> TOKEN" so this works fine for paper bot.
    """
    for tc in token_changes:
        mint = tc.get("mint")
        chg = safe_float(tc.get("token_change"), 0.0)
        if not mint or mint == SOL_MINT:
            continue
        if abs(chg) > 0:
            return {"mint": mint, "token_change": chg}
    return None

def implied_usd_value_from_sol(sol_delta: float) -> float:
    # sol_delta is change in SOL for tracked wallet
    return sol_delta * SOL_PRICE_USD

# -----------------------------
# Paper trading engine
# -----------------------------
def ensure_position(state: Dict[str, Any], mint: str) -> Dict[str, Any]:
    pos = state["positions"].get(mint)
    if not pos:
        pos = {"qty": 0.0, "avg_cost_usd": 0.0, "realized_pnl_usd": 0.0}
        state["positions"][mint] = pos
    return pos

def execute_paper_buy(state: Dict[str, Any], mint: str, qty: float, total_cost_usd: float, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buy rule:
      - Spend up to MAX_BUY_USD, not more than observed swap cost
      - Ensure cash_usd - spend >= MIN_CASH_LEFT_USD
    """
    cash = state["cash_usd"]

    spend = min(MAX_BUY_USD, total_cost_usd)
    if spend <= 0:
        return {"action": "ignored", "reason": "non_positive_spend"}

    # Always leave MIN_CASH_LEFT_USD
    if cash - spend < MIN_CASH_LEFT_USD:
        return {"action": "ignored", "reason": "min_cash_left_guard", "cash_usd": cash, "attempt_spend": spend}

    # Scale qty to match the spend if the observed swap cost is larger than spend
    # (if observed cost is smaller than spend, we keep qty as-is)
    scaled_qty = qty
    if total_cost_usd > 0 and spend < total_cost_usd:
        scale = spend / total_cost_usd
        scaled_qty = qty * scale

    if scaled_qty <= 0:
        return {"action": "ignored", "reason": "non_positive_qty_after_scale"}

    pos = ensure_position(state, mint)

    # Weighted average cost
    old_qty = pos["qty"]
    old_cost_total = old_qty * pos["avg_cost_usd"]
    new_cost_total = old_cost_total + spend
    new_qty = old_qty + scaled_qty

    pos["qty"] = new_qty
    pos["avg_cost_usd"] = (new_cost_total / new_qty) if new_qty > 0 else 0.0

    state["cash_usd"] = cash - spend
    state["stats"]["buys"] += 1

    return {
        "action": "buy",
        "mint": mint,
        "qty_bought": scaled_qty,
        "spend_usd": spend,
        "new_cash_usd": state["cash_usd"],
        "new_pos_qty": pos["qty"],
        "new_avg_cost_usd": pos["avg_cost_usd"],
        "meta": meta,
    }

def execute_paper_sell(state: Dict[str, Any], mint: str, qty: float, proceeds_usd: float, meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sell:
      - If we hold less than qty, sell what we have (paper).
      - Principal (cost basis) always goes back to cash_usd.
      - Profit split: 40% to cash_usd, 60% to reserve_cash_usd.
      - Loss reduces cash_usd only.
    """
    pos = ensure_position(state, mint)
    held = pos["qty"]
    if held <= 0:
        return {"action": "ignored", "reason": "no_position"}

    sell_qty = min(held, qty)
    if sell_qty <= 0:
        return {"action": "ignored", "reason": "non_positive_sell_qty"}

    # If proceeds_usd corresponds to qty, we scale proceeds if we sold less than observed qty
    scaled_proceeds = proceeds_usd
    if qty > 0 and sell_qty < qty:
        scaled_proceeds = proceeds_usd * (sell_qty / qty)

    cost_basis = sell_qty * pos["avg_cost_usd"]
    pnl = scaled_proceeds - cost_basis

    # Return principal always
    state["cash_usd"] += cost_basis

    if pnl > 0:
        to_cash = pnl * PROFIT_TO_CASH_PCT
        to_reserve = pnl * PROFIT_TO_RESERVE_PCT
        state["cash_usd"] += to_cash
        state["reserve_cash_usd"] += to_reserve
    else:
        # Loss hits tradable cash only
        state["cash_usd"] += pnl  # pnl is negative

    # Reduce position qty (avg_cost stays)
    pos["qty"] = held - sell_qty
    pos["realized_pnl_usd"] = safe_float(pos.get("realized_pnl_usd"), 0.0) + pnl

    state["stats"]["sells"] += 1

    return {
        "action": "sell",
        "mint": mint,
        "qty_sold": sell_qty,
        "proceeds_usd": scaled_proceeds,
        "cost_basis_usd": cost_basis,
        "pnl_usd": pnl,
        "profit_split": {
            "profit_to_cash_pct": PROFIT_TO_CASH_PCT,
            "profit_to_reserve_pct": PROFIT_TO_RESERVE_PCT,
        },
        "new_cash_usd": state["cash_usd"],
        "new_reserve_cash_usd": state["reserve_cash_usd"],
        "remaining_pos_qty": pos["qty"],
        "meta": meta,
    }

def handle_swap_event(state: Dict[str, Any], tx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a tracked-wallet SWAP into a paper BUY/SELL decision.
    Uses tracked wallet SOL delta + token delta to estimate USD costs/proceeds.
    """
    changes = extract_tracked_wallet_changes(tx, TRACKED_WALLET)
    signature = changes.get("signature")
    if not signature:
        state["stats"]["ignored"] += 1
        return {"action": "ignored", "reason": "missing_signature"}

    # Deduplicate
    if signature in (state.get("last_signatures") or []):
        state["stats"]["ignored"] += 1
        return {"action": "ignored", "reason": "duplicate_signature", "signature": signature}

    state["last_signatures"].append(signature)
    clamp_last_signatures(state)

    sol_change = safe_float(changes.get("sol_change"), 0.0)
    token_pick = pick_non_sol_token_change(changes.get("token_changes") or [])
    if not token_pick:
        state["stats"]["ignored"] += 1
        return {"action": "ignored", "reason": "no_non_sol_token_change", "signature": signature}

    mint = token_pick["mint"]
    token_change = safe_float(token_pick["token_change"], 0.0)

    # BUY heuristic:
    # - SOL decreases (spent), token increases
    # SELL heuristic:
    # - SOL increases (received), token decreases
    meta = {
        "signature": signature,
        "source": tx.get("source"),
        "timestamp": tx.get("timestamp"),
        "sol_price_usd": SOL_PRICE_USD,
        "sol_change": sol_change,
        "token_change": token_change,
    }

    # estimated USD value moved in/out of SOL
    sol_usd_value = implied_usd_value_from_sol(sol_change)

    # BUY
    if sol_change < 0 and token_change > 0:
        total_cost_usd = abs(sol_usd_value)
        # Note: token_change is total token qty acquired in the swap
        return execute_paper_buy(state, mint, qty=token_change, total_cost_usd=total_cost_usd, meta=meta)

    # SELL
    if sol_change > 0 and token_change < 0:
        proceeds_usd = abs(sol_usd_value)
        sell_qty = abs(token_change)
        return execute_paper_sell(state, mint, qty=sell_qty, proceeds_usd=proceeds_usd, meta=meta)

    state["stats"]["ignored"] += 1
    return {"action": "ignored", "reason": "not_buy_or_sell_pattern", "meta": meta}

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="sol-paper-bot", version="1.0.0")

@app.get("/health")
def health():
    return {"ok": True, "ts": now_ts()}

@app.get("/paper/state")
def paper_state():
    state = load_state()
    return {
        "cash_usd": state["cash_usd"],
        "reserve_cash_usd": state["reserve_cash_usd"],
        "min_cash_left_usd": MIN_CASH_LEFT_USD,
        "max_buy_usd": MAX_BUY_USD,
        "profit_split": {
            "profit_to_cash_pct": PROFIT_TO_CASH_PCT,
            "profit_to_reserve_pct": PROFIT_TO_RESERVE_PCT,
        },
        "positions": state["positions"],
        "stats": state["stats"],
        "tracked_wallet": TRACKED_WALLET,
        "sol_price_usd": SOL_PRICE_USD,
    }

@app.get("/events")
def events_preview(limit: int = 50):
    """
    Shows last N raw webhook lines (best-effort).
    """
    if not os.path.exists(EVENTS_PATH):
        return {"count": 0, "events": []}

    lines: List[str] = []
    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())

    last = lines[-max(0, min(limit, 500)):]
    parsed = []
    for ln in last:
        try:
            parsed.append(json.loads(ln))
        except Exception:
            parsed.append({"raw": ln})

    return {"count": len(parsed), "events": parsed}

@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(default=None, convert_underscores=False),
):
    # Security
    if WEBHOOK_SECRET:
        # Accept either x-webhook-secret or X-Webhook-Secret (some systems vary)
        provided = x_webhook_secret
        if not provided:
            provided = request.headers.get("X-Webhook-Secret")

        if provided != WEBHOOK_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")

    if not TRACKED_WALLET:
        raise HTTPException(status_code=500, detail="TRACKED_WALLET not configured")

    body = await request.json()
    state = load_state()

    state["stats"]["webhooks_received"] += 1
    state["stats"]["last_event_ts"] = now_ts()

    # Helius sometimes sends:
    #  - a list of transactions
    #  - or {"payload":[...]} when proxied
    payload: List[Dict[str, Any]] = []

    if isinstance(body, list):
        payload = body
    elif isinstance(body, dict):
        # your earlier event samples show {"payload":[{...}]}
        if isinstance(body.get("payload"), list):
            payload = body["payload"]
        else:
            # sometimes the tx itself is the dict
            payload = [body]
    else:
        payload = []

    results: List[Dict[str, Any]] = []

    for tx in payload:
        if not isinstance(tx, dict):
            continue

        tx_type = tx.get("type")
        if tx_type != "SWAP":
            state["stats"]["ignored"] += 1
            continue

        state["stats"]["swaps_seen"] += 1
        res = handle_swap_event(state, tx)
        results.append(res)

    save_state(state)

    # Log trimmed event for debugging
    append_event_log({
        "ts": now_ts(),
        "count": len(payload),
        "results": results[:20],  # keep small
    })

    return JSONResponse({"ok": True, "handled": len(results), "results": results})

# For local dev (Render uses Start Command with uvicorn)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
