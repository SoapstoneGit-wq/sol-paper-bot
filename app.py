import os
import time
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

# ----------------------------
# Config (ENV)
# ----------------------------
def env_str(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return default if v is None else str(v)

def env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or v == "":
        return float(default)
    return float(v)

def env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or v == "":
        return int(default)
    return int(v)

def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

DEBUG_WEBHOOK = env_bool("DEBUG_WEBHOOK", False)

WEBHOOK_PATH_TOKEN = env_str("WEBHOOK_PATH_TOKEN", "")  # e.g. hook_93f2a8c17b
WEBHOOK_SECRET = env_str("WEBHOOK_SECRET", "")          # what you paste in Helius "Authentication Header"

# Trading cash config
START_CASH_USD = env_float("START_CASH_USD", 800.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 25.0)
MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)

# Reserve behavior:
# Reserve starts at 0. It only grows from PROFIT on sells.
RESERVE_PCT = env_float("RESERVE_PCT", 0.60)  # e.g. 0.60 means 60% of PROFIT goes to reserve

TRADABLE_PCT = env_float("TRADABLE_PCT", 0.40)  # not required by logic, but kept for config parity

HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)  # time-exit fallback
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)

SOL_PRICE_USD = env_float("SOL_PRICE_USD", 100.0)

# Wallets
# Put comma-separated list in env TRACKED_WALLETS
TRACKED_WALLETS = [
    w.strip() for w in env_str("TRACKED_WALLETS", "").split(",") if w.strip()
]

# ----------------------------
# Tweak C: TP/SL/Trailing Stop
# ----------------------------
TAKE_PROFIT_1 = env_float("TAKE_PROFIT_1", 0.30)       # +30%
TP1_SELL_PCT = env_float("TP1_SELL_PCT", 0.50)         # sell 50% at TP1
TAKE_PROFIT_2 = env_float("TAKE_PROFIT_2", 0.50)       # +50%
STOP_LOSS = env_float("STOP_LOSS", -0.15)              # -15%

ENABLE_TRAILING = env_bool("ENABLE_TRAILING", True)
TRAIL_ACTIVATE = env_float("TRAIL_ACTIVATE", 0.20)     # only start trailing after +20%
TRAIL_PCT = env_float("TRAIL_PCT", 0.10)               # trail by 10% from peak

# ----------------------------
# Constants / Helpers
# ----------------------------
SOL_MINT = "So11111111111111111111111111111111111111112"

# "Stable-like" mints (treat as $1)
STABLE_MINTS = {
    # USDC
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    # USDT
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    # USD1 (you see this one)
    "USD1ttGY1N17NEEHLmELoaybftRBUSErhqYiQzvEmuB",
}

def now_ts() -> int:
    return int(time.time())

def _lamports_to_sol(lamports: int) -> float:
    return float(lamports) / 1_000_000_000.0

# ----------------------------
# In-memory State
# ----------------------------
STATE: Dict[str, Any] = {
    "started_at": now_ts(),
    "cash_usd": float(START_CASH_USD),
    "reserve_cash_usd": 0.0,              # IMPORTANT: starts at 0 (no initial split)
    "positions": {},                      # mint -> position dict
    "trades_count": 0,
    "counters": {
        "webhooks_received": 0,
        "webhooks_unauthorized": 0,
        "skipped_no_secret": 0,
        "skipped_bad_payload": 0,
        "skipped_bad_path": 0,
        "skipped_low_cash": 0,
        "skip_no_trade_extract": 0,
        "buys": 0,
        "sells": 0,
        "forced_exits": 0,
    },
    "recent_trades": [],                  # small list of last N trades
}

EVENTS: List[Dict[str, Any]] = []
EVENTS_MAX = 500

def push_event(e: Dict[str, Any]) -> None:
    e = dict(e)
    e.setdefault("ts", now_ts())
    EVENTS.append(e)
    if len(EVENTS) > EVENTS_MAX:
        del EVENTS[: len(EVENTS) - EVENTS_MAX]

def add_recent_trade(t: Dict[str, Any]) -> None:
    STATE["recent_trades"].append(t)
    if len(STATE["recent_trades"]) > 50:
        del STATE["recent_trades"][: len(STATE["recent_trades"]) - 50]

def get_config_snapshot() -> Dict[str, Any]:
    return {
        "SOL_PRICE_USD": SOL_PRICE_USD,
        "START_CASH_USD": START_CASH_USD,
        "MAX_BUY_USD": MAX_BUY_USD,
        "MIN_CASH_LEFT_USD": MIN_CASH_LEFT_USD,
        "RESERVE_PCT": RESERVE_PCT,
        "TRADABLE_PCT": TRADABLE_PCT,
        "HOLD_MAX_SECONDS": HOLD_MAX_SECONDS,
        "FORCED_EXIT_FALLBACK_MULTI": FORCED_EXIT_FALLBACK_MULTI,
        "TAKE_PROFIT_1": TAKE_PROFIT_1,
        "TP1_SELL_PCT": TP1_SELL_PCT,
        "TAKE_PROFIT_2": TAKE_PROFIT_2,
        "STOP_LOSS": STOP_LOSS,
        "ENABLE_TRAILING": ENABLE_TRAILING,
        "TRAIL_ACTIVATE": TRAIL_ACTIVATE,
        "TRAIL_PCT": TRAIL_PCT,
        "TRACKED_WALLETS_COUNT": len(TRACKED_WALLETS),
        "DEBUG_WEBHOOK": DEBUG_WEBHOOK,
        "WEBHOOK_PATH_TOKEN_SET": bool(WEBHOOK_PATH_TOKEN),
        "WEBHOOK_SECRET_SET": bool(WEBHOOK_SECRET),
    }

# ----------------------------
# Paper Trading Core
# ----------------------------
def can_buy(usd: float) -> bool:
    return (STATE["cash_usd"] - usd) >= MIN_CASH_LEFT_USD

def ensure_position(mint: str) -> Dict[str, Any]:
    pos = STATE["positions"].get(mint)
    if pos is None:
        pos = {
            "mint": mint,
            "qty": 0.0,
            "cost_usd": 0.0,
            "avg_px": 0.0,
            "opened_ts": None,
            "last_buy_ts": None,
            "peak_px": 0.0,
            "tp1_done": False,
            "tp2_done": False,
            "trail_active": False,
        }
        STATE["positions"][mint] = pos
    return pos

def paper_buy(mint: str, usd: float, price_usd: float, reason: str, meta: Dict[str, Any]) -> None:
    usd = float(usd)
    price_usd = float(price_usd)
    if usd <= 0 or price_usd <= 0:
        return

    if not can_buy(usd):
        STATE["counters"]["skipped_low_cash"] += 1
        push_event({"kind": "skip_low_cash", "cash": STATE["cash_usd"], "want": usd, "mint": mint})
        return

    qty = usd / price_usd

    pos = ensure_position(mint)

    # Update position weighted average
    new_cost = pos["cost_usd"] + usd
    new_qty = pos["qty"] + qty
    pos["cost_usd"] = new_cost
    pos["qty"] = new_qty
    pos["avg_px"] = (new_cost / new_qty) if new_qty > 0 else 0.0

    t = now_ts()
    if pos["opened_ts"] is None:
        pos["opened_ts"] = t
    pos["last_buy_ts"] = t

    # Update peak price tracking
    pos["peak_px"] = max(pos.get("peak_px") or 0.0, price_usd)

    # Deduct cash
    STATE["cash_usd"] -= usd

    STATE["trades_count"] += 1
    STATE["counters"]["buys"] += 1

    push_event({
        "kind": "paper_buy",
        "mint": mint,
        "usd": usd,
        "qty": qty,
        "price_usd": price_usd,
        "cash_after": STATE["cash_usd"],
        "reason": reason,
        "meta": meta,
    })

    add_recent_trade({
        "side": "BUY",
        "mint": mint,
        "usd": usd,
        "qty": qty,
        "price_usd": price_usd,
        "ts": t,
        "reason": reason,
    })

def paper_sell(mint: str, sell_qty: float, price_usd: float, reason: str, meta: Dict[str, Any]) -> None:
    price_usd = float(price_usd)
    if price_usd <= 0:
        return

    pos = STATE["positions"].get(mint)
    if not pos or pos["qty"] <= 0:
        return

    sell_qty = float(min(sell_qty, pos["qty"]))
    if sell_qty <= 0:
        return

    # Cost basis portion
    avg_px = float(pos["avg_px"] or 0.0)
    cost_basis = sell_qty * avg_px
    proceeds = sell_qty * price_usd
    pnl = proceeds - cost_basis

    # Reserve grows ONLY from PROFIT (not from initial cash)
    reserve_add = 0.0
    if pnl > 0:
        reserve_add = pnl * float(RESERVE_PCT)

    # Cash gets proceeds minus reserve_add (so cash still goes up on winners)
    STATE["cash_usd"] += (proceeds - reserve_add)
    STATE["reserve_cash_usd"] += reserve_add

    # Reduce position
    pos["qty"] -= sell_qty
    pos["cost_usd"] -= cost_basis
    if pos["qty"] <= 1e-12:
        # close
        pos["qty"] = 0.0
        pos["cost_usd"] = 0.0
        pos["avg_px"] = 0.0
        pos["opened_ts"] = None
        pos["last_buy_ts"] = None
        pos["peak_px"] = 0.0
        pos["tp1_done"] = False
        pos["tp2_done"] = False
        pos["trail_active"] = False
    else:
        pos["avg_px"] = (pos["cost_usd"] / pos["qty"]) if pos["qty"] > 0 else 0.0

    STATE["trades_count"] += 1
    STATE["counters"]["sells"] += 1

    push_event({
        "kind": "paper_sell",
        "mint": mint,
        "qty": sell_qty,
        "price_usd": price_usd,
        "proceeds_usd": proceeds,
        "pnl_usd": pnl,
        "reserve_add_usd": reserve_add,
        "cash_after": STATE["cash_usd"],
        "reserve_after": STATE["reserve_cash_usd"],
        "reason": reason,
        "meta": meta,
    })

    add_recent_trade({
        "side": "SELL",
        "mint": mint,
        "qty": sell_qty,
        "price_usd": price_usd,
        "proceeds_usd": proceeds,
        "pnl_usd": pnl,
        "reserve_add_usd": reserve_add,
        "ts": now_ts(),
        "reason": reason,
    })

def on_price_update(mint: str, price_usd: float, reason: str, meta: Dict[str, Any]) -> None:
    """Apply TP/SL/trailing logic on new observed price."""
    pos = STATE["positions"].get(mint)
    if not pos or pos["qty"] <= 0:
        return

    price_usd = float(price_usd)
    avg_px = float(pos["avg_px"] or 0.0)
    if avg_px <= 0:
        return

    # Update peak price
    pos["peak_px"] = max(float(pos.get("peak_px") or 0.0), price_usd)

    ret = (price_usd / avg_px) - 1.0
    peak_ret = (pos["peak_px"] / avg_px) - 1.0 if pos["peak_px"] > 0 else ret

    # Stop loss (sell all)
    if ret <= float(STOP_LOSS):
        paper_sell(mint, pos["qty"], price_usd, "stop_loss", {**meta, "ret": ret})
        return

    # Take profit 2 (sell rest)
    if (not pos.get("tp2_done")) and ret >= float(TAKE_PROFIT_2):
        pos["tp2_done"] = True
        paper_sell(mint, pos["qty"], price_usd, "take_profit_2", {**meta, "ret": ret})
        return

    # Take profit 1 (sell partial)
    if (not pos.get("tp1_done")) and ret >= float(TAKE_PROFIT_1):
        pos["tp1_done"] = True
        sell_qty = pos["qty"] * float(TP1_SELL_PCT)
        # Guard against microscopic leftovers
        sell_qty = max(0.0, min(sell_qty, pos["qty"]))
        if sell_qty > 0:
            paper_sell(mint, sell_qty, price_usd, "take_profit_1", {**meta, "ret": ret})
        # don't return; trailing may still apply later

    # Trailing stop
    if ENABLE_TRAILING:
        if (not pos.get("trail_active")) and peak_ret >= float(TRAIL_ACTIVATE):
            pos["trail_active"] = True

        if pos.get("trail_active"):
            # drawdown from peak
            if peak_ret > 0:
                drawdown = (price_usd / pos["peak_px"]) - 1.0  # negative when below peak
                if drawdown <= -float(TRAIL_PCT):
                    paper_sell(mint, pos["qty"], price_usd, "trailing_stop", {**meta, "ret": ret, "peak_ret": peak_ret})
                    return

def time_exit_checks() -> None:
    """If a position is held longer than HOLD_MAX_SECONDS, force exit using last known price."""
    if HOLD_MAX_SECONDS <= 0:
        return

    t = now_ts()
    for mint, pos in list(STATE["positions"].items()):
        if not pos or pos["qty"] <= 0:
            continue
        opened = pos.get("opened_ts")
        if not opened:
            continue
        age = t - int(opened)
        if age >= int(HOLD_MAX_SECONDS):
            # Force exit at a conservative price:
            # Use avg_px * FORCED_EXIT_FALLBACK_MULTI (i.e., assume bad fill)
            avg_px = float(pos.get("avg_px") or 0.0)
            if avg_px <= 0:
                continue
            forced_px = avg_px * float(FORCED_EXIT_FALLBACK_MULTI)
            STATE["counters"]["forced_exits"] += 1
            paper_sell(mint, pos["qty"], forced_px, "time_exit", {"hold_max_seconds": HOLD_MAX_SECONDS})
            push_event({"kind": "forced_exit", "mint": mint, "age_s": age, "forced_px": forced_px})

# ----------------------------
# Trade extraction from Helius Enhanced tx
# ----------------------------
def extract_implied_trade_from_enhanced_tx(tx: dict, tracked_wallet: str, sol_price_usd: float) -> Optional[dict]:
    """
    Returns None if we can't confidently extract a trade.
    Otherwise returns:
      {
        "mint": <token mint>,
        "side": "BUY" or "SELL",
        "token_qty": float,
        "usd_value": float,
        "price_usd": float,
        "type": <tx type>,
        "signature": <sig>,
      }
    """
    tx_type = (tx.get("type") or "").upper()
    if tx_type != "SWAP":
        return None

    token_transfers = tx.get("tokenTransfers") or []
    native_transfers = tx.get("nativeTransfers") or []

    # Find candidate non-SOL, non-stable token movement involving tracked_wallet
    candidates = []
    for t in token_transfers:
        mint = t.get("mint")
        if not mint or mint == SOL_MINT or mint in STABLE_MINTS:
            continue
        from_user = t.get("fromUserAccount")
        to_user = t.get("toUserAccount")
        if from_user == tracked_wallet or to_user == tracked_wallet:
            candidates.append(t)

    if not candidates:
        return None

    # Pick the largest absolute token transfer as "the traded token"
    def abs_amt(x):
        try:
            return abs(float(x.get("tokenAmount", 0.0)))
        except Exception:
            return 0.0

    tok = sorted(candidates, key=abs_amt, reverse=True)[0]
    mint = tok["mint"]
    token_qty = float(tok.get("tokenAmount") or 0.0)
    if token_qty == 0.0:
        return None

    side = "BUY" if tok.get("toUserAccount") == tracked_wallet else "SELL"

    # 1) Prefer stablecoin value
    stable_value = 0.0
    for t in token_transfers:
        mint2 = t.get("mint")
        if mint2 not in STABLE_MINTS:
            continue
        amt = float(t.get("tokenAmount") or 0.0)
        if side == "BUY" and t.get("fromUserAccount") == tracked_wallet:
            stable_value += abs(amt)
        if side == "SELL" and t.get("toUserAccount") == tracked_wallet:
            stable_value += abs(amt)

    usd_value = stable_value

    # 2) Otherwise approximate with SOL transfers involving tracked wallet
    if usd_value <= 0.0:
        sol_moved = 0.0
        for nt in native_transfers:
            lamports = int(nt.get("amount") or 0)
            from_u = nt.get("fromUserAccount")
            to_u = nt.get("toUserAccount")
            if side == "BUY" and from_u == tracked_wallet:
                sol_moved += _lamports_to_sol(lamports)
            if side == "SELL" and to_u == tracked_wallet:
                sol_moved += _lamports_to_sol(lamports)
        usd_value = sol_moved * float(sol_price_usd)

    if usd_value <= 0.0:
        return None

    price_usd = usd_value / abs(token_qty)

    return {
        "mint": mint,
        "side": side,
        "token_qty": abs(token_qty),
        "usd_value": float(usd_value),
        "price_usd": float(price_usd),
        "type": tx_type,
        "signature": tx.get("signature"),
    }

# ----------------------------
# Auth helpers
# ----------------------------
def masked(s: str, keep: int = 4) -> str:
    if not s:
        return ""
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + ("*" * (len(s) - keep))

def check_webhook_auth(req: Request) -> bool:
    """
    Accepts either:
      - Authorization: <secret>
      - X-Webhook-Secret: <secret>
    """
    if not WEBHOOK_SECRET:
        STATE["counters"]["skipped_no_secret"] += 1
        return False

    auth = req.headers.get("authorization") or ""
    xsec = req.headers.get("x-webhook-secret") or ""
    # Helius UI sometimes calls it "Authentication Header" but you control the value.
    # We'll accept either header.

    got = auth.strip() or xsec.strip()
    return got == WEBHOOK_SECRET

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True, "ts": now_ts()}

@app.get("/events")
def get_events():
    return {"count": len(EVENTS), "events": EVENTS}

@app.get("/paper/state")
def paper_state():
    # run time-exit checks on reads too
    time_exit_checks()

    return {
        "cash_usd": round(float(STATE["cash_usd"]), 6),
        "reserve_cash_usd": round(float(STATE["reserve_cash_usd"]), 6),
        "positions": STATE["positions"],
        "trades_count": STATE["trades_count"],
        "counters": STATE["counters"],
        "started_at": STATE["started_at"],
        "config": get_config_snapshot(),
        "recent_trades": STATE["recent_trades"],
    }

@app.post("/webhook/{path_token}")
async def webhook(path_token: str, request: Request):
    # Path token check (prevents random hits)
    if WEBHOOK_PATH_TOKEN and path_token != WEBHOOK_PATH_TOKEN:
        STATE["counters"]["skipped_bad_path"] += 1
        if DEBUG_WEBHOOK:
            push_event({
                "kind": "webhook_bad_path",
                "got": path_token,
                "want": WEBHOOK_PATH_TOKEN,
            })
        raise HTTPException(status_code=404, detail="Not found")

    # Auth check
    if not check_webhook_auth(request):
        STATE["counters"]["webhooks_unauthorized"] += 1
        if DEBUG_WEBHOOK:
            auth = request.headers.get("authorization")
            xsec = request.headers.get("x-webhook-secret")
            push_event({
                "kind": "webhook_unauthorized_debug",
                "reason": "missing_or_mismatch_header",
                "auth_present": bool(auth),
                "x_present": bool(xsec),
                "auth_len": len(auth or ""),
                "x_len": len(xsec or ""),
                "server_secret_len": len(WEBHOOK_SECRET or ""),
                "got_masked": masked((auth or xsec or ""), keep=6),
            })
        raise HTTPException(status_code=401, detail="Unauthorized")

    STATE["counters"]["webhooks_received"] += 1

    # Parse payload
    try:
        payload = await request.json()
    except Exception:
        STATE["counters"]["skipped_bad_payload"] += 1
        push_event({"kind": "skip_bad_payload"})
        raise HTTPException(status_code=400, detail="Bad JSON")

    # Helius Enhanced webhooks often send a LIST of tx objects
    if isinstance(payload, dict):
        # sometimes wrapped
        txs = payload.get("transactions") or payload.get("data") or payload.get("txs") or payload.get("result")
        if txs is None:
            # maybe it's a single tx dict
            txs = [payload]
    else:
        txs = payload

    if not isinstance(txs, list):
        STATE["counters"]["skipped_bad_payload"] += 1
        push_event({"kind": "skip_bad_payload_type", "payload_type": str(type(payload))})
        raise HTTPException(status_code=400, detail="Bad payload format")

    matched = 0

    # Optional raw sample log (helps debugging pricing)
    if DEBUG_WEBHOOK:
        push_event({
            "kind": "raw_payload_sample",
            "payload_type": "list",
            "sample": txs[:1],  # only log 1 to avoid huge spam
        })

    # Process each tx
    for tx in txs:
        if not isinstance(tx, dict):
            continue

        sig = tx.get("signature")
        tx_type = (tx.get("type") or "").upper()

        # Filter by tracked wallets: if TRACKED_WALLETS empty, match nothing.
        if not TRACKED_WALLETS:
            continue

        # We'll treat a tx as "related" if feePayer is tracked OR any tokenTransfer touches tracked wallet
        fee_payer = tx.get("feePayer")
        related_wallets = set()
        if fee_payer:
            related_wallets.add(fee_payer)

        for t in (tx.get("tokenTransfers") or []):
            if t.get("fromUserAccount"):
                related_wallets.add(t.get("fromUserAccount"))
            if t.get("toUserAccount"):
                related_wallets.add(t.get("toUserAccount"))

        # Find which tracked wallet this tx is for
        wallet = None
        for w in TRACKED_WALLETS:
            if w in related_wallets:
                wallet = w
                break

        if not wallet:
            continue

        # Only SWAP drives pricing + decisions
        trade = extract_implied_trade_from_enhanced_tx(tx, tracked_wallet=wallet, sol_price_usd=float(SOL_PRICE_USD))
        if not trade:
            continue

        matched += 1
        mint = trade["mint"]
        price_usd = trade["price_usd"]

        # If tracked wallet is BUYing the token, we follow with our own paper buy (fixed MAX_BUY_USD)
        if trade["side"] == "BUY":
            paper_buy(
                mint=mint,
                usd=min(float(MAX_BUY_USD), float(STATE["cash_usd"]) - float(MIN_CASH_LEFT_USD)),
                price_usd=price_usd,
                reason="matched_helius_swap",
                meta={"signature": sig, "type": tx_type, "wallet": wallet, "mint": mint},
            )

        # Regardless of side, use this observed price as a "price update" to trigger TP/SL/trailing
        on_price_update(
            mint=mint,
            price_usd=price_usd,
            reason="swap_price_update",
            meta={"signature": sig, "type": tx_type, "wallet": wallet, "mint": mint},
        )

    # Time exits
    time_exit_checks()

    push_event({
        "kind": "webhook_ok",
        "payload_type": "list",
        "matched": matched,
        "tracked_wallets_count": len(TRACKED_WALLETS),
    })

    return JSONResponse({"ok": True, "matched": matched})
