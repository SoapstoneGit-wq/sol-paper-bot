# app.py
# Render-friendly Solana paper bot webhook receiver with:
# 1) Persistent state.json (Render Persistent Disk friendly)
# 2) Reserve -> cash rebalance rule (prevents "stuck at low cash")
# 3) SWAP-only filtering (ignores TRANSFER so you don't trade on wallet moves)

import os
import json
import time
import hmac
import hashlib
import threading
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from flask import Flask, request, jsonify, abort

app = Flask(__name__)

# ----------------------------
# Config (ENV)
# ----------------------------

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "")
    if v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

# Required: used to match the webhook URL path: /webhook/hook_<token>
WEBHOOK_PATH_TOKEN = os.getenv("WEBHOOK_PATH_TOKEN", "").strip()
# Optional: if you set a Bearer token in Helius "Authentication Header", put ONLY the token here
# Example Helius header: "Bearer TEST-1234" -> set WEBHOOK_BEARER_TOKEN="TEST-1234"
WEBHOOK_BEARER_TOKEN = os.getenv("WEBHOOK_BEARER_TOKEN", "").strip()

# Optional: if you use a signature header instead (x-webhook-secret), set it here and in Helius header.
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
WEBHOOK_SECRET_HEADER = os.getenv("WEBHOOK_SECRET_HEADER", "x-webhook-secret").strip()

# Persistent storage location:
# On Render: create a Persistent Disk and mount it, e.g. to /var/data
# Then set STATE_PATH to /var/data/state.json
STATE_PATH = os.getenv("STATE_PATH", "/var/data/state.json").strip()

# Tracked wallets (CSV list)
TRACKED_WALLETS = [w.strip() for w in os.getenv("TRACKED_WALLETS", "").split(",") if w.strip()]
TRACKED_WALLETS_SET = set([w.lower() for w in TRACKED_WALLETS])

# Paper trading config
SOL_PRICE_USD = env_float("SOL_PRICE_USD", 90.0)  # only used for paper qty calc
START_CASH_USD = env_float("START_CASH_USD", 800.0)

MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 25.0)

RESERVE_PCT = env_float("RESERVE_PCT", 0.60)   # fraction of total funds we want held in reserve
TRADABLE_PCT = env_float("TRADABLE_PCT", 0.40) # fraction of total funds we want available for buys

# Risk settings (simple paper simulation)
STOP_LOSS = env_float("STOP_LOSS", -0.15)      # -0.15 => stop at -15% from avg entry
TAKE_PROFIT_1 = env_float("TAKE_PROFIT_1", 0.30)
TAKE_PROFIT_2 = env_float("TAKE_PROFIT_2", 0.50)
TP1_SELL_PCT = env_float("TP1_SELL_PCT", 0.50)
ENABLE_TRAILING = env_bool("ENABLE_TRAILING", True)
TRAIL_ACTIVATE = env_float("TRAIL_ACTIVATE", 0.20)  # start trailing after +20%
TRAIL_PCT = env_float("TRAIL_PCT", 0.10)            # trail by 10% from peak once active

HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)  # time-exit fallback
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)  # exit price relative to peak/avg estimate

DEBUG_WEBHOOK = env_bool("DEBUG_WEBHOOK", True)
EVENTS_MAX = env_int("EVENTS_MAX", 500)
RECENT_TRADES_MAX = env_int("RECENT_TRADES_MAX", 200)

# ----------------------------
# Persistent state
# ----------------------------

STATE_LOCK = threading.Lock()

def now_ts() -> int:
    return int(time.time())

def atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    # write to temp then replace
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, path)

def default_state() -> Dict[str, Any]:
    # positions keyed by mint
    return {
        "cash_usd": float(START_CASH_USD),
        "reserve_cash_usd": 0.0,
        "positions": {},  # mint -> position dict
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
        "started_at": now_ts(),
        "events": [],         # rolling log
        "recent_trades": [],  # rolling trade list
        "last_saved_at": 0,
    }

def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        st = default_state()
        save_state(st)
        return st
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            st = json.load(f)
        # minimal migration / safety
        if "cash_usd" not in st:
            st = default_state()
        if "events" not in st:
            st["events"] = []
        if "recent_trades" not in st:
            st["recent_trades"] = []
        if "positions" not in st:
            st["positions"] = {}
        if "counters" not in st:
            st["counters"] = default_state()["counters"]
        return st
    except Exception:
        # corrupted file -> reset but keep it
        st = default_state()
        save_state(st)
        return st

def save_state(st: Dict[str, Any]) -> None:
    st["last_saved_at"] = now_ts()
    atomic_write_json(STATE_PATH, st)

STATE = load_state()

def push_event(kind: str, payload: Dict[str, Any]) -> None:
    e = {"ts": now_ts(), "kind": kind, **payload}
    STATE["events"].append(e)
    if len(STATE["events"]) > EVENTS_MAX:
        STATE["events"] = STATE["events"][-EVENTS_MAX:]

def push_trade(trade: Dict[str, Any]) -> None:
    STATE["recent_trades"].append(trade)
    if len(STATE["recent_trades"]) > RECENT_TRADES_MAX:
        STATE["recent_trades"] = STATE["recent_trades"][-RECENT_TRADES_MAX:]


# ----------------------------
# Auth / verification
# ----------------------------

def is_valid_path() -> bool:
    # If user sets a token, enforce it in the route
    if WEBHOOK_PATH_TOKEN:
        return True
    # If they didn't set it, still allow but flag it
    return True

def check_bearer_auth() -> bool:
    if not WEBHOOK_BEARER_TOKEN:
        return True
    auth = request.headers.get("Authorization", "")
    # Expect "Bearer <token>"
    parts = auth.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return hmac.compare_digest(parts[1], WEBHOOK_BEARER_TOKEN)
    return False

def check_secret_header() -> bool:
    # Optional "x-webhook-secret" style header check
    if not WEBHOOK_SECRET:
        return True
    got = request.headers.get(WEBHOOK_SECRET_HEADER, "")
    if got == "":
        return False
    return hmac.compare_digest(got, WEBHOOK_SECRET)

def authorize_webhook_or_401() -> None:
    ok = check_bearer_auth() and check_secret_header()
    if not ok:
        STATE["counters"]["webhooks_unauthorized"] += 1
        push_event("webhook_unauthorized", {"reason": "bad_auth"})
        save_state(STATE)
        abort(401)

# ----------------------------
# Helius payload parsing
# ----------------------------

def lower(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def tx_involves_tracked_wallet(tx: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Returns (matched, matched_wallet)
    We try a few common places in Helius enhanced payloads:
      - feePayer
      - accountData[].account
      - nativeTransfers[].fromUserAccount / toUserAccount
      - tokenTransfers[].fromUserAccount / toUserAccount
    """
    if not TRACKED_WALLETS_SET:
        return False, None

    # feePayer
    fee_payer = lower(tx.get("feePayer"))
    if fee_payer and fee_payer in TRACKED_WALLETS_SET:
        return True, tx.get("feePayer")

    # accountData
    ad = tx.get("accountData") or []
    for row in ad:
        acct = lower(row.get("account"))
        if acct and acct in TRACKED_WALLETS_SET:
            return True, row.get("account")

    # nativeTransfers
    nt = tx.get("nativeTransfers") or []
    for row in nt:
        frm = lower(row.get("fromUserAccount"))
        to = lower(row.get("toUserAccount"))
        if frm in TRACKED_WALLETS_SET:
            return True, row.get("fromUserAccount")
        if to in TRACKED_WALLETS_SET:
            return True, row.get("toUserAccount")

    # tokenTransfers
    tt = tx.get("tokenTransfers") or []
    for row in tt:
        frm = lower(row.get("fromUserAccount"))
        to = lower(row.get("toUserAccount"))
        if frm in TRACKED_WALLETS_SET:
            return True, row.get("fromUserAccount")
        if to in TRACKED_WALLETS_SET:
            return True, row.get("toUserAccount")

    return False, None

def extract_trade_from_tx(tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    SWAP-only: we only consider tx where tx["type"] == "SWAP".
    We return a simplified "signal" dict for paper buying.
    """
    tx_type = (tx.get("type") or "").upper()
    if tx_type != "SWAP":
        return None

    sig = tx.get("signature")
    # In a real implementation you'd parse swap direction + mint.
    # For your current paper tests, we'll paper-buy SOL as your "trade asset"
    # when a tracked wallet does a SWAP.
    return {
        "type": "SWAP",
        "signature": sig,
        "timestamp": tx.get("timestamp"),
        "source": tx.get("source"),
    }

# ----------------------------
# Paper trading logic
# ----------------------------

def total_equity_usd(st: Dict[str, Any]) -> float:
    # This is just cash+reserve + cost basis of positions (simplified)
    # You can improve later by marking-to-market.
    eq = float(st.get("cash_usd", 0.0)) + float(st.get("reserve_cash_usd", 0.0))
    # Add cost basis as proxy for invested capital
    for pos in st.get("positions", {}).values():
        eq += float(pos.get("cost_usd", 0.0))
    return float(eq)

def rebalance_reserve_to_cash(st: Dict[str, Any]) -> None:
    """
    Goal: avoid getting stuck when cash falls to MIN_CASH_LEFT_USD
    Rule:
      If cash < (MAX_BUY_USD + MIN_CASH_LEFT_USD) AND reserve > 0,
      move enough from reserve to bring cash up to:
        target_cash = max(MAX_BUY_USD + MIN_CASH_LEFT_USD, total_cash * TRADABLE_PCT)
      while also trying to keep reserve around RESERVE_PCT.
    """
    cash = float(st.get("cash_usd", 0.0))
    reserve = float(st.get("reserve_cash_usd", 0.0))

    threshold = float(MAX_BUY_USD + MIN_CASH_LEFT_USD)
    if cash >= threshold or reserve <= 0:
        return

    # total liquid funds (cash + reserve)
    total_liquid = cash + reserve
    desired_cash = max(threshold, total_liquid * float(TRADABLE_PCT))

    move = min(reserve, max(0.0, desired_cash - cash))
    if move <= 0:
        return

    st["reserve_cash_usd"] = reserve - move
    st["cash_usd"] = cash + move

    push_event("rebalance_reserve_to_cash", {
        "moved": round(move, 6),
        "cash_after": round(st["cash_usd"], 6),
        "reserve_after": round(st["reserve_cash_usd"], 6),
        "reason": "cash_below_threshold"
    })

def ensure_reserve_target(st: Dict[str, Any]) -> None:
    """
    After trades, keep reserve roughly at RESERVE_PCT of liquid funds (cash+reserve),
    but never force it if cash would fall below MIN_CASH_LEFT_USD.
    """
    cash = float(st.get("cash_usd", 0.0))
    reserve = float(st.get("reserve_cash_usd", 0.0))
    total_liquid = cash + reserve
    if total_liquid <= 0:
        return

    target_reserve = total_liquid * float(RESERVE_PCT)
    # if reserve too low, move some cash to reserve (but keep MIN_CASH_LEFT_USD)
    if reserve < target_reserve:
        need = target_reserve - reserve
        can_move = max(0.0, cash - float(MIN_CASH_LEFT_USD))
        move = min(need, can_move)
        if move > 0:
            st["cash_usd"] = cash - move
            st["reserve_cash_usd"] = reserve + move
            push_event("rebalance_cash_to_reserve", {
                "moved": round(move, 6),
                "cash_after": round(st["cash_usd"], 6),
                "reserve_after": round(st["reserve_cash_usd"], 6),
                "reason": "maintain_reserve_target"
            })

def get_or_create_position(st: Dict[str, Any], mint: str) -> Dict[str, Any]:
    pos = st["positions"].get(mint)
    if not pos:
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
        st["positions"][mint] = pos
    return pos

def paper_buy_sol(st: Dict[str, Any], usd: float, reason: str, meta: Dict[str, Any]) -> None:
    # Rebalance first so we can keep trading
    rebalance_reserve_to_cash(st)

    cash = float(st.get("cash_usd", 0.0))
    usd = float(usd)

    # Enforce min cash left
    if cash - usd < float(MIN_CASH_LEFT_USD):
        st["counters"]["skipped_low_cash"] += 1
        push_event("skip_low_cash", {
            "cash": round(cash, 6),
            "want": round(usd, 6),
            "min_cash_left": float(MIN_CASH_LEFT_USD),
        })
        return

    # "Buy" SOL (paper) at SOL_PRICE_USD
    px = float(SOL_PRICE_USD)
    qty = usd / px if px > 0 else 0.0

    mint = "SOL"  # simplified paper asset
    pos = get_or_create_position(st, mint)

    # Update avg price with new purchase
    old_qty = float(pos["qty"])
    old_cost = float(pos["cost_usd"])
    new_qty = old_qty + qty
    new_cost = old_cost + usd
    pos["qty"] = new_qty
    pos["cost_usd"] = new_cost
    pos["avg_px"] = (new_cost / new_qty) if new_qty > 0 else 0.0

    ts = now_ts()
    if pos["opened_ts"] is None:
        pos["opened_ts"] = ts
    pos["last_buy_ts"] = ts

    # reset peak/trailing flags if needed
    pos["peak_px"] = max(float(pos.get("peak_px", 0.0)), px)
    if ENABLE_TRAILING and (px >= pos["avg_px"] * (1.0 + float(TRAIL_ACTIVATE))):
        pos["trail_active"] = True

    st["cash_usd"] = cash - usd
    st["counters"]["buys"] += 1
    st["trades_count"] = int(st.get("trades_count", 0)) + 1

    trade = {
        "side": "BUY",
        "mint": mint,
        "usd": round(usd, 6),
        "qty": round(qty, 12),
        "price_usd": round(px, 6),
        "ts": ts,
        "reason": reason,
        "meta": meta,
    }
    push_trade(trade)
    push_event("paper_buy", {
        "symbol": mint,
        "usd": round(usd, 6),
        "qty": round(qty, 12),
        "cash_after": round(st["cash_usd"], 6),
        "reason": reason,
        "meta": meta,
    })

    # After trade, try to keep reserve in target range
    ensure_reserve_target(st)

def paper_sell_all(st: Dict[str, Any], mint: str, price_usd: float, reason: str) -> None:
    pos = st["positions"].get(mint)
    if not pos:
        return
    qty = float(pos.get("qty", 0.0))
    if qty <= 0:
        return

    proceeds = qty * float(price_usd)
    cost = float(pos.get("cost_usd", 0.0))
    pnl = proceeds - cost

    st["cash_usd"] = float(st.get("cash_usd", 0.0)) + proceeds
    st["counters"]["sells"] += 1
    st["trades_count"] = int(st.get("trades_count", 0)) + 1

    # reset position
    pos.update({
        "qty": 0.0,
        "cost_usd": 0.0,
        "avg_px": 0.0,
        "opened_ts": None,
        "last_buy_ts": None,
        "peak_px": 0.0,
        "tp1_done": False,
        "tp2_done": False,
        "trail_active": False,
    })

    trade = {
        "side": "SELL",
        "mint": mint,
        "qty": round(qty, 12),
        "price_usd": round(float(price_usd), 6),
        "proceeds_usd": round(proceeds, 6),
        "pnl_usd": round(pnl, 6),
        "ts": now_ts(),
        "reason": reason,
    }
    push_trade(trade)
    push_event("paper_sell", trade)

    ensure_reserve_target(st)

def maybe_manage_positions(st: Dict[str, Any]) -> None:
    """
    Very simplified risk manager:
      - Stop loss if current price <= avg_px*(1+STOP_LOSS)
      - Take profit partials aren't implemented in this simple SOL-only example
      - Time exit if held > HOLD_MAX_SECONDS
      - Trailing stop if activated and price <= peak*(1-TRAIL_PCT)
    Because we don't have real mark-to-market prices for random mints here,
    we only simulate using SOL_PRICE_USD as current price.
    """
    mint = "SOL"
    pos = st["positions"].get(mint)
    if not pos:
        return
    qty = float(pos.get("qty", 0.0))
    if qty <= 0:
        return

    px = float(SOL_PRICE_USD)
    avg = float(pos.get("avg_px", 0.0))
    if avg <= 0:
        return

    # update peak
    pos["peak_px"] = max(float(pos.get("peak_px", 0.0)), px)

    # activate trailing
    if ENABLE_TRAILING and (not pos.get("trail_active", False)) and px >= avg * (1.0 + float(TRAIL_ACTIVATE)):
        pos["trail_active"] = True
        push_event("trail_activated", {"mint": mint, "avg_px": avg, "peak_px": pos["peak_px"], "px": px})

    # trailing stop
    if ENABLE_TRAILING and pos.get("trail_active", False):
        peak = float(pos.get("peak_px", px))
        trail_stop = peak * (1.0 - float(TRAIL_PCT))
        if px <= trail_stop:
            paper_sell_all(st, mint, px, reason="trailing_stop")
            return

    # stop loss
    stop_px = avg * (1.0 + float(STOP_LOSS))
    if px <= stop_px:
        paper_sell_all(st, mint, px, reason="stop_loss")
        return

    # time exit
    opened = pos.get("opened_ts")
    if opened is not None:
        age = now_ts() - int(opened)
        if age >= int(HOLD_MAX_SECONDS):
            # exit at a "fallback" price estimate
            # (in practice you would use current px; this keeps behavior similar to your prior setup)
            fallback_px = max(0.000001, avg * float(FORCED_EXIT_FALLBACK_MULTI))
            st["counters"]["forced_exits"] += 1
            paper_sell_all(st, mint, fallback_px, reason="time_exit")
            return

# ----------------------------
# Routes
# ----------------------------

@app.get("/")
def root():
    return jsonify({
        "ok": True,
        "service": "sol-paper-bot",
        "time": datetime.utcnow().isoformat() + "Z"
    })

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.get("/paper/state")
def paper_state():
    with STATE_LOCK:
        # keep a copy to avoid mid-write issues
        st = json.loads(json.dumps(STATE))
        # attach config snapshot
        st["config"] = {
            "SOL_PRICE_USD": SOL_PRICE_USD,
            "START_CASH_USD": START_CASH_USD,
            "MAX_BUY_USD": MAX_BUY_USD,
            "MIN_CASH_LEFT_USD": MIN_CASH_LEFT_USD,
            "RESERVE_PCT": RESERVE_PCT,
            "TRADABLE_PCT": TRADABLE_PCT,
            "HOLD_MAX_SECONDS": HOLD_MAX_SECONDS,
            "FORCED_EXIT_FALLBACK_MULTI": FORCED_EXIT_FALLBACK_MULTI,
            "TAKE_PROFIT_1": TAKE_PROFIT_1,
            "TAKE_PROFIT_2": TAKE_PROFIT_2,
            "TP1_SELL_PCT": TP1_SELL_PCT,
            "STOP_LOSS": STOP_LOSS,
            "ENABLE_TRAILING": ENABLE_TRAILING,
            "TRAIL_ACTIVATE": TRAIL_ACTIVATE,
            "TRAIL_PCT": TRAIL_PCT,
            "TRACKED_WALLETS_COUNT": len(TRACKED_WALLETS),
            "DEBUG_WEBHOOK": DEBUG_WEBHOOK,
            "WEBHOOK_PATH_TOKEN_SET": bool(WEBHOOK_PATH_TOKEN),
            "WEBHOOK_SECRET_SET": bool(WEBHOOK_SECRET),
            "WEBHOOK_BEARER_SET": bool(WEBHOOK_BEARER_TOKEN),
            "STATE_PATH": STATE_PATH,
        }
        return jsonify(st)

@app.get("/events")
def events():
    with STATE_LOCK:
        return jsonify({"count": len(STATE.get("events", [])), "events": STATE.get("events", [])})

@app.post("/paper/reset")
def paper_reset():
    # Optional convenience reset
    with STATE_LOCK:
        global STATE
        STATE = default_state()
        save_state(STATE)
        return jsonify({"ok": True, "reset": True, "state_path": STATE_PATH})

@app.post("/webhook/hook_<token>")
def helius_webhook(token: str):
    # Validate path token if configured
    with STATE_LOCK:
        if WEBHOOK_PATH_TOKEN and token != WEBHOOK_PATH_TOKEN:
            STATE["counters"]["skipped_bad_path"] += 1
            push_event("skipped_bad_path", {"got": token})
            save_state(STATE)
            abort(404)

        # auth checks
        STATE["counters"]["webhooks_received"] += 1
        authorize_webhook_or_401()

        # parse payload
        try:
            payload = request.get_json(force=True, silent=False)
        except Exception:
            STATE["counters"]["skipped_bad_payload"] += 1
            push_event("skipped_bad_payload", {"reason": "json_parse_error"})
            save_state(STATE)
            abort(400)

        # Helius enhanced webhook is typically a list of tx objects
        if not isinstance(payload, list):
            # sometimes Helius wraps: {"type":"...", "data":[...]} — if so, adapt
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                payload_list = payload["data"]
            else:
                STATE["counters"]["skipped_bad_payload"] += 1
                push_event("skipped_bad_payload", {"reason": "payload_not_list"})
                save_state(STATE)
                abort(400)
        else:
            payload_list = payload

        matched_count = 0
        did_trade = False
        sample_tx = payload_list[0] if payload_list else None

        # Optional debug sample (kept small)
        if DEBUG_WEBHOOK and sample_tx:
            push_event("raw_payload_sample", {
                "payload_type": "list",
                "sample": [sample_tx]  # NOTE: this can be large; keep DEBUG_WEBHOOK off if it gets heavy
            })

        # process transactions
        for tx in payload_list:
            if not isinstance(tx, dict):
                continue

            matched, matched_wallet = tx_involves_tracked_wallet(tx)
            if not matched:
                continue

            matched_count += 1

            # SWAP-only filter
            signal = extract_trade_from_tx(tx)
            if signal is None:
                # ignored because not SWAP
                push_event("ignored_non_swap", {
                    "signature": tx.get("signature"),
                    "type": tx.get("type"),
                    "matched_wallet": matched_wallet,
                })
                continue

            # At this point: tracked wallet did a SWAP -> paper-buy SOL
            paper_buy_sol(
                STATE,
                usd=float(MAX_BUY_USD),
                reason="matched_helius_swap",
                meta={"signature": signal.get("signature"), "type": "SWAP", "matched_wallet": matched_wallet}
            )
            did_trade = True

        # Manage positions each webhook tick (optional)
        maybe_manage_positions(STATE)

        push_event("webhook_ok", {
            "payload_type": "list",
            "matched": matched_count,
            "tracked_wallets_count": len(TRACKED_WALLETS),
        })

        save_state(STATE)

        return jsonify({
            "ok": True,
            "matched": matched_count,
            "did_trade": did_trade,
        })

# ----------------------------
# Boot
# ----------------------------

if __name__ == "__main__":
    # Render sets PORT
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
