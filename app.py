import os
import json
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# -----------------------------
# Config helpers
# -----------------------------
def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default

def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v

def now_ts() -> int:
    return int(time.time())

# -----------------------------
# State + persistence
# -----------------------------
DEFAULT_STATE = {
    "cash_usd": 800.0,
    "reserve_cash_usd": 0.0,
    "positions": {},  # symbol -> position dict
    "trades_count": 0,
    "counters": {
        "webhooks_received": 0,
        "webhooks_unauthorized": 0,
        "skipped_no_secret": 0,
        "skipped_bad_payload": 0,
        "skipped_bad_path": 0,
        "skipped_low_cash": 0,
        "skipped_no_trade_extract": 0,
        "buys": 0,
        "sells": 0,
        "forced_exits": 0,
    },
    "started_at": now_ts(),
    "recent_trades": [],
}

STATE_LOCK = threading.Lock()
EVENTS_LOCK = threading.Lock()
EVENTS: List[Dict[str, Any]] = []
EVENTS_MAX = 500

def log_event(kind: str, **kwargs: Any) -> None:
    evt = {"ts": now_ts(), "kind": kind, **kwargs}
    with EVENTS_LOCK:
        EVENTS.append(evt)
        if len(EVENTS) > EVENTS_MAX:
            del EVENTS[: len(EVENTS) - EVENTS_MAX]

def state_path() -> str:
    # Render persistent disk should be mounted to /var/data
    return env_str("STATE_PATH", "/var/data/state.json")

def ensure_state_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def atomic_write_json(path: str, obj: Any) -> None:
    ensure_state_dir(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def load_state() -> Dict[str, Any]:
    path = state_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = json.load(f)
        # merge defaults for missing keys
        merged = dict(DEFAULT_STATE)
        merged.update(s if isinstance(s, dict) else {})
        if "counters" not in merged or not isinstance(merged["counters"], dict):
            merged["counters"] = dict(DEFAULT_STATE["counters"])
        else:
            c = dict(DEFAULT_STATE["counters"])
            c.update(merged["counters"])
            merged["counters"] = c
        if "positions" not in merged or not isinstance(merged["positions"], dict):
            merged["positions"] = {}
        if "recent_trades" not in merged or not isinstance(merged["recent_trades"], list):
            merged["recent_trades"] = []
        return merged
    except FileNotFoundError:
        s = dict(DEFAULT_STATE)
        # initialize cash from env on first boot
        s["cash_usd"] = env_float("START_CASH_USD", s["cash_usd"])
        s["reserve_cash_usd"] = 0.0
        return s
    except Exception:
        # if corrupted, fall back safely
        s = dict(DEFAULT_STATE)
        s["cash_usd"] = env_float("START_CASH_USD", s["cash_usd"])
        s["reserve_cash_usd"] = 0.0
        return s

def save_state(st: Dict[str, Any]) -> None:
    atomic_write_json(state_path(), st)

STATE: Dict[str, Any] = load_state()

# -----------------------------
# Trading / policy
# -----------------------------
def cfg_snapshot() -> Dict[str, Any]:
    return {
        "SOL_PRICE_USD": env_float("SOL_PRICE_USD", 90.0),
        "START_CASH_USD": env_float("START_CASH_USD", 800.0),
        "MAX_BUY_USD": env_float("MAX_BUY_USD", 25.0),
        "MIN_CASH_LEFT_USD": env_float("MIN_CASH_LEFT_USD", 5.0),
        "RESERVE_PCT": env_float("RESERVE_PCT", 0.40),   # <-- 40% reserve target
        "TRADABLE_PCT": env_float("TRADABLE_PCT", 0.60), # optional / informational
        "HOLD_MAX_SECONDS": env_int("HOLD_MAX_SECONDS", 900),
        "FORCED_EXIT_FALLBACK_MULTI": env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50),
        "DEBUG_WEBHOOK": env_str("DEBUG_WEBHOOK", "true").lower() == "true",
    }

def equity_usd(st: Dict[str, Any], px_usd: float) -> float:
    eq = float(st.get("cash_usd", 0.0)) + float(st.get("reserve_cash_usd", 0.0))
    for _, p in st.get("positions", {}).items():
        qty = float(p.get("qty", 0.0))
        eq += qty * px_usd
    return eq

def target_reserve_usd(st: Dict[str, Any], px_usd: float, reserve_pct: float) -> float:
    eq = equity_usd(st, px_usd)
    if reserve_pct < 0.0:
        reserve_pct = 0.0
    if reserve_pct > 0.95:
        reserve_pct = 0.95
    return eq * reserve_pct

def enforce_profit_split(st: Dict[str, Any], px_usd: float, reserve_pct: float) -> None:
    """
    Keep reserve around reserve_pct of equity.
    If reserve is above target, move excess to cash.
    If reserve is below target and cash is comfortably above MIN_CASH_LEFT, top up reserve.
    """
    cfg = cfg_snapshot()
    min_cash_left = cfg["MIN_CASH_LEFT_USD"]

    cash = float(st.get("cash_usd", 0.0))
    reserve = float(st.get("reserve_cash_usd", 0.0))
    target = target_reserve_usd(st, px_usd, reserve_pct)

    # Allow small jitter to avoid constant micro-moves
    eps = max(1.0, 0.0025 * max(1.0, target))

    # If reserve too high, move excess to cash
    if reserve > target + eps:
        move = reserve - target
        st["reserve_cash_usd"] = reserve - move
        st["cash_usd"] = cash + move
        log_event("reserve_to_cash_split", usd=round(move, 6), target_reserve=round(target, 6))

    # If reserve too low, move from cash -> reserve (but don't starve cash)
    cash = float(st.get("cash_usd", 0.0))
    reserve = float(st.get("reserve_cash_usd", 0.0))
    if reserve + eps < target and cash > (min_cash_left + eps):
        need = target - reserve
        can = max(0.0, cash - min_cash_left)
        move = min(need, can)
        if move > 0:
            st["cash_usd"] = cash - move
            st["reserve_cash_usd"] = reserve + move
            log_event("cash_to_reserve_split", usd=round(move, 6), target_reserve=round(target, 6))

def rebalance_reserve_to_cash_if_needed(st: Dict[str, Any]) -> None:
    """
    If cash is too low to place a new trade while keeping MIN_CASH_LEFT, pull from reserve.
    This is the 'rebalance reserve → cash if cash too low' rule.
    """
    cfg = cfg_snapshot()
    cash = float(st.get("cash_usd", 0.0))
    reserve = float(st.get("reserve_cash_usd", 0.0))

    max_buy = cfg["MAX_BUY_USD"]
    min_left = cfg["MIN_CASH_LEFT_USD"]

    # "ready to trade" threshold: enough for MAX_BUY + MIN_CASH_LEFT
    want_cash = max_buy + min_left
    if cash >= want_cash:
        return
    need = want_cash - cash
    move = min(need, reserve)
    if move > 0:
        st["cash_usd"] = cash + move
        st["reserve_cash_usd"] = reserve - move
        log_event("rebalance_reserve_to_cash", usd=round(move, 6), cash_target=round(want_cash, 6))

def get_or_init_position(st: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    pos = st["positions"].get(symbol)
    if not isinstance(pos, dict):
        pos = {
            "symbol": symbol,
            "qty": 0.0,
            "cost_usd": 0.0,
            "avg_px": 0.0,
            "opened_ts": None,
            "last_buy_ts": None,
            "peak_px": 0.0,
        }
        st["positions"][symbol] = pos
    return pos

def paper_buy(st: Dict[str, Any], symbol: str, usd: float, px_usd: float, reason: str) -> None:
    cfg = cfg_snapshot()
    cash = float(st.get("cash_usd", 0.0))
    if cash - usd < cfg["MIN_CASH_LEFT_USD"]:
        st["counters"]["skipped_low_cash"] += 1
        log_event("paper_buy_skipped_low_cash", usd=usd)
        return

    qty = usd / px_usd if px_usd > 0 else 0.0
    if qty <= 0:
        st["counters"]["skipped_no_trade_extract"] += 1
        log_event("paper_buy_skipped_bad_price", px_usd=px_usd)
        return

    pos = get_or_init_position(st, symbol)
    old_qty = float(pos.get("qty", 0.0))
    old_cost = float(pos.get("cost_usd", 0.0))
    new_qty = old_qty + qty
    new_cost = old_cost + usd
    pos["qty"] = new_qty
    pos["cost_usd"] = new_cost
    pos["avg_px"] = (new_cost / new_qty) if new_qty > 0 else 0.0
    if pos.get("opened_ts") is None:
        pos["opened_ts"] = now_ts()
    pos["last_buy_ts"] = now_ts()
    pos["peak_px"] = max(float(pos.get("peak_px", 0.0)), px_usd)

    st["cash_usd"] = cash - usd
    st["trades_count"] += 1
    st["counters"]["buys"] += 1
    st["recent_trades"].append({
        "side": "BUY",
        "symbol": symbol,
        "usd": usd,
        "qty": qty,
        "price_usd": px_usd,
        "ts": now_ts(),
        "reason": reason,
    })
    if len(st["recent_trades"]) > 200:
        st["recent_trades"] = st["recent_trades"][-200:]

    log_event("paper_buy", symbol=symbol, usd=round(usd, 6), qty=round(qty, 10), price_usd=round(px_usd, 6), reason=reason)

def paper_sell_all(st: Dict[str, Any], symbol: str, px_usd: float, reason: str) -> None:
    pos = get_or_init_position(st, symbol)
    qty = float(pos.get("qty", 0.0))
    if qty <= 0:
        return

    proceeds = qty * px_usd
    cost = float(pos.get("cost_usd", 0.0))
    pnl = proceeds - cost

    st["cash_usd"] = float(st.get("cash_usd", 0.0)) + proceeds
    st["trades_count"] += 1
    st["counters"]["sells"] += 1
    st["recent_trades"].append({
        "side": "SELL",
        "symbol": symbol,
        "qty": qty,
        "price_usd": px_usd,
        "proceeds_usd": proceeds,
        "pnl_usd": pnl,
        "ts": now_ts(),
        "reason": reason,
    })
    if len(st["recent_trades"]) > 200:
        st["recent_trades"] = st["recent_trades"][-200:]

    # reset position
    pos["qty"] = 0.0
    pos["cost_usd"] = 0.0
    pos["avg_px"] = 0.0
    pos["opened_ts"] = None
    pos["last_buy_ts"] = None
    pos["peak_px"] = 0.0

    log_event("paper_sell", symbol=symbol, proceeds_usd=round(proceeds, 6), pnl_usd=round(pnl, 6), reason=reason)

def evaluate_exits(st: Dict[str, Any], px_usd: float) -> None:
    """
    IMPORTANT: This must run BEFORE any buy logic.
    Currently you were seeing 'stuck not selling' because buy checks were returning early.
    """
    cfg = cfg_snapshot()
    hold_max = cfg["HOLD_MAX_SECONDS"]
    fallback_multi = cfg["FORCED_EXIT_FALLBACK_MULTI"]

    positions = st.get("positions", {})
    for symbol, p in list(positions.items()):
        if not isinstance(p, dict):
            continue
        qty = float(p.get("qty", 0.0))
        if qty <= 0:
            continue
        opened_ts = p.get("opened_ts")
        if opened_ts is None:
            continue

        age = now_ts() - int(opened_ts)
        if age >= hold_max:
            st["counters"]["forced_exits"] += 1
            # Forced exit uses fallback multiplier (simulates worse fill if you want)
            sell_px = px_usd * fallback_multi
            log_event("time_exit_triggered", symbol=symbol, age_seconds=age, hold_max=hold_max, sell_px=round(sell_px, 6))
            paper_sell_all(st, symbol, sell_px, reason="time_exit")

# -----------------------------
# Webhook parsing (Helius enhanced)
# -----------------------------
def normalize_authorization(req: Request) -> str:
    # Helius "Authentication Header" typically sets Authorization header
    # We'll check Authorization first, then x-webhook-secret optionally.
    auth = req.headers.get("authorization") or ""
    return auth.strip()

def is_authorized(req: Request) -> bool:
    expected = env_str("WEBHOOK_AUTH", "").strip()
    secret = env_str("WEBHOOK_SECRET", "").strip()

    if secret:
        got = (req.headers.get("x-webhook-secret") or "").strip()
        if got != secret:
            return False
        return True

    if not expected:
        # If you didn't set a secret/token, accept (not recommended)
        return True

    got = normalize_authorization(req)
    if got == expected:
        return True
    # Also allow "Bearer <expected>" or expected already includes Bearer
    if got.lower().startswith("bearer ") and got[7:].strip() == expected:
        return True
    if expected.lower().startswith("bearer ") and got == expected:
        return True
    return False

def extract_swaps(payload: Any) -> List[Dict[str, Any]]:
    """
    Helius enhanced webhooks often send a LIST of tx objects.
    We only keep items that are SWAP.
    """
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        # sometimes wrapped
        items = payload.get("events") or payload.get("data") or payload.get("transactions") or []
        if not isinstance(items, list):
            items = []
    else:
        return []

    swaps = []
    for tx in items:
        if not isinstance(tx, dict):
            continue
        # Common fields in Helius enhanced payloads
        t1 = (tx.get("type") or "").upper()
        t2 = (tx.get("transactionType") or "").upper()
        if t1 == "SWAP" or t2 == "SWAP":
            swaps.append(tx)
    return swaps

def matches_tracked_wallet(tx: Dict[str, Any], tracked: List[str]) -> bool:
    """
    Best-effort matching: try 'feePayer' and/or account keys.
    """
    if not tracked:
        return True
    fee = (tx.get("feePayer") or tx.get("fee_payer") or "").strip()
    if fee and fee in tracked:
        return True
    # Try accountData array if present
    ad = tx.get("accountData")
    if isinstance(ad, list):
        for a in ad:
            if isinstance(a, dict) and (a.get("account") in tracked):
                return True
    return False

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "service": "sol-paper-bot", "ts": now_ts()}

@app.get("/paper/state")
def paper_state():
    cfg = cfg_snapshot()
    with STATE_LOCK:
        st = STATE
        # include config snapshot + some "set" indicators for quick debugging
        resp = dict(st)
        resp["config"] = {
            **cfg,
            "WEBHOOK_PATH_TOKEN_SET": bool(env_str("WEBHOOK_PATH_TOKEN", "").strip()),
            "WEBHOOK_AUTH_SET": bool(env_str("WEBHOOK_AUTH", "").strip()),
            "WEBHOOK_SECRET_SET": bool(env_str("WEBHOOK_SECRET", "").strip()),
        }
        return JSONResponse(resp)

@app.get("/paper/events")
def paper_events():
    with EVENTS_LOCK:
        return {"count": len(EVENTS), "events": list(EVENTS)[-EVENTS_MAX:]}

@app.post("/paper/tick")
def paper_tick():
    """
    Manual endpoint to force exit checks + policy enforcement,
    useful if you want the bot to sell even when webhooks slow down.
    """
    cfg = cfg_snapshot()
    px = cfg["SOL_PRICE_USD"]
    reserve_pct = cfg["RESERVE_PCT"]

    with STATE_LOCK:
        st = STATE
        evaluate_exits(st, px)
        rebalance_reserve_to_cash_if_needed(st)
        enforce_profit_split(st, px, reserve_pct)
        save_state(st)
    return {"ok": True}

@app.post("/webhook/hook_{token}")
async def helius_webhook(token: str, request: Request):
    expected_token = env_str("WEBHOOK_PATH_TOKEN", "").strip()
    if expected_token and token != expected_token:
        with STATE_LOCK:
            STATE["counters"]["skipped_bad_path"] += 1
        log_event("webhook_bad_path", got=token)
        raise HTTPException(status_code=404, detail="bad path")

    with STATE_LOCK:
        STATE["counters"]["webhooks_received"] += 1

    if not is_authorized(request):
        with STATE_LOCK:
            STATE["counters"]["webhooks_unauthorized"] += 1
        log_event("webhook_unauthorized")
        raise HTTPException(status_code=401, detail="unauthorized")

    try:
        payload = await request.json()
    except Exception:
        with STATE_LOCK:
            STATE["counters"]["skipped_bad_payload"] += 1
        log_event("webhook_bad_payload")
        raise HTTPException(status_code=400, detail="bad json")

    cfg = cfg_snapshot()
    px = cfg["SOL_PRICE_USD"]
    reserve_pct = cfg["RESERVE_PCT"]

    tracked = [w.strip() for w in env_str("TRACKED_WALLETS", "").split(",") if w.strip()]
    swaps = extract_swaps(payload)

    # debug logging
    if cfg["DEBUG_WEBHOOK"]:
        log_event("webhook_ok", payload_type=("list" if isinstance(payload, list) else "dict"), tracked_wallets_count=len(tracked), swaps=len(swaps))

    # ---- IMPORTANT ORDER ----
    # 1) Exit checks (SELLS) must happen first so we don't get stuck in buy-skip loops
    # 2) Rebalance reserve -> cash if needed
    # 3) Enforce 60/40 policy (reserve_pct=0.40)
    # 4) Only then attempt buys from matched SWAPs
    with STATE_LOCK:
        st = STATE

        evaluate_exits(st, px)
        rebalance_reserve_to_cash_if_needed(st)
        enforce_profit_split(st, px, reserve_pct)

        matched = 0
        for tx in swaps:
            if not matches_tracked_wallet(tx, tracked):
                continue
            matched += 1

            # Paper strategy: buy SOL for MAX_BUY_USD on each matched SWAP
            usd = cfg["MAX_BUY_USD"]
            paper_buy(st, "SOL", usd=usd, px_usd=px, reason="matched_helius_swap")

            # After each buy, re-run policies
            rebalance_reserve_to_cash_if_needed(st)
            enforce_profit_split(st, px, reserve_pct)

        save_state(st)

    # Return how many swap txs matched tracked wallets
    return {"ok": True, "matched": matched, "swaps": len(swaps)}
