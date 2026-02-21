import os
import json
import time
import tempfile
import threading
from typing import Any, Dict, List, Optional, Tuple

import urllib.request
import urllib.error

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# ----------------------------
# Config helpers
# ----------------------------

def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v)

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _now_ts() -> int:
    return int(time.time())

# ----------------------------
# Runtime config (Render env vars)
# ----------------------------

STATE_PATH = _env_str("STATE_PATH", "/var/data/state.json")

# Static fallback if live price fails
SOL_PRICE_USD_STATIC_FALLBACK = _env_float("SOL_PRICE_USD", 90.0)

# Price source (pyth)
PRICE_SOURCE = _env_str("PRICE_SOURCE", "pyth").strip().lower()  # "pyth" or "static"
PYTH_HERMES_BASE_URL = _env_str("PYTH_HERMES_BASE_URL", "https://hermes.pyth.network").strip().rstrip("/")
PYTH_SOL_PRICE_ID = _env_str(
    "PYTH_SOL_PRICE_ID",
    "ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d",
).strip()
PRICE_REFRESH_SECONDS = _env_int("PRICE_REFRESH_SECONDS", 5)
PRICE_MAX_STALENESS_SECONDS = _env_int("PRICE_MAX_STALENESS_SECONDS", 60)

START_CASH_USD = _env_float("START_CASH_USD", 800.0)
MAX_BUY_USD = _env_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = _env_float("MIN_CASH_LEFT_USD", 5.0)

# Profit policy targets (equity split):
RESERVE_PCT = _env_float("RESERVE_PCT", 0.40)
TRADABLE_PCT = _env_float("TRADABLE_PCT", 0.60)

# Safety: time-based exit simulation
HOLD_MAX_SECONDS = _env_int("HOLD_MAX_SECONDS", 900)

# kept for compatibility (not used for sell pricing)
FORCED_EXIT_FALLBACK_MULTI = _env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)

DEBUG_WEBHOOK = _env_bool("DEBUG_WEBHOOK", False)

# Webhook protection
WEBHOOK_PATH_TOKEN = _env_str("WEBHOOK_PATH_TOKEN", "").strip()
WEBHOOK_SECRET = _env_str("WEBHOOK_SECRET", "").strip()

# Wallet filter
TRACKED_WALLETS_RAW = _env_str("TRACKED_WALLETS", "")
TRACKED_WALLETS = [w.strip() for w in TRACKED_WALLETS_RAW.split(",") if w.strip()]

# wSOL mint (used in many swaps)
WSOL_MINT = "So11111111111111111111111111111111111111112"

# ----------------------------
# Live price (Pyth Hermes)
# ----------------------------

_LIVE_PRICE_LOCK = threading.Lock()
_LIVE_SOL_PRICE_USD: float = SOL_PRICE_USD_STATIC_FALLBACK
_LIVE_SOL_PRICE_META: Dict[str, Any] = {
    "source": "static",
    "price": SOL_PRICE_USD_STATIC_FALLBACK,
    "conf": None,
    "publish_time": None,
    "error": None,
    "ts_fetched": _now_ts(),
}

def _http_get_json(url: str, timeout: int = 5) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "accept": "application/json",
            "user-agent": "sol-paper-bot/1.0",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))

def _fetch_pyth_sol_price() -> Tuple[float, Dict[str, Any]]:
    if not PYTH_SOL_PRICE_ID:
        raise RuntimeError("PYTH_SOL_PRICE_ID not set")

    url = f"{PYTH_HERMES_BASE_URL}/v2/updates/price/latest?ids[]={PYTH_SOL_PRICE_ID}"
    j = _http_get_json(url, timeout=5)

    parsed = j.get("parsed")
    if not isinstance(parsed, list) or not parsed:
        raise RuntimeError("Unexpected Hermes response: missing parsed")

    item = parsed[0]
    price_obj = item.get("price", {})
    if not isinstance(price_obj, dict):
        raise RuntimeError("Unexpected Hermes response: missing price object")

    price_raw = price_obj.get("price")
    expo = price_obj.get("expo")
    conf_raw = price_obj.get("conf")
    publish_time = price_obj.get("publish_time")

    if price_raw is None or expo is None:
        raise RuntimeError("Unexpected Hermes response: price/expo missing")

    try:
        pr = int(price_raw)
        ex = int(expo)
        conf = int(conf_raw) if conf_raw is not None else None
        px = float(pr) * (10.0 ** float(ex))
        conf_px = float(conf) * (10.0 ** float(ex)) if conf is not None else None
    except Exception as e:
        raise RuntimeError(f"Failed parsing Hermes price: {e}")

    meta = {
        "source": "pyth",
        "price": px,
        "conf": conf_px,
        "publish_time": int(publish_time) if publish_time is not None else None,
        "error": None,
        "ts_fetched": _now_ts(),
    }
    return px, meta

def get_live_sol_price() -> Tuple[float, Dict[str, Any]]:
    global _LIVE_SOL_PRICE_USD, _LIVE_SOL_PRICE_META

    if PRICE_SOURCE != "pyth":
        meta = {
            "source": "static",
            "price": SOL_PRICE_USD_STATIC_FALLBACK,
            "conf": None,
            "publish_time": None,
            "error": None,
            "ts_fetched": _now_ts(),
        }
        return SOL_PRICE_USD_STATIC_FALLBACK, meta

    with _LIVE_PRICE_LOCK:
        now = _now_ts()
        ts_fetched = int(_LIVE_SOL_PRICE_META.get("ts_fetched") or 0)
        if now - ts_fetched < max(1, PRICE_REFRESH_SECONDS):
            return float(_LIVE_SOL_PRICE_USD), dict(_LIVE_SOL_PRICE_META)

        try:
            px, meta = _fetch_pyth_sol_price()
            pub = meta.get("publish_time")
            if isinstance(pub, int) and (now - pub) > PRICE_MAX_STALENESS_SECONDS:
                meta["error"] = f"stale_price_publish_time age={now - pub}s"
                return float(_LIVE_SOL_PRICE_USD), dict(_LIVE_SOL_PRICE_META)

            _LIVE_SOL_PRICE_USD = float(px)
            _LIVE_SOL_PRICE_META = dict(meta)
            return float(_LIVE_SOL_PRICE_USD), dict(_LIVE_SOL_PRICE_META)
        except Exception as e:
            err = str(e)
            _LIVE_SOL_PRICE_META = {
                "source": "pyth",
                "price": float(_LIVE_SOL_PRICE_USD),
                "conf": None,
                "publish_time": _LIVE_SOL_PRICE_META.get("publish_time"),
                "error": err,
                "ts_fetched": _now_ts(),
            }
            if float(_LIVE_SOL_PRICE_USD) > 0:
                return float(_LIVE_SOL_PRICE_USD), dict(_LIVE_SOL_PRICE_META)
            return SOL_PRICE_USD_STATIC_FALLBACK, {
                "source": "static_fallback",
                "price": SOL_PRICE_USD_STATIC_FALLBACK,
                "conf": None,
                "publish_time": None,
                "error": err,
                "ts_fetched": _now_ts(),
            }

# ----------------------------
# State persistence
# ----------------------------

def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def load_state() -> Dict[str, Any]:
    base = {
        "cash_usd": float(START_CASH_USD),
        "reserve_cash_usd": 0.0,
        "positions": {},
        "trades_count": 0,
        "counters": {
            "webhooks_received": 0,
            "webhooks_unauthorized": 0,
            "skipped_no_secret": 0,
            "skipped_bad_payload": 0,
            "skipped_bad_path": 0,
            "skipped_low_cash": 0,
            "buys": 0,
            "sells": 0,
            "forced_exits": 0,
            "copy_sells": 0,
            "rebalances": 0,
        },
        "started_at": _now_ts(),
        "recent_trades": [],
    }

    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in base.items():
                if k not in data:
                    data[k] = v
            if "counters" not in data or not isinstance(data["counters"], dict):
                data["counters"] = base["counters"]
            else:
                for ck, cv in base["counters"].items():
                    if ck not in data["counters"]:
                        data["counters"][ck] = cv
            return data
    except Exception:
        return base

    return base

STATE: Dict[str, Any] = load_state()
EVENTS: List[Dict[str, Any]] = []

def save_state() -> None:
    try:
        _ensure_dir_for_file(STATE_PATH)
        tmp_dir = os.path.dirname(STATE_PATH) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="state_", suffix=".json", dir=tmp_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(STATE, f, ensure_ascii=False)
            os.replace(tmp_path, STATE_PATH)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    except Exception:
        pass

def log_event(kind: str, **fields: Any) -> None:
    evt = {"ts": _now_ts(), "kind": kind, **fields}
    EVENTS.append(evt)
    if len(EVENTS) > 5000:
        del EVENTS[: len(EVENTS) - 5000]

def push_recent_trade(rec: Dict[str, Any]) -> None:
    STATE["recent_trades"].append(rec)
    if len(STATE["recent_trades"]) > 200:
        STATE["recent_trades"] = STATE["recent_trades"][-200:]

# ----------------------------
# Profit policy + rebalance logic
# ----------------------------

def equity_usd() -> float:
    eq = float(STATE.get("cash_usd", 0.0)) + float(STATE.get("reserve_cash_usd", 0.0))
    positions = STATE.get("positions", {}) or {}
    for sym, p in positions.items():
        qty = float(p.get("qty", 0.0) or 0.0)
        if qty <= 0:
            continue
        px = float(p.get("mark_px", 0.0) or 0.0)
        if px <= 0:
            px = float(p.get("avg_px", 0.0) or 0.0)
        eq += qty * px
    return float(eq)

def rebalance_targets() -> None:
    eq = equity_usd()
    target_reserve = max(0.0, eq * float(RESERVE_PCT))
    cash = float(STATE.get("cash_usd", 0.0))
    reserve = float(STATE.get("reserve_cash_usd", 0.0))

    if reserve > target_reserve + 1e-9:
        move = reserve - target_reserve
        STATE["reserve_cash_usd"] = reserve - move
        STATE["cash_usd"] = cash + move
        STATE["counters"]["rebalances"] += 1
        log_event("rebalance_reserve_to_cash_target", move=round(move, 6), target_reserve=round(target_reserve, 6))
        return

    if reserve + 1e-9 < target_reserve:
        needed = target_reserve - reserve
        min_cash_floor = max(MIN_CASH_LEFT_USD, (MAX_BUY_USD + MIN_CASH_LEFT_USD))
        available = max(0.0, cash - min_cash_floor)
        move = min(needed, available)
        if move > 0:
            STATE["cash_usd"] = cash - move
            STATE["reserve_cash_usd"] = reserve + move
            STATE["counters"]["rebalances"] += 1
            log_event("rebalance_cash_to_reserve_target", move=round(move, 6), target_reserve=round(target_reserve, 6))

def ensure_cash_for_buy(buy_usd: float) -> None:
    cash = float(STATE.get("cash_usd", 0.0))
    reserve = float(STATE.get("reserve_cash_usd", 0.0))

    required = float(buy_usd) + float(MIN_CASH_LEFT_USD)
    if cash >= required:
        return

    short = required - cash
    if reserve <= 0:
        return

    move = min(short, reserve)
    if move > 0:
        STATE["reserve_cash_usd"] = reserve - move
        STATE["cash_usd"] = cash + move
        STATE["counters"]["rebalances"] += 1
        log_event("rebalance_reserve_to_cash_low_cash", move=round(move, 6), required=round(required, 6))

# ----------------------------
# SELL / BUY helpers
# ----------------------------

def paper_sell_all_sol(meta: Dict[str, Any], reason: str = "wallet_sell") -> bool:
    pos = (STATE.get("positions", {}) or {}).get("SOL")
    if not isinstance(pos, dict):
        return False
    qty = float(pos.get("qty", 0.0) or 0.0)
    if qty <= 0:
        return False

    live_px, live_meta = get_live_sol_price()
    sell_px = float(pos.get("mark_px", 0.0) or 0.0)
    if sell_px <= 0:
        sell_px = float(live_px)

    proceeds = qty * sell_px
    now = _now_ts()

    STATE["cash_usd"] = float(STATE.get("cash_usd", 0.0)) + proceeds

    # reset position fully (prevents insane avg_px carryover)
    pos["qty"] = 0.0
    pos["cost_usd"] = 0.0
    pos["avg_px"] = 0.0
    pos["mark_px"] = sell_px
    pos["closed_ts"] = now
    pos["last_sell_ts"] = now
    pos["opened_ts"] = None

    STATE["trades_count"] += 1
    STATE["counters"]["sells"] += 1
    STATE["counters"]["copy_sells"] += 1

    meta2 = dict(meta or {})
    meta2["sol_price_meta"] = live_meta

    log_event(
        "paper_sell_copy_wallet",
        symbol="SOL",
        qty=qty,
        px=sell_px,
        proceeds_usd=round(proceeds, 6),
        reason=reason,
        meta=meta2,
    )
    push_recent_trade({
        "side": "SELL",
        "symbol": "SOL",
        "qty": qty,
        "price_usd": sell_px,
        "proceeds_usd": proceeds,
        "ts": now,
        "reason": reason,
        "meta": meta2,
    })

    rebalance_targets()
    return True

# ----------------------------
# Selling logic (time-based forced exit)
# ----------------------------

def maybe_run_forced_exits() -> None:
    now = _now_ts()
    positions = STATE.get("positions", {}) or {}

    live_px, live_meta = get_live_sol_price()

    for sym, p in list(positions.items()):
        qty = float(p.get("qty", 0.0) or 0.0)
        if qty <= 0:
            continue
        opened = p.get("opened_ts")
        if not opened:
            continue

        age = now - int(opened)
        if age < HOLD_MAX_SECONDS:
            continue

        sell_px = float(p.get("mark_px", 0.0) or 0.0)
        if sell_px <= 0:
            sell_px = float(live_px)

        proceeds = qty * sell_px

        STATE["cash_usd"] = float(STATE.get("cash_usd", 0.0)) + proceeds

        p["qty"] = 0.0
        p["cost_usd"] = 0.0
        p["avg_px"] = 0.0
        p["closed_ts"] = now
        p["last_sell_ts"] = now
        p["opened_ts"] = None

        STATE["trades_count"] += 1
        STATE["counters"]["sells"] += 1
        STATE["counters"]["forced_exits"] += 1

        log_event(
            "paper_sell_forced_exit",
            symbol=sym,
            qty=qty,
            px=sell_px,
            proceeds_usd=round(proceeds, 6),
            age_s=age,
            price_meta=live_meta,
        )
        push_recent_trade({
            "side": "SELL",
            "symbol": sym,
            "qty": qty,
            "price_usd": sell_px,
            "proceeds_usd": proceeds,
            "ts": now,
            "reason": "time_exit",
        })

    rebalance_targets()

# ----------------------------
# Webhook parsing helpers (Helius Enhanced)
# ----------------------------

def is_swap_event(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    t = str(item.get("type") or item.get("transactionType") or item.get("txType") or "").upper()
    if t:
        return t == "SWAP"
    if "tokenSwap" in item:
        return True
    return False

def find_matched_wallet(item: Dict[str, Any]) -> Optional[str]:
    if not TRACKED_WALLETS:
        return None
    s = json.dumps(item, separators=(",", ":"), ensure_ascii=False)
    for w in TRACKED_WALLETS:
        if w and w in s:
            return w
    return None

def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def _detect_wallet_sol_delta(item: Dict[str, Any], wallet: str) -> Tuple[float, Dict[str, Any]]:
    """
    Attempts to compute net SOL delta for a given wallet from an enhanced tx object.
    Returns (delta_sol, debug_meta).

    We look in multiple common fields:
      - accountData[].nativeBalanceChange  (lamports delta)
      - nativeTransfers (lamports or SOL amounts)
      - tokenBalanceChanges for wSOL mint (So111...)
    """
    dbg: Dict[str, Any] = {"method_hits": [], "delta_native_sol": 0.0, "delta_wsol_sol": 0.0}

    # 1) accountData nativeBalanceChange (lamports)
    acc = item.get("accountData")
    if isinstance(acc, list):
        for a in acc:
            if not isinstance(a, dict):
                continue
            acct = a.get("account") or a.get("pubkey") or a.get("address")
            if acct != wallet:
                continue
            lamports_delta = a.get("nativeBalanceChange")
            if lamports_delta is not None:
                d = _safe_float(lamports_delta) / 1_000_000_000.0
                dbg["delta_native_sol"] += d
                dbg["method_hits"].append("accountData.nativeBalanceChange")

    # 2) nativeTransfers (some payloads include these)
    nt = item.get("nativeTransfers")
    if isinstance(nt, list):
        for t in nt:
            if not isinstance(t, dict):
                continue
            from_acct = t.get("fromUserAccount") or t.get("from")
            to_acct = t.get("toUserAccount") or t.get("to")
            amt = t.get("amount")  # often lamports; sometimes SOL
            if amt is None:
                continue
            # Heuristic: if amount is huge, treat as lamports; else treat as SOL
            a = _safe_float(amt)
            if abs(a) > 1_000_000:  # likely lamports
                a = a / 1_000_000_000.0
            if from_acct == wallet:
                dbg["delta_native_sol"] -= a
                dbg["method_hits"].append("nativeTransfers")
            if to_acct == wallet:
                dbg["delta_native_sol"] += a
                dbg["method_hits"].append("nativeTransfers")

    # 3) tokenBalanceChanges / tokenTransfers for wSOL mint
    # Different providers name this differently.
    tbc = item.get("tokenBalanceChanges") or item.get("tokenTransfers") or item.get("tokenBalanceChange")
    if isinstance(tbc, list):
        for ch in tbc:
            if not isinstance(ch, dict):
                continue
            mint = ch.get("mint") or ch.get("tokenMint") or ch.get("token")
            if mint != WSOL_MINT:
                continue
            owner = ch.get("userAccount") or ch.get("owner") or ch.get("accountOwner")
            if owner != wallet:
                continue

            # Try common delta fields
            # Some payloads: "rawTokenAmount": {"tokenAmount":"123", "decimals":9}
            raw = ch.get("rawTokenAmount")
            if isinstance(raw, dict):
                ta = raw.get("tokenAmount")
                dec = raw.get("decimals", 9)
                # If provider uses raw as absolute (not delta), this won't help.
                # We'll also check "tokenAmount" or "changeAmount".
                # Here we treat "tokenAmount" as delta if a "changeType" exists.
                change_type = str(ch.get("changeType") or "").lower()
                if ta is not None and change_type in ("inc", "increase", "dec", "decrease"):
                    val = _safe_float(ta) / (10.0 ** float(dec))
                    if change_type in ("dec", "decrease"):
                        val = -abs(val)
                    else:
                        val = abs(val)
                    dbg["delta_wsol_sol"] += val
                    dbg["method_hits"].append("tokenBalanceChanges.rawTokenAmount+changeType")

            # Alternative: direct "tokenAmount" delta
            if ch.get("tokenAmount") is not None and ch.get("decimals") is not None and ch.get("changeType"):
                val = _safe_float(ch.get("tokenAmount")) / (10.0 ** float(ch.get("decimals")))
                change_type = str(ch.get("changeType") or "").lower()
                if "dec" in change_type:
                    val = -abs(val)
                elif "inc" in change_type:
                    val = abs(val)
                dbg["delta_wsol_sol"] += val
                dbg["method_hits"].append("tokenBalanceChanges.tokenAmount+changeType")

            # Alternative: "amount" (may already be float SOL)
            if ch.get("amount") is not None and ch.get("direction"):
                val = _safe_float(ch.get("amount"))
                direction = str(ch.get("direction") or "").lower()
                if direction in ("out", "send", "sent"):
                    val = -abs(val)
                elif direction in ("in", "receive", "received"):
                    val = abs(val)
                dbg["delta_wsol_sol"] += val
                dbg["method_hits"].append("tokenTransfers.amount+direction")

    delta = float(dbg["delta_native_sol"]) + float(dbg["delta_wsol_sol"])
    dbg["delta_total_sol"] = delta
    return delta, dbg

def _classify_swap_side_for_wallet(item: Dict[str, Any], wallet: str) -> Tuple[str, Dict[str, Any]]:
    """
    Returns ("buy"|"sell"|"unknown", debug_meta)
    buy  => wallet net SOL increases (native+wSOL)
    sell => wallet net SOL decreases
    """
    delta, dbg = _detect_wallet_sol_delta(item, wallet)
    # small threshold to ignore dust/noise
    eps = 1e-6
    if delta > eps:
        return "buy", dbg
    if delta < -eps:
        return "sell", dbg
    return "unknown", dbg

# ----------------------------
# Paper trade execution
# ----------------------------

def paper_buy_sol(usd: float, meta: Dict[str, Any]) -> bool:
    usd = float(usd)

    ensure_cash_for_buy(usd)

    cash = float(STATE.get("cash_usd", 0.0))
    if cash < (usd + MIN_CASH_LEFT_USD):
        STATE["counters"]["skipped_low_cash"] += 1
        log_event("paper_buy_skipped_low_cash", usd=usd)
        return False

    px, px_meta = get_live_sol_price()
    px = float(px) if px and float(px) > 0 else float(SOL_PRICE_USD_STATIC_FALLBACK)

    qty = usd / px if px > 0 else 0.0

    pos = STATE.setdefault("positions", {}).setdefault("SOL", {
        "symbol": "SOL",
        "qty": 0.0,
        "cost_usd": 0.0,
        "avg_px": 0.0,
        "opened_ts": None,
        "last_buy_ts": None,
        "mark_px": px,
        "closed_ts": None,
        "last_sell_ts": None,
    })

    prev_qty = float(pos.get("qty", 0.0) or 0.0)

    if prev_qty <= 0.0:
        pos["qty"] = 0.0
        pos["cost_usd"] = 0.0
        pos["avg_px"] = 0.0
        pos["opened_ts"] = None
        pos["closed_ts"] = None
        pos["last_sell_ts"] = None

    prev_qty = float(pos.get("qty", 0.0) or 0.0)
    prev_cost = float(pos.get("cost_usd", 0.0) or 0.0)

    new_cost = prev_cost + usd
    new_qty = prev_qty + qty
    new_avg = (new_cost / new_qty) if new_qty > 0 else 0.0

    pos["qty"] = new_qty
    pos["cost_usd"] = new_cost
    pos["avg_px"] = new_avg
    pos["mark_px"] = px

    now = _now_ts()
    if not pos.get("opened_ts"):
        pos["opened_ts"] = now
    pos["last_buy_ts"] = now

    STATE["cash_usd"] = cash - usd
    STATE["trades_count"] += 1
    STATE["counters"]["buys"] += 1

    meta2 = dict(meta or {})
    meta2["sol_price_meta"] = px_meta

    log_event(
        "paper_buy_executed",
        symbol="SOL",
        usd=usd,
        px=px,
        qty=qty,
        cash_usd_after=round(float(STATE["cash_usd"]), 6),
        reason="matched_helius_event",
        meta=meta2
    )

    push_recent_trade({
        "side": "BUY",
        "symbol": "SOL",
        "usd": usd,
        "qty": qty,
        "price_usd": px,
        "ts": now,
        "reason": "matched_helius_swap",
        "meta": meta2,
    })

    rebalance_targets()
    return True

# ----------------------------
# FastAPI app
# ----------------------------

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "service": "sol-paper-bot", "ts": _now_ts()}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/paper/state")
def paper_state():
    maybe_run_forced_exits()
    save_state()

    live_px, live_meta = get_live_sol_price()

    cfg = {
        "SOL_PRICE_USD_STATIC_FALLBACK": SOL_PRICE_USD_STATIC_FALLBACK,
        "PRICE_SOURCE": PRICE_SOURCE,
        "PYTH_HERMES_BASE_URL": PYTH_HERMES_BASE_URL,
        "PYTH_SOL_PRICE_ID_SET": bool(PYTH_SOL_PRICE_ID),
        "PRICE_REFRESH_SECONDS": PRICE_REFRESH_SECONDS,
        "PRICE_MAX_STALENESS_SECONDS": PRICE_MAX_STALENESS_SECONDS,
        "LIVE_SOL_PRICE_USD": float(live_px),
        "LIVE_SOL_PRICE_META": live_meta,

        "START_CASH_USD": START_CASH_USD,
        "MAX_BUY_USD": MAX_BUY_USD,
        "MIN_CASH_LEFT_USD": MIN_CASH_LEFT_USD,
        "RESERVE_PCT": RESERVE_PCT,
        "TRADABLE_PCT": TRADABLE_PCT,
        "HOLD_MAX_SECONDS": HOLD_MAX_SECONDS,
        "FORCED_EXIT_FALLBACK_MULTI": FORCED_EXIT_FALLBACK_MULTI,

        "TRACKED_WALLETS_COUNT": len(TRACKED_WALLETS),
        "DEBUG_WEBHOOK": DEBUG_WEBHOOK,
        "WEBHOOK_PATH_TOKEN_SET": bool(WEBHOOK_PATH_TOKEN),
        "WEBHOOK_SECRET_SET": bool(WEBHOOK_SECRET),
        "STATE_PATH": STATE_PATH,
    }

    out = dict(STATE)
    out["config"] = cfg
    return JSONResponse(out)

@app.get("/paper/events")
def paper_events(count: int = 200):
    count = max(1, min(int(count), 2000))
    return {"count": count, "events": EVENTS[-count:]}

@app.get("/events")
def events_alias(count: int = 200):
    return paper_events(count=count)

def _is_authorized(req: Request) -> bool:
    if not WEBHOOK_SECRET:
        STATE["counters"]["skipped_no_secret"] += 1
        return True

    auth = (req.headers.get("authorization") or "").strip()
    x_secret = (req.headers.get("x-webhook-secret") or "").strip()

    if x_secret and x_secret == WEBHOOK_SECRET:
        return True
    if auth == WEBHOOK_SECRET:
        return True
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        return token == WEBHOOK_SECRET
    return False

def _token_matches(path_token: str) -> bool:
    if not WEBHOOK_PATH_TOKEN:
        return True
    a = WEBHOOK_PATH_TOKEN.strip()
    b = (path_token or "").strip()
    if a == b:
        return True
    def norm(x: str) -> str:
        return x[5:] if x.startswith("hook_") else x
    return norm(a) == norm(b)

@app.post("/webhook/{token}")
async def webhook(token: str, request: Request):
    if not _token_matches(token):
        STATE["counters"]["skipped_bad_path"] += 1
        log_event("webhook_bad_path", got=token)
        save_state()
        raise HTTPException(status_code=404, detail="bad path")

    if not _is_authorized(request):
        STATE["counters"]["webhooks_unauthorized"] += 1
        log_event("webhook_unauthorized")
        save_state()
        raise HTTPException(status_code=401, detail="unauthorized")

    STATE["counters"]["webhooks_received"] += 1

    try:
        payload = await request.json()
    except Exception:
        STATE["counters"]["skipped_bad_payload"] += 1
        log_event("webhook_bad_payload")
        save_state()
        raise HTTPException(status_code=400, detail="bad json")

    items: List[Dict[str, Any]] = []
    payload_type = "unknown"
    if isinstance(payload, list):
        payload_type = "list"
        items = [x for x in payload if isinstance(x, dict)]
    elif isinstance(payload, dict):
        payload_type = "dict"
        if isinstance(payload.get("events"), list):
            items = [x for x in payload["events"] if isinstance(x, dict)]
        else:
            items = [payload]
    else:
        STATE["counters"]["skipped_bad_payload"] += 1
        log_event("webhook_bad_payload_type", payload_type=str(type(payload)))
        save_state()
        raise HTTPException(status_code=400, detail="bad payload")

    matched = 0

    # run forced exits first
    maybe_run_forced_exits()

    for item in items:
        if not is_swap_event(item):
            continue

        wallet = find_matched_wallet(item)
        if not wallet:
            continue

        matched += 1

        meta = {
            "wallet": wallet,
            "signature": item.get("signature") or item.get("txSignature") or item.get("transactionSignature"),
            "type": "SWAP",
        }

        side, dbg = _classify_swap_side_for_wallet(item, wallet)
        meta2 = dict(meta)
        meta2["side_detected"] = side
        meta2["side_detect_debug"] = dbg

        if DEBUG_WEBHOOK:
            log_event("debug_webhook_item", note="matched_swap", wallet=wallet, side=side, debug=dbg)

        if side == "sell":
            sold = paper_sell_all_sol(meta=meta2, reason="wallet_sell")
            if not sold and DEBUG_WEBHOOK:
                log_event("debug_wallet_sell_no_position", wallet=wallet, signature=meta.get("signature"))
        elif side == "buy":
            paper_buy_sol(MAX_BUY_USD, meta=meta2)
        else:
            # unknown -> default BUY to keep behavior simple, but log it
            log_event("swap_side_unknown_default_buy", wallet=wallet, signature=meta.get("signature"), debug=dbg)
            paper_buy_sol(MAX_BUY_USD, meta=meta2)

    log_event("webhook_ok", payload_type=payload_type, tracked_wallets_count=len(TRACKED_WALLETS), matched=matched)

    save_state()
    return {"ok": True, "matched": matched}
