import os
import json
import time
import tempfile
from typing import Any, Dict, List, Optional, Tuple

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

# Static fallback (used if live price fails / disabled)
SOL_PRICE_USD = _env_float("SOL_PRICE_USD", 90.0)

START_CASH_USD = _env_float("START_CASH_USD", 800.0)
MAX_BUY_USD = _env_float("MAX_BUY_USD", 25.0)
MIN_CASH_LEFT_USD = _env_float("MIN_CASH_LEFT_USD", 5.0)

# Profit policy targets (equity split):
RESERVE_PCT = _env_float("RESERVE_PCT", 0.40)
TRADABLE_PCT = _env_float("TRADABLE_PCT", 0.60)

# Safety: time-based exit simulation
HOLD_MAX_SECONDS = _env_int("HOLD_MAX_SECONDS", 900)
FORCED_EXIT_FALLBACK_MULTI = _env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)

DEBUG_WEBHOOK = _env_bool("DEBUG_WEBHOOK", False)

# Webhook protection
WEBHOOK_PATH_TOKEN = _env_str("WEBHOOK_PATH_TOKEN", "").strip()
WEBHOOK_SECRET = _env_str("WEBHOOK_SECRET", "").strip()

# Wallet filter
TRACKED_WALLETS_RAW = _env_str("TRACKED_WALLETS", "")
TRACKED_WALLETS = [w.strip() for w in TRACKED_WALLETS_RAW.split(",") if w.strip()]

# ----------------------------
# Live SOL Price via Pyth Hermes (Option B)
# ----------------------------
# You will set these env vars in Render:
#
# PRICE_SOURCE=pyth
# PYTH_SOL_PRICE_ID=<feed id for SOL/USD>
#
# Optional:
# PYTH_HERMES_BASE_URL=https://hermes.pyth.network
# PRICE_REFRESH_SECONDS=5
# PRICE_MAX_STALENESS_SECONDS=60
#
# Notes:
# - We try a couple Hermes JSON endpoints because Hermes has multiple routes depending on version.
# - If it fails, we fall back to SOL_PRICE_USD (static).
#
PRICE_SOURCE = _env_str("PRICE_SOURCE", "static").strip().lower()  # "static" or "pyth"
PYTH_HERMES_BASE_URL = _env_str("PYTH_HERMES_BASE_URL", "https://hermes.pyth.network").strip().rstrip("/")
PYTH_SOL_PRICE_ID = _env_str("PYTH_SOL_PRICE_ID", "").strip()
PRICE_REFRESH_SECONDS = _env_int("PRICE_REFRESH_SECONDS", 5)
PRICE_MAX_STALENESS_SECONDS = _env_int("PRICE_MAX_STALENESS_SECONDS", 60)

# Tiny in-memory cache
_PRICE_CACHE: Dict[str, Any] = {
    "ts_fetched": 0,
    "price": None,        # float
    "conf": None,         # float
    "publish_time": None, # int epoch
    "source": "static",
    "error": None,
}

def _http_get_json(url: str, timeout_s: float = 3.0) -> Any:
    """
    Avoids adding new deps. Uses stdlib urllib.
    """
    import urllib.request
    import urllib.error

    req = urllib.request.Request(
        url,
        headers={
            "accept": "application/json",
            "user-agent": "sol-paper-bot/pyth-hermes",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(str(e))

def _parse_pyth_price_obj(obj: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Hermes responses vary. We try to extract:
      price (float), conf (float), publish_time (int epoch)
    """
    if not isinstance(obj, dict):
        return None, None, None

    # Common nesting patterns:
    # - {"price": {"price": "123.45", "conf": "0.12", "publish_time": 123456}}
    # - {"price": {"price": 12345, "expo": -2, "conf": 12, "publish_time": 123456}}
    price_node = obj.get("price") if isinstance(obj.get("price"), dict) else obj

    if not isinstance(price_node, dict):
        return None, None, None

    publish_time = price_node.get("publish_time") or price_node.get("publishTime") or obj.get("publish_time") or obj.get("publishTime")
    try:
        publish_time = int(publish_time) if publish_time is not None else None
    except Exception:
        publish_time = None

    # If Hermes gives "price" as a string float:
    if "price" in price_node and isinstance(price_node.get("price"), (str, float, int)) and "expo" not in price_node:
        try:
            p = float(price_node.get("price"))
        except Exception:
            p = None
        try:
            c = float(price_node.get("conf")) if price_node.get("conf") is not None else None
        except Exception:
            c = None
        return p, c, publish_time

    # If Hermes gives price as integer + expo (pyth style):
    if "price" in price_node and "expo" in price_node:
        try:
            p_i = float(price_node.get("price"))
            expo = int(price_node.get("expo"))
            p = p_i * (10 ** expo)
        except Exception:
            p = None
        try:
            c_i = float(price_node.get("conf")) if price_node.get("conf") is not None else None
            expo = int(price_node.get("expo"))
            c = (c_i * (10 ** expo)) if c_i is not None else None
        except Exception:
            c = None
        return p, c, publish_time

    return None, None, publish_time

def fetch_pyth_sol_price() -> Tuple[Optional[float], Optional[str]]:
    """
    Tries a few Hermes endpoints that commonly exist.
    Returns (price, error_message)
    """
    if not PYTH_SOL_PRICE_ID:
        return None, "PYTH_SOL_PRICE_ID not set"

    # Try endpoints (Hermes has different routes depending on version/deploy)
    candidates = [
        f"{PYTH_HERMES_BASE_URL}/v2/price_feeds?ids[]={PYTH_SOL_PRICE_ID}",
        f"{PYTH_HERMES_BASE_URL}/v2/price_feeds?ids={PYTH_SOL_PRICE_ID}",
        f"{PYTH_HERMES_BASE_URL}/api/latest_price_feeds?ids[]={PYTH_SOL_PRICE_ID}",
        f"{PYTH_HERMES_BASE_URL}/api/latest_price_feeds?ids={PYTH_SOL_PRICE_ID}",
    ]

    last_err = None
    for url in candidates:
        try:
            data = _http_get_json(url, timeout_s=3.0)

            # Possible shapes:
            # 1) {"price_feeds":[{...}]}
            # 2) [{...}]
            # 3) {"data":[{...}]}
            # 4) {"parsed":[{...}]}
            feeds = None
            if isinstance(data, dict):
                for k in ("price_feeds", "data", "parsed", "result"):
                    if isinstance(data.get(k), list):
                        feeds = data.get(k)
                        break
            elif isinstance(data, list):
                feeds = data

            if not feeds or not isinstance(feeds, list):
                last_err = f"unexpected response shape from {url}"
                continue

            feed0 = feeds[0] if feeds else None
            if not isinstance(feed0, dict):
                last_err = f"unexpected feed item from {url}"
                continue

            # Sometimes the feed object contains id + price object
            # Sometimes it IS the price object
            p, conf, pub = _parse_pyth_price_obj(feed0)
            if p is None:
                # Try nested known key
                if isinstance(feed0.get("price"), dict):
                    p, conf, pub = _parse_pyth_price_obj(feed0.get("price"))
            if p is None:
                last_err = f"could not parse price from {url}"
                continue

            # staleness check
            now = _now_ts()
            if pub is not None and (now - int(pub)) > PRICE_MAX_STALENESS_SECONDS:
                last_err = f"stale price (publish_time too old) from {url}"
                continue

            # Success: update cache fields we can
            _PRICE_CACHE["conf"] = conf
            _PRICE_CACHE["publish_time"] = pub
            return float(p), None

        except Exception as e:
            last_err = f"{url}: {str(e)}"
            continue

    return None, last_err or "unknown error"

def get_sol_price_usd() -> Tuple[float, Dict[str, Any]]:
    """
    Returns (price, meta)
    meta includes source + errors.
    """
    now = _now_ts()

    # If not using live price, return static
    if PRICE_SOURCE != "pyth":
        meta = {
            "source": "static",
            "price": float(SOL_PRICE_USD),
            "conf": None,
            "publish_time": None,
            "error": None,
            "ts_fetched": None,
        }
        return float(SOL_PRICE_USD), meta

    # Cache check
    ts_fetched = int(_PRICE_CACHE.get("ts_fetched") or 0)
    cached_price = _PRICE_CACHE.get("price")
    if cached_price is not None and (now - ts_fetched) < max(1, PRICE_REFRESH_SECONDS):
        meta = {
            "source": _PRICE_CACHE.get("source") or "pyth",
            "price": float(cached_price),
            "conf": _PRICE_CACHE.get("conf"),
            "publish_time": _PRICE_CACHE.get("publish_time"),
            "error": _PRICE_CACHE.get("error"),
            "ts_fetched": ts_fetched,
        }
        return float(cached_price), meta

    # Fetch fresh
    p, err = fetch_pyth_sol_price()
    if p is not None:
        _PRICE_CACHE["ts_fetched"] = now
        _PRICE_CACHE["price"] = float(p)
        _PRICE_CACHE["source"] = "pyth"
        _PRICE_CACHE["error"] = None
        meta = {
            "source": "pyth",
            "price": float(p),
            "conf": _PRICE_CACHE.get("conf"),
            "publish_time": _PRICE_CACHE.get("publish_time"),
            "error": None,
            "ts_fetched": now,
        }
        return float(p), meta

    # If Pyth fails, fall back to static
    _PRICE_CACHE["ts_fetched"] = now
    _PRICE_CACHE["price"] = float(SOL_PRICE_USD)
    _PRICE_CACHE["source"] = "static_fallback"
    _PRICE_CACHE["error"] = err

    meta = {
        "source": "static_fallback",
        "price": float(SOL_PRICE_USD),
        "conf": None,
        "publish_time": None,
        "error": err,
        "ts_fetched": now,
    }
    return float(SOL_PRICE_USD), meta

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
        "positions": {},   # symbol -> dict
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
            "rebalances": 0,
        },
        "started_at": _now_ts(),
        "recent_trades": [],  # append small records
    }

    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # merge keys conservatively
            for k, v in base.items():
                if k not in data:
                    data[k] = v
            # counters merge
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

    live_px, _meta = get_sol_price_usd()

    for sym, p in positions.items():
        qty = float(p.get("qty", 0.0) or 0.0)
        if qty <= 0:
            continue
        px = float(p.get("mark_px", 0.0) or 0.0)
        if px <= 0:
            px = live_px if sym == "SOL" else float(p.get("avg_px", 0.0) or 0.0)
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
# Selling logic (simple time-based forced exit)
# ----------------------------

def maybe_run_forced_exits() -> None:
    now = _now_ts()
    positions = STATE.get("positions", {}) or {}

    live_px, _meta = get_sol_price_usd()

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

        avg_px = float(p.get("avg_px", 0.0) or 0.0)
        if avg_px <= 0:
            continue

        sell_px = avg_px * float(FORCED_EXIT_FALLBACK_MULTI)

        # keep mark_px fresh-ish
        if sym == "SOL":
            p["mark_px"] = float(live_px)

        proceeds = qty * sell_px

        STATE["cash_usd"] = float(STATE.get("cash_usd", 0.0)) + proceeds
        p["qty"] = 0.0
        p["closed_ts"] = now
        p["last_sell_ts"] = now

        STATE["trades_count"] += 1
        STATE["counters"]["sells"] += 1
        STATE["counters"]["forced_exits"] += 1

        log_event("paper_sell_forced_exit", symbol=sym, qty=qty, px=sell_px, proceeds_usd=round(proceeds, 6), age_s=age)
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

    px, px_meta = get_sol_price_usd()
    qty = usd / px if px > 0 else 0.0

    pos = STATE.setdefault("positions", {}).setdefault("SOL", {
        "symbol": "SOL",
        "qty": 0.0,
        "cost_usd": 0.0,
        "avg_px": 0.0,
        "opened_ts": None,
        "last_buy_ts": None,
        "mark_px": px,
    })

    prev_qty = float(pos.get("qty", 0.0) or 0.0)
    prev_cost = float(pos.get("cost_usd", 0.0) or 0.0)

    new_cost = prev_cost + usd
    new_qty = prev_qty + qty
    new_avg = (new_cost / new_qty) if new_qty > 0 else 0.0

    pos["qty"] = new_qty
    pos["cost_usd"] = new_cost
    pos["avg_px"] = new_avg
    pos["mark_px"] = float(px)
    now = _now_ts()
    if not pos.get("opened_ts"):
        pos["opened_ts"] = now
    pos["last_buy_ts"] = now

    STATE["cash_usd"] = cash - usd
    STATE["trades_count"] += 1
    STATE["counters"]["buys"] += 1

    # include price source info in logs so you can confirm it's actually live
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

    live_px, live_meta = get_sol_price_usd()

    cfg = {
        "SOL_PRICE_USD_STATIC_FALLBACK": SOL_PRICE_USD,
        "PRICE_SOURCE": PRICE_SOURCE,
        "PYTH_HERMES_BASE_URL": PYTH_HERMES_BASE_URL,
        "PYTH_SOL_PRICE_ID_SET": bool(PYTH_SOL_PRICE_ID),
        "PRICE_REFRESH_SECONDS": PRICE_REFRESH_SECONDS,
        "PRICE_MAX_STALENESS_SECONDS": PRICE_MAX_STALENESS_SECONDS,
        "LIVE_SOL_PRICE_USD": live_px,
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

    maybe_run_forced_exits()

    for item in items:
        if not is_swap_event(item):
            continue

        wallet = find_matched_wallet(item)
        if not wallet:
            continue

        matched += 1

        if DEBUG_WEBHOOK:
            log_event("debug_webhook_item", note="matched_swap", wallet=wallet)

        meta = {
            "wallet": wallet,
            "signature": item.get("signature") or item.get("txSignature") or item.get("transactionSignature"),
            "type": "SWAP",
        }

        paper_buy_sol(MAX_BUY_USD, meta=meta)

    log_event("webhook_ok", payload_type=payload_type, tracked_wallets_count=len(TRACKED_WALLETS), matched=matched)
    save_state()
    return {"ok": True, "matched": matched}
