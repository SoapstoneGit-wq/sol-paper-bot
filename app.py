import os
import json
import time
from collections import deque
from typing import Dict, Any, List, Optional, Tuple

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# -----------------------------
# Config (easy to tune)
# -----------------------------
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()

# Supports:
# - TRACKED_WALLET="..."
# - TRACKED_WALLETS="w1,w2,w3"
_single = os.getenv("TRACKED_WALLET", "").strip()
_multi = os.getenv("TRACKED_WALLETS", "").strip()
TRACKED_WALLETS = set()

if _single:
    TRACKED_WALLETS.add(_single)
if _multi:
    for w in _multi.split(","):
        w = w.strip()
        if w:
            TRACKED_WALLETS.add(w)

# Paper settings
START_CASH_USD = float(os.getenv("START_CASH_USD", "500"))
BUY_USD_PER_TRADE = float(os.getenv("BUY_USD_PER_TRADE", "100"))
FEE_USD_EST = float(os.getenv("FEE_USD_EST", "3.0"))

# Risk / exits (easy knobs)
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.20"))          # 0.20 = -20%
TRAIL_START_PCT = float(os.getenv("TRAIL_START_PCT", "0.30"))       # start trailing once +30% from avg cost
TRAIL_GAP_PCT = float(os.getenv("TRAIL_GAP_PCT", "0.15"))           # trail by 15% from peak
SELL_ON_WHALE_SELL = os.getenv("SELL_ON_WHALE_SELL", "true").lower() == "true"
SELL_FRACTION_ON_WHALE_SELL = float(os.getenv("SELL_FRACTION_ON_WHALE_SELL", "1.0"))  # 1.0 = sell all
MIN_USD_TO_TRADE = float(os.getenv("MIN_USD_TO_TRADE", "25"))

# Price source (Jupiter)
JUP_PRICE_URL = os.getenv("JUP_PRICE_URL", "https://price.jup.ag/v6/price")
PRICE_CACHE_TTL_SEC = int(os.getenv("PRICE_CACHE_TTL_SEC", "20"))

# State persistence
STATE_PATH = os.getenv("STATE_PATH", "/tmp/paper_state.json")

# Store last N raw webhook events (for debugging)
EVENTS_MAX = int(os.getenv("EVENTS_MAX", "300"))
EVENTS = deque(maxlen=EVENTS_MAX)


# -----------------------------
# Helpers: state load/save
# -----------------------------
def default_state() -> Dict[str, Any]:
    return {
        "cash": START_CASH_USD,
        "positions": {},  # mint -> position
        "trades": [],     # list of trade records
        "realized_pnl_usd": 0.0,
        "unrealized_pnl_usd": 0.0,
        "equity_usd": START_CASH_USD,
        "positions_value_usd": 0.0,
        "updated_at": time.time(),
    }


def load_state() -> Dict[str, Any]:
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return default_state()


def save_state(state: Dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f)
    except Exception:
        # Render /tmp should work; if not, we still keep in-memory behavior via reload each request.
        pass


# -----------------------------
# Helpers: webhook auth (ROBUST FIX)
# -----------------------------
def is_authorized(req) -> bool:
    """
    Accepts webhook if ANY of these match WEBHOOK_SECRET:
    - header x-webhook-secret == secret
    - header authorization == secret (or Bearer secret)
    - ANY header value equals secret (fallback for weird provider formatting)
    """
    if not WEBHOOK_SECRET:
        # If user forgot to set secret, don't accidentally open the endpoint.
        return False

    xws = req.headers.get("x-webhook-secret", "").strip()
    if xws == WEBHOOK_SECRET:
        return True

    auth = req.headers.get("authorization", "").strip()
    if auth == WEBHOOK_SECRET:
        return True
    if auth.lower().startswith("bearer "):
        if auth.split(" ", 1)[1].strip() == WEBHOOK_SECRET:
            return True

    # Fallback: sometimes people paste "x-webhook-secret: value" into provider UI.
    # Providers then send it as a single header line or odd mapping; this catches it.
    for k, v in req.headers.items():
        if isinstance(v, str) and WEBHOOK_SECRET in v:
            return True

    return False


# -----------------------------
# Helpers: price fetching
# -----------------------------
_price_cache: Dict[str, Tuple[float, float]] = {}  # mint -> (ts, price)

def get_price_usd(mint: str) -> Optional[float]:
    now = time.time()
    cached = _price_cache.get(mint)
    if cached and (now - cached[0]) <= PRICE_CACHE_TTL_SEC:
        return cached[1]

    try:
        r = requests.get(JUP_PRICE_URL, params={"ids": mint}, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        px = data.get("data", {}).get(mint, {}).get("price")
        if px is None:
            return None
        px = float(px)
        _price_cache[mint] = (now, px)
        return px
    except Exception:
        return None


# -----------------------------
# Parsing: detect "buy" or "sell" from Helius enhanced payload
# -----------------------------
STABLE_MINTS = {
    "So11111111111111111111111111111111111111112",  # SOL (wrapped)
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
}

def extract_wallets_in_payload(payload: Any) -> set:
    wallets = set()
    try:
        if isinstance(payload, list):
            for item in payload:
                w = item.get("user") or item.get("feePayer") or item.get("account")
                if isinstance(w, str):
                    wallets.add(w)
                # Sometimes nested
                fp = item.get("feePayer")
                if isinstance(fp, str):
                    wallets.add(fp)
        elif isinstance(payload, dict):
            w = payload.get("user") or payload.get("feePayer") or payload.get("account")
            if isinstance(w, str):
                wallets.add(w)
    except Exception:
        pass
    return wallets

def detect_trade_signal(payload: Any) -> Optional[Dict[str, Any]]:
    """
    Returns a signal dict like:
      {"action": "BUY"|"SELL", "mint": "...", "wallet": "...", "signature": "..."}
    If we can't confidently detect, return None.
    """
    # Helius enhanced webhooks are usually a list of tx objects
    if not isinstance(payload, list):
        return None

    for tx in payload:
        if not isinstance(tx, dict):
            continue

        # Only care about SWAP-like events
        tx_type = (tx.get("type") or "").upper()
        if tx_type not in {"SWAP"}:
            continue

        fee_payer = tx.get("feePayer") or tx.get("user") or ""
        sig = tx.get("signature") or tx.get("signatureHash") or tx.get("transactionHash") or ""

        # tokenBalanceChanges typically include per-userAccount mint deltas.
        # We'll infer the "target mint" as the non-stable mint with the biggest abs change for the tracked wallet.
        tbc = tx.get("tokenBalanceChanges") or []
        if not isinstance(tbc, list):
            continue

        # Gather net deltas by mint for this wallet (if userAccount is present)
        deltas: Dict[str, float] = {}
        for c in tbc:
            if not isinstance(c, dict):
                continue
            mint = c.get("mint")
            user_acct = c.get("userAccount")
            raw = c.get("rawTokenAmount") or {}
            # Helius uses tokenAmount sometimes
            amt = None
            if isinstance(raw, dict):
                amt = raw.get("tokenAmount")
            if amt is None:
                amt = c.get("tokenAmount")
            if mint and isinstance(user_acct, str) and isinstance(amt, (int, float)):
                # Only use if it's our tracked wallet
                if TRACKED_WALLETS and user_acct not in TRACKED_WALLETS and fee_payer not in TRACKED_WALLETS:
                    continue
                deltas[mint] = deltas.get(mint, 0.0) + float(amt)

        if not deltas:
            continue

        # pick a non-stable mint with largest abs delta
        best_mint = None
        best_abs = 0.0
        best_delta = 0.0
        for m, d in deltas.items():
            if m in STABLE_MINTS:
                continue
            if abs(d) > best_abs:
                best_abs = abs(d)
                best_mint = m
                best_delta = d

        if not best_mint:
            continue

        action = "BUY" if best_delta > 0 else "SELL"
        return {
            "action": action,
            "mint": best_mint,
            "wallet": fee_payer,
            "signature": sig,
        }

    return None


# -----------------------------
# Paper engine
# -----------------------------
def ensure_position(state: Dict[str, Any], mint: str) -> Dict[str, Any]:
    pos = state["positions"].get(mint)
    if not pos:
        pos = {
            "qty": 0.0,
            "avg_cost_usd_per_token": 0.0,
            "last_price_usd": None,
            "peak_price_usd": None,
            "trail_stop_price_usd": None,
            "tp_done": False,
        }
        state["positions"][mint] = pos
    return pos


def buy_paper(state: Dict[str, Any], mint: str, usd: float, sig: str = "", desc: str = "") -> Optional[Dict[str, Any]]:
    if usd < MIN_USD_TO_TRADE:
        return None
    if state["cash"] < usd + FEE_USD_EST:
        return None

    px = get_price_usd(mint)
    if px is None or px <= 0:
        return None

    qty = usd / px
    pos = ensure_position(state, mint)

    # update avg cost
    new_cost = (pos["qty"] * pos["avg_cost_usd_per_token"]) + usd
    new_qty = pos["qty"] + qty
    pos["qty"] = new_qty
    pos["avg_cost_usd_per_token"] = (new_cost / new_qty) if new_qty > 0 else 0.0

    pos["last_price_usd"] = px
    if pos["peak_price_usd"] is None or px > pos["peak_price_usd"]:
        pos["peak_price_usd"] = px

    state["cash"] -= (usd + FEE_USD_EST)

    trade = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "type": "SWAP",
        "side": "BUY",
        "mint": mint,
        "qty": qty,
        "price_usd_per_token": px,
        "trade_usd_est": usd,
        "fees_usd_est": FEE_USD_EST,
        "sig": sig,
        "desc": desc,
    }
    state["trades"].append(trade)
    return trade


def sell_paper(state: Dict[str, Any], mint: str, fraction: float = 1.0, sig: str = "", reason: str = "") -> Optional[Dict[str, Any]]:
    fraction = max(0.0, min(1.0, fraction))
    pos = state["positions"].get(mint)
    if not pos or pos["qty"] <= 0:
        return None

    px = get_price_usd(mint)
    if px is None or px <= 0:
        return None

    qty_sell = pos["qty"] * fraction
    if qty_sell <= 0:
        return None

    proceeds = qty_sell * px
    cost_basis = qty_sell * pos["avg_cost_usd_per_token"]
    realized = proceeds - cost_basis - FEE_USD_EST

    pos["qty"] -= qty_sell
    if pos["qty"] <= 1e-12:
        # clear position
        del state["positions"][mint]
    else:
        pos["last_price_usd"] = px

    state["cash"] += (proceeds - FEE_USD_EST)
    state["realized_pnl_usd"] += realized

    trade = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "type": "SWAP",
        "side": "SELL",
        "mint": mint,
        "qty": qty_sell,
        "price_usd_per_token": px,
        "trade_usd_est": proceeds,
        "fees_usd_est": FEE_USD_EST,
        "realized_pnl_net": realized,
        "sig": sig,
        "reason": reason,
    }
    state["trades"].append(trade)
    return trade


def apply_risk_checks(state: Dict[str, Any]) -> None:
    """
    Stop loss + trailing stop.
    """
    to_sell: List[Tuple[str, str]] = []

    for mint, pos in list(state["positions"].items()):
        if pos["qty"] <= 0:
            continue

        px = get_price_usd(mint)
        if px is None or px <= 0:
            continue

        pos["last_price_usd"] = px

        avg = float(pos["avg_cost_usd_per_token"] or 0.0)
        if avg <= 0:
            continue

        # STOP LOSS
        stop_price = avg * (1.0 - STOP_LOSS_PCT)
        if px <= stop_price:
            to_sell.append((mint, "STOP_LOSS"))
            continue

        # TRAILING STOP activates after price is up enough
        trail_start = avg * (1.0 + TRAIL_START_PCT)
        if px >= trail_start:
            peak = pos["peak_price_usd"] or px
            if px > peak:
                peak = px
            pos["peak_price_usd"] = peak
            trail_stop = peak * (1.0 - TRAIL_GAP_PCT)
            pos["trail_stop_price_usd"] = trail_stop

            if px <= trail_stop:
                to_sell.append((mint, "TRAILING_STOP"))

    for mint, reason in to_sell:
        sell_paper(state, mint, fraction=1.0, reason=reason)


def refresh_equity(state: Dict[str, Any]) -> None:
    pos_value = 0.0
    unreal = 0.0

    for mint, pos in state["positions"].items():
        if pos["qty"] <= 0:
            continue
        px = get_price_usd(mint)
        if px is None:
            continue
        pos["last_price_usd"] = px
        pos_value += pos["qty"] * px
        unreal += pos["qty"] * (px - pos["avg_cost_usd_per_token"])

    state["positions_value_usd"] = pos_value
    state["unrealized_pnl_usd"] = unreal
    state["equity_usd"] = state["cash"] + pos_value


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "sol-paper-bot", "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})


@app.post("/webhook")
def webhook():
    if not is_authorized(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401

    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"ok": False, "error": "invalid json"}), 400

    # Keep raw event
    EVENTS.appendleft({
        "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "helius",
        "payload": payload,
    })

    state = load_state()

    # Detect buy/sell from whale
    sig = ""
    signal = detect_trade_signal(payload)
    if signal:
        sig = signal.get("signature", "") or ""
        mint = signal["mint"]
        action = signal["action"]

        if action == "BUY":
            buy_paper(state, mint, BUY_USD_PER_TRADE, sig=sig)
        elif action == "SELL" and SELL_ON_WHALE_SELL:
            sell_paper(state, mint, fraction=SELL_FRACTION_ON_WHALE_SELL, sig=sig, reason="WHALE_SELL")

    # Always run risk checks after ingest
    apply_risk_checks(state)
    refresh_equity(state)
    save_state(state)

    return jsonify({"ok": True, "ingested": True, "signal": signal}), 200


@app.get("/events")
def events():
    return jsonify({"count": len(EVENTS), "events": list(EVENTS)}), 200


@app.get("/paper/state")
def paper_state():
    state = load_state()
    refresh_equity(state)
    save_state(state)
    return jsonify(state), 200


@app.post("/paper/reset")
def paper_reset():
    # OPTIONAL: lets you reset state without redeploy
    if not is_authorized(request):
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    state = default_state()
    save_state(state)
    return jsonify({"ok": True, "state": state}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
