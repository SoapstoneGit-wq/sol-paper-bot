import os
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="sol-paper-bot")

# ----------------------------
# Env helpers
# ----------------------------
def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else v

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return float(default)
    try:
        return float(v)
    except ValueError:
        return float(default)

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return int(default)
    try:
        return int(v)
    except ValueError:
        return int(default)

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def parse_wallets(csv: str) -> List[str]:
    # Accept comma-separated list, trim whitespace, ignore empties
    parts = [p.strip() for p in csv.split(",")]
    return [p for p in parts if p]

# ----------------------------
# Config
# ----------------------------
WEBHOOK_SECRET = env_str("WEBHOOK_SECRET", "").strip()
TRACKED_WALLETS = set(parse_wallets(env_str("TRACKED_WALLET", "")))

# If you don’t care about USD estimates, you can leave this as-is.
SOL_PRICE_USD = env_float("SOL_PRICE_USD", 115.0)

START_CASH_USD = env_float("START_CASH_USD", 500.0)
MAX_BUY_USD = env_float("MAX_BUY_USD", 25.0)

# Always leave at least this much in TRADABLE cash (so you can keep room / avoid zeroing out)
MIN_CASH_LEFT_USD = env_float("MIN_CASH_LEFT_USD", 100.0)

# 60 reserve / 40 tradable
RESERVE_PCT = env_float("RESERVE_PCT", 0.60)
TRADABLE_PCT = env_float("TRADABLE_PCT", 0.40)

# Forced exit timer
HOLD_MAX_SECONDS = env_int("HOLD_MAX_SECONDS", 900)  # 900 seconds = 15 minutes

# If we force-exit without a real market price, we simulate an exit at this multiplier of entry price.
FORCED_EXIT_FALLBACK_MULTI = env_float("FORCED_EXIT_FALLBACK_MULTI", 0.50)

# 🔧 Drop-in patch switch (safe debug)
DEBUG_WEBHOOK = env_bool("DEBUG_WEBHOOK", False)

# ----------------------------
# In-memory paper state
# ----------------------------
def now_ts() -> int:
    return int(time.time())

STATE: Dict[str, Any] = {
    "started_at": now_ts(),
    "cash_usd": round(START_CASH_USD * TRADABLE_PCT, 2),
    "reserve_cash_usd": round(START_CASH_USD * RESERVE_PCT, 2),
    "positions": {},  # mint -> position dict
    "trades_count": 0,
    "events": [],  # recent events summary
    "counters": {
        "webhooks_received": 0,
        "webhooks_unauthorized": 0,
        "skipped_no_secret": 0,
        "skipped_bad_payload": 0,
        "buys": 0,
        "sells": 0,
        "forced_exits": 0,
        "skipped_low_cash": 0,
    },
}

def config_snapshot() -> Dict[str, Any]:
    return {
        "SOL_PRICE_USD": SOL_PRICE_USD,
        "START_CASH_USD": START_CASH_USD,
        "MAX_BUY_USD": MAX_BUY_USD,
        "MIN_CASH_LEFT_USD": MIN_CASH_LEFT_USD,
        "RESERVE_PCT": RESERVE_PCT,
        "TRADABLE_PCT": TRADABLE_PCT,
        "HOLD_MAX_SECONDS": HOLD_MAX_SECONDS,
        "FORCED_EXIT_FALLBACK_MULTI": FORCED_EXIT_FALLBACK_MULTI,
        "TRACKED_WALLETS_COUNT": len(TRACKED_WALLETS),
        "DEBUG_WEBHOOK": DEBUG_WEBHOOK,
    }

def add_event(kind: str, data: Dict[str, Any]) -> None:
    # Keep a small rolling window
    e = {"ts": now_ts(), "kind": kind, **data}
    STATE["events"].append(e)
    if len(STATE["events"]) > 200:
        STATE["events"] = STATE["events"][-200:]

def scrub(s: str) -> str:
    # Never print full secrets; keep only safe info
    s = s or ""
    if len(s) <= 4:
        return "*" * len(s)
    return s[:2] + "*" * (len(s) - 4) + s[-2:]

def get_header_secret(req: Request) -> Optional[str]:
    # Header name is case-insensitive in HTTP
    # We accept both x-webhook-secret and x-webhook_secret just in case.
    v = req.headers.get("x-webhook-secret")
    if v is None:
        v = req.headers.get("x-webhook_secret")
    if v is None:
        return None
    return v.strip()

def is_authorized(req: Request) -> Tuple[bool, str]:
    if not WEBHOOK_SECRET:
        return False, "server_missing_WEBHOOK_SECRET"
    got = get_header_secret(req)
    if got is None:
        return False, "missing_header"
    if got != WEBHOOK_SECRET:
        return False, "mismatch"
    return True, "ok"

# ----------------------------
# Parsing Helius payload -> simple buy/sell signals
# ----------------------------
def extract_payload_list(body: Any) -> Optional[List[Dict[str, Any]]]:
    # Helius typically sends {"payload":[{...},{...}]} in your examples.
    if isinstance(body, dict):
        payload = body.get("payload")
        if isinstance(payload, list):
            return [p for p in payload if isinstance(p, dict)]
    return None

def tx_involves_tracked_wallet(tx: Dict[str, Any]) -> bool:
    # Fast heuristic: feePayer or any accountData.account matches tracked
    fee_payer = tx.get("feePayer")
    if isinstance(fee_payer, str) and fee_payer in TRACKED_WALLETS:
        return True

    acct_data = tx.get("accountData")
    if isinstance(acct_data, list):
        for row in acct_data:
            if isinstance(row, dict):
                acct = row.get("account")
                if isinstance(acct, str) and acct in TRACKED_WALLETS:
                    return True
    return False

def estimate_usd_spent_by_tracked(tx: Dict[str, Any]) -> Optional[float]:
    # Look for nativeBalanceChange on a tracked account (negative = spent)
    acct_data = tx.get("accountData")
    if not isinstance(acct_data, list):
        return None

    spent_lamports = 0
    for row in acct_data:
        if not isinstance(row, dict):
            continue
        acct = row.get("account")
        if acct in TRACKED_WALLETS:
            nbc = row.get("nativeBalanceChange")
            if isinstance(nbc, int) and nbc < 0:
                spent_lamports += (-nbc)

    if spent_lamports <= 0:
        return None

    spent_sol = spent_lamports / 1_000_000_000
    return float(spent_sol * SOL_PRICE_USD)

def find_token_transfer_for_tracked(tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Prefer tokenTransfers entries that involve tracked wallet as toUserAccount or fromUserAccount
    transfers = tx.get("tokenTransfers")
    if not isinstance(transfers, list):
        return None

    for t in transfers:
        if not isinstance(t, dict):
            continue
        to_user = t.get("toUserAccount")
        from_user = t.get("fromUserAccount")
        if to_user in TRACKED_WALLETS or from_user in TRACKED_WALLETS:
            return t
    return None

def classify_trade(tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Return a dict with {"side": "buy"/"sell", "mint": str, "qty": float, "est_price": float}
    t = find_token_transfer_for_tracked(tx)
    if not t:
        return None

    mint = t.get("mint")
    qty = t.get("tokenAmount")
    to_user = t.get("toUserAccount")
    from_user = t.get("fromUserAccount")

    if not isinstance(mint, str) or not mint:
        return None
    if not isinstance(qty, (int, float)) or qty == 0:
        return None

    side: Optional[str] = None
    if to_user in TRACKED_WALLETS and from_user not in TRACKED_WALLETS:
        side = "buy"
    elif from_user in TRACKED_WALLETS and to_user not in TRACKED_WALLETS:
        side = "sell"
    else:
        # ambiguous
        return None

    usd_spent = estimate_usd_spent_by_tracked(tx)
    est_price = None
    if usd_spent is not None and qty > 0 and side == "buy":
        est_price = usd_spent / float(qty)

    return {
        "side": side,
        "mint": mint,
        "qty": float(qty),
        "est_entry_price": float(est_price) if est_price is not None else None,
        "sig": tx.get("signature"),
        "source": tx.get("source"),
        "type": tx.get("type"),
        "timestamp": tx.get("timestamp"),
    }

# ----------------------------
# Paper engine
# ----------------------------
def tradable_cash() -> float:
    return float(STATE["cash_usd"])

def reserve_cash() -> float:
    return float(STATE["reserve_cash_usd"])

def can_buy() -> bool:
    return tradable_cash() > MIN_CASH_LEFT_USD + 0.01

def paper_buy(mint: str, entry_price: Optional[float]) -> None:
    if not can_buy():
        STATE["counters"]["skipped_low_cash"] += 1
        add_event("skip_buy_low_cash", {"mint": mint, "cash_usd": tradable_cash()})
        return

    budget = min(MAX_BUY_USD, max(0.0, tradable_cash() - MIN_CASH_LEFT_USD))
    if budget <= 0:
        STATE["counters"]["skipped_low_cash"] += 1
        add_event("skip_buy_low_cash", {"mint": mint, "cash_usd": tradable_cash()})
        return

    # If we don't have an entry price estimate, treat as unknown and just store cost.
    pos = STATE["positions"].get(mint)
    if pos is None:
        pos = {
            "mint": mint,
            "opened_at": now_ts(),
            "cost_usd": 0.0,
            "entry_price": entry_price,  # may be None
        }
        STATE["positions"][mint] = pos

    pos["cost_usd"] = round(float(pos["cost_usd"]) + budget, 6)

    STATE["cash_usd"] = round(tradable_cash() - budget, 6)
    STATE["trades_count"] += 1
    STATE["counters"]["buys"] += 1

    add_event("buy", {
        "mint": mint,
        "budget_usd": round(budget, 4),
        "cash_usd": round(tradable_cash(), 4),
        "reserve_cash_usd": round(reserve_cash(), 4),
        "entry_price": entry_price,
    })

def paper_sell(mint: str, exit_price: Optional[float], forced: bool = False) -> None:
    pos = STATE["positions"].get(mint)
    if not pos:
        return

    cost = float(pos.get("cost_usd", 0.0))
    if cost <= 0:
        STATE["positions"].pop(mint, None)
        return

    # If we have an exit price AND entry price, simulate P&L. Otherwise just return cost (flat).
    entry_price = pos.get("entry_price")
    proceeds = cost

    if exit_price is not None and entry_price is not None and entry_price > 0:
        # Simulate proceeds proportionally
        proceeds = cost * (float(exit_price) / float(entry_price))

    proceeds = float(proceeds)
    profit = proceeds - cost

    # Split proceeds back into tradable/reserve by configured ratio.
    tradable_add = proceeds * TRADABLE_PCT
    reserve_add = proceeds * RESERVE_PCT

    STATE["cash_usd"] = round(tradable_cash() + tradable_add, 6)
    STATE["reserve_cash_usd"] = round(reserve_cash() + reserve_add, 6)

    STATE["positions"].pop(mint, None)
    STATE["trades_count"] += 1
    STATE["counters"]["sells"] += 1
    if forced:
        STATE["counters"]["forced_exits"] += 1

    add_event("sell_forced" if forced else "sell", {
        "mint": mint,
        "cost_usd": round(cost, 4),
        "proceeds_usd": round(proceeds, 4),
        "profit_usd": round(profit, 4),
        "cash_usd": round(tradable_cash(), 4),
        "reserve_cash_usd": round(reserve_cash(), 4),
        "exit_price": exit_price,
        "entry_price": entry_price,
    })

def run_forced_exits() -> None:
    ts = now_ts()
    to_close = []
    for mint, pos in list(STATE["positions"].items()):
        opened = int(pos.get("opened_at", ts))
        if ts - opened >= HOLD_MAX_SECONDS:
            to_close.append(mint)

    for mint in to_close:
        pos = STATE["positions"].get(mint)
        if not pos:
            continue
        entry = pos.get("entry_price")
        exit_price = None
        if entry is not None:
            exit_price = float(entry) * float(FORCED_EXIT_FALLBACK_MULTI)
        paper_sell(mint, exit_price=exit_price, forced=True)

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "sol-paper-bot"}

@app.get("/events")
def events() -> Dict[str, Any]:
    run_forced_exits()
    return {"count": len(STATE["events"]), "events": STATE["events"]}

@app.get("/paper/state")
def paper_state() -> Dict[str, Any]:
    run_forced_exits()
    return {
        "cash_usd": round(tradable_cash(), 4),
        "reserve_cash_usd": round(reserve_cash(), 4),
        "positions": STATE["positions"],
        "trades_count": STATE["trades_count"],
        "counters": STATE["counters"],
        "started_at": STATE["started_at"],
        "config": config_snapshot(),
    }

@app.post("/webhook")
async def webhook(req: Request) -> JSONResponse:
    STATE["counters"]["webhooks_received"] += 1

    ok, reason = is_authorized(req)
    if not ok:
        STATE["counters"]["webhooks_unauthorized"] += 1

        # ---- DROP-IN PATCH (SAFE DEBUG) ----
        if DEBUG_WEBHOOK:
            got = get_header_secret(req)
            add_event("webhook_unauthorized_debug", {
                "reason": reason,
                "header_present": got is not None,
                "header_len": (len(got) if got is not None else 0),
                "server_secret_len": len(WEBHOOK_SECRET),
                "server_secret_masked": scrub(WEBHOOK_SECRET),
                "got_masked": scrub(got) if got else None,
                "headers_keys_sample": sorted(list(req.headers.keys()))[:20],
            })
        # -----------------------------------

        if reason == "missing_header":
            STATE["counters"]["skipped_no_secret"] += 1

        return JSONResponse(status_code=401, content={"ok": False, "error": "unauthorized", "reason": reason})

    try:
        body = await req.json()
    except Exception:
        STATE["counters"]["skipped_bad_payload"] += 1
        return JSONResponse(status_code=400, content={"ok": False, "error": "bad_json"})

    payload_list = extract_payload_list(body)
    if payload_list is None:
        # Accept also direct list payloads if someone forwards it differently
        if isinstance(body, list):
            payload_list = [b for b in body if isinstance(b, dict)]
        else:
            STATE["counters"]["skipped_bad_payload"] += 1
            add_event("skip_bad_payload", {"type": str(type(body))})
            return JSONResponse(status_code=400, content={"ok": False, "error": "bad_payload_shape"})

    processed = 0

    for tx in payload_list:
        if not tx_involves_tracked_wallet(tx):
            continue

        trade = classify_trade(tx)
        if not trade:
            continue

        mint = trade["mint"]
        side = trade["side"]
        entry_price = trade.get("est_entry_price")

        if side == "buy":
            paper_buy(mint=mint, entry_price=entry_price)
            processed += 1
        elif side == "sell":
            # For sells, if we have the position, close it at a neutral price (or you can enhance later)
            paper_sell(mint=mint, exit_price=None, forced=False)
            processed += 1

        add_event("tx_seen", {
            "side": side,
            "mint": mint,
            "sig": trade.get("sig"),
            "source": trade.get("source"),
            "type": trade.get("type"),
            "timestamp": trade.get("timestamp"),
            "entry_price": entry_price,
        })

    run_forced_exits()

    return JSONResponse(status_code=200, content={"ok": True, "processed": processed})
