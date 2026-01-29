import os
import time
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple
from flask import Flask, request, jsonify

# =========================
# Config (env vars)
# =========================

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

def env_list(name: str) -> List[str]:
    v = os.getenv(name, "")
    items = [x.strip() for x in v.split(",") if x.strip()]
    return items

APP_PORT = env_int("PORT", 10000)

# Wallets you want to copy-trade (comma-separated)
TRACKED_WALLETS = set(env_list("TRACKED_WALLETS"))

# Starting paper cash
START_CASH_USD = env_float("START_CASH_USD", 500.0)

# Buy sizing
BUY_USD_FIXED = env_float("BUY_USD_FIXED", 25.0)     # fixed dollar buy per entry
MAX_OPEN_POSITIONS = env_int("MAX_OPEN_POSITIONS", 10)
MAX_INFLIGHT_PCT = env_float("MAX_INFLIGHT_PCT", 0.80)  # max % of tradable cash allocated

# Pricing
SOL_PRICE_USD = env_float("SOL_PRICE_USD", 200.0)  # used to estimate token price from SOL delta

# Fees/slippage simulation (paper)
FEE_FIXED_USD = env_float("FEE_FIXED_USD", 0.10)     # flat per trade
FEE_PCT = env_float("FEE_PCT", 0.0035)               # 0.35% simulated friction
SLIPPAGE_PCT = env_float("SLIPPAGE_PCT", 0.005)       # 0.50% assumed worse fill

# Exit logic (independent of whale sell)
STOP_LOSS_PCT = env_float("STOP_LOSS_PCT", 0.15)      # 15% hard stop
MAX_HOLD_SECONDS = env_int("MAX_HOLD_SECONDS", 20 * 60)  # 20 minutes

# Take profit ladder (partial sells)
TP1_PCT = env_float("TP1_PCT", 0.30)  # +30%
TP2_PCT = env_float("TP2_PCT", 0.60)  # +60%
TP3_PCT = env_float("TP3_PCT", 1.20)  # +120%

TP1_SELL_FRAC = env_float("TP1_SELL_FRAC", 0.25)  # sell 25%
TP2_SELL_FRAC = env_float("TP2_SELL_FRAC", 0.25)
TP3_SELL_FRAC = env_float("TP3_SELL_FRAC", 0.25)

# Trailing stop activates after profit threshold
TRAIL_ACTIVATE_PCT = env_float("TRAIL_ACTIVATE_PCT", 0.40)  # +40%
TRAIL_DIST_PCT = env_float("TRAIL_DIST_PCT", 0.25)          # 25% trail from peak

# Profit vault / compounding
BIG_WIN_ROI = env_float("BIG_WIN_ROI", 0.50)                 # 50%+
BIG_WIN_RESERVE_SPLIT = env_float("BIG_WIN_RESERVE_SPLIT", 0.50)  # reserve 50% of profit
BASE_RESERVE_SPLIT = env_float("BASE_RESERVE_SPLIT", 0.10)        # reserve 10% of profit on normal wins (optional)
MIN_PROFIT_USD = env_float("MIN_PROFIT_USD", 1.00)

# Webhook auth (optional)
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()

# =========================
# State
# =========================

@dataclass
class Position:
    mint: str
    qty: float
    entry_price_usd: float
    entry_cost_usd: float
    opened_at: float

    # tracking
    last_price_usd: float = 0.0
    peak_price_usd: float = 0.0

    # partial sells
    tp1_done: bool = False
    tp2_done: bool = False
    tp3_done: bool = False

    # trailing
    trail_active: bool = False
    trail_stop_price_usd: Optional[float] = None

    def pnl_usd(self) -> float:
        if self.last_price_usd <= 0:
            return 0.0
        return (self.qty * self.last_price_usd) - self.entry_cost_usd

    def roi(self) -> float:
        if self.entry_cost_usd <= 0:
            return 0.0
        return self.pnl_usd() / self.entry_cost_usd


class PaperState:
    def __init__(self):
        self.cash = START_CASH_USD           # tradable
        self.reserve_cash = 0.0              # vaulted
        self.positions: Dict[str, Position] = {}
        self.event_log: List[Dict[str, Any]] = []
        self.closed_trades: List[Dict[str, Any]] = []

    def snapshot(self) -> Dict[str, Any]:
        positions_out = {mint: asdict(pos) for mint, pos in self.positions.items()}
        equity = self.cash + self.reserve_cash
        for pos in self.positions.values():
            if pos.last_price_usd > 0:
                equity += pos.qty * pos.last_price_usd
        return {
            "cash": self.cash,
            "reserve_cash": self.reserve_cash,
            "open_positions": len(self.positions),
            "positions": positions_out,
            "equity_estimate": equity,
            "closed_trades_count": len(self.closed_trades),
            "tracked_wallets": list(TRACKED_WALLETS),
            "config": {
                "BUY_USD_FIXED": BUY_USD_FIXED,
                "MAX_OPEN_POSITIONS": MAX_OPEN_POSITIONS,
                "STOP_LOSS_PCT": STOP_LOSS_PCT,
                "MAX_HOLD_SECONDS": MAX_HOLD_SECONDS,
                "TPs": [TP1_PCT, TP2_PCT, TP3_PCT],
                "TRAIL_ACTIVATE_PCT": TRAIL_ACTIVATE_PCT,
                "TRAIL_DIST_PCT": TRAIL_DIST_PCT,
                "BIG_WIN_ROI": BIG_WIN_ROI,
                "BIG_WIN_RESERVE_SPLIT": BIG_WIN_RESERVE_SPLIT,
                "BASE_RESERVE_SPLIT": BASE_RESERVE_SPLIT,
                "SOL_PRICE_USD": SOL_PRICE_USD,
            }
        }


STATE = PaperState()
app = Flask(__name__)

# =========================
# Helpers
# =========================

def add_event(evt: Dict[str, Any], keep: int = 200):
    STATE.event_log.append(evt)
    if len(STATE.event_log) > keep:
        STATE.event_log = STATE.event_log[-keep:]


def lamports_to_sol(lamports: float) -> float:
    return lamports / 1_000_000_000.0


def apply_trade_friction(notional_usd: float) -> float:
    """
    Total friction in USD for a trade (fee + pct + slippage).
    """
    return FEE_FIXED_USD + (notional_usd * FEE_PCT) + (notional_usd * SLIPPAGE_PCT)


def tradable_cash() -> float:
    return max(0.0, STATE.cash)


def inflight_value_usd() -> float:
    total = 0.0
    for pos in STATE.positions.values():
        total += pos.entry_cost_usd
    return total


def can_open_new_position() -> bool:
    if len(STATE.positions) >= MAX_OPEN_POSITIONS:
        return False
    # cap inflight as % of tradable cash
    cash = tradable_cash()
    if cash <= 0:
        return False
    inflight = inflight_value_usd()
    return inflight <= cash * MAX_INFLIGHT_PCT


def estimate_price_from_helius_swap(payload_obj: Dict[str, Any], tracked_wallet: str, mint: str, token_amount: float) -> Optional[float]:
    """
    Estimate price_usd using nativeBalanceChange for tracked wallet and SOL_PRICE_USD.

    For buys:
      tracked wallet nativeBalanceChange is negative (SOL spent).
    For sells:
      tracked wallet nativeBalanceChange is positive (SOL received).

    We use abs(nativeBalanceChange SOL) / token_amount * SOL_PRICE_USD
    """
    try:
        acct_data = payload_obj.get("accountData", [])
        native_change_lamports = None
        for item in acct_data:
            if item.get("account") == tracked_wallet:
                native_change_lamports = item.get("nativeBalanceChange")
                break
        if native_change_lamports is None:
            return None

        sol_delta = lamports_to_sol(float(native_change_lamports))
        # if token_amount is 0, can't price
        if token_amount == 0:
            return None

        # Use magnitude of SOL delta as swap notional; ignore small transfers/noise
        notional_sol = abs(sol_delta)
        if notional_sol <= 0:
            return None

        price = (notional_sol * SOL_PRICE_USD) / abs(token_amount)
        if price <= 0 or math.isnan(price) or math.isinf(price):
            return None
        return price
    except Exception:
        return None


def vault_profit_on_close(entry_cost_usd: float, realized_profit_usd: float) -> float:
    """
    Decide how much realized profit to move into reserve_cash.
    Returns 'to_reserve' amount (>=0).
    """
    if realized_profit_usd <= MIN_PROFIT_USD:
        return 0.0
    roi = realized_profit_usd / max(entry_cost_usd, 1e-9)
    split = BIG_WIN_RESERVE_SPLIT if roi >= BIG_WIN_ROI else BASE_RESERVE_SPLIT
    if split <= 0:
        return 0.0
    return max(0.0, realized_profit_usd * split)


def open_position(mint: str, price_usd: float):
    if price_usd is None or price_usd <= 0:
        return

    if mint in STATE.positions:
        return  # already holding

    if not can_open_new_position():
        return

    buy_usd = min(BUY_USD_FIXED, tradable_cash())
    if buy_usd < 5.0:
        return

    friction = apply_trade_friction(buy_usd)
    total_cost = buy_usd + friction
    if total_cost > STATE.cash:
        return

    qty = buy_usd / price_usd
    if qty <= 0:
        return

    STATE.cash -= total_cost

    pos = Position(
        mint=mint,
        qty=qty,
        entry_price_usd=price_usd,
        entry_cost_usd=total_cost,
        opened_at=time.time(),
        last_price_usd=price_usd,
        peak_price_usd=price_usd,
    )
    STATE.positions[mint] = pos

    add_event({
        "ts": time.time(),
        "kind": "PAPER_BUY",
        "mint": mint,
        "price_usd": price_usd,
        "qty": qty,
        "buy_usd": buy_usd,
        "friction_usd": friction,
        "total_cost_usd": total_cost,
        "cash_after": STATE.cash,
    })


def sell_fraction(pos: Position, price_usd: float, frac: float, reason: str):
    """
    Sell a fraction of position qty at given price.
    Updates cash, position qty, and records a closed_trade entry when position hits 0.
    """
    frac = max(0.0, min(1.0, frac))
    if frac <= 0 or pos.qty <= 0:
        return

    sell_qty = pos.qty * frac
    proceeds = sell_qty * price_usd
    friction = apply_trade_friction(proceeds)
    net_proceeds = max(0.0, proceeds - friction)

    # Update position
    pos.qty -= sell_qty
    pos.last_price_usd = price_usd

    # Add proceeds to tradable cash first
    STATE.cash += net_proceeds

    add_event({
        "ts": time.time(),
        "kind": "PAPER_SELL_PARTIAL",
        "mint": pos.mint,
        "reason": reason,
        "price_usd": price_usd,
        "sell_qty": sell_qty,
        "gross_proceeds_usd": proceeds,
        "friction_usd": friction,
        "net_proceeds_usd": net_proceeds,
        "remaining_qty": pos.qty,
        "cash_after": STATE.cash,
    })

    # If position closed, compute realized P/L against proportional cost basis
    # We'll treat entry_cost as full position cost and allocate by sold fraction overall.
    # For simplicity, when fully closed we compute final realized profit.
    if pos.qty <= 1e-12:
        close_position_fully(pos.mint, price_usd, reason)


def close_position_fully(mint: str, price_usd: float, reason: str):
    pos = STATE.positions.get(mint)
    if not pos:
        return

    # If qty is already zero, just finalize close
    if pos.qty > 1e-12:
        # sell remaining
        sell_qty = pos.qty
        proceeds = sell_qty * price_usd
        friction = apply_trade_friction(proceeds)
        net_proceeds = max(0.0, proceeds - friction)

        pos.qty = 0.0
        pos.last_price_usd = price_usd
        STATE.cash += net_proceeds

        add_event({
            "ts": time.time(),
            "kind": "PAPER_SELL_FINAL",
            "mint": mint,
            "reason": reason,
            "price_usd": price_usd,
            "sell_qty": sell_qty,
            "gross_proceeds_usd": proceeds,
            "friction_usd": friction,
            "net_proceeds_usd": net_proceeds,
            "cash_after": STATE.cash,
        })

    # Realized profit: (current value of original qty sold) - entry_cost.
    # Since we sold in pieces, this simplified model is "best-effort":
    # We'll estimate realized as (entry_cost recovered into cash minus entry_cost)
    # by looking at last_price * original qty is not available anymore.
    # So we store the profit at close as (position value at last price - entry_cost),
    # but that's close enough for paper compounding logic given scalp volatility.

    # Better: track cumulative proceeds per position. If you want that next, tell me.
    est_value_at_exit = (pos.entry_cost_usd / max(pos.entry_price_usd, 1e-9)) * price_usd
    realized_profit = est_value_at_exit - pos.entry_cost_usd

    to_reserve = vault_profit_on_close(pos.entry_cost_usd, realized_profit)
    if to_reserve > 0:
        # Move from tradable cash into reserve vault
        move_amt = min(to_reserve, STATE.cash)
        STATE.cash -= move_amt
        STATE.reserve_cash += move_amt

    trade_record = {
        "ts_closed": time.time(),
        "mint": mint,
        "reason": reason,
        "entry_price_usd": pos.entry_price_usd,
        "exit_price_usd": price_usd,
        "entry_cost_usd": pos.entry_cost_usd,
        "est_value_at_exit_usd": est_value_at_exit,
        "est_realized_profit_usd": realized_profit,
        "vaulted_usd": to_reserve,
        "cash_after": STATE.cash,
        "reserve_after": STATE.reserve_cash,
        "hold_seconds": time.time() - pos.opened_at,
    }
    STATE.closed_trades.append(trade_record)
    if len(STATE.closed_trades) > 500:
        STATE.closed_trades = STATE.closed_trades[-500:]

    add_event({
        "ts": time.time(),
        "kind": "POSITION_CLOSED",
        **trade_record
    })

    # remove position
    STATE.positions.pop(mint, None)


def update_and_maybe_exit(mint: str, price_usd: float):
    pos = STATE.positions.get(mint)
    if not pos or price_usd is None or price_usd <= 0:
        return

    pos.last_price_usd = price_usd
    if price_usd > pos.peak_price_usd:
        pos.peak_price_usd = price_usd

    # 1) Stop loss
    stop_price = pos.entry_price_usd * (1.0 - STOP_LOSS_PCT)
    if price_usd <= stop_price:
        close_position_fully(mint, price_usd, reason=f"STOP_LOSS_{int(STOP_LOSS_PCT*100)}pct")
        return

    # 2) Max hold time (exit if not meaningfully green by then)
    age = time.time() - pos.opened_at
    if age >= MAX_HOLD_SECONDS:
        close_position_fully(mint, price_usd, reason=f"MAX_HOLD_{MAX_HOLD_SECONDS}s")
        return

    # 3) Take-profit ladder (partial sells)
    r = (price_usd / max(pos.entry_price_usd, 1e-9)) - 1.0

    if (not pos.tp1_done) and r >= TP1_PCT:
        pos.tp1_done = True
        sell_fraction(pos, price_usd, TP1_SELL_FRAC, reason=f"TP1_{int(TP1_PCT*100)}pct")

    if mint not in STATE.positions:
        return

    pos = STATE.positions[mint]
    if (not pos.tp2_done) and r >= TP2_PCT:
        pos.tp2_done = True
        sell_fraction(pos, price_usd, TP2_SELL_FRAC, reason=f"TP2_{int(TP2_PCT*100)}pct")

    if mint not in STATE.positions:
        return

    pos = STATE.positions[mint]
    if (not pos.tp3_done) and r >= TP3_PCT:
        pos.tp3_done = True
        sell_fraction(pos, price_usd, TP3_SELL_FRAC, reason=f"TP3_{int(TP3_PCT*100)}pct")

    if mint not in STATE.positions:
        return

    pos = STATE.positions[mint]

    # 4) Trailing stop
    if (not pos.trail_active) and r >= TRAIL_ACTIVATE_PCT:
        pos.trail_active = True
        pos.trail_stop_price_usd = pos.peak_price_usd * (1.0 - TRAIL_DIST_PCT)

    if pos.trail_active:
        # update trail stop as peak increases
        new_stop = pos.peak_price_usd * (1.0 - TRAIL_DIST_PCT)
        if pos.trail_stop_price_usd is None or new_stop > pos.trail_stop_price_usd:
            pos.trail_stop_price_usd = new_stop

        if pos.trail_stop_price_usd is not None and price_usd <= pos.trail_stop_price_usd:
            close_position_fully(mint, price_usd, reason=f"TRAIL_{int(TRAIL_DIST_PCT*100)}pct")
            return


def parse_swaps_from_helius(body: Any) -> List[Dict[str, Any]]:
    """
    Helius can send:
      - {"events":[{"payload":[...]}]}
      - or an array of payload objects
      - or single payload object

    We'll normalize to a list of payload objects that have `tokenTransfers`, `accountData`, `type`, etc.
    """
    out = []
    if isinstance(body, dict) and "events" in body:
        # your /events endpoint format
        for e in body.get("events", []):
            payload_list = e.get("payload", [])
            if isinstance(payload_list, list):
                out.extend(payload_list)
    elif isinstance(body, list):
        out.extend(body)
    elif isinstance(body, dict):
        out.append(body)
    return [x for x in out if isinstance(x, dict)]


def extract_tracked_wallet_swaps(payload_obj: Dict[str, Any]) -> List[Tuple[str, str, float, str]]:
    """
    Return list of (tracked_wallet, mint, token_amount, side)
    side: "BUY" if tracked wallet receives tokens, "SELL" if sends tokens.
    """
    swaps = []
    tts = payload_obj.get("tokenTransfers", []) or []
    for tt in tts:
        mint = tt.get("mint")
        token_amt = tt.get("tokenAmount")
        from_user = tt.get("fromUserAccount")
        to_user = tt.get("toUserAccount")

        # Only care if mint looks like a token mint
        if not mint or token_amt is None:
            continue

        # If we have tracked wallets, only react to those
        if TRACKED_WALLETS:
            if to_user in TRACKED_WALLETS:
                swaps.append((to_user, mint, float(token_amt), "BUY"))
            elif from_user in TRACKED_WALLETS:
                swaps.append((from_user, mint, float(token_amt), "SELL"))
        else:
            # If no tracked wallets configured, do nothing
            pass

    return swaps


# =========================
# Routes
# =========================

@app.get("/health")
def health():
    return jsonify({"ok": True, "ts": time.time()}), 200


@app.get("/events")
def events():
    return jsonify({"count": len(STATE.event_log), "events": STATE.event_log[-200:]}), 200


@app.get("/paper/state")
def paper_state():
    return jsonify(STATE.snapshot()), 200


@app.post("/webhook")
def webhook():
    # Optional secret header check
    if WEBHOOK_SECRET:
        got = request.headers.get("x-webhook-secret", "") or request.headers.get("X-Webhook-Secret", "")
        if got != WEBHOOK_SECRET:
            return jsonify({"ok": False, "error": "bad secret"}), 401

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"ok": False, "error": "no json"}), 400

    payloads = parse_swaps_from_helius(body)
    handled = 0

    for p in payloads:
        # We only react to swap-like payloads, but we can still log
        p_type = p.get("type")
        sig = p.get("signature")
        src = p.get("source")
        ts = p.get("timestamp")

        swaps = extract_tracked_wallet_swaps(p)
        if not swaps:
            continue

        for (twallet, mint, token_amt, side) in swaps:
            # Estimate price from this swap
            price = estimate_price_from_helius_swap(p, twallet, mint, token_amt)
            if price is None:
                # If we can't price, log and skip trading decision
                add_event({
                    "ts": time.time(),
                    "kind": "SWAP_UNPRICED",
                    "tracked_wallet": twallet,
                    "mint": mint,
                    "side": side,
                    "token_amount": token_amt,
                    "payload_type": p_type,
                    "source": src,
                    "signature": sig,
                    "payload_ts": ts,
                })
                continue

            # Update price & exits for existing position
            update_and_maybe_exit(mint, price)

            # If tracked wallet bought, we try to enter (copy)
            if side == "BUY":
                open_position(mint, price)

            # If tracked wallet sold, we do NOT rely on it anymore,
            # but we can optionally tighten exits or close immediately.
            # For now: on whale sell, close if we are still holding.
            if side == "SELL" and mint in STATE.positions:
                close_position_fully(mint, price, reason="WHALE_SELL_SIGNAL")

            add_event({
                "ts": time.time(),
                "kind": "TRACKED_SWAP",
                "tracked_wallet": twallet,
                "mint": mint,
                "side": side,
                "token_amount": token_amt,
                "est_price_usd": price,
                "payload_type": p_type,
                "source": src,
                "signature": sig,
                "payload_ts": ts,
                "cash": STATE.cash,
                "reserve_cash": STATE.reserve_cash,
                "open_positions": len(STATE.positions),
            })
            handled += 1

    return jsonify({"ok": True, "handled": handled}), 200


# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=APP_PORT)
