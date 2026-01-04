import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

APP_NAME = "sol-paper-bot"
LOG_DIR = Path("logs")
EVENT_LOG = LOG_DIR / "events.jsonl"

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

app = FastAPI(title=APP_NAME)

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def ensure_logfile() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not EVENT_LOG.exists():
        EVENT_LOG.write_text("", encoding="utf-8")

def append_jsonl(record: Dict[str, Any]) -> None:
    ensure_logfile()
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record))
        f.write("\n")
        
# =======================
# PAPER TRADING STATE
# =======================

PAPER_STATE_FILE = Path("logs/paper_state.json")

def load_paper_state():
    if not PAPER_STATE_FILE.exists():
        return {
            "cash": 500.0,
            "positions": {},
            "trades": []
        }
    return json.loads(PAPER_STATE_FILE.read_text())

def save_paper_state(state):
    PAPER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAPER_STATE_FILE.write_text(json.dumps(state, indent=2))

@app.get("/paper/state")
def paper_state():
    return load_paper_state()

@app.get("/health")
def health():
    return {"ok": True, "service": APP_NAME, "time": utc_now_iso()}

@app.get("/events")
def get_events():
    ensure_logfile()
    lines = EVENT_LOG.read_text(encoding="utf-8").splitlines()[-200:]
    return {
        "count": len(lines),
        "events": [json.loads(l) for l in lines]
    }

@app.post("/webhook")
async def webhook(
    request: Request,
    x_webhook_secret: Optional[str] = Header(default=None, convert_underscores=False),
    authorization: Optional[str] = Header(default=None),
):
    # Accept secret from multiple possible headers
    provided = x_webhook_secret or authorization or ""

    # Handle: Authorization: Bearer <secret>
    if provided.lower().startswith("bearer "):
        provided = provided[7:].strip()

    # Handle badly formatted "x-webhook-secret: value"
    if provided.lower().startswith("x-webhook-secret"):
        parts = provided.split(":", 1)
        if len(parts) == 2:
            provided = parts[1].strip()

    if WEBHOOK_SECRET and provided != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    payload = await request.json()

    record = {
        "received_at": utc_now_iso(),
        "source": "helius",
        "payload": payload,
    }
    append_jsonl(record)

    return JSONResponse({"ok": True})
