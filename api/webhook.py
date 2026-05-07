"""
Sector Command — Twilio Webhook (Vercel Serverless Function, Flask)

Receives SMS replies, parses commands, updates portfolio in Google Sheets,
generates journal entries.

Deploy to Vercel as api/webhook.py
"""

import os
import json
import base64
import tempfile
from datetime import datetime
from flask import Flask, request, Response


def _init_google_creds():
    """Decode GOOGLE_CREDS_B64 env var into a temp file for gspread auth."""
    b64 = os.environ.get("GOOGLE_CREDS_B64", "")
    if not b64:
        return
    try:
        decoded = base64.b64decode(b64).decode("utf-8")
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        tmp.write(decoded)
        tmp.close()
        os.environ["GOOGLE_CREDS_PATH"] = tmp.name
        print(f"[Init] Wrote Google creds to {tmp.name}")
    except Exception as e:
        print(f"[Init] Failed to decode GOOGLE_CREDS_B64: {e}")


_init_google_creds()

app = Flask(__name__)
SYSTEM_VERSION = "v4.2"


@app.route("/api/webhook", methods=["GET", "POST"])
@app.route("/api/webhook/", methods=["GET", "POST"])
def webhook():
    """Flask entry point for Vercel."""
    if request.method != "POST":
        return Response("Sector Command Webhook Active", status=200)

    from_number = request.form.get("From", "")
    message_body = request.form.get("Body", "").strip().upper()
    timestamp = datetime.now().isoformat()

    print(f"[Webhook] {timestamp} | From: {from_number} | Message: {message_body}")

    allowed = os.environ.get("MY_PHONE", "+14702728228")
    if from_number != allowed:
        return _twiml_response("Unauthorized.")

    response_text = process_command(message_body, timestamp)
    return _twiml_response(response_text)


def process_command(command, timestamp):
    """Parse and execute a text command."""

    # Load portfolio state from Google Sheets
    portfolio = load_portfolio()

    if command == "BUY":
        return execute_trade("BUY", None, None, portfolio, timestamp)

    elif command.startswith("BUY $"):
        try:
            amount = float(command.replace("BUY $", "").strip())
            return execute_trade("BUY", None, amount, portfolio, timestamp)
        except ValueError:
            return "Could not parse amount. Use: BUY $30"

    elif command.startswith("BUY "):
        ticker = command.replace("BUY ", "").strip()
        return execute_trade("BUY", ticker, None, portfolio, timestamp)

    elif command == "SELL":
        return execute_trade("SELL", None, None, portfolio, timestamp)

    elif command.startswith("SELL ALL "):
        ticker = command.replace("SELL ALL ", "").strip()
        return execute_sell_all(ticker, portfolio, timestamp)

    elif command.startswith("SELL "):
        ticker = command.replace("SELL ", "").strip()
        return execute_trade("SELL", ticker, None, portfolio, timestamp)

    elif command == "SKIP":
        log_journal_entry({
            "timestamp": timestamp,
            "action": "SKIP",
            "decision": "Signal declined — no reason provided",
            "version": SYSTEM_VERSION,
        })
        return "Signal skipped. Logged."

    elif command == "STATUS":
        return format_status(portfolio)

    elif command == "WHY":
        return "Check the Streamlit dashboard for detailed SHAP analysis and model reasoning."

    elif command == "PAUSE":
        portfolio["paused"] = True
        save_portfolio(portfolio)
        return "System PAUSED. No signals will be sent. Text RESUME to reactivate."

    elif command == "RESUME":
        portfolio["paused"] = False
        save_portfolio(portfolio)
        return "System RESUMED. Signals will continue at next scheduled run."

    elif command == "FORCE SELL ALL":
        return force_sell_all(portfolio, timestamp)

    else:
        return f"Unknown command: {command}\nOptions: BUY, SELL, SKIP, STATUS, WHY, PAUSE, RESUME"


def execute_trade(action, ticker, custom_amount, portfolio, timestamp):
    """Execute a buy or sell trade."""
    # Load latest recommendations from sheets
    latest_signal = load_latest_signal()

    if not latest_signal:
        return "No active signal to execute. Wait for next signal."

    recs = latest_signal.get("recommendations", [])

    if action == "BUY":
        buy_recs = [r for r in recs if r["action"] == "BUY" and not r.get("vetoed")]
        if not buy_recs:
            return "No buy signals available."
        rec = buy_recs[0]  # Highest conviction

        if ticker and ticker in [r["ticker"] for r in buy_recs]:
            rec = next(r for r in buy_recs if r["ticker"] == ticker)

    elif action == "SELL":
        sell_recs = [r for r in recs if r["action"] == "SELL"]
        if not sell_recs:
            return "No sell signals available."
        rec = sell_recs[0]

    # Calculate amount
    balance = portfolio.get("balance", 400)
    if custom_amount:
        amount = min(custom_amount, balance)
    else:
        amount = balance * rec["target_weight"] * rec["sizing_multiplier"]

    # Apply slippage
    slippage = amount * 10 / 10000  # 10 bps
    net_amount = amount - slippage

    # Log the trade
    trade = {
        "timestamp": timestamp,
        "version": SYSTEM_VERSION,
        "action": action,
        "ticker": rec["ticker"],
        "sector": rec.get("sector", ""),
        "amount": round(net_amount, 2),
        "conviction": rec["conviction_label"],
        "regime": latest_signal.get("regime", "unknown"),
        "vix": latest_signal.get("vix", 0),
        "rsi": rec.get("rsi", 0),
        "agent_weights": json.dumps(rec.get("agent_weights", {})),
        "decision": f"FOLLOWED signal — {action} at {'custom' if custom_amount else 'recommended'} size",
        "slippage_applied": round(slippage, 2),
    }

    log_journal_entry(trade)

    # Update portfolio
    if action == "BUY":
        holdings = portfolio.get("holdings", {})
        if rec["ticker"] not in holdings:
            holdings[rec["ticker"]] = {"shares": 0, "cost_basis": 0, "market_value": 0}
        # Simplified: track dollar amount as "shares" for paper trading
        holdings[rec["ticker"]]["market_value"] = holdings[rec["ticker"]].get("market_value", 0) + net_amount
        portfolio["holdings"] = holdings
        portfolio["cash"] = portfolio.get("cash", balance) - amount
    elif action == "SELL":
        holdings = portfolio.get("holdings", {})
        if rec["ticker"] in holdings:
            holdings[rec["ticker"]]["market_value"] = max(0, holdings[rec["ticker"]].get("market_value", 0) - amount)
            if holdings[rec["ticker"]]["market_value"] < 1:
                del holdings[rec["ticker"]]
            portfolio["holdings"] = holdings
            portfolio["cash"] = portfolio.get("cash", 0) + net_amount

    portfolio["trades_today"] = portfolio.get("trades_today", 0) + 1
    save_portfolio(portfolio)

    return (f"✓ {action} {rec['ticker']} — ${net_amount:.2f}\n"
            f"Conviction: {rec['conviction_label']}\n"
            f"Slippage: ${slippage:.2f}\n"
            f"Balance: ${portfolio.get('balance', 0):.2f}")


def execute_sell_all(ticker, portfolio, timestamp):
    """Sell entire position in a ticker."""
    holdings = portfolio.get("holdings", {})
    if ticker not in holdings:
        return f"No position in {ticker}."

    amount = holdings[ticker].get("market_value", 0)
    slippage = amount * 10 / 10000
    net = amount - slippage

    del holdings[ticker]
    portfolio["holdings"] = holdings
    portfolio["cash"] = portfolio.get("cash", 0) + net
    portfolio["trades_today"] = portfolio.get("trades_today", 0) + 1

    log_journal_entry({
        "timestamp": timestamp, "version": SYSTEM_VERSION,
        "action": "SELL ALL", "ticker": ticker, "amount": round(net, 2),
        "decision": "Manual full liquidation",
    })
    save_portfolio(portfolio)
    return f"✓ Sold all {ticker} — ${net:.2f} (after ${slippage:.2f} slippage)"


def force_sell_all(portfolio, timestamp):
    """Liquidate everything to cash."""
    holdings = portfolio.get("holdings", {})
    total = sum(h.get("market_value", 0) for h in holdings.values())
    slippage = total * 10 / 10000
    net = total - slippage

    portfolio["holdings"] = {}
    portfolio["cash"] = portfolio.get("cash", 0) + net
    portfolio["trades_today"] = portfolio.get("trades_today", 0) + len(holdings)

    log_journal_entry({
        "timestamp": timestamp, "version": SYSTEM_VERSION,
        "action": "FORCE SELL ALL", "amount": round(net, 2),
        "decision": "Emergency liquidation — all positions closed",
    })
    save_portfolio(portfolio)
    return f"✓ All positions liquidated — ${net:.2f} to cash"


def format_status(portfolio):
    """Format current portfolio status."""
    bal = portfolio.get("balance", 0)
    cash = portfolio.get("cash", bal)
    ghost = portfolio.get("ghost_balance", 0)
    holdings = portfolio.get("holdings", {})
    paused = portfolio.get("paused", False)

    lines = [f"PORTFOLIO STATUS {'[PAUSED]' if paused else ''}",
             f"Balance: ${bal:.2f} | Cash: ${cash:.2f}",
             f"Ghost SPY: ${ghost:.2f}"]

    if holdings:
        lines.append("Holdings:")
        for t, h in holdings.items():
            mv = h.get("market_value", 0)
            pct = mv / bal * 100 if bal > 0 else 0
            lines.append(f"  {t}: ${mv:.2f} ({pct:.1f}%)")
    else:
        lines.append("Holdings: ALL CASH")

    return "\n".join(lines)


# ── Google Sheets helpers ──

def load_portfolio():
    """Load portfolio from Google Sheets."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_file(
            os.environ.get("GOOGLE_CREDS_PATH", "creds.json"),
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
        gc = gspread.authorize(creds)
        sheet = gc.open("Sector Command").worksheet("Portfolio")
        records = sheet.get_all_records()
        if records:
            return json.loads(records[-1].get("state", "{}"))
    except Exception as e:
        print(f"[Sheets] Load error: {e}")

    return {"balance": 400, "initial": 400, "ghost_balance": 400,
            "holdings": {}, "cash": 400, "paused": False}


def save_portfolio(portfolio):
    """Save portfolio to Google Sheets."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_file(
            os.environ.get("GOOGLE_CREDS_PATH", "creds.json"),
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
        gc = gspread.authorize(creds)
        sheet = gc.open("Sector Command").worksheet("Portfolio")
        sheet.append_row([datetime.now().isoformat(), SYSTEM_VERSION,
                          json.dumps(portfolio, default=str)])
    except Exception as e:
        print(f"[Sheets] Save error: {e}")


def load_latest_signal():
    """Load most recent signal from Google Sheets."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_file(
            os.environ.get("GOOGLE_CREDS_PATH", "creds.json"),
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
        gc = gspread.authorize(creds)
        sheet = gc.open("Sector Command").worksheet("Signals")
        records = sheet.get_all_records()
        if records:
            return json.loads(records[-1].get("data", "{}"))
    except Exception as e:
        print(f"[Sheets] Signal load error: {e}")
    return None


def log_journal_entry(entry):
    """Write journal entry to Google Sheets."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_file(
            os.environ.get("GOOGLE_CREDS_PATH", "creds.json"),
            scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
        gc = gspread.authorize(creds)
        sheet = gc.open("Sector Command").worksheet("Journal")
        row = [entry.get(k, "") for k in [
            "timestamp", "version", "action", "ticker", "amount",
            "conviction", "regime", "vix", "rsi", "agent_weights",
            "decision", "slippage_applied"
        ]]
        sheet.append_row(row)
    except Exception as e:
        print(f"[Journal] Error: {e}")


def _twiml_response(text):
    """Format a TwiML response for Twilio."""
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/xml"},
        "body": f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{text}</Message></Response>'
    }
