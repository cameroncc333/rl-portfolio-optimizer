"""
Sector Command — Signal Server v1.0

Runs 4x daily via GitHub Actions. Pulls fresh data, runs all 3 RL agents,
computes technical signals, applies hard veto, sizes by conviction,
sends SMS via Twilio, updates Google Sheets, runs ghost portfolio.

Usage:
    python signal_server.py                  # Live mode
    python signal_server.py --replay 2025-06-01 2025-12-31   # Replay mode
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import yfinance as yf

try:
    from twilio.rest import Client as TwilioClient
    HAS_TWILIO = True
except ImportError:
    HAS_TWILIO = False

try:
    import gspread
    from google.oauth2.service_account import Credentials
    HAS_GSHEETS = True
except ImportError:
    HAS_GSHEETS = False

from stable_baselines3 import PPO, A2C, SAC
from config import (
    TICKERS, SECTOR_ETFS, NUM_ASSETS, LOOKBACK_WINDOW,
    WALK_FORWARD_WINDOWS, AGENTS, SEEDS, MODEL_DIR,
    REGIME_THRESHOLDS, KILL_SWITCH_THRESHOLD,
)
from data_loader import prepare_data
from portfolio_env import PortfolioEnv
from evaluate import run_agent, _load_norm_stats

AGENT_CLASSES = {"PPO": PPO, "A2C": A2C, "SAC": SAC}

# ── System version (logged with every trade)
SYSTEM_VERSION = "v4.2"

# ── Agent priority (best to worst based on v4 evaluation)
AGENT_PRIORITY = ["A2C", "SAC", "PPO"]

# ── Slippage on paper trades
SLIPPAGE_BPS = 10  # 0.1%

# ── Conviction sizing
CONVICTION_SIZING = {3: 1.0, 2: 0.5, 1: 0.25}

# ── Veto thresholds
VETO_RSI_THRESHOLD = 80
VETO_REGIME = "stressed"


class SectorCommand:
    """The brain of the signal pipeline."""

    def __init__(self, mode="live", twilio_client=None, sheets_client=None):
        self.mode = mode
        self.twilio = twilio_client
        self.sheets = sheets_client
        self.models = {}
        self.norm_stats = {}
        self.features = None
        self.returns = None
        self.close = None
        self.regimes = None
        self.regime_numeric = None
        self.vix = None
        self.sector_tickers = []

    def load_data(self):
        """Pull fresh market data."""
        print(f"[Signal] Loading data (mode={self.mode})...")
        self.features, self.returns, self.close, self.regimes, \
            self.regime_numeric, self.vix = prepare_data()
        self.sector_tickers = [t for t in TICKERS if t in self.returns.columns]
        print(f"[Signal] {len(self.features)} days, {len(self.sector_tickers)} sectors")

    def load_models(self):
        """Load trained RL agents."""
        print("[Signal] Loading agents...")
        w_idx = len(WALK_FORWARD_WINDOWS) - 1
        for agent_name in AGENT_PRIORITY:
            for seed in SEEDS:
                run_id = f"{agent_name}_w{w_idx}_s{seed}"
                for suffix in ["best_model", f"{run_id}_final"]:
                    path = os.path.join(MODEL_DIR, run_id, f"{suffix}.zip")
                    if os.path.exists(path):
                        AgentClass = AGENT_CLASSES[AGENTS[agent_name]["class"]]
                        self.models[agent_name] = AgentClass.load(path)
                        self.norm_stats[agent_name] = _load_norm_stats(run_id)
                        print(f"  ✓ {agent_name}")
                        break
                if agent_name in self.models:
                    break
        print(f"[Signal] {len(self.models)} agents loaded")

    def get_current_signals(self):
        """Run all agents and compute signals for today."""
        signals = {}

        # Current regime
        current_regime = self.regimes.iloc[-1]
        current_vix = float(self.vix.iloc[-1])

        # VIX term structure (if available in features)
        vix_term = 0.0
        if ("vix_term", self.sector_tickers[0]) in self.features.columns:
            vix_term = float(self.features[("vix_term", self.sector_tickers[0])].iloc[-1])

        signals["regime"] = current_regime
        signals["vix"] = current_vix
        signals["vix_term"] = vix_term
        signals["vix_term_label"] = "contango" if vix_term > 0 else "backwardation"
        signals["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Run each agent
        agent_weights = {}
        recent_feat = self.features.iloc[-LOOKBACK_WINDOW - 100:]
        recent_ret = self.returns.loc[recent_feat.index]
        recent_regime = self.regime_numeric.loc[recent_feat.index]

        for agent_name, model in self.models.items():
            try:
                _, weights = run_agent(
                    model, recent_feat, recent_ret[self.sector_tickers],
                    recent_regime, norm_stats=self.norm_stats.get(agent_name)
                )
                agent_weights[agent_name] = weights[-1] if len(weights) > 0 else None
            except Exception as e:
                print(f"  ⚠ {agent_name} failed: {e}")
                agent_weights[agent_name] = None

        signals["agent_weights"] = agent_weights

        # Technical signals per sector
        tech_signals = {}
        for ticker in self.sector_tickers:
            ts = {}
            if ("rsi", ticker) in self.features.columns:
                ts["rsi"] = float(self.features[("rsi", ticker)].iloc[-1]) * 100
            if ("sharpe", ticker) in self.features.columns:
                ts["sharpe"] = float(self.features[("sharpe", ticker)].iloc[-1])
            if ("beta", ticker) in self.features.columns:
                ts["beta"] = float(self.features[("beta", ticker)].iloc[-1])
            if ("ret_20d", ticker) in self.features.columns:
                ts["momentum_20d"] = float(self.features[("ret_20d", ticker)].iloc[-1])
            tech_signals[ticker] = ts

        signals["tech"] = tech_signals

        # Compute recommendations
        signals["recommendations"] = self._compute_recommendations(
            agent_weights, tech_signals, current_regime
        )

        return signals

    def _compute_recommendations(self, agent_weights, tech_signals, regime):
        """Generate buy/sell recommendations with conviction and veto."""
        recs = []
        ew = 1.0 / len(self.sector_tickers)

        for i, ticker in enumerate(self.sector_tickers):
            # Get each agent's weight for this sector
            weights = {}
            for agent_name, w in agent_weights.items():
                if w is not None:
                    weights[agent_name] = float(w[i])

            if not weights:
                continue

            avg_weight = np.mean(list(weights.values()))
            overweight = avg_weight > ew * 1.1  # >10% above equal weight
            underweight = avg_weight < ew * 0.9  # >10% below equal weight

            # Conviction: how many agents agree on direction
            if overweight:
                agreement = sum(1 for w in weights.values() if w > ew * 1.05)
            elif underweight:
                agreement = sum(1 for w in weights.values() if w < ew * 0.95)
            else:
                agreement = 0

            conviction = min(agreement, 3)
            conviction_label = {3: "HIGH", 2: "MEDIUM", 1: "LOW"}.get(conviction, "NONE")

            # Hard veto check
            tech = tech_signals.get(ticker, {})
            rsi = tech.get("rsi", 50)
            vetoed = False
            veto_reason = ""

            if overweight and rsi > VETO_RSI_THRESHOLD and regime == VETO_REGIME:
                vetoed = True
                veto_reason = f"RSI {rsi:.0f} + {regime} regime"
                avg_weight = ew  # Force to equal weight

            if overweight and rsi > 85:  # Extreme overbought regardless of regime
                vetoed = True
                veto_reason = f"RSI {rsi:.0f} (extreme overbought)"
                avg_weight = ew * 0.5

            # Size by conviction
            sizing = CONVICTION_SIZING.get(conviction, 0.25)

            # Risk level
            beta = tech.get("beta", 1.0)
            if beta > 1.3 or rsi > 70:
                risk = "HIGH"
            elif beta > 0.8 and rsi < 70:
                risk = "MODERATE"
            else:
                risk = "LOW"

            if overweight or underweight:
                action = "BUY" if overweight else "SELL"
                recs.append({
                    "ticker": ticker,
                    "sector": SECTOR_ETFS.get(ticker, ticker),
                    "action": action,
                    "target_weight": avg_weight,
                    "conviction": conviction,
                    "conviction_label": conviction_label,
                    "sizing_multiplier": sizing,
                    "risk": risk,
                    "rsi": rsi,
                    "beta": beta,
                    "sharpe": tech.get("sharpe", 0),
                    "agent_weights": weights,
                    "vetoed": vetoed,
                    "veto_reason": veto_reason,
                })

        # Sort by conviction (highest first), then by action (BUY before SELL)
        recs.sort(key=lambda x: (-x["conviction"], x["action"]))
        return recs

    def format_sms(self, signals, portfolio=None):
        """Format signals into an SMS message."""
        ts = signals["timestamp"]
        regime = signals["regime"].upper()
        vix = signals["vix"]
        term = signals["vix_term_label"]

        lines = [f"SECTOR COMMAND {ts.split(' ')[1] if ' ' in ts else ''}"]

        # Portfolio status (if we have it)
        if portfolio:
            bal = portfolio.get("balance", 0)
            init = portfolio.get("initial", 400)
            pnl_pct = (bal - init) / init * 100 if init > 0 else 0
            ghost = portfolio.get("ghost_balance", init)
            ghost_pnl = (ghost - init) / init * 100 if init > 0 else 0
            alpha = pnl_pct - ghost_pnl

            lines.append(f"Portfolio: ${bal:.2f} ({pnl_pct:+.1f}%)")
            lines.append(f"Ghost SPY: ${ghost:.2f} ({ghost_pnl:+.1f}%)")
            lines.append(f"Alpha: {alpha:+.1f}%")

        lines.append(f"Regime: {regime} | VIX: {vix:.1f} | {term}")
        lines.append("")

        # Recommendations
        recs = signals["recommendations"]
        if not recs:
            lines.append("NO SIGNALS — hold current positions")
        else:
            for r in recs[:3]:  # Max 3 recs per text (SMS length limit)
                emoji = "📈" if r["action"] == "BUY" else "📉"
                vetoed_tag = " [VETOED]" if r["vetoed"] else ""

                if portfolio:
                    bal = portfolio.get("balance", 400)
                    dollar_amt = bal * r["target_weight"] * r["sizing_multiplier"]
                    lines.append(f"{emoji} {r['action']} ${dollar_amt:.0f} {r['ticker']} "
                                 f"({r['sector'][:12]}){vetoed_tag}")
                else:
                    lines.append(f"{emoji} {r['action']} {r['ticker']} → "
                                 f"{r['target_weight']:.1%}{vetoed_tag}")

                lines.append(f"  Conviction: {r['conviction_label']} ({r['conviction']}/3)")
                lines.append(f"  Risk: {r['risk']} | RSI: {r['rsi']:.0f} | β: {r['beta']:.1f}")

                if r["vetoed"]:
                    lines.append(f"  VETO: {r['veto_reason']}")

                # Agent breakdown
                aw = r["agent_weights"]
                agent_str = " | ".join(f"{a}: {w:.0%}" for a, w in aw.items())
                lines.append(f"  {agent_str}")
                lines.append("")

        lines.append("Reply: BUY, SELL, SKIP, STATUS, WHY")

        return "\n".join(lines)

    def format_eod_summary(self, signals, portfolio):
        """End-of-day summary text."""
        bal = portfolio.get("balance", 0)
        init = portfolio.get("initial", 400)
        ghost = portfolio.get("ghost_balance", init)
        today_pnl = portfolio.get("today_pnl", 0)
        today_pct = portfolio.get("today_pct", 0)
        spy_today = portfolio.get("spy_today_pct", 0)

        lines = [
            "MARKET CLOSED",
            f"Today: {today_pct:+.1f}% (${today_pnl:+.2f}) | SPY: {spy_today:+.1f}%",
            f"Alpha today: {today_pct - spy_today:+.1f}%",
            f"Portfolio: ${bal:.2f} ({(bal-init)/init*100:+.1f}% all-time)",
            f"Ghost SPY: ${ghost:.2f} ({(ghost-init)/init*100:+.1f}% all-time)",
        ]

        # Holdings breakdown
        holdings = portfolio.get("holdings", {})
        if holdings:
            best_ticker = max(holdings, key=lambda t: holdings[t].get("today_return", 0))
            worst_ticker = min(holdings, key=lambda t: holdings[t].get("today_return", 0))
            lines.append(f"Best: {best_ticker} {holdings[best_ticker].get('today_return',0):+.1f}%")
            lines.append(f"Worst: {worst_ticker} {holdings[worst_ticker].get('today_return',0):+.1f}%")

        trades_today = portfolio.get("trades_today", 0)
        lines.append(f"Trades today: {trades_today}")

        # Streak tracking
        streaks = portfolio.get("model_streaks", {})
        if streaks:
            streak_parts = []
            for agent, s in streaks.items():
                if s > 0:
                    streak_parts.append(f"{agent}: {s}d winning")
                elif s < 0:
                    streak_parts.append(f"{agent}: {abs(s)}d losing")
            if streak_parts:
                lines.append(" | ".join(streak_parts))

        return "\n".join(lines)

    def format_weekly_summary(self, portfolio):
        """Saturday morning weekly recap."""
        bal = portfolio.get("balance", 0)
        init = portfolio.get("initial", 400)
        ghost = portfolio.get("ghost_balance", init)
        week_pnl = portfolio.get("week_pnl_pct", 0)
        ghost_week = portfolio.get("ghost_week_pct", 0)

        lines = [
            f"WEEKLY RECAP",
            f"Portfolio: ${bal:.2f} ({week_pnl:+.1f}% this week)",
            f"Ghost SPY: ${ghost:.2f} ({ghost_week:+.1f}% this week)",
            f"Net alpha: {week_pnl - ghost_week:+.1f}%",
            f"Trades: {portfolio.get('week_trades', 0)}",
        ]

        return "\n".join(lines)

    def send_sms(self, message):
        """Send SMS via Twilio."""
        if not HAS_TWILIO or not self.twilio:
            print(f"[SMS] (no Twilio) Message:\n{message}")
            return False

        try:
            twilio_number = os.environ.get("TWILIO_PHONE")
            my_number = os.environ.get("MY_PHONE", "+14702728228")

            self.twilio.messages.create(
                body=message,
                from_=twilio_number,
                to=my_number,
            )
            print("[SMS] Sent successfully")
            return True
        except Exception as e:
            print(f"[SMS] FAILED: {e}")
            return False

    def load_portfolio_from_sheets(self):
        """Load current portfolio state from Google Sheets."""
        if not HAS_GSHEETS or not self.sheets:
            # Return default cold start
            return {
                "balance": 400.0,
                "initial": 400.0,
                "ghost_balance": 400.0,
                "holdings": {},
                "trades_today": 0,
                "today_pnl": 0,
                "today_pct": 0,
                "spy_today_pct": 0,
                "paused": False,
            }

        try:
            ws = self.sheets.worksheet("Portfolio")
            data = ws.get_all_records()
            if data:
                latest = data[-1]
                return json.loads(latest.get("state", "{}"))
        except Exception as e:
            print(f"[Sheets] Error loading portfolio: {e}")

        return {"balance": 400.0, "initial": 400.0, "ghost_balance": 400.0,
                "holdings": {}, "trades_today": 0, "paused": False}

    def save_portfolio_to_sheets(self, portfolio):
        """Save portfolio state to Google Sheets."""
        if not HAS_GSHEETS or not self.sheets:
            print("[Sheets] (no connection) Portfolio state:")
            print(json.dumps(portfolio, indent=2, default=str))
            return

        try:
            ws = self.sheets.worksheet("Portfolio")
            ws.append_row([
                datetime.now().isoformat(),
                SYSTEM_VERSION,
                json.dumps(portfolio, default=str),
            ])
        except Exception as e:
            print(f"[Sheets] Error saving: {e}")

    def log_trade(self, trade_data):
        """Log a trade to Google Sheets journal."""
        if not HAS_GSHEETS or not self.sheets:
            print(f"[Journal] {json.dumps(trade_data, indent=2, default=str)}")
            return

        try:
            ws = self.sheets.worksheet("Journal")
            ws.append_row([
                trade_data.get("timestamp", ""),
                SYSTEM_VERSION,
                trade_data.get("action", ""),
                trade_data.get("ticker", ""),
                trade_data.get("amount", 0),
                trade_data.get("price", 0),
                trade_data.get("conviction", ""),
                trade_data.get("regime", ""),
                trade_data.get("vix", 0),
                trade_data.get("rsi", 0),
                trade_data.get("agent_weights", ""),
                trade_data.get("decision", ""),
                trade_data.get("reasoning", ""),
            ])
        except Exception as e:
            print(f"[Journal] Error: {e}")

    def update_ghost_portfolio(self, portfolio):
        """Update ghost SPY buy-and-hold portfolio."""
        if "SPY" in self.close.columns:
            spy_prices = self.close["SPY"]
            if len(spy_prices) >= 2:
                spy_return = float(spy_prices.iloc[-1] / spy_prices.iloc[-2] - 1)
                ghost = portfolio.get("ghost_balance", portfolio.get("initial", 400))
                portfolio["ghost_balance"] = ghost * (1 + spy_return)
                portfolio["spy_today_pct"] = spy_return * 100
        return portfolio

    def update_holdings_prices(self, portfolio):
        """Update all holdings with current prices."""
        holdings = portfolio.get("holdings", {})
        today_pnl = 0

        for ticker, pos in holdings.items():
            if ticker in self.close.columns:
                current_price = float(self.close[ticker].iloc[-1])
                prev_price = float(self.close[ticker].iloc[-2]) if len(self.close[ticker]) > 1 else current_price
                shares = pos.get("shares", 0)
                cost_basis = pos.get("cost_basis", current_price)

                pos["current_price"] = current_price
                pos["market_value"] = shares * current_price
                pos["unrealized_pnl"] = shares * (current_price - cost_basis)
                pos["today_return"] = (current_price / prev_price - 1) * 100
                today_pnl += shares * (current_price - prev_price)

        # Update total balance
        cash = portfolio.get("cash", portfolio.get("balance", 400))
        invested = sum(h.get("market_value", 0) for h in holdings.values())
        portfolio["balance"] = cash + invested if holdings else cash
        portfolio["today_pnl"] = today_pnl
        portfolio["today_pct"] = (today_pnl / portfolio["balance"]) * 100 if portfolio["balance"] > 0 else 0

        return portfolio

    def run(self, replay_start=None, replay_end=None):
        """Main execution."""
        print("=" * 50)
        print(f"  SECTOR COMMAND — {self.mode.upper()} MODE")
        print(f"  Version: {SYSTEM_VERSION}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 50)

        # Load data and models
        self.load_data()
        self.load_models()

        if not self.models:
            msg = "ERROR: No trained models found. Run train.py first."
            print(msg)
            self.send_sms(msg)
            return

        # Check if paused
        portfolio = self.load_portfolio_from_sheets()
        if portfolio.get("paused", False):
            print("[Signal] System PAUSED. Send RESUME to reactivate.")
            return

        # Generate signals
        signals = self.get_current_signals()

        # Update portfolio prices
        portfolio = self.update_holdings_prices(portfolio)
        portfolio = self.update_ghost_portfolio(portfolio)

        # [REGIME CHANGE ALERT] Check if regime changed since last run
        prev_regime = portfolio.get("last_regime", "normal")
        current_regime = signals["regime"]
        regime_changed = prev_regime != current_regime
        portfolio["last_regime"] = current_regime

        if regime_changed:
            regime_alert = (f"⚠️ REGIME SHIFT: {prev_regime.upper()} → {current_regime.upper()}\n"
                           f"VIX: {signals['vix']:.1f} | Term: {signals['vix_term_label']}\n"
                           f"Consider defensive rebalance.\n\n")
        else:
            regime_alert = ""

        # [STREAK TRACKING] Track model accuracy
        streaks = portfolio.get("model_streaks", {"A2C": 0, "SAC": 0, "PPO": 0})
        # Update streaks based on whether today's recommended direction was correct
        if len(self.returns) > 1:
            today_market = float(self.returns.iloc[-1].mean())  # Avg sector return
            for agent_name in streaks:
                if agent_name in signals.get("agent_weights", {}):
                    w = signals["agent_weights"][agent_name]
                    if w is not None:
                        agent_return = float(np.dot(w, self.returns.iloc[-1].values))
                        ew_return = today_market
                        if agent_return > ew_return:
                            streaks[agent_name] = max(0, streaks[agent_name]) + 1
                        else:
                            streaks[agent_name] = min(0, streaks[agent_name]) - 1
        portfolio["model_streaks"] = streaks
        signals["streaks"] = streaks

        # Determine message type based on time
        now = datetime.now()
        hour = now.hour

        if now.weekday() == 5 and hour < 12:  # Saturday morning
            message = self.format_weekly_summary(portfolio)
        elif hour >= 16:  # After market close
            message = self.format_eod_summary(signals, portfolio)
        else:  # During market hours
            message = regime_alert + self.format_sms(signals, portfolio)

        # Send
        print(f"\n[Message]\n{message}\n")
        self.send_sms(message)

        # Save state
        self.save_portfolio_to_sheets(portfolio)

        # Log run
        run_log = {
            "timestamp": datetime.now().isoformat(),
            "version": SYSTEM_VERSION,
            "mode": self.mode,
            "regime": signals["regime"],
            "vix": signals["vix"],
            "n_recommendations": len(signals["recommendations"]),
            "status": "success",
        }
        print(f"\n[Run Log] {json.dumps(run_log)}")

        return signals, portfolio


def main():
    parser = argparse.ArgumentParser(description="Sector Command Signal Server")
    parser.add_argument("--replay", nargs=2, metavar=("START", "END"),
                        help="Replay mode with date range")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without sending SMS")
    args = parser.parse_args()

    # Initialize Twilio
    twilio_client = None
    if HAS_TWILIO and not args.dry_run:
        sid = os.environ.get("TWILIO_SID")
        token = os.environ.get("TWILIO_TOKEN")
        if sid and token:
            twilio_client = TwilioClient(sid, token)

    # Initialize Google Sheets
    sheets_client = None
    if HAS_GSHEETS:
        creds_path = os.environ.get("GOOGLE_CREDS_PATH")
        if creds_path and os.path.exists(creds_path):
            creds = Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
            )
            gc = gspread.authorize(creds)
            sheets_client = gc.open("Sector Command")

    mode = "replay" if args.replay else "live"
    sc = SectorCommand(mode=mode, twilio_client=twilio_client, sheets_client=sheets_client)
    sc.run()


if __name__ == "__main__":
    main()
