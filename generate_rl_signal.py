#!/usr/bin/env python3
"""
generate_rl_signal.py — RL inference bridge for Sector Command Live

Loads the production RL ensemble (PPO/A2C/SAC), runs today's feature vector,
and writes the decision JSON that sector-command-live reads via RL_SIGNAL_JSON.

Usage:
    python generate_rl_signal.py                       # write to default path
    python generate_rl_signal.py --dry-run             # print, don't write
    python generate_rl_signal.py --out /path/to/signal.json

Default out path: ../sector-command-live/data/rl_signal.json
(set RL_SIGNAL_OUT env var to override)

Run from rl-portfolio-optimizer/ with the venv active:
    source rl_env/bin/activate
    caffeinate -i python generate_rl_signal.py
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime, timezone

# Production models: ordered by priority (A2C performed best in v4 eval)
PRODUCTION_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models-live")
PRODUCTION_RUNS = ["A2C_w2_s42", "SAC_w2_s42", "PPO_w2_s42"]

# How many recent data rows to feed the agent (lookback=60 + buffer for steps)
INFERENCE_SLICE = 250

# Agent must exceed EW × this factor to earn a BUY vote
BUY_THRESHOLD_FACTOR = 1.10


def _load_agents():
    from stable_baselines3 import PPO, A2C, SAC
    CLASSES = {"PPO": PPO, "A2C": A2C, "SAC": SAC}
    agents, norm_stats = {}, {}

    for run_id in PRODUCTION_RUNS:
        agent_name = run_id.split("_")[0]
        run_dir = os.path.join(PRODUCTION_MODEL_DIR, run_id)
        if not os.path.isdir(run_dir):
            print(f"  ✗ {run_id}: directory not found ({run_dir})")
            continue

        for fname in [f"{run_id}_final.zip", "best_model.zip"]:
            model_path = os.path.join(run_dir, fname)
            if os.path.exists(model_path):
                try:
                    agents[agent_name] = CLASSES[agent_name].load(model_path)
                    print(f"  ✓ {agent_name} ← {fname}")
                except Exception as e:
                    print(f"  ✗ {agent_name} load failed: {e}")
                break

        npz = os.path.join(run_dir, "norm_stats.npz")
        if os.path.exists(npz):
            d = np.load(npz)
            norm_stats[agent_name] = (d["means"], d["stds"])

    return agents, norm_stats


def _run_inference(agents, norm_stats, features, returns, regime_numeric):
    """Run each agent through the data slice; return {agent: final_weights}."""
    from config import LOOKBACK_WINDOW
    from evaluate import run_agent
    results = {}
    for agent_name, model in agents.items():
        try:
            _, weight_hist = run_agent(
                model, features, returns, regime_numeric,
                norm_stats=norm_stats.get(agent_name),
            )
            if len(weight_hist) > 0:
                results[agent_name] = np.array(weight_hist[-1])
                top_idx = int(np.argmax(results[agent_name]))
                print(f"  {agent_name}: top weight → index {top_idx} ({float(results[agent_name][top_idx]):.1%})")
            else:
                print(f"  {agent_name}: no weights returned")
        except Exception as e:
            print(f"  {agent_name} inference failed: {e}")
    return results


def _build_signal(weights_by_agent, tickers, vix, regime, features):
    """Turn per-agent weight arrays into the JSON signal dict."""
    n = len(tickers)
    ew = 1.0 / n

    # Each agent votes: if its top-allocation sector exceeds EW × factor → BUY vote
    votes = {}
    for agent_name, w in weights_by_agent.items():
        if w is None:
            continue
        w = np.array(w)
        top_idx = int(np.argmax(w))
        top_ticker = tickers[top_idx]
        if float(w[top_idx]) > ew * BUY_THRESHOLD_FACTOR:
            votes[agent_name] = f"BUY {top_ticker}"
        else:
            votes[agent_name] = "HOLD"

    # Count BUY votes per ticker
    buy_counts = {}
    for v in votes.values():
        if v.startswith("BUY"):
            t = v.split()[1]
            buy_counts[t] = buy_counts.get(t, 0) + 1

    if buy_counts:
        target = max(buy_counts, key=buy_counts.get)
        action = "BUY"
    else:
        # All agents holding → abstain to broad market
        target = "SPY" if "SPY" in tickers else tickers[0]
        action = "BUY"
        buy_counts[target] = 0

    # Confidence: based on avg overweight of the top pick across agents
    target_idx = tickers.index(target) if target in tickers else -1
    if target_idx >= 0:
        weights_for_target = [float(w[target_idx]) for w in weights_by_agent.values() if w is not None]
        avg_w = float(np.mean(weights_for_target)) if weights_for_target else ew
        avg_overweight = max(0.0, avg_w - ew)
        # Scale: 0 overweight → 40% confidence, 2× ew overweight → 90% confidence
        confidence = min(95, int(40 + (avg_overweight / ew) * 25))
        current_weight = round(avg_w * 100, 1)
    else:
        confidence = 40
        current_weight = 0.0

    # Pull RSI and rel_strength from the feature matrix (live values)
    rsi = None
    rel_strength = None
    if target_idx >= 0 and features is not None and target in tickers:
        try:
            rsi_raw = float(features[("rsi", target)].iloc[-1])
            rsi = round(rsi_raw * 100, 1)  # stored as 0-1 in features
        except Exception:
            pass
        try:
            rel_strength = round(float(features[("rel_strength", target)].iloc[-1]), 3)
        except Exception:
            pass

    # Determine regime string
    regime_str = str(regime).upper() if regime else "NORMAL"
    if regime_str in ("CALM",):
        regime_label = "CALM"
    elif regime_str in ("STRESSED",):
        regime_label = "STRESSED"
    else:
        regime_label = "NORMAL"

    return {
        "votes": votes,
        "target": target,
        "action": action,
        "confidence": confidence,
        "current_weight": current_weight,
        "rsi": rsi,
        "rel_strength": rel_strength,
        "vix": round(float(vix), 1),
        "regime": regime_label,
        "ghost_alpha": 0.0,
        "_source": "LIVE",
        "_generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "_models": " / ".join(PRODUCTION_RUNS),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_out = os.environ.get(
        "RL_SIGNAL_OUT",
        os.path.join(script_dir, "../sector-command-live/data/rl_signal.json"),
    )

    parser = argparse.ArgumentParser(description="Generate RL signal JSON for Sector Command Live")
    parser.add_argument("--out", default=default_out, help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true", help="Print instead of writing")
    args = parser.parse_args()

    sys.path.insert(0, script_dir)

    # ── 1. Load data ───────────────────────────────────────────────
    print("[RL Signal] Fetching market data...")
    from data_loader import prepare_data
    from config import TICKERS, LOOKBACK_WINDOW

    features, returns, close, regimes, regime_numeric, vix_series = prepare_data()

    # Slice to recent data — enough for lookback + inference steps
    features = features.iloc[-INFERENCE_SLICE:]
    returns = returns.loc[features.index]
    regime_numeric = regime_numeric.loc[features.index]
    regimes_slice = regimes.loc[features.index]

    current_regime = str(regimes_slice.iloc[-1])
    current_vix = float(vix_series.iloc[-1])
    sector_tickers = [t for t in TICKERS if t in returns.columns]

    print(f"  Data slice: {features.index[0].date()} → {features.index[-1].date()}")
    print(f"  Regime: {current_regime.upper()} | VIX: {current_vix:.1f}")

    # ── 2. Load agents ─────────────────────────────────────────────
    print("[RL Signal] Loading agents...")
    agents, norm_stats = _load_agents()

    if not agents:
        print("ERROR: No agents loaded. Check models-live/ directory.")
        sys.exit(1)

    # Detect model action dim from first loaded agent; trim tickers if needed.
    # Existing 11-asset models still work after BTC/ETH are added to config.
    sample_model = next(iter(agents.values()))
    model_n_assets = sample_model.action_space.shape[0]
    if model_n_assets < len(sector_tickers):
        print(f"  Model has {model_n_assets}-asset action space; "
              f"trimming tickers from {len(sector_tickers)} → {model_n_assets} "
              f"(retrain to use full universe)")
        sector_tickers = sector_tickers[:model_n_assets]
        # Also drop BTC/ETH columns from features so obs-dim matches model
        keep_tickers = set(sector_tickers)
        cols_to_keep = [c for c in features.columns
                        if c[1] in keep_tickers]
        features = features[cols_to_keep]

    # ── 3. Run inference ───────────────────────────────────────────
    print("[RL Signal] Running ensemble inference...")
    weights_by_agent = _run_inference(
        agents, norm_stats,
        features,
        returns[sector_tickers],
        regime_numeric,
    )

    if not weights_by_agent:
        print("ERROR: All inference attempts failed.")
        sys.exit(1)

    # ── 4. Build signal ────────────────────────────────────────────
    signal = _build_signal(weights_by_agent, sector_tickers, current_vix, current_regime, features)

    print(f"\n[RL Signal] Decision: {signal['action']} {signal['target']} "
          f"@ {signal['confidence']}% confidence ({current_regime.upper()})")
    print(f"  Votes: {signal['votes']}")
    if signal.get('rsi'):
        print(f"  {signal['target']} RSI={signal['rsi']}  rel_strength={signal['rel_strength']}")

    # ── 5. Output ──────────────────────────────────────────────────
    if args.dry_run:
        print("\n[RL Signal] DRY RUN output:")
        print(json.dumps(signal, indent=2))
    else:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(signal, f, indent=2)
        print(f"\n[RL Signal] Written → {out_path}")

    return signal


if __name__ == "__main__":
    main()
