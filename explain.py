"""
SHAP Explainability Module (Upgrade 4)

Generates feature importance and force plots explaining why the RL agent
made specific allocation decisions. Based on de-la-Rica-Escudero et al. (2025)
PLOS ONE methodology: PPO + SHAP for explainable portfolio management.

Usage:
    python explain.py                    # Explain best PPO model
    python explain.py --agent SAC        # Explain specific agent
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. Run: pip install shap")

from stable_baselines3 import PPO, A2C, SAC
from config import (
    MODEL_DIR, RESULTS_DIR, SEEDS, LOOKBACK_WINDOW, TICKERS,
    SECTOR_ETFS, WALK_FORWARD_WINDOWS, AGENTS
)
from data_loader import prepare_data
from portfolio_env import PortfolioEnv
from evaluate import _load_norm_stats

AGENT_CLASSES = {"PPO": PPO, "A2C": A2C, "SAC": SAC}


def collect_state_action_pairs(model, features, returns, regime_numeric,
                                norm_stats=None, max_steps=500):
    """Run agent and collect (state, action) pairs for SHAP analysis."""
    env = PortfolioEnv(features, returns, regime_numeric,
                       lookback=LOOKBACK_WINDOW, norm_stats=norm_stats,
                       random_start=False, tc_curriculum=False)
    obs, info = env.reset()

    states = []
    actions = []
    step = 0
    done = False

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        states.append(obs.copy())
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    return np.array(states), np.array(actions)


def build_feature_names(n_features, n_assets, feature_names, tickers, lookback):
    """Build human-readable feature names for the observation vector."""
    names = []

    # Flattened lookback window features
    for day in range(lookback):
        for feat in feature_names:
            for tick in tickers:
                sector = SECTOR_ETFS.get(tick, tick)
                names.append(f"d-{lookback - day}_{feat}_{sector[:4]}")

    # Current weights
    for tick in tickers:
        names.append(f"w_{SECTOR_ETFS.get(tick, tick)[:4]}")

    # Transition cost estimates
    for tick in tickers:
        names.append(f"tc_{SECTOR_ETFS.get(tick, tick)[:4]}")

    # Meta features
    names.extend(["regime", "rolling_sharpe", "drawdown", "tc_multiplier"])

    return names


def run_shap_analysis(agent_name="PPO", n_background=100, n_explain=50):
    """Run SHAP analysis on a trained agent."""
    if not HAS_SHAP:
        print("ERROR: Install shap first: pip install shap")
        return

    print("=" * 60)
    print(f"  SHAP EXPLAINABILITY — {agent_name}")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    print("\n[1/5] Loading data...")
    features, returns, close, regimes, regime_numeric, vix = prepare_data()
    sector_tickers = [t for t in TICKERS if t in returns.columns]

    # Load model
    print("\n[2/5] Loading model...")
    window = WALK_FORWARD_WINDOWS[-1]
    ts = pd.Timestamp
    test_feat = features.loc[ts(window[2]):ts(window[3])]
    test_ret = returns.loc[ts(window[2]):ts(window[3])]
    test_regime = regime_numeric.loc[ts(window[2]):ts(window[3])]

    model = None
    norm_stats = None
    w_idx = len(WALK_FORWARD_WINDOWS) - 1
    for seed in SEEDS:
        run_id = f"{agent_name}_w{w_idx}_s{seed}"
        for suffix in ["best_model", f"{run_id}_final"]:
            path = os.path.join(MODEL_DIR, run_id, f"{suffix}.zip")
            if os.path.exists(path):
                AgentClass = AGENT_CLASSES[AGENTS[agent_name]["class"]]
                model = AgentClass.load(path)
                norm_stats = _load_norm_stats(run_id)
                print(f"  Loaded: {path}")
                break
        if model:
            break

    if model is None:
        print(f"  ERROR: No model found for {agent_name}")
        return

    # Collect state-action pairs
    print("\n[3/5] Collecting state-action pairs...")
    states, actions = collect_state_action_pairs(
        model, test_feat, test_ret[sector_tickers], test_regime,
        norm_stats=norm_stats
    )
    print(f"  Collected {len(states)} state-action pairs")

    # Build feature names
    env_temp = PortfolioEnv(test_feat, test_ret[sector_tickers], test_regime,
                             lookback=LOOKBACK_WINDOW, norm_stats=norm_stats,
                             random_start=False, tc_curriculum=False)
    feat_names = build_feature_names(
        env_temp.num_features, env_temp.num_assets,
        env_temp.feature_names, env_temp.tickers, LOOKBACK_WINDOW
    )

    # Truncate feature names to match actual obs dim
    if len(feat_names) > states.shape[1]:
        feat_names = feat_names[:states.shape[1]]
    elif len(feat_names) < states.shape[1]:
        feat_names.extend([f"feat_{i}" for i in range(len(feat_names), states.shape[1])])

    # Create prediction function (maps state → action)
    def predict_fn(X):
        actions = []
        for x in X:
            action, _ = model.predict(x, deterministic=True)
            actions.append(action)
        return np.array(actions)

    # Run SHAP
    print("\n[4/5] Computing SHAP values (this takes a few minutes)...")
    background = states[:n_background]
    explain_data = states[:n_explain]

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain_data, nsamples=100)

    # Generate plots
    print("\n[5/5] Generating plots...")

    # Summary plot (mean |SHAP| across all decisions)
    # shap_values is (n_outputs, n_samples, n_features) for multi-output
    if isinstance(shap_values, list):
        # Average across output dimensions (sectors), then across samples
        stacked = np.array([np.abs(sv) for sv in shap_values])  # (n_outputs, n_samples, n_features)
        mean_shap = np.mean(stacked, axis=0)  # (n_samples, n_features)
    else:
        mean_shap = np.abs(shap_values)

    # Ensure mean_shap is 2D (n_samples, n_features)
    if mean_shap.ndim > 2:
        mean_shap = mean_shap.reshape(mean_shap.shape[0], -1)

    # Top 30 most important features
    mean_importance = np.mean(mean_shap, axis=0).flatten()  # (n_features,) guaranteed 1D
    top_k = min(30, len(mean_importance))
    top_idx = np.argsort(mean_importance)[-top_k:][::-1].flatten()
    top_names = [feat_names[idx] if idx < len(feat_names) else f"feat_{idx}" for idx in top_idx.tolist()]
    top_values = mean_importance[top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_k), top_values[::-1], color="#2563eb", alpha=0.8)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"{agent_name} — Top {top_k} Feature Importances (SHAP)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"shap_summary_{agent_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/shap_summary_{agent_name}.png")

    # Feature group importance (aggregate SHAP by feature type)
    group_importance = {}
    for i, name in enumerate(feat_names):
        if i >= len(mean_importance):
            break
        # Extract feature type from name (e.g., "d-5_rsi_Tech" → "rsi")
        parts = name.split("_")
        if len(parts) >= 2 and parts[0].startswith("d-"):
            feat_type = parts[1]
        elif parts[0].startswith("w_"):
            feat_type = "current_weights"
        elif parts[0].startswith("tc_"):
            feat_type = "transition_cost"
        else:
            feat_type = name
        group_importance[feat_type] = group_importance.get(feat_type, 0) + mean_importance[i]

    sorted_groups = sorted(group_importance.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    g_names = [g[0] for g in sorted_groups[:15]]
    g_vals = [g[1] for g in sorted_groups[:15]]
    ax.barh(range(len(g_names)), g_vals[::-1], color="#10b981", alpha=0.8)
    ax.set_yticks(range(len(g_names)))
    ax.set_yticklabels(g_names[::-1], fontsize=10)
    ax.set_xlabel("Aggregated |SHAP Value|")
    ax.set_title(f"{agent_name} — Feature Group Importance (SHAP)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"shap_groups_{agent_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/shap_groups_{agent_name}.png")

    print(f"\n  Top 5 feature groups driving {agent_name} decisions:")
    for name, val in sorted_groups[:5]:
        print(f"    {name}: {val:.4f}")

    print("\n  SHAP analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP Explainability for RL Portfolio Optimizer")
    parser.add_argument("--agent", type=str, default="PPO", choices=list(AGENTS.keys()))
    args = parser.parse_args()
    run_shap_analysis(agent_name=args.agent)
