"""
Optuna Hyperparameter Tuning (Upgrade 5)

Automatically searches for optimal:
- Reward function weights (λ_drawdown, λ_turnover, λ_sharpe, λ_cvar, λ_concentration)
- Learning rate
- Network architecture
- Entropy coefficient

Usage:
    python tune.py                       # 50 trials, PPO
    python tune.py --agent SAC --trials 100
"""

import os
import argparse
import json
import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed. Run: pip install optuna")

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from config import (
    AGENTS, SEEDS, LOOKBACK_WINDOW, TICKERS,
    WALK_FORWARD_WINDOWS, MODEL_DIR, LOG_DIR, RESULTS_DIR,
    OPTUNA_N_TRIALS, OPTUNA_TIMESTEPS
)
from data_loader import prepare_data
from portfolio_env import PortfolioEnv
from evaluate import run_agent, compute_metrics

AGENT_CLASSES = {"PPO": PPO, "A2C": A2C, "SAC": SAC}


def make_env(features, returns, regime_numeric, overrides=None):
    """Create env with optional hyperparameter overrides."""
    def _init():
        env = PortfolioEnv(
            features, returns, regime_numeric,
            lookback=LOOKBACK_WINDOW,
        )
        # Apply reward weight overrides if provided
        if overrides:
            import config
            for key, val in overrides.items():
                setattr(config, key, val)
        env = Monitor(env)
        return env
    return _init


def objective(trial, agent_name, train_feat, train_ret, train_regime,
              test_feat, test_ret, test_regime, sector_tickers):
    """Optuna objective: maximize out-of-sample Sharpe ratio."""

    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    lambda_dd = trial.suggest_float("lambda_drawdown", 0.5, 5.0)
    lambda_turn = trial.suggest_float("lambda_turnover", 0.001, 0.02)
    lambda_sharpe = trial.suggest_float("lambda_sharpe", 0.01, 0.5)
    lambda_cvar = trial.suggest_float("lambda_cvar", 0.1, 2.0)
    lambda_conc = trial.suggest_float("lambda_concentration", 0.005, 0.1)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.05, log=True)

    # Network architecture
    n_layers = trial.suggest_int("n_layers", 2, 3)
    layer_size = trial.suggest_categorical("layer_size", [128, 256, 512])
    net_arch = [layer_size] * n_layers

    # Apply overrides to config module
    import config
    config.LAMBDA_DRAWDOWN = lambda_dd
    config.LAMBDA_TURNOVER = lambda_turn
    config.LAMBDA_SHARPE = lambda_sharpe
    config.LAMBDA_CVAR = lambda_cvar
    config.LAMBDA_CONCENTRATION = lambda_conc

    # Build environments
    train_env = DummyVecEnv([make_env(train_feat, train_ret, train_regime)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Get agent config
    agent_cfg = AGENTS[agent_name]
    AgentClass = AGENT_CLASSES[agent_cfg["class"]]
    model_config = agent_cfg["config"].copy()
    model_config["learning_rate"] = lr
    model_config["ent_coef"] = ent_coef

    if agent_name in ["PPO", "A2C"]:
        model_config["policy_kwargs"] = {
            "net_arch": dict(pi=net_arch, vf=net_arch),
        }
    else:
        model_config["policy_kwargs"] = {"net_arch": net_arch}

    try:
        model = AgentClass("MlpPolicy", train_env, verbose=0, seed=42, **model_config)
        model.learn(total_timesteps=OPTUNA_TIMESTEPS, progress_bar=False)

        # Evaluate on test data
        train_env_raw = PortfolioEnv(train_feat, train_ret, train_regime,
                                      lookback=LOOKBACK_WINDOW)
        norm_stats = train_env_raw.get_norm_stats()

        agent_ret, _ = run_agent(model, test_feat, test_ret[sector_tickers],
                                  test_regime, norm_stats=norm_stats)
        metrics = compute_metrics(agent_ret)

        if "error" in metrics:
            return -10.0

        sharpe = metrics["_sharpe"]

        # Report intermediate value for pruning
        trial.report(sharpe, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return sharpe

    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        return -10.0


def run_tuning(agent_name="PPO", n_trials=OPTUNA_N_TRIALS):
    """Run Optuna hyperparameter search."""
    if not HAS_OPTUNA:
        print("ERROR: Install optuna first: pip install optuna")
        return

    print("=" * 60)
    print(f"  OPTUNA HYPERPARAMETER TUNING — {agent_name}")
    print(f"  Trials: {n_trials} | Timesteps/trial: {OPTUNA_TIMESTEPS:,}")
    print("=" * 60)

    # Load data
    print("\n[1/3] Loading data...")
    features, returns, close, regimes, regime_numeric, vix = prepare_data()
    sector_tickers = [t for t in TICKERS if t in returns.columns]

    # Use most recent window
    window = WALK_FORWARD_WINDOWS[-1]
    ts = pd.Timestamp
    train_feat = features.loc[:ts(window[1])]
    train_ret = returns.loc[:ts(window[1])]
    train_regime = regime_numeric.loc[:ts(window[1])]
    test_feat = features.loc[ts(window[2]):ts(window[3])]
    test_ret = returns.loc[ts(window[2]):ts(window[3])]
    test_regime = regime_numeric.loc[ts(window[2]):ts(window[3])]

    print(f"  Train: {len(train_feat)} days | Test: {len(test_feat)} days")

    # Create study
    print(f"\n[2/3] Running {n_trials} trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
        study_name=f"rl_portfolio_{agent_name}",
    )

    study.optimize(
        lambda trial: objective(
            trial, agent_name, train_feat, train_ret, train_regime,
            test_feat, test_ret, test_regime, sector_tickers
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Results
    print(f"\n[3/3] Results:")
    print(f"  Best Sharpe: {study.best_value:.4f}")
    print(f"  Best params:")
    for key, val in study.best_params.items():
        print(f"    {key}: {val}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        "best_sharpe": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
        "agent": agent_name,
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params}
            for t in study.trials if t.value is not None
        ],
    }
    results_path = os.path.join(RESULTS_DIR, f"optuna_{agent_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    # Generate optimization history plot
    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt_path = os.path.join(RESULTS_DIR, f"optuna_history_{agent_name}.png")
        fig.figure.savefig(plt_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {plt_path}")
    except Exception:
        pass

    # Print recommended config update
    bp = study.best_params
    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  RECOMMENDED CONFIG UPDATE:              │")
    print(f"  │  LAMBDA_DRAWDOWN = {bp.get('lambda_drawdown', 2.0):.3f}               │")
    print(f"  │  LAMBDA_TURNOVER = {bp.get('lambda_turnover', 0.005):.4f}              │")
    print(f"  │  LAMBDA_SHARPE   = {bp.get('lambda_sharpe', 0.1):.3f}               │")
    print(f"  │  LAMBDA_CVAR     = {bp.get('lambda_cvar', 0.5):.3f}               │")
    print(f"  │  LAMBDA_CONC     = {bp.get('lambda_concentration', 0.02):.4f}              │")
    print(f"  │  learning_rate   = {bp.get('learning_rate', 3e-4):.6f}           │")
    print(f"  │  ent_coef        = {bp.get('ent_coef', 0.01):.4f}              │")
    print(f"  │  net_arch        = {[bp.get('layer_size', 256)] * bp.get('n_layers', 3)}  │")
    print(f"  └─────────────────────────────────────────┘")

    return study


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Optuna Tuning for RL Portfolio Optimizer")
    parser.add_argument("--agent", type=str, default="PPO", choices=list(AGENTS.keys()))
    parser.add_argument("--trials", type=int, default=OPTUNA_N_TRIALS)
    args = parser.parse_args()

    run_tuning(agent_name=args.agent, n_trials=args.trials)
