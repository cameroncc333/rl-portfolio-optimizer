"""
Training Pipeline — Multi-Agent, Walk-Forward, Seed-Robust

FIXES vs v2:
- VecNormalize eval env uses training=False and syncs stats from train env
- Normalization stats from train env are passed to test env (no look-ahead)
- Walk-forward windows use 4-tuple (train_start, train_end, test_start, test_end)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from config import (
    AGENTS, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES,
    SEEDS, MODEL_DIR, LOG_DIR, RESULTS_DIR, LOOKBACK_WINDOW,
    TICKERS, WALK_FORWARD_WINDOWS
)
from data_loader import prepare_data
from portfolio_env import PortfolioEnv

AGENT_CLASSES = {"PPO": PPO, "A2C": A2C, "SAC": SAC}


class PortfolioMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                self.episode_rewards.append(info["episode"]["r"])
                if self.verbose > 0 and self.episode_count % 5 == 0:
                    recent = np.mean(self.episode_rewards[-5:])
                    print(f"    Ep {self.episode_count}: avg_reward={recent:.2f}")
        return True


def make_env(features, returns, regime_numeric, norm_stats=None, spy_returns=None):
    """Factory for monitored portfolio environment with optional norm stats."""
    def _init():
        env = PortfolioEnv(
            features, returns, regime_numeric,
            lookback=LOOKBACK_WINDOW,
            norm_stats=norm_stats,
            spy_returns=spy_returns,
        )
        env = Monitor(env)
        return env
    return _init


def split_by_dates(features, returns, regime_numeric,
                    train_start, train_end, test_start, test_end,
                    spy_returns=None):
    """Split data for walk-forward: non-overlapping train and test."""
    ts = pd.Timestamp
    train_feat = features.loc[ts(train_start):ts(train_end)]
    train_ret = returns.loc[ts(train_start):ts(train_end)]
    train_regime = regime_numeric.loc[ts(train_start):ts(train_end)]

    test_feat = features.loc[ts(test_start):ts(test_end)]
    test_ret = returns.loc[ts(test_start):ts(test_end)]
    test_regime = regime_numeric.loc[ts(test_start):ts(test_end)]

    if spy_returns is not None:
        train_spy = spy_returns.loc[ts(train_start):ts(train_end)]
        test_spy = spy_returns.loc[ts(test_start):ts(test_end)]
    else:
        train_spy = test_spy = None

    return (train_feat, train_ret, train_regime,
            test_feat, test_ret, test_regime,
            train_spy, test_spy)


def train_single(agent_name, train_feat, train_ret, train_regime,
                 test_feat, test_ret, test_regime,
                 seed, timesteps, window_id, verbose=1,
                 train_spy=None, test_spy=None):
    """Train a single agent. Returns model path and norm stats."""

    agent_cfg = AGENTS[agent_name]
    AgentClass = AGENT_CLASSES[agent_cfg["class"]]
    config = agent_cfg["config"].copy()

    # Create train environment and extract normalization stats
    train_env_raw = PortfolioEnv(train_feat, train_ret, train_regime,
                                  lookback=LOOKBACK_WINDOW, spy_returns=train_spy)
    train_norm_stats = train_env_raw.get_norm_stats()

    # Create vectorized envs — eval env uses TRAIN normalization stats
    train_env = DummyVecEnv([make_env(train_feat, train_ret, train_regime,
                                       spy_returns=train_spy)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(test_feat, test_ret, test_regime,
                                      norm_stats=train_norm_stats, spy_returns=test_spy)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                             clip_obs=10.0, training=False)

    # Directories
    run_id = f"{agent_name}_w{window_id}_s{seed}"
    run_model_dir = os.path.join(MODEL_DIR, run_id)
    run_log_dir = os.path.join(LOG_DIR, run_id)
    os.makedirs(run_model_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    model = AgentClass(
        "MlpPolicy", train_env, verbose=0, seed=seed,
        tensorboard_log=run_log_dir, **config,
    )

    callbacks = CallbackList([
        PortfolioMetricsCallback(verbose=verbose),
        EvalCallback(
            eval_env, best_model_save_path=run_model_dir,
            log_path=run_log_dir, eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES, deterministic=True, render=False,
        ),
        CheckpointCallback(
            save_freq=max(50_000, timesteps // 5),
            save_path=run_model_dir, name_prefix=run_id,
        ),
    ])

    model.learn(total_timesteps=timesteps, callback=callbacks,
                tb_log_name=run_id, progress_bar=False)

    # Save
    final_path = os.path.join(run_model_dir, f"{run_id}_final")
    model.save(final_path)
    train_env.save(os.path.join(run_model_dir, "vec_normalize.pkl"))

    # Save train norm stats for evaluation
    np.savez(os.path.join(run_model_dir, "norm_stats.npz"),
             means=train_norm_stats[0], stds=train_norm_stats[1])

    return final_path, run_id


def train_full(agents=None, timesteps=TOTAL_TIMESTEPS, seeds=None,
               windows=None, verbose=1):
    if agents is None:
        agents = list(AGENTS.keys())
    if seeds is None:
        seeds = SEEDS
    if windows is None:
        windows = WALK_FORWARD_WINDOWS

    print("=" * 70)
    print("  REGIME-ADAPTIVE RL PORTFOLIO OPTIMIZER — TRAINING")
    print("=" * 70)
    print(f"  Agents:    {agents}")
    print(f"  Seeds:     {seeds}")
    print(f"  Windows:   {len(windows)}")
    print(f"  Timesteps: {timesteps:,} per run")
    total_runs = len(agents) * len(seeds) * len(windows)
    print(f"  Total runs: {total_runs}")
    print("=" * 70)

    print("\n[1/3] Loading data...")
    features, returns, close, regimes, regime_numeric, vix = prepare_data()

    # SPY daily returns for relative reward shaping
    from config import BENCHMARK_TICKER
    spy_returns = close[BENCHMARK_TICKER].pct_change().fillna(0.0)
    spy_returns = spy_returns.loc[features.index]

    print("\n[2/3] Training agents...")
    results_log = []
    run_count = 0

    for w_idx, (t_start, t_end, v_start, v_end) in enumerate(windows):
        print(f"\n{'─' * 60}")
        print(f"  Window {w_idx + 1}: Train {t_start}→{t_end} | Test {v_start}→{v_end}")
        print(f"{'─' * 60}")

        (train_feat, train_ret, train_regime,
         test_feat, test_ret, test_regime,
         train_spy, test_spy) = split_by_dates(
            features, returns, regime_numeric, t_start, t_end, v_start, v_end,
            spy_returns=spy_returns,
        )

        print(f"  Train: {len(train_feat)} days | Test: {len(test_feat)} days")

        if len(train_feat) < LOOKBACK_WINDOW + 100 or len(test_feat) < LOOKBACK_WINDOW + 10:
            print("  ⚠ Insufficient data, skipping.")
            continue

        for agent_name in agents:
            for seed in seeds:
                run_count += 1
                print(f"\n  [{run_count}/{total_runs}] {agent_name} | "
                      f"seed={seed} | window={w_idx + 1}")
                try:
                    model_path, run_id = train_single(
                        agent_name, train_feat, train_ret, train_regime,
                        test_feat, test_ret, test_regime,
                        seed, timesteps, w_idx, verbose,
                        train_spy=train_spy, test_spy=test_spy,
                    )
                    results_log.append({
                        "agent": agent_name, "seed": seed, "window": w_idx,
                        "train_period": f"{t_start} → {t_end}",
                        "test_period": f"{v_start} → {v_end}",
                        "model_path": model_path, "run_id": run_id,
                        "status": "success",
                    })
                    print(f"    ✓ Saved: {model_path}")
                except Exception as e:
                    print(f"    ✗ FAILED: {e}")
                    results_log.append({
                        "agent": agent_name, "seed": seed, "window": w_idx,
                        "status": "failed", "error": str(e),
                    })

    print("\n[3/3] Saving manifest...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "training_manifest.json"), "w") as f:
        json.dump(results_log, f, indent=2)

    n_ok = sum(1 for r in results_log if r["status"] == "success")
    n_fail = sum(1 for r in results_log if r["status"] == "failed")
    print(f"\n{'=' * 70}")
    print(f"  COMPLETE — {n_ok} succeeded, {n_fail} failed")
    print(f"  TensorBoard: tensorboard --logdir {LOG_DIR}")
    print(f"{'=' * 70}")

    return results_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--agent", type=str, choices=list(AGENTS.keys()))
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    if args.full:
        train_full(timesteps=args.timesteps, verbose=args.verbose)
    elif args.agent:
        train_full(agents=[args.agent], timesteps=args.timesteps,
                   seeds=SEEDS[:args.seeds], verbose=args.verbose)
    else:
        train_full(agents=["PPO"], timesteps=args.timesteps,
                   seeds=SEEDS[:args.seeds],
                   windows=[WALK_FORWARD_WINDOWS[-1]], verbose=args.verbose)
