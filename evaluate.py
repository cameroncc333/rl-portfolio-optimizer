"""
Evaluation & Backtesting Framework

FIXES vs v2:
- True risk parity (equalize risk contributions via covariance, not just inverse vol)
- Information ratio and tracking error added
- Ablation uses zero-mask (obs dimension stays constant)
- Loads train normalization stats for test evaluation (no look-ahead)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO, A2C, SAC

from config import (
    MODEL_DIR, RESULTS_DIR, SEEDS, LOOKBACK_WINDOW, TICKERS, SECTOR_ETFS,
    WALK_FORWARD_WINDOWS, AGENTS, BOOTSTRAP_SAMPLES, CONFIDENCE_LEVEL,
    RISK_FREE_RATE, ABLATION_GROUPS, NUM_ASSETS
)
from data_loader import prepare_data
from portfolio_env import PortfolioEnv

AGENT_CLASSES = {"PPO": PPO, "A2C": A2C, "SAC": SAC}


# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(returns_series, benchmark_returns=None, rf=RISK_FREE_RATE):
    """Full performance metrics. If benchmark provided, also computes IR."""
    if len(returns_series) < 2:
        return {"error": "Insufficient data"}

    r = returns_series.values if hasattr(returns_series, 'values') else np.array(returns_series)

    total_return = np.prod(1 + r) - 1
    n_years = max(len(r) / 252, 0.01)
    cagr = (1 + total_return) ** (1 / n_years) - 1
    annual_vol = np.std(r) * np.sqrt(252)
    sharpe = (cagr - rf) / (annual_vol + 1e-8)

    downside = r[r < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-8
    sortino = (cagr - rf) / (downside_vol + 1e-8)

    cumulative = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_dd = np.min(drawdown)
    calmar = cagr / (abs(max_dd) + 1e-8)
    win_rate = np.mean(r > 0)

    result = {
        "Total Return": f"{total_return:.2%}",
        "CAGR": f"{cagr:.2%}",
        "Annual Vol": f"{annual_vol:.2%}",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Calmar": f"{calmar:.2f}",
        "Win Rate": f"{win_rate:.2%}",
        "Days": len(r),
        "_cagr": cagr, "_sharpe": sharpe, "_sortino": sortino,
        "_max_dd": max_dd, "_calmar": calmar, "_annual_vol": annual_vol,
        "_cumulative": pd.Series(cumulative, index=getattr(returns_series, 'index', None)),
        "_drawdown": pd.Series(drawdown, index=getattr(returns_series, 'index', None)),
    }

    # Information ratio vs benchmark
    if benchmark_returns is not None:
        b = benchmark_returns.values if hasattr(benchmark_returns, 'values') else np.array(benchmark_returns)
        min_l = min(len(r), len(b))
        active = r[:min_l] - b[:min_l]
        tracking_error = np.std(active) * np.sqrt(252)
        active_return = np.mean(active) * 252
        info_ratio = active_return / (tracking_error + 1e-8)
        result["Info Ratio"] = f"{info_ratio:.2f}"
        result["Tracking Error"] = f"{tracking_error:.2%}"
        result["_info_ratio"] = info_ratio
        result["_tracking_error"] = tracking_error

    return result


def bootstrap_sharpe_ci(returns_series, n_bootstrap=BOOTSTRAP_SAMPLES,
                         confidence=CONFIDENCE_LEVEL, rf=RISK_FREE_RATE):
    r = returns_series.values if hasattr(returns_series, 'values') else np.array(returns_series)
    n = len(r)
    rng = np.random.RandomState(42)
    sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(r, size=n, replace=True)
        ann_ret = np.mean(sample) * 252
        ann_vol = np.std(sample) * np.sqrt(252) + 1e-8
        sharpes[i] = (ann_ret - rf) / ann_vol

    alpha = (1 - confidence) / 2
    return np.mean(sharpes), np.percentile(sharpes, alpha * 100), np.percentile(sharpes, (1 - alpha) * 100)


# ═══════════════════════════════════════════════════════════════════
# BENCHMARKS
# ═══════════════════════════════════════════════════════════════════

def equal_weight(ret_df):
    w = np.ones(ret_df.shape[1]) / ret_df.shape[1]
    return pd.Series(ret_df.values @ w, index=ret_df.index, name="Equal Weight")


def risk_parity(ret_df, lookback=60, rebal_freq=20):
    """TRUE risk parity: equalize marginal risk contributions via covariance.
    NOT just inverse-volatility weighting."""
    n = ret_df.shape[1]
    result = []
    w = np.ones(n) / n

    for i in range(len(ret_df)):
        if i % rebal_freq == 0 and i >= lookback:
            recent = ret_df.iloc[max(0, i - lookback):i].values
            cov = np.cov(recent.T) + np.eye(n) * 1e-8

            # Iterative risk-parity: minimize sum( (w_i * (Cov @ w)_i - target_risk)^2 )
            # Simple approximation: w_i ∝ 1 / sqrt(Cov_ii * (Cov @ w_prev)_i)
            for _ in range(10):  # Iterate to convergence
                port_vol_contrib = cov @ w
                marginal_risk = w * port_vol_contrib
                target = np.mean(marginal_risk)
                # Adjust weights inversely proportional to their risk contribution
                adjustment = target / (marginal_risk + 1e-10)
                w = w * adjustment
                w = np.clip(w, 0, 1)
                w /= w.sum() + 1e-8

        result.append(ret_df.iloc[i].values @ w)

    return pd.Series(result, index=ret_df.index, name="Risk Parity")


def momentum_top3(ret_df, lookback=20, rebal_freq=20):
    n = ret_df.shape[1]
    result = []
    w = np.ones(n) / n
    for i in range(len(ret_df)):
        if i % rebal_freq == 0 and i >= lookback:
            trailing = ret_df.iloc[max(0, i - lookback):i].sum()
            top_k = trailing.nlargest(3).index
            w = np.zeros(n)
            for t in top_k:
                w[ret_df.columns.get_loc(t)] = 1.0 / 3
        result.append(ret_df.iloc[i].values @ w)
    return pd.Series(result, index=ret_df.index, name="Momentum Top-3")


def min_variance(ret_df, lookback=60, rebal_freq=20):
    n = ret_df.shape[1]
    result = []
    w = np.ones(n) / n
    for i in range(len(ret_df)):
        if i % rebal_freq == 0 and i >= lookback:
            recent = ret_df.iloc[max(0, i - lookback):i].values
            cov = np.cov(recent.T) + np.eye(n) * 1e-6
            try:
                cov_inv = np.linalg.inv(cov)
                ones = np.ones(n)
                w = cov_inv @ ones / (ones @ cov_inv @ ones)
                w = np.clip(w, 0, 1)
                w /= w.sum() + 1e-8
            except np.linalg.LinAlgError:
                pass
        result.append(ret_df.iloc[i].values @ w)
    return pd.Series(result, index=ret_df.index, name="Min Variance")


def inverse_volatility(ret_df, lookback=60, rebal_freq=20):
    n = ret_df.shape[1]
    result = []
    w = np.ones(n) / n
    for i in range(len(ret_df)):
        if i % rebal_freq == 0 and i >= lookback:
            vols = ret_df.iloc[max(0, i - lookback):i].std().values + 1e-8
            inv = 1.0 / vols
            w = inv / inv.sum()
        result.append(ret_df.iloc[i].values @ w)
    return pd.Series(result, index=ret_df.index, name="Inverse Volatility")


def spy_buy_hold(close_df, index):
    return close_df["SPY"].pct_change().loc[index].fillna(0).rename("SPY Buy & Hold")


# ═══════════════════════════════════════════════════════════════════
# AGENT EVALUATION
# ═══════════════════════════════════════════════════════════════════

def _load_norm_stats(run_id):
    """Load training normalization stats for a given run."""
    path = os.path.join(MODEL_DIR, run_id, "norm_stats.npz")
    if os.path.exists(path):
        data = np.load(path)
        return (data["means"], data["stds"])
    return None


def run_agent(model, features, returns, regime_numeric,
              norm_stats=None, deterministic=True):
    """Execute agent on data with proper normalization."""
    env = PortfolioEnv(features, returns, regime_numeric,
                       lookback=LOOKBACK_WINDOW, norm_stats=norm_stats)
    obs, info = env.reset()

    port_returns, weight_hist = [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        port_returns.append(info["portfolio_return"])
        weight_hist.append(info["weights"])

    start = LOOKBACK_WINDOW + 1
    valid_idx = returns.index[start:start + len(port_returns)]
    min_l = min(len(valid_idx), len(port_returns))

    return (
        pd.Series(port_returns[:min_l], index=valid_idx[:min_l]),
        np.array(weight_hist[:min_l]),
    )


# ═══════════════════════════════════════════════════════════════════
# SEED ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════

def seed_robustness_analysis(agent_name, features, returns, regime_numeric,
                              close, window_idx=2):
    results = {}
    window = WALK_FORWARD_WINDOWS[window_idx]
    ts = pd.Timestamp

    test_feat = features.loc[ts(window[2]):ts(window[3])]
    test_ret = returns.loc[ts(window[2]):ts(window[3])]
    test_regime = regime_numeric.loc[ts(window[2]):ts(window[3])]
    sector_tickers = [t for t in TICKERS if t in test_ret.columns]

    for seed in SEEDS:
        run_id = f"{agent_name}_w{window_idx}_s{seed}"
        model_path = os.path.join(MODEL_DIR, run_id, f"{run_id}_final.zip")
        if not os.path.exists(model_path):
            continue

        norm_stats = _load_norm_stats(run_id)
        AgentClass = AGENT_CLASSES[AGENTS[agent_name]["class"]]
        model = AgentClass.load(model_path)

        agent_ret, weights = run_agent(model, test_feat, test_ret[sector_tickers],
                                        test_regime, norm_stats=norm_stats)
        metrics = compute_metrics(agent_ret)
        results[seed] = metrics

    if not results:
        return None

    sharpes = [m["_sharpe"] for m in results.values()]
    cagrs = [m["_cagr"] for m in results.values()]
    max_dds = [m["_max_dd"] for m in results.values()]

    return {
        "agent": agent_name, "n_seeds": len(results),
        "sharpe_mean": np.mean(sharpes), "sharpe_std": np.std(sharpes),
        "cagr_mean": np.mean(cagrs), "cagr_std": np.std(cagrs),
        "max_dd_mean": np.mean(max_dds), "max_dd_std": np.std(max_dds),
        "seed_results": results,
    }


# ═══════════════════════════════════════════════════════════════════
# ABLATION (ZERO-MASK, NOT DIMENSION REMOVAL)
# ═══════════════════════════════════════════════════════════════════

def run_ablation(model_path, agent_class_name, features, returns,
                 regime_numeric, norm_stats=None):
    """Zero-mask each feature group and re-evaluate. Obs dimension stays constant."""
    AgentClass = AGENT_CLASSES[agent_class_name]
    model = AgentClass.load(model_path)
    sector_tickers = [t for t in TICKERS if t in returns.columns]
    results = {}

    # Baseline
    base_ret, _ = run_agent(model, features, returns[sector_tickers],
                            regime_numeric, norm_stats=norm_stats)
    results["all_features"] = compute_metrics(base_ret)
    print(f"  Baseline Sharpe: {results['all_features']['Sharpe']}")

    for group_name, group_features in ABLATION_GROUPS.items():
        print(f"  Ablating: {group_name} ({group_features})")

        ablation_mask = {f: True for f in group_features}
        env = PortfolioEnv(features, returns[sector_tickers], regime_numeric,
                           lookback=LOOKBACK_WINDOW,
                           ablation_mask=ablation_mask,
                           norm_stats=norm_stats)
        obs, info = env.reset()

        port_returns = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            port_returns.append(info["portfolio_return"])

        start = LOOKBACK_WINDOW + 1
        valid_idx = returns.index[start:start + len(port_returns)]
        min_l = min(len(valid_idx), len(port_returns))
        ablated_ret = pd.Series(port_returns[:min_l], index=valid_idx[:min_l])
        results[f"no_{group_name}"] = compute_metrics(ablated_ret)

        sharpe_drop = results["all_features"]["_sharpe"] - results[f"no_{group_name}"]["_sharpe"]
        print(f"    Sharpe Δ: {sharpe_drop:+.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate(run_ablation_study=False):
    print("=" * 70)
    print("  REGIME-ADAPTIVE RL PORTFOLIO OPTIMIZER — EVALUATION")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[1/5] Loading data...")
    features, returns, close, regimes, regime_numeric, vix = prepare_data()
    sector_tickers = [t for t in TICKERS if t in returns.columns]

    window = WALK_FORWARD_WINDOWS[-1]
    ts = pd.Timestamp
    test_feat = features.loc[ts(window[2]):ts(window[3])]
    test_ret = returns.loc[ts(window[2]):ts(window[3])]
    test_ret_sectors = test_ret[sector_tickers]
    test_regime = regime_numeric.loc[ts(window[2]):ts(window[3])]

    print(f"  Test: {window[2]} → {window[3]} ({len(test_ret)} days)")

    # Load and run agents
    print("\n[2/5] Evaluating agents...")
    agent_results = {}
    agent_weights = {}
    w_idx = len(WALK_FORWARD_WINDOWS) - 1

    for agent_name in AGENTS:
        for seed in SEEDS:
            run_id = f"{agent_name}_w{w_idx}_s{seed}"
            for suffix in ["best_model", f"{run_id}_final"]:
                path = os.path.join(MODEL_DIR, run_id, f"{suffix}.zip")
                if os.path.exists(path):
                    print(f"  {agent_name}: {path}")
                    norm_stats = _load_norm_stats(run_id)
                    AgentClass = AGENT_CLASSES[AGENTS[agent_name]["class"]]
                    model = AgentClass.load(path)
                    ret, w = run_agent(model, test_feat, test_ret_sectors,
                                       test_regime, norm_stats=norm_stats)
                    agent_results[f"{agent_name} Agent"] = ret
                    agent_weights[agent_name] = w
                    break
            if agent_name in agent_weights:
                break

    # Benchmarks
    print("\n[3/5] Computing benchmarks...")
    ref_idx = list(agent_results.values())[0].index if agent_results else test_ret_sectors.index
    bench_ret = test_ret_sectors.loc[ref_idx]

    benchmarks = {
        "Equal Weight": equal_weight(bench_ret),
        "Risk Parity": risk_parity(bench_ret),
        "Momentum Top-3": momentum_top3(bench_ret),
        "Min Variance": min_variance(bench_ret),
        "Inverse Volatility": inverse_volatility(bench_ret),
    }
    if "SPY" in close.columns:
        benchmarks["SPY Buy & Hold"] = spy_buy_hold(close, ref_idx)

    min_len = min(
        *([len(v) for v in agent_results.values()] or [len(bench_ret)]),
        *[len(v) for v in benchmarks.values()]
    )

    all_strats = {}
    for n, r in {**agent_results, **benchmarks}.items():
        all_strats[n] = r.iloc[:min_len]

    # Metrics + bootstrap CIs + information ratio vs EW
    print("\n[4/5] Metrics + bootstrap CIs + information ratio...")
    ew_ret = all_strats.get("Equal Weight")
    all_metrics = {}
    for name, ret in all_strats.items():
        m = compute_metrics(ret, benchmark_returns=ew_ret if "Agent" in name else None)
        sh_mean, ci_lo, ci_hi = bootstrap_sharpe_ci(ret)
        m["Sharpe 95% CI"] = f"[{ci_lo:.2f}, {ci_hi:.2f}]"
        m["_ci_lo"] = ci_lo
        m["_ci_hi"] = ci_hi
        all_metrics[name] = m

    # Print
    print(f"\n{'=' * 105}")
    print(f"{'Strategy':<22} {'CAGR':>8} {'Sharpe':>8} {'95% CI':>16} "
          f"{'Max DD':>10} {'Sortino':>8} {'Info Ratio':>12}")
    print(f"{'-' * 105}")
    for name, m in all_metrics.items():
        if "error" in m:
            continue
        ir = m.get("Info Ratio", "—")
        print(f"{name:<22} {m['CAGR']:>8} {m['Sharpe']:>8} "
              f"{m['Sharpe 95% CI']:>16} {m['Max Drawdown']:>10} "
              f"{m['Sortino']:>8} {ir:>12}")
    print(f"{'=' * 105}")

    # Seed robustness
    print("\n[5/5] Seed robustness...")
    seed_summaries = {}
    for agent_name in AGENTS:
        summary = seed_robustness_analysis(
            agent_name, features, returns[sector_tickers], regime_numeric, close)
        if summary:
            seed_summaries[agent_name] = summary
            print(f"  {agent_name}: Sharpe = {summary['sharpe_mean']:.3f} "
                  f"± {summary['sharpe_std']:.3f} (n={summary['n_seeds']})")

    # Ablation
    if run_ablation_study:
        print("\n[ABLATION] Feature group importance...")
        for agent_name in AGENTS:
            for seed in SEEDS:
                run_id = f"{agent_name}_w{w_idx}_s{seed}"
                path = os.path.join(MODEL_DIR, run_id, f"{run_id}_final.zip")
                if os.path.exists(path):
                    norm_stats = _load_norm_stats(run_id)
                    abl = run_ablation(path, AGENTS[agent_name]["class"],
                                       test_feat, test_ret_sectors, test_regime,
                                       norm_stats=norm_stats)
                    save_data = {k: {kk: vv for kk, vv in v.items()
                                     if not kk.startswith("_")}
                                 for k, v in abl.items()}
                    abl_path = os.path.join(RESULTS_DIR, f"ablation_{agent_name}.json")
                    with open(abl_path, "w") as f:
                        json.dump(save_data, f, indent=2)
                    print(f"  Saved: {abl_path}")
                    break

    # Save
    save_metrics = {n: {k: v for k, v in m.items() if not k.startswith("_")}
                    for n, m in all_metrics.items() if "error" not in m}
    with open(os.path.join(RESULTS_DIR, "performance_comparison.json"), "w") as f:
        json.dump(save_metrics, f, indent=2)

    _generate_plots(all_metrics, agent_weights, sector_tickers, seed_summaries)
    return all_metrics, agent_weights, seed_summaries


def _generate_plots(all_metrics, agent_weights, sector_tickers, seed_summaries):
    """Publication-quality evaluation plots."""
    colors = {
        "PPO Agent": "#2563eb", "A2C Agent": "#8b5cf6", "SAC Agent": "#06b6d4",
        "Equal Weight": "#6b7280", "Risk Parity": "#10b981",
        "Momentum Top-3": "#f59e0b", "Min Variance": "#ef4444",
        "Inverse Volatility": "#ec4899", "SPY Buy & Hold": "#a3a3a3",
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle("Regime-Adaptive RL Portfolio Optimizer — Out-of-Sample", fontsize=16, weight="bold")

    # Cumulative returns
    ax = axes[0, 0]
    for name, m in all_metrics.items():
        if "_cumulative" not in m or m["_cumulative"] is None:
            continue
        cum = m["_cumulative"]
        ax.plot(cum.index, cum.values, label=name, color=colors.get(name, "#333"),
                linewidth=2 if "Agent" in name else 1, linestyle="-" if "Agent" in name else "--")
    ax.set_title("Cumulative Returns (Growth of $1)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha=0.3)

    # Drawdown
    ax = axes[0, 1]
    for name, m in all_metrics.items():
        if "_drawdown" in m and m["_drawdown"] is not None:
            dd = m["_drawdown"]
            ax.fill_between(dd.index, dd.values * 100, alpha=0.25,
                            color=colors.get(name, "#333"), label=name)
    ax.set_title("Drawdown (%)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Sharpe with CIs
    ax = axes[1, 0]
    strat_names = sorted(all_metrics.keys(), key=lambda n: all_metrics[n].get("_sharpe", -99))
    sharpes = [all_metrics[n]["_sharpe"] for n in strat_names if "_sharpe" in all_metrics[n]]
    ci_lows = [all_metrics[n].get("_ci_lo", all_metrics[n]["_sharpe"]) for n in strat_names if "_sharpe" in all_metrics[n]]
    ci_highs = [all_metrics[n].get("_ci_hi", all_metrics[n]["_sharpe"]) for n in strat_names if "_sharpe" in all_metrics[n]]
    valid_names = [n for n in strat_names if "_sharpe" in all_metrics[n]]
    y_pos = range(len(valid_names))
    ax.barh(y_pos, sharpes, color=[colors.get(n, "#333") for n in valid_names], alpha=0.8, height=0.6)
    ax.errorbar(sharpes, y_pos,
                xerr=[np.array(sharpes) - np.array(ci_lows), np.array(ci_highs) - np.array(sharpes)],
                fmt="none", ecolor="black", capsize=3, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid_names, fontsize=8)
    ax.set_title("Sharpe Ratio with 95% Bootstrap CI")
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    # Seed robustness
    ax = axes[1, 1]
    if seed_summaries:
        agents = list(seed_summaries.keys())
        means = [seed_summaries[a]["sharpe_mean"] for a in agents]
        stds = [seed_summaries[a]["sharpe_std"] for a in agents]
        ax.bar(agents, means, yerr=stds,
               color=[colors.get(f"{a} Agent", "#333") for a in agents],
               alpha=0.8, capsize=5)
        ax.set_title("Seed Robustness (Sharpe ± σ)")
        ax.set_ylabel("Sharpe Ratio")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "Train with multiple seeds first",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Seed Robustness")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "evaluation_results.png"), dpi=150, bbox_inches="tight")
    plt.close()

    for agent_name, weights in agent_weights.items():
        if len(weights) == 0:
            continue
        fig, ax = plt.subplots(figsize=(14, 5))
        labels = [SECTOR_ETFS.get(t, t) for t in sector_tickers]
        step = max(1, len(weights) // 80)
        sns.heatmap(weights[::step].T, yticklabels=labels, cmap="YlOrRd", ax=ax,
                    cbar_kws={"label": "Weight"})
        ax.set_title(f"{agent_name} — Sector Allocation Over Time")
        ax.set_xlabel("Time (sampled)")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"weights_{agent_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\nPlots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()
    evaluate(run_ablation_study=args.ablation)
