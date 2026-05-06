# Regime-Adaptive Reinforcement Learning for Dynamic Sector Allocation

**A Multi-Agent Comparison with Walk-Forward Validation, Ablation Analysis, and Bootstrap Inference**

Cameron Camarotti · Class of 2027, Mill Creek High School
[GitHub](https://github.com/cameroncc333) · [All Around Services](https://allaroundservice.com)

---

## Abstract

This paper trains and evaluates three deep reinforcement learning agents — **PPO** (Proximal Policy Optimization), **A2C** (Advantage Actor-Critic), and **SAC** (Soft Actor-Critic) — on the problem of dynamically allocating capital across 11 S&P 500 sector ETFs. The agents learn to map a 132-dimensional market signal vector (12 features × 11 sectors) to portfolio weights through 500,000 simulated trading steps, receiving a **regime-adaptive reward** that increases drawdown penalties during periods of elevated market stress. We validate each agent across **3 non-overlapping walk-forward windows** spanning 2008–2026, train each configuration across **5 random seeds** for robustness, and report **bootstrap 95% confidence intervals** on all Sharpe ratios. Performance is benchmarked against 6 passive and systematic strategies. A feature ablation study quantifies the marginal Sharpe contribution of each signal group.

> **Results populate after training.** This is a live research pipeline — run `python train.py && python evaluate.py` to generate all metrics and populate the tables below with real, reproducible numbers. No placeholder data is presented as real.

---

## Table of Contents

1. [Methodology](#methodology)
2. [Observation Space](#observation-space)
3. [Reward Function](#reward-function)
4. [Training Protocol](#training-protocol)
5. [Evaluation Framework](#evaluation-framework)
6. [Visual Gallery](#visual-gallery)
7. [Limitations & Future Work](#limitations--future-work)
8. [Installation & Reproduction](#installation--reproduction)
9. [References](#references)

---

## Methodology

### Problem Formulation

At each trading day *t*, the agent observes a state **s_t** comprising 60 days of multi-signal market history, its current portfolio weights, and three regime indicators. It selects an action **a_t** ∈ ℝ¹¹ representing raw allocation logits, which are softmax-normalized and constrained to produce valid portfolio weights. The environment returns a shaped reward signal and transitions to state **s_{t+1}**.

This is formulated as a **Markov Decision Process** where the transition dynamics are determined by realized market returns — a non-stationary, partially observable environment that makes the problem substantially harder than standard RL benchmarks.

### Data Pipeline

1. **Source**: Yahoo Finance via `yfinance` — 11 SPDR sector ETFs + SPY (benchmark) + CBOE VIX (volatility index)
2. **Period**: January 2008 – April 2026 (~4,500 trading days). Starting from 2008 captures the Global Financial Crisis, providing critical stressed-regime training data.
3. **Handling of XLRE**: The Real Estate sector ETF (XLRE) began trading in October 2015. Prior to that date, XLRE data is excluded and the universe operates on 10 sectors. Post-2015, all 11 sectors are active. This is handled explicitly — no forward-fill imputation of nonexistent data.
4. **Feature Engineering**: 12 technical signals computed per sector (detailed below), with z-score normalization computed **only on training data** and applied to test data to prevent look-ahead bias.

### Regime Detection

Market regimes are classified daily using a blended volatility measure:

```
blended_vol_t = 0.6 × realized_vol(SPY, 20d) + 0.4 × (VIX_t / 100)
```

| Regime | Blended Vol | Reward Multiplier | Historical Frequency |
|--------|-------------|-------------------|---------------------|
| Calm | < 10% | κ = 1.0 | ~35% of days |
| Normal | 10–20% | κ = 1.2 | ~45% of days |
| Stressed | > 20% | κ = 1.5 (losses penalized 2.25×) | ~20% of days |

This dual-confirmation approach (backward-looking realized vol + forward-looking implied vol via VIX) is more robust than either measure alone.

---

## Observation Space

Each observation at time *t* contains:

| Signal | Computation | Dimension | Rationale |
|--------|-------------|-----------|-----------|
| Log Returns | ln(P_t / P_{t-k}), k ∈ {5, 20, 60} | 3 × 11 | Multi-timescale momentum |
| RSI | 14-day Relative Strength Index / 100 | 1 × 11 | Mean-reversion signal |
| MACD | Normalized MACD histogram / price | 1 × 11 | Trend strength and direction |
| Bollinger Width | (Upper − Lower band) / price | 1 × 11 | Volatility expansion/compression |
| Realized Volatility | 20-day rolling σ, annualized | 1 × 11 | Current risk level |
| Rolling Sharpe | 60-day rolling Sharpe ratio | 1 × 11 | Risk-adjusted momentum |
| Volume Ratio | log(V_t / V_{t-20}) | 1 × 11 | Liquidity/attention shift |
| Beta vs SPY | 60-day rolling CAPM β | 1 × 11 | Systematic risk loading |
| Relative Strength | 20-day sector return − SPY return | 1 × 11 | Sector rotation signal |
| VIX Level | VIX / 100 (market-wide) | 1 × 11 | Market fear gauge |
| Cross-Sector ρ | Mean pairwise correlation (60d) | 1 × 11 | Dispersion regime |

**Total per-step observation**: `60 × 12 × 11 + 11 + 3 = 7,934 dimensions`

The 60-day lookback window, 12 engineered features per sector, current portfolio weights (11), and 3 meta-features (regime level, rolling agent Sharpe, current drawdown) combine to create a high-dimensional state representation that captures both cross-sectional and temporal market structure.

---

## Reward Function

```
r_t = R_portfolio × κ(regime)                            [regime-scaled return]
    − λ_dd × max(0, ΔDD_t) × (1 + regime_stress_t)     [adaptive drawdown penalty]
    − λ_turn × ||Δw_t||₁                                 [turnover penalty]
    + λ_S × max(0, Ŝ_60d)                                [Sharpe consistency bonus]
    − λ_HHI × max(0, HHI(w_t) − 1/N)                    [concentration penalty]
```

| Parameter | Value | Purpose |
|-----------|-------|---------|
| λ_dd | 2.0 | Penalize increasing drawdown; amplified in stressed regimes |
| λ_turn | 0.005 | Discourage excessive rebalancing (proxy for market impact) |
| λ_S | 0.1 | Reward positive risk-adjusted consistency |
| λ_HHI | 0.02 | Penalize Herfindahl concentration above equal-weight baseline |
| Transaction cost | 6 bps/trade | 5 bps proportional + 1 bps spread |
| Position limit | 30% max | No single sector > 30% of portfolio |
| Cash reserve | 2% | Minimum cash buffer |

The key innovation is **regime-adaptive penalty scaling**: during stressed markets, losses are penalized at κ × 1.5 = 2.25× the calm-market rate, teaching the agent defensive positioning without explicit regime-switching rules.

---

## Training Protocol

### Multi-Agent Tournament

| Algorithm | Key Property | Exploration Strategy |
|-----------|--------------|---------------------|
| **PPO** | Clipped surrogate objective; stable updates | Entropy bonus (ε = 0.01) |
| **A2C** | Synchronous advantage estimation; fast | Entropy regularization |
| **SAC** | Maximum entropy RL; off-policy replay | Automatic temperature tuning |

All agents share the same architecture: `[256, 256, 128]` hidden units with ReLU activations for both actor and critic networks.

### Walk-Forward Validation

Three **non-overlapping** test windows ensure robustness across different macro regimes:

| Window | Training Period | Test Period | Macro Context |
|--------|----------------|-------------|---------------|
| W1 | 2008–2018 | 2019–2020 | Pre-COVID bull → COVID crash |
| W2 | 2008–2021 | 2021–2023 | Recovery → Fed tightening cycle |
| W3 | 2008–2023 | 2024–2026 | Rate plateau → current market |

### Seed Robustness

Each agent × window configuration is trained with 5 random seeds: `{42, 123, 456, 789, 1337}`. We report mean ± standard deviation across seeds to quantify initialization sensitivity.

---

## Evaluation Framework

### Benchmark Strategies

| Strategy | Description | Rebalance |
|----------|-------------|-----------|
| Equal Weight (1/N) | Uniform allocation across all sectors | Daily (implicit) |
| Risk Parity | Equalize risk contribution via covariance | Monthly (20 days) |
| Momentum Top-3 | Equal-weight top 3 sectors by 20-day return | Monthly |
| Minimum Variance | Optimize for minimum portfolio variance | Monthly |
| Inverse Volatility | Weight by 1/σ (realized volatility) | Monthly |
| SPY Buy & Hold | S&P 500 index passive benchmark | N/A |

### Statistical Inference

- **Bootstrap Confidence Intervals**: 10,000 bootstrap samples of the daily return series; report 95% CI on annualized Sharpe ratio. An agent's outperformance is considered **statistically significant** only if its Sharpe CI does not overlap with the benchmark CI.
- **Information Ratio**: Annualized active return / tracking error vs. equal-weight benchmark.
- **Seed Robustness**: Mean ± σ of Sharpe across 5 seeds per agent.

### Feature Ablation

Each signal group is systematically zeroed out (not removed — observation dimension stays constant) and the trained agent is re-evaluated:

| Group | Features | Hypothesis |
|-------|----------|------------|
| Momentum | ret_5d, ret_20d, ret_60d | Core alpha signal |
| Volatility | volatility, bb_width | Risk estimation |
| Trend | rsi, macd_norm | Mean reversion / trend following |
| Risk-adjusted | sharpe, beta | Quality filter |
| Regime | corr_regime, vix | Macro awareness |
| Relative | rel_strength, vol_change | Sector rotation |

---

## Visual Gallery

After training, the evaluation pipeline generates these outputs in `results/`:

| Output | Description |
|--------|-------------|
| `evaluation_results.png` | 4-panel: cumulative returns, drawdown, Sharpe w/ CI, seed robustness |
| `weights_PPO.png` | Sector allocation heatmap over time |
| `weights_A2C.png` | A2C allocation comparison |
| `weights_SAC.png` | SAC allocation comparison |
| `performance_comparison.json` | Full metrics for all strategies |
| `ablation_PPO.json` | Feature group importance ranking |

*Screenshots of dashboard available after running `streamlit run dashboard.py`*

---

## Limitations & Future Work

### Known Limitations

1. **Sector-level granularity**: Operating at the ETF level abstracts away individual stock selection. This is by design (reduces dimensionality), but limits alpha capacity compared to a stock-level system.
2. **Transaction cost model**: The 6 bps flat cost is a simplification. Real execution involves non-linear market impact that depends on trade size relative to volume (Almgren-Chriss framework). The March 2026 paper by Riera Abbade et al. demonstrates this materially changes algorithm rankings.
3. **Single-episode training**: Each episode consumes the full training window sequentially. Random-start training (beginning episodes at random points in the data) would improve sample efficiency.
4. **No alternative data**: The feature set uses only price, volume, and VIX. Incorporating sentiment (e.g., FinBERT on Fed statements), options flow, or macroeconomic indicators would expand the information set.
5. **Stationarity assumption**: Z-score normalization assumes feature distributions are time-invariant. An adaptive normalization scheme (e.g., exponential moving statistics) would handle structural breaks better.

### Planned Extensions

- **FinBERT Sentiment Integration**: Parse FOMC statements and earnings call transcripts to add NLP-derived sentiment features to the observation space (see companion project: `fed-rate-sector-analysis`).
- **Almgren-Chriss Market Impact**: Replace flat transaction costs with a square-root market impact model calibrated to sector ETF ADV.
- **Reinforcement Learning from Human Feedback (RLHF)**: Use personal investment thesis as a preference signal to fine-tune the policy (connects to custodial brokerage account decisions).

---

## Installation & Reproduction

### Requirements

- Python 3.10+ (SB3 v2.3+ requires it; Python 3.9 reached end-of-life October 2025)
- ~4 GB disk for data + models
- Training: ~10 min per agent (100K steps), ~45 min for full tournament

```bash
python -m venv rl_env && source rl_env/bin/activate
pip install -r requirements.txt
```

### Commands

```bash
# Quick test: single agent, 1 seed, most recent window (~10 min)
python train.py --agent PPO --timesteps 100000 --seeds 1

# Full tournament: 3 agents × 5 seeds × 3 windows (run overnight)
python train.py --full --timesteps 500000

# Evaluate all models + generate plots
python evaluate.py

# Feature ablation study
python evaluate.py --ablation

# Interactive dashboard
streamlit run dashboard.py

# TensorBoard training curves
tensorboard --logdir logs/
```

### Project Structure

```
rl-portfolio-optimizer/
├── README.md              ← This document
├── requirements.txt       ← Pinned dependencies
├── config.py              ← All hyperparameters, thresholds, experiment grid
├── data_loader.py         ← Data pipeline, feature engineering, regime detection
├── portfolio_env.py       ← Custom Gymnasium environment (core contribution)
├── train.py               ← Multi-agent walk-forward training loop
├── evaluate.py            ← Benchmarks, bootstrap inference, ablation, plotting
├── dashboard.py           ← Streamlit dashboard (5-tab interactive visualization)
├── models/                ← Saved checkpoints (agent_window_seed)
├── results/               ← Metrics JSON, evaluation plots
└── logs/                  ← TensorBoard training logs
```

---

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*
2. Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning. *ICML 2018*
3. Raffin, A., Hill, A., Gleave, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR 22(268):1-8*
4. Liu, X.Y., Yang, H., Chen, Q., et al. (2020). FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading. *NeurIPS 2020 Deep RL Workshop*
5. Riera Abbade, L. et al. (2026). Realistic Market Impact Modeling for Reinforcement Learning Trading Environments. *arXiv:2603.29086*
6. de-la-Rica-Escudero, A. et al. (2025). Explainable Post Hoc Portfolio Management Policy of a DRL Agent. *PLOS ONE 20(1)*
7. Costa, C. & Costa, A. (2025). RLPortfolio: Reinforcement Learning for Financial Portfolio Optimization. *BRACIS 2024, LNCS vol. 15414*

---

**License**: MIT
