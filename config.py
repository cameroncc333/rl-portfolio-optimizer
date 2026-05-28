"""
Configuration — Regime-Adaptive RL Portfolio Optimizer (v4)
12-upgrade version: CVaR, kill switch, TC curriculum, random-start,
SHAP, Optuna, dynamic ensemble, VIX term structure, cosine LR,
reward clipping, transition cost obs, "What Didn't Work" docs.
"""

# ═══════════════════════════════════════════════════════════════════
# ASSET UNIVERSE
# ═══════════════════════════════════════════════════════════════════
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Discret.",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Svcs",
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}
TICKERS = list(SECTOR_ETFS.keys())
NUM_ASSETS = len(TICKERS)

# ── v1.1 Abstain universe (SPY = market-neutral, BIL = cash/defensive)
# After retraining on 13+2=15 assets (incl BTC/ETH), change TICKERS = UNIVERSE below.
# Current models (w2_s42) were trained on 11 assets — generate_rl_signal.py
# auto-detects model action dim and trims tickers to match for backward compat.
ABSTAIN_ASSETS = ["SPY", "BIL"]
UNIVERSE = TICKERS + ABSTAIN_ASSETS    # 15 assets for v1.1 retrain
N_ASSETS = len(UNIVERSE)               # 15; current models use NUM_ASSETS=11

LATE_INCEPTION = {
    "XLRE":    "2015-10-08",
    "XLC":     "2018-06-18",
    "BTC-USD": "2014-09-17",
    "ETH-USD": "2015-08-07",
}

BENCHMARK_TICKER = "SPY"
VIX_TICKER = "^VIX"
VIX3M_TICKER = "^VIX3M"  # 3-month VIX for term structure slope

# ═══════════════════════════════════════════════════════════════════
# DATA SETTINGS
# ═══════════════════════════════════════════════════════════════════
DATA_START = "2008-01-01"
# DATA_END is dynamic so inference always uses today's market data.
# Historical backtests that need a fixed end date can override by setting
# the DATA_END_OVERRIDE env var: DATA_END_OVERRIDE=2026-03-31 python evaluate.py
import os as _os, datetime as _dt
DATA_END = _os.environ.get("DATA_END_OVERRIDE", _dt.date.today().isoformat())
LOOKBACK_WINDOW = 60

WALK_FORWARD_WINDOWS = [
    ("2008-01-01", "2018-12-31", "2019-01-01", "2020-12-31"),
    ("2008-01-01", "2021-06-30", "2021-07-01", "2023-06-30"),
    ("2008-01-01", "2023-12-31", "2024-01-01", "2026-03-31"),
]

# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
RETURN_WINDOWS = [5, 20, 60]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
VOLATILITY_WINDOW = 20
SHARPE_WINDOW = 60
CORRELATION_WINDOW = 60

# ═══════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════
REGIME_THRESHOLDS = {
    "calm":     0.10,
    "normal":   0.20,
}

REGIME_REWARD_MULTIPLIERS = {
    "calm":     1.0,
    "normal":   1.2,
    "stressed": 1.5,
}

# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT SETTINGS
# ═══════════════════════════════════════════════════════════════════
INITIAL_BALANCE = 1_000_000
TRANSACTION_COST_BPS = 5
SPREAD_COST_BPS = 1
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.30
CASH_RESERVE = 0.02

# [UPGRADE 1] Random-start episodes
RANDOM_START = True

# [UPGRADE 3] Daily loss limit kill switch
KILL_SWITCH_THRESHOLD = -0.025  # Force to equal-weight if daily loss exceeds 2.5%

# [UPGRADE 10] Transaction cost curriculum
TC_CURRICULUM = True
TC_CURRICULUM_STEPS = 200_000   # Ramp TC from 0% to 100% over this many steps

# ═══════════════════════════════════════════════════════════════════
# REWARD FUNCTION WEIGHTS
# ═══════════════════════════════════════════════════════════════════
LAMBDA_DRAWDOWN = 2.0
LAMBDA_TURNOVER = 0.005
LAMBDA_SHARPE = 0.1
LAMBDA_CONCENTRATION = 0.02
LAMBDA_CVAR = 0.5              # [UPGRADE 2] CVaR tail-risk penalty weight
CVAR_ALPHA = 0.05              # Bottom 5% of returns for CVaR
REWARD_SCALING = 100.0
REWARD_CLIP = 5.0              # [UPGRADE 12] Clip final reward to [-5, 5]

# Relative reward: use excess return vs SPY as the base signal.
# Set to True when training with spy_returns passed to PortfolioEnv.
# Existing models trained with USE_RELATIVE_REWARD=False are unaffected.
USE_RELATIVE_REWARD = True

# Cross-sectional Z-score features strip broad market noise so the agent
# reads pure idiosyncratic alpha rather than sector-wide momentum.
# Adds 4 extra features per asset (14 → 18) — requires retraining to use.
# Set False to match existing 11-asset models (14-feature obs space).
USE_CROSS_SECTIONAL_FEATURES = True

# ═══════════════════════════════════════════════════════════════════
# AGENT CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════
AGENTS = {
    "PPO": {
        "class": "PPO",
        "config": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch": dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            },
        },
    },
    "A2C": {
        "class": "A2C",
        "config": {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "vf_coef": 0.25,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch": dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            },
        },
    },
    "SAC": {
        "class": "SAC",
        "config": {
            "learning_rate": 3e-4,
            "buffer_size": 100_000,
            "learning_starts": 1000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "ent_coef": "auto",
            "policy_kwargs": {
                "net_arch": [256, 256, 128],
            },
        },
    },
}

# ═══════════════════════════════════════════════════════════════════
# TRAINING SETTINGS
# ═══════════════════════════════════════════════════════════════════
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 3
SEEDS = [42, 123, 456, 789, 1337]
MODEL_DIR = "models"
LOG_DIR = "logs"
RESULTS_DIR = "results"

USE_COSINE_LR = True           # [UPGRADE 8] Cosine-annealing learning rate

# ═══════════════════════════════════════════════════════════════════
# EVALUATION SETTINGS
# ═══════════════════════════════════════════════════════════════════
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
RISK_FREE_RATE = 0.0375

ABLATION_GROUPS = {
    "momentum":    ["ret_5d", "ret_20d", "ret_60d"],
    "volatility":  ["volatility", "bb_width"],
    "trend":       ["rsi", "macd_norm"],
    "risk":        ["sharpe", "beta"],
    "regime":      ["corr_regime", "vix", "vix_term"],
    "relative":    ["rel_strength", "vol_change"],
}

# ═══════════════════════════════════════════════════════════════════
# OPTUNA SETTINGS [UPGRADE 5]
# ═══════════════════════════════════════════════════════════════════
OPTUNA_N_TRIALS = 50
OPTUNA_TIMESTEPS = 200_000
