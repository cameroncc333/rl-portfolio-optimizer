"""
Custom Gymnasium Environment for Portfolio Optimization (v4)

UPGRADES:
1. Random-start episodes (break sequence memorization)
2. CVaR tail-risk penalty (institutional risk standard)
3. Daily loss limit kill switch (force defensive on -2.5%)
10. Transaction cost curriculum (ramp from 0 to full)
11. Explicit transition cost estimate in observation
12. Reward clipping for gradient stability
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import (
    INITIAL_BALANCE, TRANSACTION_COST_BPS, SPREAD_COST_BPS,
    MIN_WEIGHT, MAX_WEIGHT, CASH_RESERVE,
    LAMBDA_DRAWDOWN, LAMBDA_TURNOVER, LAMBDA_SHARPE, LAMBDA_CONCENTRATION,
    LAMBDA_CVAR, CVAR_ALPHA,
    REWARD_SCALING, REWARD_CLIP, LOOKBACK_WINDOW, REGIME_REWARD_MULTIPLIERS,
    RANDOM_START, KILL_SWITCH_THRESHOLD,
    TC_CURRICULUM, TC_CURRICULUM_STEPS, USE_RELATIVE_REWARD
)


class PortfolioEnv(gym.Env):
    """
    RL environment for dynamic sector allocation.

    Obs:  flattened (lookback × features × assets) + weights + transition_cost + meta
    Act:  continuous logits → softmax → clipped weights
    Rew:  regime-scaled return - DD - turnover - HHI - CVaR + Sharpe (clipped)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, features, returns, regime_numeric,
                 lookback=LOOKBACK_WINDOW,
                 ablation_mask=None,
                 norm_stats=None,
                 random_start=RANDOM_START,
                 tc_curriculum=TC_CURRICULUM,
                 spy_returns=None):
        super().__init__()

        self.lookback = lookback
        self.random_start = random_start
        self.tc_curriculum = tc_curriculum
        self.global_step = 0  # For TC curriculum scheduling
        # spy_returns: daily SPY returns aligned to same index as features/returns
        # Used for relative reward shaping (reward = excess return vs SPY)
        self.spy_returns = (spy_returns.values if hasattr(spy_returns, "values")
                            else spy_returns)

        # Parse features
        all_features = sorted(features.columns.get_level_values("feature").unique())
        tickers = sorted(features.columns.get_level_values("ticker").unique())
        self.num_features = len(all_features)
        self.num_assets = len(tickers)
        self.feature_names = all_features
        self.tickers = tickers

        # Build 3D feature array: (T, F, A)
        feature_array = np.zeros((len(features), self.num_features, self.num_assets))
        for i, feat in enumerate(all_features):
            for j, tick in enumerate(tickers):
                if (feat, tick) in features.columns:
                    feature_array[:, i, j] = features[(feat, tick)].values

        if ablation_mask:
            for i, feat in enumerate(all_features):
                if feat in ablation_mask:
                    feature_array[:, i, :] = 0.0

        self.features_raw = feature_array
        self.returns = returns[tickers].values
        self.regime_values = regime_numeric.values

        # Normalization
        if norm_stats is not None:
            self._means, self._stds = norm_stats
        else:
            self._means = np.nanmean(self.features_raw, axis=0, keepdims=True)
            self._stds = np.nanstd(self.features_raw, axis=0, keepdims=True) + 1e-8

        self.features_norm = np.clip(
            (self.features_raw - self._means) / self._stds, -5.0, 5.0
        )
        np.nan_to_num(self.features_norm, copy=False, nan=0.0, posinf=5.0, neginf=-5.0)

        # [UPGRADE 11] Obs: features + weights + transition_cost_est + meta(4)
        # meta = [regime, rolling_sharpe, drawdown, tc_multiplier]
        obs_dim = (lookback * self.num_features * self.num_assets
                    + self.num_assets      # current weights
                    + self.num_assets      # estimated transition cost per sector
                    + 4)                   # regime, rolling_sharpe, drawdown, tc_mult
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_assets,), dtype=np.float32
        )

        self.current_step = None
        self.weights = None
        self.portfolio_value = None
        self.peak_value = None
        self.returns_history = None
        self._prev_dd = None

    def get_norm_stats(self):
        return (self._means, self._stds)

    def _get_tc_multiplier(self):
        """[UPGRADE 10] Transaction cost curriculum: ramp from 0 to 1."""
        if not self.tc_curriculum:
            return 1.0
        return min(1.0, self.global_step / TC_CURRICULUM_STEPS)

    def _get_obs(self):
        start = self.current_step - self.lookback
        end = self.current_step
        flat = self.features_norm[start:end].flatten().astype(np.float32)

        regime = np.float32(self.regime_values[self.current_step])

        if len(self.returns_history) >= 20:
            arr = np.array(self.returns_history[-60:])
            rolling_sharpe = np.float32(
                np.mean(arr) / (np.std(arr) + 1e-8) * np.sqrt(252)
            )
        else:
            rolling_sharpe = np.float32(0.0)

        dd = np.float32(
            (self.peak_value - self.portfolio_value) / self.peak_value
        )

        tc_mult = np.float32(self._get_tc_multiplier())

        # [UPGRADE 11] Transition cost estimate: how much it would cost
        # to move from current weights to equal-weight (proxy for "cost awareness")
        ew = np.ones(self.num_assets, dtype=np.float32) / self.num_assets * (1 - CASH_RESERVE)
        transition_cost_est = np.abs(self.weights - ew).astype(np.float32)

        obs = np.concatenate([
            flat,
            self.weights.astype(np.float32),
            transition_cost_est,
            np.array([regime, rolling_sharpe, dd, tc_mult], dtype=np.float32),
        ])
        return obs

    def _action_to_weights(self, action):
        exp_a = np.exp(action - np.max(action))
        weights = exp_a / (exp_a.sum() + 1e-8)
        weights = np.clip(weights, MIN_WEIGHT, MAX_WEIGHT)
        investable = 1.0 - CASH_RESERVE
        total = weights.sum()
        if total > 1e-8:
            weights = weights / total * investable
        else:
            weights = np.ones(self.num_assets) / self.num_assets * investable
        return weights

    def _compute_cvar(self):
        """[UPGRADE 2] Conditional Value-at-Risk: expected loss in worst α% of days."""
        if len(self.returns_history) < 20:
            return 0.0
        recent = np.array(self.returns_history[-60:])
        cutoff = int(len(recent) * CVAR_ALPHA)
        if cutoff < 1:
            cutoff = 1
        sorted_returns = np.sort(recent)
        cvar = np.mean(sorted_returns[:cutoff])  # Mean of worst α%
        return cvar  # Negative number (it's a loss)

    def _compute_reward(self, portfolio_return, new_weights, old_weights):
        regime_val = self.regime_values[self.current_step]
        if regime_val < 0.25:
            regime_label = "calm"
        elif regime_val < 0.75:
            regime_label = "normal"
        else:
            regime_label = "stressed"

        regime_mult = REGIME_REWARD_MULTIPLIERS[regime_label]

        # 1. Regime-scaled return (optionally relative to SPY benchmark)
        if USE_RELATIVE_REWARD and self.spy_returns is not None:
            spy_r = float(self.spy_returns[self.current_step])
            base_return = portfolio_return - spy_r
        else:
            base_return = portfolio_return

        if base_return < 0 and regime_label == "stressed":
            reward = base_return * regime_mult * 1.5
        else:
            reward = base_return * regime_mult

        # 2. Transaction costs with curriculum [UPGRADE 10]
        turnover = np.sum(np.abs(new_weights - old_weights))
        tc_mult = self._get_tc_multiplier()
        tc = turnover * (TRANSACTION_COST_BPS + SPREAD_COST_BPS) / 10000.0 * tc_mult
        self.portfolio_value *= (1.0 + portfolio_return) * (1.0 - tc)

        # 3. Drawdown penalty (regime-adaptive)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        current_dd = (self.peak_value - self.portfolio_value) / self.peak_value
        dd_increase = max(0.0, current_dd - self._prev_dd)
        self._prev_dd = current_dd
        stress_multiplier = 1.0 + regime_val
        reward -= LAMBDA_DRAWDOWN * dd_increase * stress_multiplier

        # 4. Turnover penalty (also scaled by curriculum)
        reward -= LAMBDA_TURNOVER * turnover * tc_mult

        # 5. Concentration penalty
        hhi = np.sum(new_weights ** 2)
        hhi_baseline = 1.0 / self.num_assets
        reward -= LAMBDA_CONCENTRATION * max(0, hhi - hhi_baseline)

        # 6. Sharpe consistency bonus
        self.returns_history.append(portfolio_return)
        if len(self.returns_history) >= 20:
            recent = np.array(self.returns_history[-60:])
            rolling_sharpe = np.mean(recent) / (np.std(recent) + 1e-8) * np.sqrt(252)
            reward += LAMBDA_SHARPE * max(0, rolling_sharpe)

        # 7. [UPGRADE 2] CVaR tail-risk penalty
        cvar = self._compute_cvar()
        if cvar < 0:
            reward += LAMBDA_CVAR * cvar  # cvar is negative, so this penalizes

        # Scale and clip [UPGRADE 12]
        reward = reward * REWARD_SCALING
        reward = np.clip(reward, -REWARD_CLIP, REWARD_CLIP)

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # [UPGRADE 1] Random-start: begin at random point in data
        if self.random_start and self.np_random is not None:
            max_start = len(self.returns) - self.lookback - 100  # Need at least 100 days
            if max_start > self.lookback:
                self.current_step = self.np_random.integers(self.lookback, max_start)
            else:
                self.current_step = self.lookback
        else:
            self.current_step = self.lookback

        investable = 1.0 - CASH_RESERVE
        self.weights = np.ones(self.num_assets) / self.num_assets * investable

        self.portfolio_value = INITIAL_BALANCE
        self.peak_value = INITIAL_BALANCE
        self.returns_history = []
        self._prev_dd = 0.0

        self._ep_returns = []
        self._ep_weights = []
        self._ep_values = [INITIAL_BALANCE]
        self._ep_regimes = []

        obs = self._get_obs()
        info = {"portfolio_value": self.portfolio_value, "weights": self.weights.copy()}
        return obs, info

    def step(self, action):
        old_weights = self.weights.copy()
        new_weights = self._action_to_weights(action)

        day_returns = self.returns[self.current_step]
        portfolio_return = np.dot(new_weights, day_returns)

        # [UPGRADE 3] Kill switch: force to equal-weight if daily loss exceeds threshold
        if portfolio_return < KILL_SWITCH_THRESHOLD:
            investable = 1.0 - CASH_RESERVE
            new_weights = np.ones(self.num_assets) / self.num_assets * investable

        reward = self._compute_reward(portfolio_return, new_weights, old_weights)

        self.weights = new_weights
        self.current_step += 1
        self.global_step += 1  # For TC curriculum

        self._ep_returns.append(portfolio_return)
        self._ep_weights.append(new_weights.copy())
        self._ep_values.append(self.portfolio_value)
        self._ep_regimes.append(self.regime_values[self.current_step - 1])

        terminated = self.current_step >= len(self.returns) - 1
        truncated = False

        obs = (self._get_obs() if not terminated
               else np.zeros(self.observation_space.shape, dtype=np.float32))

        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "weights": self.weights.copy(),
            "turnover": np.sum(np.abs(new_weights - old_weights)),
            "drawdown": (self.peak_value - self.portfolio_value) / self.peak_value,
            "regime": self.regime_values[self.current_step - 1],
            "step": self.current_step,
            "kill_switch_triggered": portfolio_return < KILL_SWITCH_THRESHOLD,
        }
        return obs, reward, terminated, truncated, info

    def get_episode_data(self):
        return {
            "returns": np.array(self._ep_returns),
            "weights": np.array(self._ep_weights),
            "values": np.array(self._ep_values),
            "regimes": np.array(self._ep_regimes),
            "tickers": self.tickers,
        }
