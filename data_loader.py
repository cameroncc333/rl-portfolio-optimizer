"""
Data Loader & Feature Engineering Pipeline (v4)

Additions: VIX term structure slope (VIX vs VIX3M).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from config import (
    TICKERS, BENCHMARK_TICKER, VIX_TICKER, VIX3M_TICKER,
    DATA_START, DATA_END, RETURN_WINDOWS, RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, BOLLINGER_PERIOD,
    BOLLINGER_STD, VOLATILITY_WINDOW, SHARPE_WINDOW,
    CORRELATION_WINDOW, LATE_INCEPTION, ABSTAIN_ASSETS
)


def fetch_data(tickers=None, start=DATA_START, end=DATA_END, include_abstain=False):
    """Download sector ETFs, SPY, VIX, and VIX3M.
    Set include_abstain=True when training the 13-asset v1.1 universe.
    """
    if tickers is None:
        base = TICKERS + [BENCHMARK_TICKER]
        if include_abstain:
            # Add BIL (SPY already in base as benchmark)
            extra = [t for t in ABSTAIN_ASSETS if t not in base]
            base = base + extra
        tickers = base

    print(f"[Data] Fetching {len(tickers)} tickers: {start} → {end}")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
        volume = data["Volume"]
    else:
        close = data[["Close"]].copy()
        volume = data[["Volume"]].copy()

    # Fetch VIX + VIX3M
    print("[Data] Fetching VIX + VIX3M...")
    vix_tickers = [VIX_TICKER, VIX3M_TICKER]
    vix_data = yf.download(vix_tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_close = vix_data["Close"]
        vix = vix_close[VIX_TICKER] if VIX_TICKER in vix_close.columns else vix_close.iloc[:, 0]
        if VIX3M_TICKER in vix_close.columns:
            vix3m = vix_close[VIX3M_TICKER]
        else:
            # VIX3M may not be available for full history — fallback to VIX
            print("[Data] VIX3M not available, using VIX as proxy")
            vix3m = vix.copy()
    else:
        vix = vix_data["Close"]
        vix3m = vix.copy()

    # Handle late-inception ETFs
    for ticker, inception in LATE_INCEPTION.items():
        if ticker in close.columns:
            inception_dt = pd.Timestamp(inception)
            close.loc[close.index < inception_dt, ticker] = np.nan
            volume.loc[volume.index < inception_dt, ticker] = np.nan

    close = close.ffill(limit=5)
    volume = volume.ffill(limit=5).fillna(0)
    vix = vix.ffill().dropna()
    vix3m = vix3m.ffill().dropna()

    common_idx = close.index.intersection(vix.index).intersection(vix3m.index)
    close = close.loc[common_idx]
    volume = volume.loc[common_idx]
    vix = vix.loc[common_idx]
    vix3m = vix3m.loc[common_idx]

    print(f"[Data] {len(close)} trading days, {close.shape[1]} tickers + VIX + VIX3M")
    return close, volume, vix, vix3m


def detect_regimes(close, vix):
    from config import REGIME_THRESHOLDS
    spy_ret = np.log(close[BENCHMARK_TICKER] / close[BENCHMARK_TICKER].shift(1))
    realized_vol = spy_ret.rolling(20).std() * np.sqrt(252)
    vix_normalized = vix / 100.0
    blended_vol = 0.6 * realized_vol + 0.4 * vix_normalized

    regimes = pd.Series("normal", index=close.index, name="regime")
    regimes[blended_vol < REGIME_THRESHOLDS["calm"]] = "calm"
    regimes[blended_vol > REGIME_THRESHOLDS["normal"]] = "stressed"
    regime_numeric = regimes.map({"calm": 0.0, "normal": 0.5, "stressed": 1.0})

    print(f"[Regime] Calm: {(regimes == 'calm').sum()} | "
          f"Normal: {(regimes == 'normal').sum()} | "
          f"Stressed: {(regimes == 'stressed').sum()}")
    return regimes, regime_numeric


def compute_features(close, volume, vix, vix3m):
    """Engineer 14 features per sector (added: vix_term structure slope)."""
    sector_tickers = [t for t in close.columns if t in TICKERS]
    features = {}

    for ticker in sector_tickers:
        px = close[ticker]
        vol = volume[ticker] if ticker in volume.columns else pd.Series(0, index=close.index)

        for w in RETURN_WINDOWS:
            features[(f"ret_{w}d", ticker)] = np.log(px / px.shift(w))

        rsi = RSIIndicator(close=px, window=RSI_PERIOD)
        features[("rsi", ticker)] = rsi.rsi() / 100.0

        macd = MACD(close=px, window_slow=MACD_SLOW, window_fast=MACD_FAST,
                     window_sign=MACD_SIGNAL)
        features[("macd_norm", ticker)] = macd.macd_diff() / px

        bb = BollingerBands(close=px, window=BOLLINGER_PERIOD, window_dev=BOLLINGER_STD)
        features[("bb_width", ticker)] = (bb.bollinger_hband() - bb.bollinger_lband()) / px

        log_ret = np.log(px / px.shift(1))
        features[("volatility", ticker)] = log_ret.rolling(VOLATILITY_WINDOW).std() * np.sqrt(252)

        rmean = log_ret.rolling(SHARPE_WINDOW).mean() * np.sqrt(252)
        rstd = log_ret.rolling(SHARPE_WINDOW).std() * np.sqrt(252)
        features[("sharpe", ticker)] = rmean / (rstd + 1e-8)

        features[("vol_change", ticker)] = np.log((vol + 1) / (vol.shift(20) + 1))

    # Cross-sector correlation
    ret_df = close[sector_tickers].pct_change()
    rolling_corr_means = _vectorized_mean_correlation(ret_df, CORRELATION_WINDOW)
    for ticker in sector_tickers:
        features[("corr_regime", ticker)] = rolling_corr_means

    # VIX (normalized)
    vix_norm = vix / 100.0
    for ticker in sector_tickers:
        features[("vix", ticker)] = vix_norm

    # [UPGRADE 7] VIX term structure slope: (VIX3M - VIX) / VIX
    # Positive = contango (normal), Negative = backwardation (crash signal)
    vix_term_slope = (vix3m - vix) / (vix + 1e-8)
    for ticker in sector_tickers:
        features[("vix_term", ticker)] = vix_term_slope

    # Beta and relative strength vs SPY
    spy = close[BENCHMARK_TICKER]
    spy_ret = np.log(spy / spy.shift(1))
    for ticker in sector_tickers:
        px_ret = np.log(close[ticker] / close[ticker].shift(1))
        cov = px_ret.rolling(60).cov(spy_ret)
        var = spy_ret.rolling(60).var()
        features[("beta", ticker)] = cov / (var + 1e-8)
        features[("rel_strength", ticker)] = (
            np.log(close[ticker] / close[ticker].shift(20)) -
            np.log(spy / spy.shift(20))
        )

    feature_df = pd.DataFrame(features, index=close.index)
    feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns,
                                                     names=["feature", "ticker"])
    feature_df = feature_df.dropna(how="all")
    feature_df = feature_df.fillna(0.0)

    regimes, regime_numeric = detect_regimes(close, vix)
    regimes = regimes.reindex(feature_df.index)
    regime_numeric = regime_numeric.reindex(feature_df.index)

    n_feat = len(feature_df.columns.get_level_values("feature").unique())
    print(f"[Features] {n_feat} features × {len(sector_tickers)} sectors "
          f"= {len(feature_df.columns)} signals")
    print(f"[Features] {feature_df.index[0].date()} → {feature_df.index[-1].date()} "
          f"({len(feature_df)} days)")

    return feature_df, regimes, regime_numeric


def _vectorized_mean_correlation(ret_df, window):
    n_assets = ret_df.shape[1]
    n_pairs = n_assets * (n_assets - 1) / 2
    corr_sum = pd.Series(0.0, index=ret_df.index)
    cols = ret_df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairwise = ret_df[cols[i]].rolling(window).corr(ret_df[cols[j]])
            corr_sum = corr_sum.add(pairwise.fillna(0), fill_value=0)
    return corr_sum / max(n_pairs, 1)


def get_returns(close, tickers=None):
    if tickers is None:
        tickers = [t for t in TICKERS if t in close.columns]
    returns = close[tickers].pct_change().fillna(0.0).iloc[1:]
    return returns


def prepare_data(include_abstain=False):
    """
    Main entry point.
    Pass include_abstain=True when retraining the 13-asset v1.1 universe
    so SPY/BIL are treated as tradeable assets in the feature matrix.
    """
    close, volume, vix, vix3m = fetch_data(include_abstain=include_abstain)
    features, regimes, regime_numeric = compute_features(close, volume, vix, vix3m)
    returns = get_returns(close)
    common_idx = features.index.intersection(returns.index)
    features = features.loc[common_idx]
    returns = returns.loc[common_idx]
    regimes = regimes.loc[common_idx]
    regime_numeric = regime_numeric.loc[common_idx]
    return features, returns, close, regimes, regime_numeric, vix


if __name__ == "__main__":
    features, returns, close, regimes, regime_numeric, vix = prepare_data()
    print(f"\nFeature shape: {features.shape}")
    print(f"Returns shape: {returns.shape}")
    print(f"\nFeatures: {sorted(features.columns.get_level_values('feature').unique().tolist())}")
    print(f"\nRegime distribution:\n{regimes.value_counts()}")
