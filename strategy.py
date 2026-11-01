"""Technical indicators and entry signal strategies.

Each strategy is a callable that receives a DataFrame of OHLCV data
and returns a Series of signals: 1 = BUY, -1 = SELL, 0 = HOLD.
New strategies can be added by subclassing `BaseStrategy`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import config


# ─────────────────────────────────────────────────────────────────────
# Indicator helpers (no lookahead – computed on closed candles only)
# ─────────────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx


# ─────────────────────────────────────────────────────────────────────
# Strategy base class
# ─────────────────────────────────────────────────────────────────────

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series aligned with df.index: 1=BUY, -1=SELL, 0=HOLD."""
        ...


# ─────────────────────────────────────────────────────────────────────
# Production Strategy: ADX Regime + Golden Cross + RSI Cross 50 + MACD
# ─────────────────────────────────────────────────────────────────────

class RegimeMultiStrategy(BaseStrategy):
    """Production Quantitative Trend-Follow Strategy.

    BUY Conditions (all must be True simultaneously on closed candle):
      1) Trend Filter:  EMA(50) > EMA(200) (Strong macro uptrend)
      2) Regime Filter: ADX(14) >= 20 (Trending market, filters out whipsaws)
      3) Momentum Entry: RSI(14) crosses above 50 (RSI_cross_above_50)
      4) Momentum Conf: MACD Histogram > 0 (Positive momentum confirmation)

    SELL Conditions (any one triggers exit signal):
      - RSI(14) > 70 (Overbought exhaustion)
      - MACD Histogram crosses below 0 (Bearish momentum shift)
      - EMA(50) < EMA(200) (Trend breakdown)
    """

    def __init__(
        self,
        adx_period: int = config.ADX_PERIOD,
        adx_threshold: float = config.ADX_THRESHOLD,
        ema_fast: int = config.EMA_FAST_PERIOD,
        ema_slow: int = config.EMA_TREND_PERIOD,
        rsi_period: int = config.RSI_PERIOD,
        rsi_cross_level: float = config.RSI_CROSS_LEVEL,
        rsi_overbought: float = config.RSI_OVERBOUGHT,
        macd_fast: int = config.MACD_FAST,
        macd_slow: int = config.MACD_SLOW,
        macd_signal: int = config.MACD_SIGNAL,
    ):
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_cross_level = rsi_cross_level
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        adx = compute_adx(df, self.adx_period)
        ema20 = compute_ema(close, self.ema_fast)
        ema100 = compute_ema(close, self.ema_slow)
        rsi = compute_rsi(close, self.rsi_period)
        _, _, histogram = compute_macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal
        )

        # ── BUY Conditions (Tüm şartlar aynı anda sağlanmalı) ──────────
        # 1. Trend Şartı: EMA(50) > EMA(200)
        c1_trend = ema20 > ema100

        # 2. Rejim Şartı: ADX(14) >= 20
        c2_regime = adx >= self.adx_threshold

        # 3. Momentum Teyidi: RSI(14) 50 seviyesini yukarı yönlü kessin
        c3_rsi_cross = (rsi > self.rsi_cross_level) & (rsi.shift(1) <= self.rsi_cross_level)

        # 4. Momentum Teyidi: MACD Histogram > 0
        c4_macd_hist = histogram > 0

        # ── SELL Conditions (Çıkış tetikleyicileri) ───────────────────
        sell_rsi = rsi > self.rsi_overbought
        sell_macd = (histogram < 0) & (histogram.shift(1) >= 0)
        sell_trend = ema20 < ema100

        signals = pd.Series(0, index=df.index)
        signals[c1_trend & c2_regime & c3_rsi_cross & c4_macd_hist] = 1
        signals[sell_rsi | sell_macd | sell_trend] = -1
        return signals


# ─────────────────────────────────────────────────────────────────────
# Legacy strategies (Kept for A/B testing and comparison)
# ─────────────────────────────────────────────────────────────────────

class MultiConfirmStrategy(BaseStrategy):
    def __init__(
        self,
        ema_period: int = config.EMA_TREND_PERIOD,
        rsi_period: int = config.RSI_PERIOD,
        rsi_oversold: float = config.RSI_OVERSOLD,
        rsi_overbought: float = config.RSI_OVERBOUGHT,
        macd_fast: int = config.MACD_FAST,
        macd_slow: int = config.MACD_SLOW,
        macd_signal: int = config.MACD_SIGNAL,
    ):
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ema100 = compute_ema(close, self.ema_period)
        rsi = compute_rsi(close, self.rsi_period)
        macd_line, signal_line, _ = compute_macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal
        )
        cond_a = close > ema100
        cond_b = ((rsi >= 30) & (rsi <= 50)) | ((rsi > 30) & (rsi.shift(1) <= 30))
        cond_c = macd_line > signal_line

        rsi_ob = rsi > self.rsi_overbought
        macd_bear = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        signals = pd.Series(0, index=df.index)
        signals[cond_a & cond_b & cond_c] = 1
        signals[rsi_ob | macd_bear] = -1
        return signals


class RSIStrategy(BaseStrategy):
    def __init__(self, period=config.RSI_PERIOD, oversold=config.RSI_OVERSOLD, overbought=config.RSI_OVERBOUGHT):
        self.period, self.oversold, self.overbought = period, oversold, overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = compute_rsi(df["close"], self.period)
        signals = pd.Series(0, index=df.index)
        signals[rsi < self.oversold] = 1
        signals[rsi > self.overbought] = -1
        return signals


class MACDStrategy(BaseStrategy):
    def __init__(self, fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL):
        self.fast, self.slow, self.signal = fast, slow, signal

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        macd_line, signal_line, _ = compute_macd(df["close"], self.fast, self.slow, self.signal)
        signals = pd.Series(0, index=df.index)
        signals[(macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))] = 1
        signals[(macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))] = -1
        return signals


class SMACrossStrategy(BaseStrategy):
    def __init__(self, short_period=config.SMA_SHORT, long_period=config.SMA_LONG):
        self.short_period, self.long_period = short_period, long_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sma_s = compute_sma(df["close"], self.short_period)
        sma_l = compute_sma(df["close"], self.long_period)
        signals = pd.Series(0, index=df.index)
        signals[(sma_s > sma_l) & (sma_s.shift(1) <= sma_l.shift(1))] = 1
        signals[(sma_s < sma_l) & (sma_s.shift(1) >= sma_l.shift(1))] = -1
        return signals


# ─────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────

STRATEGIES: dict[str, type[BaseStrategy]] = {
    "regime": RegimeMultiStrategy,
    "multi": MultiConfirmStrategy,
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
    "sma_cross": SMACrossStrategy,
}


def get_strategy(name: str = "regime", **kwargs) -> BaseStrategy:
    cls = STRATEGIES.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGIES)}")
    return cls(**kwargs)
