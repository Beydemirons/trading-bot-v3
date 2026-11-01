"""Market Regime Detection Engine – Hidden Markov Models (HMM) & Gaussian Mixture Models (GMM).

Categorizes real-time market microstructure into 3 quantitative regimes:
  - Regime 0: Low Volatility (Ölü Piyasa / Squeeze Compression) -> Normal Eşik (%50.0)
  - Regime 1: High Volatility (Yatay / Testere - Whipsaw & False Breakouts) -> Defansif Eşik (%65.0)
  - Regime 2: Strong Trend (Boğa / Ayı - Directional Momentum Expansion) -> Agresif Eşik (%48.0)

Features dual-engine architecture:
  - Primary Engine : GaussianHMM (hmmlearn)
  - Fallback Engine: GaussianMixture (sklearn.mixture.GaussianMixture)
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import pandas as pd

import config
from logger_setup import get_logger

log = get_logger("market_regime")

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


class MarketRegimeDetector:
    """Quantitative Regime Classification Engine."""

    REGIME_LOW_VOLATILITY = 0   # Ölü Piyasa / Sıkışma
    REGIME_CHOPPY_HIGH_VOL = 1  # Testere / Sahte Kırılım / Yatay Çalkantı
    REGIME_STRONG_TREND = 2     # Güçlü Trend / Yönlü Hareket

    REGIME_NAMES = {
        0: "DÜŞÜK VOLATİLİTE (ÖLÜ PİYASA)",
        1: "YÜKSEK VOLATİLİTE (TESTERE / YATAY)",
        2: "GÜÇLÜ TREND (BOĞA / AYI)",
    }

    # Eşik Adaptasyon Tablosu
    REGIME_THRESHOLDS = {
        0: 0.50,  # Standart / Nötr Eşik
        1: 0.65,  # Defansif Eşik (Sahte kırılımları filtrele)
        2: 0.48,  # Agresif Eşik (Trend fırsatlarını kaçırma)
    }

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self._state_map: dict[int, int] = {0: 0, 1: 1, 2: 2}

    def extract_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract stationary volatility, momentum and trend features for regime modeling."""
        if df.empty or len(df) < 30:
            return pd.DataFrame()

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # 1. Log Returns & Rolling Volatility (14 bars)
        log_ret = np.log(close / close.shift(1)).fillna(0.0)
        vol_14 = log_ret.rolling(window=14).std().fillna(0.0)

        # 2. Normalized ATR (True Range / Close)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        natr_14 = (tr.rolling(window=14).mean() / (close + 1e-9)).fillna(0.0)

        # 3. Parkinson Realized Volatility Proxy
        hl_ratio = np.log(high / (low + 1e-9))
        parkinson_vol = (hl_ratio ** 2 / (4.0 * np.log(2.0))).rolling(14).mean().fillna(0.0)

        # 4. Absolute Trend Momentum & Autocorrelation
        ret_1 = close.pct_change(1).fillna(0.0)
        ret_5 = close.pct_change(5).fillna(0.0)
        abs_trend = ret_5.abs().rolling(14).mean().fillna(0.0)

        feat_df = pd.DataFrame({
            "vol_14": vol_14,
            "natr_14": natr_14,
            "parkinson_vol": parkinson_vol,
            "abs_trend": abs_trend,
        }, index=df.index).dropna()

        return feat_df

    def fit(self, df: pd.DataFrame) -> MarketRegimeDetector:
        """Fit HMM or GMM model to multi-asset or single-asset historical candles."""
        features = self.extract_regime_features(df)
        if len(features) < 60:
            log.warning("Insufficient data points (%d) to fit market regime model.", len(features))
            return self

        X = features.values

        if HMM_AVAILABLE:
            try:
                model = GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="full",
                    n_iter=100,
                    random_state=self.random_state,
                )
                model.fit(X)
                self.model = model
                self.is_fitted = True
                log.info("GaussianHMM regime model successfully fitted on %d samples.", len(X))
            except Exception as e:
                log.warning("HMM fit failed (%s), falling back to GaussianMixture.", e)
                self._fit_gmm(X)
        else:
            self._fit_gmm(X)

        if self.is_fitted:
            self._calibrate_regime_labels(X)

        return self

    def _fit_gmm(self, X: np.ndarray) -> None:
        """Fit Gaussian Mixture Model as standard high-performance fallback."""
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=self.random_state,
        )
        model.fit(X)
        self.model = model
        self.is_fitted = True
        log.info("GaussianMixture (GMM) regime model fitted on %d samples.", len(X))

    def _calibrate_regime_labels(self, X: np.ndarray) -> None:
        """Sort unsupervised cluster states deterministically:

        - State with lowest volatility -> Regime 0 (Ölü Piyasa / Low Vol)
        - State with highest volatility & lower trend -> Regime 1 (Testere / Choppy High Vol)
        - State with highest trend momentum -> Regime 2 (Güçlü Trend)
        """
        if self.model is None or not self.is_fitted:
            return

        raw_states = self.model.predict(X)
        vol_means: dict[int, float] = {}
        trend_means: dict[int, float] = {}

        for state in range(self.n_regimes):
            mask = raw_states == state
            if np.sum(mask) > 0:
                vol_means[state] = float(np.mean(X[mask, 0]))   # vol_14
                trend_means[state] = float(np.mean(X[mask, 3])) # abs_trend
            else:
                vol_means[state] = float(state)
                trend_means[state] = float(state)

        # Volatiliteye göre sırala
        sorted_by_vol = sorted(vol_means.keys(), key=lambda s: vol_means[s])
        low_vol_state = sorted_by_vol[0]
        remaining = sorted_by_vol[1:]

        # Kalan 2 durumdan trend momentumu yüksek olan -> REGIME_STRONG_TREND (2), diğeri -> CHOP (1)
        if trend_means[remaining[0]] >= trend_means[remaining[1]]:
            trend_state = remaining[0]
            chop_state = remaining[1]
        else:
            trend_state = remaining[1]
            chop_state = remaining[0]

        self._state_map = {
            low_vol_state: self.REGIME_LOW_VOLATILITY,
            chop_state: self.REGIME_CHOPPY_HIGH_VOL,
            trend_state: self.REGIME_STRONG_TREND,
        }

    def predict_regime(self, df: pd.DataFrame) -> Tuple[int, str, float]:
        """Predict the current regime from the latest candle series.

        Returns:
            (regime_id, regime_name, recommended_probability_threshold)
        """
        if not self.is_fitted or self.model is None or df.empty or len(df) < 20:
            # Heuristik kural tabanlı acil durum tahmini
            return self._heuristic_regime_fallback(df)

        features = self.extract_regime_features(df)
        if features.empty:
            return self._heuristic_regime_fallback(df)

        try:
            X_last = features.values[-1:]
            raw_state = int(self.model.predict(X_last)[0])
            calibrated_regime = self._state_map.get(raw_state, self.REGIME_LOW_VOLATILITY)
            name = self.REGIME_NAMES[calibrated_regime]
            threshold = self.REGIME_THRESHOLDS[calibrated_regime]
            return calibrated_regime, name, threshold
        except Exception as e:
            log.debug("Regime model inference failed (%s), using heuristic.", e)
            return self._heuristic_regime_fallback(df)

    def _heuristic_regime_fallback(self, df: pd.DataFrame) -> Tuple[int, str, float]:
        """Heuristic rule-based fallback when model is not yet fitted."""
        if df.empty or len(df) < 15:
            return self.REGIME_LOW_VOLATILITY, self.REGIME_NAMES[0], 0.50

        close = df["close"].astype(float)
        ret_1 = close.pct_change(1).fillna(0.0)
        vol = ret_1.rolling(14).std().iloc[-1]
        ret_5 = close.pct_change(5).fillna(0.0).abs().iloc[-1]

        # Eşikler
        if vol < 0.010:
            regime = self.REGIME_LOW_VOLATILITY
        elif ret_5 > 0.035 and vol >= 0.010:
            regime = self.REGIME_STRONG_TREND
        else:
            regime = self.REGIME_CHOPPY_HIGH_VOL

        return regime, self.REGIME_NAMES[regime], self.REGIME_THRESHOLDS[regime]


# Module-level singleton
market_regime_detector = MarketRegimeDetector()
