"""Machine Learning Quantitative Strategy Engine – State-of-the-Art Quant Pipeline.

Features:
- Advanced "Alpha" Features (Autocorrelation, Multi-Timeframe Momentum Proxy, Volatility Regime Z-Score, Choppiness Index).
- Dynamic Triple-Barrier Method (Marcos López de Prado) with rolling volatility barriers.
- Bayesian Hyperparameter Optimization with Optuna and TimeSeriesSplit cross-validation.
- Precision Confidence Thresholding (predict_proba >= 50% sniper entry filter).
- Multi-asset pooling with strictly causal, zero look-ahead bias engineering.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

import config
from logger_setup import get_logger

log = get_logger("ml_strategy")


# ─────────────────────────────────────────────────────────────────────
# Technical Indicator Helpers
# ─────────────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    """True Range (TR)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = compute_true_range(df)
    return tr.rolling(window=period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Trend strength)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = compute_true_range(df)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / (atr + 1e-9))

    dx = (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9) * 100
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx


# ─────────────────────────────────────────────────────────────────────
# ML Strategy Core Class
# ─────────────────────────────────────────────────────────────────────

class MLStrategy:
    """State-of-the-Art Quant Strategy Engine."""

    LABEL_MAP = {-1: 0, 0: 1, 1: 2}
    INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            if os.path.exists(config.MODEL_OPT_PATH):
                self.model_path = config.MODEL_OPT_PATH
            else:
                self.model_path = config.MODEL_PATH
        else:
            self.model_path = model_path

        self.model: Any = None
        self.feature_names: list[str] = []
        self.model_type: str = "XGBOOST"
        self._last_loaded_mtime: float = 0.0
        self._load_cached_model()

    def _load_cached_model(self) -> bool:
        """Load model artifact from disk into memory."""
        if os.path.exists(self.model_path):
            try:
                self._last_loaded_mtime = os.path.getmtime(self.model_path)
                artifact = joblib.load(self.model_path)
                if isinstance(artifact, dict) and "model" in artifact:
                    self.model = artifact["model"]
                    self.feature_names = artifact.get("feature_names", [])
                    self.model_type = artifact.get("model_type", "XGBOOST")
                else:
                    self.model = artifact
                    self.feature_names = getattr(self.model, "feature_names_in_", []).tolist()
                log.info("Loaded ML model artifact from '%s' [Type: %s, Features: %d].",
                         self.model_path, self.model_type, len(self.feature_names))
                return True
            except Exception as e:
                log.warning("Could not load model from '%s': %s", self.model_path, e)
        return False

    def check_and_reload_model(self) -> bool:
        """Check if model artifact on disk was updated and hot-reload seamlessly."""
        if os.path.exists(self.model_path):
            try:
                mtime = os.path.getmtime(self.model_path)
                if mtime > self._last_loaded_mtime + 1.0:
                    log.info("[HOT-RELOAD] Newer model weights detected on disk! Reloading '%s' …", self.model_path)
                    return self._load_cached_model()
            except Exception as e:
                log.debug("Error checking model file mtime: %s", e)
        return False

    @staticmethod
    def generate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced stationary quant & alpha features from raw OHLCV.

        Advanced Alpha Feature Pipeline:
        1. Multi-horizon Returns (1, 3, 5, 10 bars)
        2. Auto-Correlation (Lag-1 return correlation over 20 bars – Trend Persistence)
        3. Multi-Timeframe Momentum Proxies:
           - 1D Proxy (6 bars x 4h): Return + Distance to EMA(6)
           - 1W Proxy (42 bars x 4h): Return + Distance to EMA(42)
        4. Volatility Regime (ATR Z-Score over 50 bars – Compression vs Expansion)
        5. Choppiness Index (CHOP 14 bars – Trend vs Consolidation)
        6. RSI(14) + RSI Slope (Momentum direction)
        7. MACD Histogram Normalized + Acceleration
        8. Bollinger Bands (%B and Bandwidth)
        9. Volume Dynamics (Volume Ratio, Volume Return, Volume Z-Score)
        10. Candle Wick Dynamics (Upper/Lower wick ratios, 3-bar and 5-bar averages)
        11. Trend Distances to EMAs (20, 50, 100) + EMA slope
        12. Market Regime (ADX) & Stochastic Channel Position (%K)
        """
        if df.empty or len(df) < 60:
            return pd.DataFrame()

        df_feat = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_p = df["open"]
        volume = df["volume"]

        # 1. Multi-horizon Returns
        ret_1 = close.pct_change(1)
        df_feat["ret_1"] = ret_1
        df_feat["ret_3"] = close.pct_change(3)
        df_feat["ret_5"] = close.pct_change(5)
        df_feat["ret_10"] = close.pct_change(10)

        # 2. [ALPHA] Auto-Correlation (Fiyatın kendisiyle korelasyonu - Trend Gücü)
        ret_lag1 = ret_1.shift(1)
        cov_20 = ret_1.rolling(20).cov(ret_lag1)
        var_ret = ret_1.rolling(20).var()
        var_lag = ret_lag1.rolling(20).var()
        df_feat["autocorr_1"] = (cov_20 / (np.sqrt(var_ret * var_lag) + 1e-9)).clip(-1.0, 1.0)

        # 3. [ALPHA] Multi-Timeframe Momentum Proxies (4h içinde 1d ve 1w eşdeğerleri)
        # 1 Gün = 6 bar x 4h
        df_feat["mtf_ret_1d"] = close.pct_change(6)
        ema_1d = close.ewm(span=6, adjust=False).mean()
        df_feat["mtf_ema_1d_dist"] = (close - ema_1d) / (ema_1d + 1e-9)

        # 1 Hafta = 42 bar x 4h (7 gün)
        df_feat["mtf_ret_1w"] = close.pct_change(42)
        ema_1w = close.ewm(span=42, adjust=False).mean()
        df_feat["mtf_ema_1w_dist"] = (close - ema_1w) / (ema_1w + 1e-9)

        # 4. [ALPHA] Oynaklık Rejimi (Volatility Regime - ATR 50 Z-Skoru)
        tr = compute_true_range(df)
        atr_14 = tr.rolling(window=14).mean()
        atr_mean_50 = atr_14.rolling(window=50).mean()
        atr_std_50 = atr_14.rolling(window=50).std()
        df_feat["vol_regime_zscore"] = ((atr_14 - atr_mean_50) / (atr_std_50 + 1e-9)).clip(-3.0, 3.0)

        # 5. [ALPHA] Choppiness Index (CHOP - Yatay vs Trend Filtresi)
        tr_sum_14 = tr.rolling(window=14).sum()
        max_hi_14 = high.rolling(window=14).max()
        min_lo_14 = low.rolling(window=14).min()
        chop_ratio = tr_sum_14 / ((max_hi_14 - min_lo_14) + 1e-9)
        chop_14 = 100.0 * (np.log10(np.maximum(chop_ratio, 1e-4)) / np.log10(14.0))
        # Normalize: -0.5 ile +0.5 arası (Pozitif = Yatay/Choppy, Negatif = Güçlü Trend)
        df_feat["chop_14"] = (chop_14.clip(0.0, 100.0) - 50.0) / 50.0

        # 6. RSI & RSI Slope
        rsi_14 = compute_rsi(close, period=14)
        df_feat["rsi_14"] = (rsi_14 - 50.0) / 50.0
        df_feat["rsi_slope_3"] = rsi_14.diff(3) / 3.0

        # 7. MACD Normalized & Acceleration
        _, _, macd_hist = compute_macd(close, fast=12, slow=26, signal=9)
        df_feat["macd_hist_norm"] = macd_hist / (atr_14 + 1e-9)
        df_feat["macd_hist_accel"] = (macd_hist - macd_hist.shift(1)) / (atr_14 + 1e-9)

        # 8. Volatility Features
        df_feat["volatility_atr_pct"] = atr_14 / (close + 1e-9)
        df_feat["candle_range_pct"] = (high - low) / (close + 1e-9)

        # 9. Bollinger Bands (%B ve Band Genişliği)
        bb_sma_20 = close.rolling(window=20).mean()
        bb_std_20 = close.rolling(window=20).std()
        bb_upper = bb_sma_20 + (2.0 * bb_std_20)
        bb_lower = bb_sma_20 - (2.0 * bb_std_20)
        bb_span = bb_upper - bb_lower + 1e-9
        df_feat["bb_pct_b"] = (close - bb_lower) / bb_span
        df_feat["bb_width"] = bb_span / (bb_sma_20 + 1e-9)

        # 10. Volume Dynamics & Hacim Z-Score
        vol_ma_20 = volume.rolling(window=20).mean()
        vol_std_20 = volume.rolling(window=20).std()
        df_feat["vol_ratio"] = volume / (vol_ma_20 + 1e-9)
        df_feat["vol_ret_1"] = volume.pct_change(1)
        df_feat["vol_zscore"] = ((volume - vol_ma_20) / (vol_std_20 + 1e-9)).clip(-3.0, 3.0)

        # 11. Fitil (Wick) Oranları
        body_max = np.maximum(open_p, close)
        body_min = np.minimum(open_p, close)
        full_range = (high - low) + 1e-9
        upper_wick = high - body_max
        lower_wick = body_min - low

        upper_wick_ratio = upper_wick / full_range
        lower_wick_ratio = lower_wick / full_range
        df_feat["upper_wick_ratio"] = upper_wick_ratio
        df_feat["lower_wick_ratio"] = lower_wick_ratio
        df_feat["upper_wick_ma3"] = upper_wick_ratio.rolling(3).mean()
        df_feat["lower_wick_ma3"] = lower_wick_ratio.rolling(3).mean()
        df_feat["upper_wick_ma5"] = upper_wick_ratio.rolling(5).mean()
        df_feat["lower_wick_ma5"] = lower_wick_ratio.rolling(5).mean()

        # 12. Trend & Moving Average Distances
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        ema_100 = close.ewm(span=100, adjust=False).mean()

        df_feat["dist_ema_20"] = (close - ema_20) / (ema_20 + 1e-9)
        df_feat["dist_ema_50"] = (close - ema_50) / (ema_50 + 1e-9)
        df_feat["dist_ema_100"] = (close - ema_100) / (ema_100 + 1e-9)
        df_feat["ema_trend_slope"] = (ema_20 - ema_50) / (ema_50 + 1e-9)

        # 13. Market Regime (ADX) & Stochastic Channel Pos
        adx_14 = compute_adx(df, period=14)
        df_feat["adx_14"] = adx_14 / 100.0

        rolling_high_14 = high.rolling(window=14).max()
        rolling_low_14 = low.rolling(window=14).min()
        df_feat["channel_pos_14"] = (close - rolling_low_14) / ((rolling_high_14 - rolling_low_14) + 1e-9)

        # 14. [ALPHA DERIVATIVE] Open Interest & Funding Rate Dynamics
        if "open_interest" in df.columns:
            oi = df["open_interest"].astype(float)
            df_feat["OI_Change"] = oi.pct_change(1).fillna(0.0).clip(-1.0, 1.0)
        elif "OI_Change" in df.columns:
            df_feat["OI_Change"] = df["OI_Change"].astype(float).fillna(0.0).clip(-1.0, 1.0)
        else:
            df_feat["OI_Change"] = 0.0

        if "funding_rate" in df.columns:
            fr = df["funding_rate"].astype(float)
            fr_ma = fr.rolling(window=3, min_periods=1).mean()
            df_feat["Funding_Trend"] = ((fr - fr_ma) * 10000.0).fillna(0.0).clip(-10.0, 10.0)
        elif "Funding_Trend" in df.columns:
            df_feat["Funding_Trend"] = df["Funding_Trend"].astype(float).fillna(0.0).clip(-10.0, 10.0)
        else:
            df_feat["Funding_Trend"] = 0.0

        df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_feat.dropna(inplace=True)

        return df_feat

    @staticmethod
    def generate_labels(
        df: pd.DataFrame,
        lookahead: int = config.LOOKAHEAD_PERIODS,
        target_profit_pct: float = config.TARGET_PROFIT_PCT,
        stop_loss_pct: float = config.STOP_LOSS_PCT,
    ) -> pd.Series:
        """Dynamic Triple-Barrier Labeling (Marcos López de Prado) or Static Fallback.

        Triple Barrier Formulation:
          Upper Barrier (TP): P_t * (1 + pt_factor * sigma_t)
          Lower Barrier (SL): P_t * (1 - sl_factor * sigma_t)
          Vertical Barrier  : t + vertical_barrier candles

        Touch Priority:
          First touch of horizontal barriers determines label (+1 or -1).
          If neither touched before vertical barrier, terminal return determines class.
        """
        if getattr(config, "USE_TRIPLE_BARRIER", True):
            return MLStrategy.generate_triple_barrier_labels(
                df,
                vol_window=config.TB_VOLATILITY_WINDOW,
                pt_factor=config.TB_PT_FACTOR,
                sl_factor=config.TB_SL_FACTOR,
                vertical_barrier=config.TB_VERTICAL_BARRIER,
            )

        # Sabit Eşikli Yedek Mantık
        close = df["close"]
        high = df["high"]
        low = df["low"]

        target_gain = target_profit_pct / 100.0
        target_loss = stop_loss_pct / 100.0

        n = len(df)
        labels = pd.Series(index=df.index, dtype=float)

        high_vals = high.values
        low_vals = low.values
        close_vals = close.values
        out_labels = np.full(n, np.nan)

        for i in range(n - lookahead):
            current_c = close_vals[i]
            window_highs = high_vals[i + 1 : i + 1 + lookahead]
            window_lows = low_vals[i + 1 : i + 1 + lookahead]

            max_gain = (np.max(window_highs) - current_c) / current_c
            max_loss = (np.min(window_lows) - current_c) / current_c

            if max_gain >= target_gain and max_loss > -target_loss:
                out_labels[i] = 1.0
            elif max_loss <= -target_loss and max_gain < target_gain:
                out_labels[i] = -1.0
            else:
                out_labels[i] = 0.0

        labels[:] = out_labels
        return labels

    @staticmethod
    def generate_triple_barrier_labels(
        df: pd.DataFrame,
        vol_window: int = 20,
        pt_factor: float = 2.0,
        sl_factor: float = 1.5,
        vertical_barrier: int = 8,
    ) -> pd.Series:
        """Marcos López de Prado's Triple-Barrier Method.

        Calculates dynamic barriers proportional to rolling daily return volatility.
        Strictly zero lookahead: volatility sigma_t is computed using closed bars <= t.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_p = df["open"]

        # Causal rolling volatility estimation
        returns = close.pct_change(1)
        vol = returns.rolling(window=vol_window).std()
        vol = vol.bfill().clip(lower=0.005)  # Minimum %0.5 volatilite tabanı

        n = len(df)
        labels = pd.Series(index=df.index, dtype=float)

        high_vals = high.values
        low_vals = low.values
        close_vals = close.values
        open_vals = open_p.values
        vol_vals = vol.values
        out_labels = np.full(n, np.nan)

        for i in range(n - vertical_barrier):
            p0 = close_vals[i]
            sigma = vol_vals[i]

            tp_price = p0 * (1.0 + (pt_factor * sigma))
            sl_price = p0 * (1.0 - (sl_factor * sigma))

            label = 0.0
            barrier_hit = False

            # Forward window boyunca ilk dokunulan bariyeri ara
            for step in range(1, vertical_barrier + 1):
                idx = i + step
                bar_hi = high_vals[idx]
                bar_lo = low_vals[idx]

                hit_upper = bar_hi >= tp_price
                hit_lower = bar_lo <= sl_price

                if hit_upper and hit_lower:
                    # Aynı mumda iki bariyer de vurulduysa mumu yönüne bak
                    label = 1.0 if close_vals[idx] >= open_vals[idx] else -1.0
                    barrier_hit = True
                    break
                elif hit_upper:
                    label = 1.0
                    barrier_hit = True
                    break
                elif hit_lower:
                    label = -1.0
                    barrier_hit = True
                    break

            # Zaman bariyeri (Vertical barrier doldu ve yatay bariyerlere dokunulmadı)
            if not barrier_hit:
                final_ret = (close_vals[i + vertical_barrier] - p0) / p0
                threshold_gain = 0.5 * pt_factor * sigma
                threshold_loss = -0.5 * sl_factor * sigma

                if final_ret >= threshold_gain:
                    label = 1.0
                elif final_ret <= threshold_loss:
                    label = -1.0
                else:
                    label = 0.0

            out_labels[i] = label

        labels[:] = out_labels
        return labels

    def optimize_hyperparameters(
        self,
        df_dict: dict[str, pd.DataFrame],
        n_trials: int = config.OPTUNA_TRIALS,
    ) -> Dict[str, Any]:
        """Bayesian Hyperparameter Optimization using Optuna with TimeSeriesSplit CV.

        Optimizes max_depth, n_estimators, learning_rate, subsample, colsample_bytree,
        gamma, and min_child_weight to maximize Out-of-Sample Macro F1 Score.
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna is not installed. Run `pip install optuna` first!")

        log.info("=" * 70)
        log.info("STARTING OPTUNA BAYESIAN HYPERPARAMETER OPTIMIZATION (%d Trials)", n_trials)
        log.info("=" * 70)

        # 1. Tüm coinlerin veri matrisini oluştur
        all_records: list[pd.DataFrame] = []
        for symbol, df in df_dict.items():
            if df.empty or len(df) < (config.TB_VERTICAL_BARRIER + 100):
                continue
            feats = self.generate_features(df)
            labels = self.generate_labels(df)
            combined = feats.join(labels.rename("label"), how="inner").dropna(subset=["label"])
            combined["symbol"] = symbol
            if not combined.empty:
                all_records.append(combined)

        if not all_records:
            raise ValueError("No valid samples could be extracted for hyperparameter optimization!")

        master_df = pd.concat(all_records, axis=0).sort_index()
        feature_cols = [c for c in master_df.columns if c not in ("label", "symbol")]
        self.feature_names = feature_cols

        # Time-Series Split: %80 Train / Val havuzu, %20 Gerçek OOS Test
        split_idx = int(len(master_df) * config.TRAIN_TEST_SPLIT_RATIO)
        train_pool = master_df.iloc[:split_idx]
        oos_test = master_df.iloc[split_idx:]

        X_train_pool = train_pool[feature_cols]
        y_train_pool = train_pool["label"].astype(int).map(self.LABEL_MAP).values

        X_test = oos_test[feature_cols]
        y_test_raw = oos_test["label"].astype(int).values
        y_test_mapped = pd.Series(y_test_raw).map(self.LABEL_MAP).values

        log.info("Optuna Pool: %d samples across %d features | Final Test: %d samples",
                 len(train_pool), len(feature_cols), len(oos_test))

        # 2. Optuna Objective Fonksiyonu
        tscv = TimeSeriesSplit(n_splits=config.OPTUNA_CV_SPLITS)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 0.95),
                "gamma": trial.suggest_float("gamma", 0.05, 3.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "random_state": 42,
                "eval_metric": "mlogloss",
                "n_jobs": -1,
            }

            cv_scores = []
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_pool)):
                X_tr = X_train_pool.iloc[train_idx].copy()
                y_tr = y_train_pool[train_idx].copy()
                X_va = X_train_pool.iloc[val_idx]
                y_va = y_train_pool[val_idx]

                # Tüm 3 sınıfın [0, 1, 2] fold içinde yer almasını sağla (sıfır ağırlıklı koruma)
                missing_classes = set([0, 1, 2]) - set(np.unique(y_tr))
                if missing_classes:
                    dummy_X = pd.DataFrame(np.zeros((len(missing_classes), X_tr.shape[1])), columns=X_tr.columns)
                    dummy_y = np.array(list(missing_classes), dtype=int)
                    w_tr = np.concatenate([np.ones(len(y_tr)), np.zeros(len(missing_classes))])
                    X_tr = pd.concat([X_tr, dummy_X], ignore_index=True)
                    y_tr = np.concatenate([y_tr, dummy_y])
                else:
                    w_tr = np.ones(len(y_tr))

                clf = xgb.XGBClassifier(**params)
                clf.fit(X_tr, y_tr, sample_weight=w_tr)

                y_val_pred = clf.predict(X_va)
                score = f1_score(y_va, y_val_pred, average="macro", zero_division=0)
                cv_scores.append(score)

            return float(np.mean(cv_scores)) if cv_scores else 0.0

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_cv_f1 = study.best_value

        print("\n" + "=" * 70)
        print("          OPTUNA BAYESIAN OPTIMIZATION RESULTS          ")
        print("=" * 70)
        print(f"En İyi TimeSeriesSplit CV Macro F1: {best_cv_f1:.4f}")
        print("Bulunan En Optimal Hiperparametreler:")
        for k, v in best_params.items():
            if isinstance(v, float):
                print(f"  - {k:<20}: {v:.5f}")
            else:
                print(f"  - {k:<20}: {v}")
        print("=" * 70 + "\n")

        # 3. Bulunan En İyi Parametrelerle Nihai Modeli Eğit
        log.info("Training final production model using Optuna best hyperparameters …")
        final_params = {
            **best_params,
            "random_state": 42,
            "eval_metric": "mlogloss",
            "n_jobs": -1,
        }
        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(X_train_pool, y_train_pool)
        self.model = final_model
        self.model_type = "XGBOOST"

        # 4. Out-of-Sample Değerlendirmesi
        y_pred_mapped = final_model.predict(X_test)
        y_probs = final_model.predict_proba(X_test)
        y_pred = pd.Series(y_pred_mapped).map(self.INV_LABEL_MAP).values

        acc = accuracy_score(y_test_raw, y_pred)
        f1_macro = f1_score(y_test_raw, y_pred, average="macro", zero_division=0)

        max_probs = np.max(y_probs, axis=1)
        confident_mask = max_probs >= config.PROBABILITY_THRESHOLD
        confident_count = int(np.sum(confident_mask))

        print("\n" + "=" * 70)
        print("      OPTIMIZED MODEL FINAL OUT-OF-SAMPLE EVALUATION      ")
        print("=" * 70)
        print(f"Genel Test Seti Doğruluğu              : {acc * 100:.2f}%")
        print(f"Macro F1-Score                         : {f1_macro:.4f}")
        print(f"Toplam Test Örneği                     : {len(X_test)}")
        print(f"[>= %{int(config.PROBABILITY_THRESHOLD*100)}] Güvenle Açılan İşlem Sayısı : {confident_count} ({confident_count/len(X_test)*100:.1f}%)")

        if confident_count > 0:
            conf_acc = accuracy_score(y_test_raw[confident_mask], y_pred[confident_mask])
            print(f"[*] KESKIN NISANCI DOGRULUGU (>= %{int(config.PROBABILITY_THRESHOLD*100)})    : {conf_acc * 100:.2f}%")

        print("\nClassification Matrix:")
        print(classification_report(y_test_raw, y_pred, zero_division=0, target_names=["SELL (-1)", "HOLD (0)", "BUY (1)"]))

        importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("Top 10 En Güçlü Quant Öznitelikleri:")
        for feat, score in importances.head(10).items():
            print(f"  - {feat:<22}: {score:.4f}")
        print("=" * 70 + "\n")

        # 5. Modeli Kaydet
        self.model_path = config.MODEL_OPT_PATH
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        artifact = {
            "model": final_model,
            "feature_names": feature_cols,
            "model_type": "XGBOOST",
            "best_params": best_params,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "use_triple_barrier": True,
                "tb_vol_window": config.TB_VOLATILITY_WINDOW,
                "tb_pt_factor": config.TB_PT_FACTOR,
                "tb_sl_factor": config.TB_SL_FACTOR,
                "tb_vertical_barrier": config.TB_VERTICAL_BARRIER,
                "probability_threshold": config.PROBABILITY_THRESHOLD,
            },
        }
        joblib.dump(artifact, self.model_path)
        log.info("Optimized model artifact successfully saved to '%s'", self.model_path)

        return best_params

    def train_model(
        self,
        df_dict: dict[str, pd.DataFrame],
        opt: bool = False,
        save_path: Optional[str] = None,
    ) -> None:
        """Standard model training pipeline with preset hyperparameters."""
        target_path = save_path or (config.MODEL_OPT_PATH if opt else config.MODEL_PATH)
        log.info("Preparing feature engineering across %d symbols (Optimized=%s) …", len(df_dict), opt)

        all_records: list[pd.DataFrame] = []

        for symbol, df in df_dict.items():
            if df.empty or len(df) < (config.TB_VERTICAL_BARRIER + 100):
                continue

            features = self.generate_features(df)
            labels = self.generate_labels(df)

            combined = features.join(labels.rename("label"), how="inner")
            combined.dropna(subset=["label"], inplace=True)
            combined["symbol"] = symbol

            if not combined.empty:
                all_records.append(combined)

        if not all_records:
            raise ValueError("No valid training samples extracted from datasets!")

        master_df = pd.concat(all_records, axis=0)
        master_df.sort_index(inplace=True)

        feature_cols = [c for c in master_df.columns if c not in ("label", "symbol")]
        self.feature_names = feature_cols

        log.info(
            "Consolidated multi-asset dataset: %d samples across %d features.",
            len(master_df), len(feature_cols),
        )

        split_idx = int(len(master_df) * config.TRAIN_TEST_SPLIT_RATIO)
        train_df = master_df.iloc[:split_idx]
        test_df = master_df.iloc[split_idx:]

        X_train, y_train_raw = train_df[feature_cols], train_df["label"].astype(int)
        X_test, y_test_raw = test_df[feature_cols], test_df["label"].astype(int)

        log.info(
            "Temporal Split -> Train: %d (%s -> %s) | Test: %d (%s -> %s)",
            len(X_train), train_df.index[0], train_df.index[-1],
            len(X_test), test_df.index[0], test_df.index[-1],
        )

        use_xgboost = (opt and config.OPT_MODEL_TYPE == "XGBOOST" and HAS_XGBOOST) or HAS_XGBOOST

        if use_xgboost:
            self.model_type = "XGBOOST"
            log.info("Training XGBoost Classifier (n_estimators=%d, max_depth=%d, lr=%.3f) …",
                     config.XGB_N_ESTIMATORS, config.XGB_MAX_DEPTH, config.XGB_LEARNING_RATE)

            y_train_mapped = y_train_raw.map(self.LABEL_MAP).values

            model = xgb.XGBClassifier(
                n_estimators=config.XGB_N_ESTIMATORS,
                max_depth=config.XGB_MAX_DEPTH,
                learning_rate=config.XGB_LEARNING_RATE,
                subsample=config.XGB_SUBSAMPLE,
                colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
                gamma=config.XGB_GAMMA,
                min_child_weight=config.XGB_MIN_CHILD_WEIGHT,
                random_state=config.RF_RANDOM_STATE,
                eval_metric="mlogloss",
                n_jobs=-1,
            )
            model.fit(X_train, y_train_mapped)
            self.model = model

            y_pred_mapped = model.predict(X_test)
            y_probs = model.predict_proba(X_test)
            y_pred = pd.Series(y_pred_mapped, index=X_test.index).map(self.INV_LABEL_MAP).values

        else:
            self.model_type = "RANDOM_FOREST"
            n_est = config.RF_OPT_N_ESTIMATORS if opt else config.RF_N_ESTIMATORS
            depth = config.RF_OPT_MAX_DEPTH if opt else config.RF_MAX_DEPTH
            split = config.RF_OPT_MIN_SAMPLES_SPLIT if opt else config.RF_MIN_SAMPLES_SPLIT
            leaf = config.RF_OPT_MIN_SAMPLES_LEAF if opt else config.RF_MIN_SAMPLES_LEAF

            log.info("Training RandomForestClassifier (n_estimators=%d, max_depth=%d) …", n_est, depth)
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth,
                min_samples_split=split,
                min_samples_leaf=leaf,
                class_weight=config.RF_CLASS_WEIGHT,
                random_state=config.RF_RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(X_train, y_train_raw)
            self.model = model

            y_pred = model.predict(X_test)
            y_probs = model.predict_proba(X_test)

        acc = accuracy_score(y_test_raw, y_pred)
        f1_macro = f1_score(y_test_raw, y_pred, average="macro", zero_division=0)

        max_probs = np.max(y_probs, axis=1)
        confident_mask = max_probs >= config.PROBABILITY_THRESHOLD
        confident_count = np.sum(confident_mask)

        print("\n" + "=" * 70)
        print(f"       QUANT ML MODEL EVALUATION [{self.model_type}] (OUT-OF-SAMPLE)       ")
        print("=" * 70)
        print(f"Tüm Test Seti Doğruluğu (Genel)        : {acc * 100:.2f}%")
        print(f"Macro F1-Score                         : {f1_macro:.4f}")
        print(f"Toplam Test Örnek Sayısı               : {len(X_test)}")
        print(f"[>= %{int(config.PROBABILITY_THRESHOLD*100)}] Guvenle Acilan Islem Sayisi : {confident_count} ({confident_count/len(X_test)*100:.1f}%)")

        if confident_count > 0:
            conf_acc = accuracy_score(y_test_raw[confident_mask], y_pred[confident_mask])
            print(f"[*] KESKIN NISANCI DOGRULUGU (>= %{int(config.PROBABILITY_THRESHOLD*100)})    : {conf_acc * 100:.2f}%")

        print("\nClassification Matrix (Tüm Örnekler):")
        print(classification_report(y_test_raw, y_pred, zero_division=0, target_names=["SELL (-1)", "HOLD (0)", "BUY (1)"]))

        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\nTop 10 En Güçlü Quant Öznitelikleri:")
        for feat, score in importances.head(10).items():
            print(f"  - {feat:<22}: {score:.4f}")
        print("=" * 70 + "\n")

        self.model_path = target_path
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        artifact = {
            "model": model,
            "feature_names": feature_cols,
            "model_type": self.model_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "use_triple_barrier": config.USE_TRIPLE_BARRIER,
                "probability_threshold": config.PROBABILITY_THRESHOLD,
            },
        }
        joblib.dump(artifact, self.model_path)
        log.info("Model artifact successfully saved to '%s'", self.model_path)

    def predict_signal_with_confidence(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> Tuple[int, float, Dict[int, float]]:
        """Predict market direction with strict confidence thresholding."""
        if threshold is None:
            threshold = config.PROBABILITY_THRESHOLD

        if self.model is None:
            if not self._load_cached_model():
                log.error("No model found. Train model first before predicting!")
                return 0, 0.0, {-1: 0.0, 0: 1.0, 1: 0.0}

        features = self.generate_features(df)
        if features.empty:
            return 0, 0.0, {-1: 0.0, 0: 1.0, 1: 0.0}

        last_features = features.iloc[[-1]]
        missing = [c for c in self.feature_names if c not in last_features.columns]
        if missing:
            log.error("Missing required features for inference: %s", missing)
            return 0, 0.0, {-1: 0.0, 0: 1.0, 1: 0.0}

        X_eval = last_features[self.feature_names]
        raw_probs = self.model.predict_proba(X_eval)[0]

        if self.model_type == "XGBOOST":
            classes = getattr(self.model, "classes_", [0, 1, 2])
            prob_dict: Dict[int, float] = {-1: 0.0, 0: 0.0, 1: 0.0}
            for cls_idx, prob in zip(classes, raw_probs):
                original_class = self.INV_LABEL_MAP.get(int(cls_idx), 0)
                prob_dict[original_class] = float(prob)
        else:
            classes = self.model.classes_
            prob_dict = {-1: 0.0, 0: 0.0, 1: 0.0}
            for cls, prob in zip(classes, raw_probs):
                prob_dict[int(cls)] = float(prob)

        raw_signal = max(prob_dict, key=prob_dict.get)
        confidence = prob_dict[raw_signal]

        # ── KESKİN NİŞANCI FİLTRESİ ─────────────────────────────────────────
        if confidence < threshold:
            final_signal = 0
        else:
            final_signal = raw_signal

        return final_signal, confidence, prob_dict

    def predict_signal(self, df: pd.DataFrame) -> int:
        signal, _, _ = self.predict_signal_with_confidence(df)
        return signal
