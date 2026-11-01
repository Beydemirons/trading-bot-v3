"""Walk-Forward Auto-Retraining Engine – Continuous Learning & Market Adaptation.

Handles rolling-window dataset updates and automated background retraining:
  - Rolling Window: Drops oldest period, appends latest 1 week of out-of-sample data.
  - Background Scheduler: Triggers Optuna Bayesian hyperparameter search periodically
    (e.g., Every Sunday at 00:00 UTC) without halting the live execution loop.
  - Hot-Reload Artifacts: Atomically replaces model artifacts on disk.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Dict

import pandas as pd

import config
from logger_setup import get_logger
from ml_strategy import MLStrategy

log = get_logger("retrainer")


class WalkForwardRetrainer:
    """Institutional Rolling-Window Continuous Learning Engine."""

    def __init__(
        self,
        symbols: Optional[list[str]] = None,
        rolling_weeks: int = config.RETRAIN_ROLLING_WEEKS,
        interval_hours: int = config.RETRAIN_INTERVAL_HOURS,
    ):
        self.symbols = symbols or config.SYMBOLS
        self.rolling_weeks = rolling_weeks
        self.interval_hours = interval_hours
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._last_retrain_time: Optional[datetime] = None

    def fetch_rolling_datasets(self) -> Dict[str, pd.DataFrame]:
        """Fetch updated OHLCV for all symbols and enforce rolling window."""
        from data_loader import load_data

        datasets: Dict[str, pd.DataFrame] = {}
        # 4h mumlar: 1 hafta = 7 gün * 6 mum = 42 mum.
        # rolling_weeks kadar en yeni mumu tut (en eskiyi kırp)
        max_candles = self.rolling_weeks * 7 * 6

        log.info(
            "[WALK-FORWARD] Updating rolling dataset for %d symbols (Window: %d weeks / ~%d candles) …",
            len(self.symbols), self.rolling_weeks, max_candles
        )

        for sym in self.symbols:
            try:
                df = load_data(symbol=sym, timeframe=config.TIMEFRAME)
                if not df.empty:
                    # En eski verileri çıkararak kayan pencereyi koru
                    if len(df) > max_candles:
                        df_rolling = df.iloc[-max_candles:].copy()
                    else:
                        df_rolling = df.copy()
                    datasets[sym] = df_rolling
            except Exception as e:
                log.error("[WALK-FORWARD] Error fetching data for %s: %s", sym, e)

        return datasets

    def execute_retrain(self, n_trials: int = 20) -> bool:
        """Execute full Bayesian hyperparameter optimization and model update."""
        log.info("=" * 75)
        log.info("[WALK-FORWARD RETRAINING] STARTING SCHEDULED OPTUNA MODEL UPDATE")
        log.info("Timestamp: %s UTC", datetime.now(timezone.utc).isoformat())
        log.info("=" * 75)

        datasets = self.fetch_rolling_datasets()
        if not datasets:
            log.error("[WALK-FORWARD] No market datasets available for retraining.")
            return False

        try:
            strategy = MLStrategy(model_path=config.MODEL_OPT_PATH)
            strategy.optimize_hyperparameters(datasets, n_trials=n_trials)
            self._last_retrain_time = datetime.now(timezone.utc)
            log.info("=" * 75)
            log.info("[WALK-FORWARD] Model retraining completed successfully. Saved to '%s'", config.MODEL_OPT_PATH)
            log.info("=" * 75)
            return True
        except Exception as e:
            log.error("[WALK-FORWARD] Model retraining encountered an error: %s", e, exc_info=True)
            return False

    def is_retrain_due(self) -> bool:
        """Check if scheduled retrain time is reached (Sunday 00:00 UTC or interval elapsed)."""
        now = datetime.now(timezone.utc)

        # 1. Pazar Günü 00:00 UTC Kontrolü
        is_sunday = now.weekday() == getattr(config, "RETRAIN_DAY_OF_WEEK", 6)
        is_midnight_hour = now.hour == getattr(config, "RETRAIN_HOUR_UTC", 0)

        if is_sunday and is_midnight_hour:
            if self._last_retrain_time is None:
                return True
            # Aynı saat diliminde tekrar tekrar tetiklemeyi önle
            elapsed = (now - self._last_retrain_time).total_seconds()
            return elapsed > 3600.0 * 2.0

        # 2. Periyodik Saat Kontrolü (Opsiyonel aralık aşımı)
        if self._last_retrain_time is not None:
            elapsed_hours = (now - self._last_retrain_time).total_seconds() / 3600.0
            if elapsed_hours >= self.interval_hours:
                return True

        return False

    def _worker_loop(self) -> None:
        """Background daemon loop monitoring retraining triggers."""
        log.info("[WALK-FORWARD] Background retraining scheduler daemon active.")
        while self._is_running:
            try:
                if self.is_retrain_due():
                    log.info("[WALK-FORWARD] Scheduled retraining trigger fired! Starting optimization …")
                    self.execute_retrain(n_trials=config.OPTUNA_TRIALS)
            except Exception as e:
                log.error("[WALK-FORWARD] Scheduler loop error: %s", e)

            # 5 dakikada bir kontrol et
            for _ in range(300):
                if not self._is_running:
                    break
                time.sleep(1)

    def start_background(self) -> None:
        """Start non-blocking daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            log.warning("[WALK-FORWARD] Background retrainer is already active.")
            return

        self._is_running = True
        self._thread = threading.Thread(target=self._worker_loop, name="RetrainerDaemon", daemon=True)
        self._thread.start()
        log.info("[WALK-FORWARD] Retrainer background thread initialized.")

    def stop_background(self) -> None:
        """Signal background thread to stop cleanly."""
        self._is_running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            log.info("[WALK-FORWARD] Retrainer background thread stopped.")


# Module-level singleton
walk_forward_retrainer = WalkForwardRetrainer()
