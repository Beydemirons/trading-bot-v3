"""Trading Bot Orchestrator – Machine Learning Pipeline & Execution.

Usage:
    # 1. [SOTA] Optuna Bayesian Hiperparametre Optimizasyonu + Triple Barrier SOTA Eğitim:
    python main.py --mode OPTIMIZE --trials 50

    # 2. Önceden optimize edilmiş parametrelerle hızlı model eğitimi:
    python main.py --mode TRAIN_OPT

    # 3. Standart Random Forest modelini eğit ve kaydet:
    python main.py --mode TRAIN

    # 4. $10.000 sanal kasa, Dinamik Kelly Pozisyonlama ve Piyasa Güven Radarı ile Paper Trading:
    python main.py --mode PAPER --balance 10000

    # 5. Canlı borsa işlem modu (API key gerektirir):
    python main.py --mode LIVE
"""

from __future__ import annotations

import argparse
import sys
import pandas as pd

import config
from logger_setup import get_logger
from ml_strategy import MLStrategy

log = get_logger("main")


def run_optimization(trials: int = config.OPTUNA_TRIALS) -> None:
    """Run Optuna Bayesian Hyperparameter Optimization across multi-asset datasets."""
    from data_loader import load_data

    log.info("=" * 75)
    log.info("STARTING SOTA QUANT ML HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    log.info("Symbols (%d)   : %s", len(config.SYMBOLS), ", ".join(config.SYMBOLS))
    log.info("Timeframe      : %s | Trials: %d | TimeSeriesSplit Folds: %d",
             config.TIMEFRAME, trials, config.OPTUNA_CV_SPLITS)
    log.info("Labeling       : Dynamic Triple-Barrier (pt=%.1fx, sl=%.1fx, max_holding=%d)",
             config.TB_PT_FACTOR, config.TB_SL_FACTOR, config.TB_VERTICAL_BARRIER)
    log.info("Target Artifact: %s", config.MODEL_OPT_PATH)
    log.info("=" * 75)

    datasets: dict[str, pd.DataFrame] = {}

    for sym in config.SYMBOLS:
        try:
            log.info("Loading / updating dataset for %s …", sym)
            df = load_data(symbol=sym, timeframe=config.TIMEFRAME)
            if not df.empty:
                datasets[sym] = df
            else:
                log.warning("Received empty dataset for %s, skipping.", sym)
        except Exception as e:
            log.error("Failed to load data for %s: %s", sym, e)

    if not datasets:
        log.error("No valid market data could be loaded. Aborting optimization.")
        sys.exit(1)

    log.info("Assembled datasets for %d symbols. Launching Optuna engine …", len(datasets))
    strategy = MLStrategy(model_path=config.MODEL_OPT_PATH)
    strategy.optimize_hyperparameters(datasets, n_trials=trials)
    log.info("Bayesian optimization completed successfully. Model saved to '%s'", config.MODEL_OPT_PATH)


def run_training(opt: bool = True) -> None:
    """Fetch multi-asset historical data from cache/exchange and train the ML model."""
    from data_loader import load_data

    target_path = config.MODEL_OPT_PATH if opt else config.MODEL_PATH
    mode_name = "OPTIMIZED SNIPER ML (XGBoost / Deep RF)" if opt else "STANDARD RANDOM FOREST"

    log.info("=" * 75)
    log.info("STARTING QUANT TRAINING PIPELINE: %s", mode_name)
    log.info("Symbols (%d)   : %s", len(config.SYMBOLS), ", ".join(config.SYMBOLS))
    log.info("Timeframe      : %s", config.TIMEFRAME)
    log.info("Labeling       : Dynamic Triple-Barrier (%s)", "Active" if config.USE_TRIPLE_BARRIER else "Static")
    log.info("Sniper Filter  : Min Confidence >= %d%%", int(config.PROBABILITY_THRESHOLD * 100))
    log.info("Target Model   : %s", target_path)
    log.info("=" * 75)

    datasets: dict[str, pd.DataFrame] = {}

    for sym in config.SYMBOLS:
        try:
            log.info("Loading / updating dataset for %s …", sym)
            df = load_data(symbol=sym, timeframe=config.TIMEFRAME)
            if not df.empty:
                datasets[sym] = df
            else:
                log.warning("Received empty dataset for %s, skipping.", sym)
        except Exception as e:
            log.error("Failed to load data for %s: %s", sym, e)

    if not datasets:
        log.error("No valid market data could be loaded. Aborting training.")
        sys.exit(1)

    log.info("Successfully assembled datasets for %d symbols. Starting training engine …", len(datasets))
    strategy = MLStrategy(model_path=target_path)
    strategy.train_model(datasets, opt=opt, save_path=target_path)
    log.info("Training pipeline finished successfully. Model saved to '%s'", target_path)


def run_execution(mode: str = "PAPER", balance: float = 10000.0, model_path: str | None = None) -> None:
    """Start the real-time execution engine in PAPER or LIVE mode."""
    from live_runner import start_live_runner

    log.info("Launching execution engine in %s mode (Starting Balance: $%.2f) …", mode, balance)
    start_live_runner(mode=mode, balance=balance, model_path=model_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="State-of-the-Art Quantitative ML Sniper Trading Bot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["OPTIMIZE", "TRAIN_OPT", "TRAIN", "PAPER", "LIVE", "WALK_FORWARD"],
        default=config.MODE,
        help="Operating mode: OPTIMIZE (Bayesian Optuna search), TRAIN_OPT (preset optimized train), TRAIN (standard RF), PAPER (simulated paper trading), LIVE (real orders), WALK_FORWARD (rolling retraining).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=config.OPTUNA_TRIALS,
        help="Number of Optuna Bayesian trials to run during --mode OPTIMIZE.",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=config.PORTFOLIO_START_BALANCE,
        help="Starting virtual cash balance in USDT for PAPER trading.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Explicit model path override for inference.",
    )
    args = parser.parse_args()

    mode = args.mode.upper()

    if mode == "OPTIMIZE":
        run_optimization(trials=args.trials)
    elif mode == "TRAIN_OPT":
        run_training(opt=True)
    elif mode == "TRAIN":
        run_training(opt=False)
    elif mode in ("PAPER", "LIVE"):
        run_execution(mode=mode, balance=args.balance, model_path=args.model_path)
    elif mode == "WALK_FORWARD":
        from walk_forward_retrainer import walk_forward_retrainer
        log.info("Launching Walk-Forward Rolling Retraining Pipeline (Trials: %d) …", args.trials)
        walk_forward_retrainer.execute_retrain(n_trials=args.trials)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
