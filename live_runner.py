"""Live / Paper Trading Runner – Sniper ML Execution & Market Confidence Radar.

Continuously monitors config.SYMBOLS across the exchange, computes advanced quant features,
filters entries by a strict >= 50% confidence threshold, maintains a real-time Market
Confidence Radar, and executes virtual spot orders with state persistence.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import ccxt
import pandas as pd

import config
from logger_setup import get_logger
from ml_strategy import MLStrategy, compute_atr
from portfolio_manager import PortfolioManager
from data_fetcher import data_fetcher
from market_regime import market_regime_detector
from walk_forward_retrainer import walk_forward_retrainer

log = get_logger("live_runner")


class LiveRunner:
    """Multi-asset live/paper trading engine powered by MLStrategy and PortfolioManager."""

    def __init__(
        self,
        mode: str = "PAPER",
        balance: Optional[float] = None,
        model_path: Optional[str] = None,
    ):
        self.mode = mode.upper()
        self.strategy = MLStrategy(model_path=model_path)
        self.portfolio = PortfolioManager(
            initial_capital=balance if balance is not None else config.PORTFOLIO_START_BALANCE,
            state_path=config.PORTFOLIO_STATE_PATH,
        )
        self._exchange: Optional[ccxt.Exchange] = None

        if self.strategy.model is None:
            log.warning("No pre-trained ML model found. Please train with `--mode TRAIN_OPT` first!")
        else:
            log.info("LiveRunner ready. Model: %s | Sniper Threshold: %d%% | Portfolio Cash: $%.2f",
                     self.strategy.model_path, int(config.PROBABILITY_THRESHOLD * 100), self.portfolio.cash_balance)

    def _get_exchange(self) -> ccxt.Exchange:
        """Initialize ccxt exchange client with rate-limiting."""
        if self._exchange is None:
            exchange_cls = getattr(ccxt, config.EXCHANGE_ID)
            opts = {
                "enableRateLimit": True,
                "timeout": 20000,
            }
            if self.mode == "LIVE":
                opts["apiKey"] = config.API_KEY
                opts["secret"] = config.API_SECRET
            self._exchange = exchange_cls(opts)
            self._exchange.load_markets()
            log.info("Connected to %s exchange (%s).", config.EXCHANGE_ID.upper(), self.mode)
        return self._exchange

    def _fetch_candles(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetch latest candles for a given symbol and parse into a DataFrame."""
        exchange = self._get_exchange()
        ohlcv = exchange.fetch_ohlcv(symbol, config.TIMEFRAME, limit=limit)
        if not ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
        return df

    def _render_sniper_radar(
        self,
        radar_records: List[Dict[str, Any]],
        market_regime_name: str = "DÜŞÜK VOLATİLİTE (ÖLÜ PİYASA)",
        active_threshold: float = config.PROBABILITY_THRESHOLD,
    ) -> str:
        """Format and render the Top-5 Market Confidence Radar banner with Regime indicator."""
        if not radar_records:
            return ""

        # AL (BUY) güvenine göre büyükten küçüğe sırala
        sorted_radar = sorted(radar_records, key=lambda x: x["buy_prob"], reverse=True)
        threshold_pct = active_threshold * 100.0

        lines: list[str] = []
        lines.append("=" * 80)
        lines.append(f" [ML SNIPER RADAR] En Yüksek Güven Sinyalleri (Eşik: %{threshold_pct:.1f}):")
        if getattr(config, "USE_MARKET_REGIME", True):
            lines.append(f"  Piyasa Rejimi: {market_regime_name:<30} | Aktif Eşik: %{threshold_pct:.1f}")
            lines.append("  " + "-" * 76)

        top5 = sorted_radar[:5]
        top5_symbols = {r["symbol"] for r in top5}

        # İlk 5 coinlik liste
        for idx, item in enumerate(top5, 1):
            sym = item["symbol"]
            prob = item["buy_prob"]
            status = item["status"]
            oi_info = f"OI: {item.get('oi_change', 0.0):+.1f}%" if "oi_change" in item else ""
            if item["is_active"]:
                line = f"  🟢 {sym:<10} : %{prob:4.1f}  [{status}] {oi_info}"
            else:
                line = f"  {idx}. {sym:<10} : %{prob:4.1f}  [{status}] {oi_info}"
            lines.append(line)

        # Eğer ilk 5 dışında kalıp da işleme girilmiş veya açık pozisyonu olan varsa ek satır bas
        active_extras = [r for r in sorted_radar[5:] if r["is_active"] and r["symbol"] not in top5_symbols]
        if active_extras:
            lines.append("  " + "-" * 76)
            for item in active_extras:
                sym = item["symbol"]
                prob = item["buy_prob"]
                status = item["status"]
                oi_info = f"OI: {item.get('oi_change', 0.0):+.1f}%" if "oi_change" in item else ""
                lines.append(f"  🟢 {sym:<10} : %{prob:4.1f}  [{status}] {oi_info}")

        lines.append("=" * 80)
        return "\n".join(lines)

    def run(self) -> None:
        """Execute the continuous paper trading loop across 20 symbols."""
        log.info("=" * 80)
        log.info("STARTING %s SNIPER TRADING LOOP (%d SYMBOLS)", self.mode, len(config.SYMBOLS))
        log.info("Confidence Threshold: %d%% | ATR Stops: %s | Poll Interval: %ds",
                 int(config.PROBABILITY_THRESHOLD * 100), config.USE_ATR_STOPS, config.POLL_INTERVAL_SECONDS)
        log.info("=" * 80)

        # Otomatik yeniden eğitim arka plan zamanlayıcısını başlat
        if getattr(config, "USE_AUTO_RETRAIN", True):
            walk_forward_retrainer.start_background()

        cycle_count = 0
        latest_prices: Dict[str, float] = {}
        cached_candles_map: Dict[str, pd.DataFrame] = {}

        while True:
            cycle_count += 1
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            log.info("--- Polling Cycle #%d [%s] ---", cycle_count, now_str)

            # Hot-reload: Arka planda eğitilen yeni model varsa bellek içine sıfır gecikmeyle aktar
            self.strategy.check_and_reload_model()

            radar_records: List[Dict[str, Any]] = []
            dominant_regime_name = "STANDART"
            dominant_threshold = config.PROBABILITY_THRESHOLD

            for symbol in config.SYMBOLS:
                try:
                    # 1. Borsa API'sinden mumları çek
                    df = self._fetch_candles(symbol, limit=100)
                    if df.empty or len(df) < 50:
                        log.warning("[%s] Insufficient candles fetched (%d), skipping.", symbol, len(df))
                        continue

                    # Alternatif Veri Entegrasyonu (Binance Futures OI & Funding Rate)
                    if getattr(config, "USE_FUTURES_DATA", True):
                        df = data_fetcher.enrich_candles(symbol, df)

                    cached_candles_map[symbol] = df
                    current_price = float(df["close"].iloc[-1])
                    latest_prices[symbol] = current_price

                    # ATR hesapla (Risk yönetimi için)
                    atr_series = compute_atr(df, period=config.ATR_PERIOD)
                    current_atr = float(atr_series.iloc[-1]) if not atr_series.empty else None

                    # 2. Açık pozisyon varsa TP/SL/Breakeven kontrolü yap
                    self.portfolio.update_market_price_and_stops(symbol, current_price, current_atr)

                    # Piyasa Rejimi Tespiti & Dinamik Eşik Modülasyonu (HMM / GMM)
                    if getattr(config, "USE_MARKET_REGIME", True):
                        regime_id, regime_name, dyn_threshold = market_regime_detector.predict_regime(df)
                    else:
                        regime_id, regime_name, dyn_threshold = 0, "STANDART", config.PROBABILITY_THRESHOLD

                    if symbol == config.SYMBOL or dominant_regime_name == "STANDART":
                        dominant_regime_name = regime_name
                        dominant_threshold = dyn_threshold

                    # 3. Son kapalı mum ile ML Tahmini al (Sızıntıyı önlemek için son canlı mumu çıkar)
                    closed_candles = df.iloc[:-1]
                    signal, confidence, probs = self.strategy.predict_signal_with_confidence(
                        closed_candles, threshold=dyn_threshold
                    )

                    # AL (BUY) olasılığı: modelin classes_ eşlemesinde '1' endeksi daima AL sınıfıdır
                    buy_prob = probs.get(1, 0.0) * 100.0
                    conf_pct = confidence * 100.0
                    just_entered = False

                    # 4. Keskin Nişancı Emir Karar Mekanizması
                    if signal == 1:
                        log.info("[%s] [SNIPER BUY] -> %-10s | Fiyat: $%.4f | Guven: %.1f%% (Eşik: %.1f%%, Rejim: %s)",
                                 self.mode, symbol, current_price, conf_pct, dyn_threshold * 100.0, regime_name)
                        if self.portfolio.can_open_position(symbol):
                            opened = self.portfolio.open_spot_position(
                                symbol=symbol,
                                current_price=current_price,
                                confidence=confidence,
                                atr_val=current_atr,
                                candidate_candles=df,
                                open_positions_candles=cached_candles_map,
                            )
                            if opened:
                                just_entered = True
                        else:
                            log.debug("[%s] Position limit or insufficient funds for %s.", self.mode, symbol)

                    elif signal == -1:
                        # %50+ Güvenli SAT Sinyali
                        log.info("[%s] [SNIPER SELL] -> %-10s | Fiyat: $%.4f | Guven: %.1f%% (P_SAT=%.1f%%)",
                                 self.mode, symbol, current_price, conf_pct, probs.get(-1, 0.0) * 100)
                        if symbol in self.portfolio.open_positions:
                            self.portfolio.close_spot_position(symbol, current_price, reason="ML_SELL_SIGNAL", confidence=confidence)

                    else:
                        # 0: Olasılık eşiği aşılmadı veya Nötr
                        log.debug("[%s] [HOLD / SKIP] -> %-10s | Fiyat: $%.4f | Guven: %.1f%% (< %.1f%%)",
                                  self.mode, symbol, current_price, conf_pct, dyn_threshold * 100.0)

                    # 5. Güven Radarı için durum belirleme
                    is_in_position = symbol in self.portfolio.open_positions
                    if just_entered:
                        status_tag = "İŞLEME GİRİLDİ"
                        is_active = True
                    elif is_in_position:
                        status_tag = "AÇIK POZİSYON"
                        is_active = True
                    elif signal == 1:
                        status_tag = "AL SİNYALİ"
                        is_active = True
                    else:
                        status_tag = "BEKLE"
                        is_active = False

                    radar_records.append({
                        "symbol": symbol,
                        "buy_prob": buy_prob,
                        "sell_prob": probs.get(-1, 0.0) * 100.0,
                        "confidence": conf_pct,
                        "status": status_tag,
                        "is_active": is_active,
                        "oi_change": df["OI_Change"].iloc[-1] if "OI_Change" in df.columns else 0.0,
                    })

                except ccxt.NetworkError as ne:
                    log.error("[%s] Network connection error on %s: %s", self.mode, symbol, ne)
                except ccxt.ExchangeError as ee:
                    log.error("[%s] Exchange API error on %s: %s", self.mode, symbol, ee)
                except Exception as e:
                    log.error("[%s] Unexpected error processing %s: %s", self.mode, symbol, e, exc_info=False)

                time.sleep(0.25)

            # 6. Terminale Piyasa Güven Radarını ve Canlı Portföy Raporunu Bas
            radar_banner = self._render_sniper_radar(
                radar_records,
                market_regime_name=dominant_regime_name,
                active_threshold=dominant_threshold,
            )
            sorted_radar = sorted(radar_records, key=lambda x: x["buy_prob"], reverse=True) if radar_records else []
            top_radar_conf = sorted_radar[0]["buy_prob"] if sorted_radar else 0.0

            status_banner = self.portfolio.format_status_banner(latest_prices, top_radar_conf=top_radar_conf)

            if radar_banner:
                print("\n" + radar_banner)
            print(f" {status_banner} ")
            print("=" * 80 + "\n")

            if sorted_radar:
                log.info("Radar Top Buy Signal: %s (%.1f%%)", sorted_radar[0]["symbol"], sorted_radar[0]["buy_prob"])
            log.info(status_banner)

            log.info("Cycle #%d completed. Sleeping for %ds …\n", cycle_count, config.POLL_INTERVAL_SECONDS)
            try:
                time.sleep(config.POLL_INTERVAL_SECONDS)
            except KeyboardInterrupt:
                log.info("Shutdown signal received during sleep. Exiting cleanly …")
                break


def start_live_runner(
    mode: str = "PAPER",
    balance: Optional[float] = None,
    model_path: Optional[str] = None,
) -> None:
    runner = LiveRunner(mode=mode, balance=balance, model_path=model_path)
    try:
        runner.run()
    except KeyboardInterrupt:
        log.info("Execution stopped by user.")
