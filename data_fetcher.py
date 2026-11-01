"""Alternative Data (Alpha Data) Fetcher – Binance Futures Derivatives Metrics.

Fetches Open Interest (OI) and Funding Rate (FR) from Binance Futures (USDT-M)
via CCXT and direct fallback endpoints. Computes 4-hour derivative dynamics:
  - OI_Change: 4-hour percentage change in aggregate open interest
  - Funding_Trend: Moving trend / deviation of funding rate from baseline
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Optional, Dict, Any, Tuple

import ccxt
import pandas as pd

import config
from logger_setup import get_logger

log = get_logger("data_fetcher")


class DataFetcher:
    """Institutional-grade Alternative Alpha Data Provider for Binance Futures."""

    def __init__(self, timeout_ms: int = 15000):
        self.timeout_ms = timeout_ms
        self._futures_exchange: Optional[ccxt.Exchange] = None
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._cache_ttl_seconds = 60.0  # 1-minute TTL for live metrics

    def _get_futures_exchange(self) -> ccxt.Exchange:
        """Initialize ccxt.binanceusdm client for futures market queries."""
        if self._futures_exchange is None:
            self._futures_exchange = ccxt.binanceusdm({
                "enableRateLimit": True,
                "timeout": self.timeout_ms,
                "options": {"defaultType": "future"},
            })
        return self._futures_exchange

    @staticmethod
    def _to_fapi_symbol(symbol: str) -> str:
        """Convert 'BTC/USDT' -> 'BTCUSDT'."""
        return symbol.replace("/", "").replace(":USDT", "").upper()

    def fetch_open_interest(self, symbol: str) -> Optional[float]:
        """Fetch current open interest value in base asset contracts."""
        fapi_sym = self._to_fapi_symbol(symbol)
        url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={fapi_sym}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuantBot/3.0"})
            with urllib.request.urlopen(req, timeout=self.timeout_ms / 1000.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return float(data.get("openInterest", 0.0))
        except Exception as e:
            log.debug("REST fetch openInterest failed for %s (%s), trying CCXT fallback.", symbol, e)
            try:
                ex = self._get_futures_exchange()
                oi_data = ex.fetch_open_interest(symbol)
                return float(oi_data.get("openInterestAmount", 0.0))
            except Exception as ce:
                log.warning("CCXT fetch_open_interest failed for %s: %s", symbol, ce)
                return None

    def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """Fetch current 8h funding rate."""
        fapi_sym = self._to_fapi_symbol(symbol)
        url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={fapi_sym}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuantBot/3.0"})
            with urllib.request.urlopen(req, timeout=self.timeout_ms / 1000.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return float(data.get("lastFundingRate", 0.0))
        except Exception as e:
            log.debug("REST fetch premiumIndex failed for %s (%s), trying CCXT fallback.", symbol, e)
            try:
                ex = self._get_futures_exchange()
                fr_data = ex.fetch_funding_rate(symbol)
                return float(fr_data.get("fundingRate", 0.0))
            except Exception as ce:
                log.warning("CCXT fetch_funding_rate failed for %s: %s", symbol, ce)
                return None

    def fetch_open_interest_history(self, symbol: str, period: str = "4h", limit: int = 30) -> list[dict]:
        """Fetch historical Open Interest series from Binance Futures Data API."""
        fapi_sym = self._to_fapi_symbol(symbol)
        url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={fapi_sym}&period={period}&limit={limit}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuantBot/3.0"})
            with urllib.request.urlopen(req, timeout=self.timeout_ms / 1000.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            log.debug("Failed to fetch historical open interest for %s: %s", symbol, e)
            return []

    def fetch_funding_rate_history(self, symbol: str, limit: int = 30) -> list[dict]:
        """Fetch historical funding rate records."""
        fapi_sym = self._to_fapi_symbol(symbol)
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={fapi_sym}&limit={limit}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuantBot/3.0"})
            with urllib.request.urlopen(req, timeout=self.timeout_ms / 1000.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            log.debug("Failed to fetch funding rate history for %s: %s", symbol, e)
            return []

    def get_derivatives_alpha(self, symbol: str) -> Dict[str, float]:
        """Compute live 4h OI_Change (%) and Funding_Trend for real-time feature injection."""
        now = time.time()
        if symbol in self._cache:
            ts, cached_val = self._cache[symbol]
            if now - ts < self._cache_ttl_seconds:
                return cached_val

        oi_hist = self.fetch_open_interest_history(symbol, period="4h", limit=5)
        fr_hist = self.fetch_funding_rate_history(symbol, limit=10)

        # 1. 4h Open Interest Change (%)
        oi_change = 0.0
        current_oi = 0.0
        if len(oi_hist) >= 2:
            prev_oi = float(oi_hist[-2].get("sumOpenInterest", 0.0))
            curr_oi = float(oi_hist[-1].get("sumOpenInterest", 0.0))
            current_oi = curr_oi
            if prev_oi > 0:
                oi_change = ((curr_oi - prev_oi) / prev_oi) * 100.0
        else:
            live_oi = self.fetch_open_interest(symbol)
            if live_oi is not None:
                current_oi = live_oi

        # 2. Funding Rate Trend
        funding_rate = 0.0
        funding_trend = 0.0
        if fr_hist:
            rates = [float(r.get("fundingRate", 0.0)) for r in fr_hist]
            funding_rate = rates[-1]
            if len(rates) >= 3:
                # Delta between latest funding rate and 3-period moving average (expressed in basis points)
                mean_fr = sum(rates[-3:]) / 3.0
                funding_trend = (funding_rate - mean_fr) * 10000.0
        else:
            live_fr = self.fetch_funding_rate(symbol)
            if live_fr is not None:
                funding_rate = live_fr

        alpha_metrics = {
            "open_interest": current_oi,
            "funding_rate": funding_rate,
            "OI_Change": float(oi_change),
            "Funding_Trend": float(funding_trend),
        }
        self._cache[symbol] = (now, alpha_metrics)
        return alpha_metrics

    def enrich_candles(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich OHLCV DataFrame with 4h Open Interest and Funding Rate series."""
        if df.empty:
            return df

        df = df.copy()

        # Son satırlara gerçek vadeli veri türevlerini ekle
        alpha = self.get_derivatives_alpha(symbol)

        if "open_interest" not in df.columns:
            df["open_interest"] = alpha["open_interest"]
        if "funding_rate" not in df.columns:
            df["funding_rate"] = alpha["funding_rate"]
        if "OI_Change" not in df.columns:
            df["OI_Change"] = 0.0
        if "Funding_Trend" not in df.columns:
            df["Funding_Trend"] = 0.0

        # En son barlara canlı alpha sinyallerini yerleştir
        df.iloc[-1, df.columns.get_loc("OI_Change")] = alpha["OI_Change"]
        df.iloc[-1, df.columns.get_loc("Funding_Trend")] = alpha["Funding_Trend"]
        if len(df) >= 2:
            df.iloc[-2, df.columns.get_loc("OI_Change")] = alpha["OI_Change"] * 0.5
            df.iloc[-2, df.columns.get_loc("Funding_Trend")] = alpha["Funding_Trend"] * 0.5

        return df


# Module-level singleton
data_fetcher = DataFetcher()
