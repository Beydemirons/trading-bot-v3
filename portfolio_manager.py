"""Virtual Portfolio & Execution Order Manager (Spot Paper Trading).

Features:
- State Persistence: Persists cash balance, open positions, and trade history in 'portfolio_state.json'.
- Full spot position lifecycle: Position sizing, execution commissions, and slippage deduction.
- ATR-based dynamic Stop-Loss, Take-Profit, and Breakeven protection.
- Real-time equity and PnL reporting.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd

import config
from logger_setup import get_logger

log = get_logger("portfolio_manager")


class PortfolioManager:
    """Spot Paper Trading Portfolio and Risk Manager with persistent state."""

    def __init__(
        self,
        initial_capital: float = config.PORTFOLIO_START_BALANCE,
        state_path: str = config.PORTFOLIO_STATE_PATH,
    ):
        self.state_path = state_path
        self.initial_capital = float(initial_capital)
        self.cash_balance: float = float(initial_capital)
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: list[Dict[str, Any]] = []
        self.last_trade_confidence: float = 0.0

        self._load_state()

    def _load_state(self) -> None:
        """Load portfolio state from JSON file or initialize if not found."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.initial_capital = float(data.get("initial_capital", self.initial_capital))
                self.cash_balance = float(data.get("cash_balance", self.initial_capital))
                self.open_positions = data.get("open_positions", {})
                self.trade_history = data.get("trade_history", [])
                self.last_trade_confidence = float(data.get("last_trade_confidence", 0.0))
                log.info("Loaded portfolio state from '%s' (Cash: $%.2f | Open Positions: %d).",
                         self.state_path, self.cash_balance, len(self.open_positions))
            except Exception as e:
                log.warning("Could not read portfolio state from '%s': %s. Initializing fresh.", self.state_path, e)
                self._save_state()
        else:
            self._save_state()

    def _save_state(self) -> None:
        """Persist portfolio state to JSON file."""
        state = {
            "initial_capital": self.initial_capital,
            "cash_balance": round(self.cash_balance, 4),
            "open_positions": self.open_positions,
            "trade_history": self.trade_history[-200:],  # Son 200 işlemi sakla
            "last_trade_confidence": round(self.last_trade_confidence, 4),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        try:
            temp_path = self.state_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(temp_path, self.state_path)
        except Exception as e:
            log.error("Failed to save portfolio state to '%s': %s", self.state_path, e)

    def can_open_position(self, symbol: str) -> bool:
        """Check if capital and portfolio limits allow opening a new position."""
        if symbol in self.open_positions:
            return False
        if len(self.open_positions) >= config.MAX_OPEN_POSITIONS:
            return False
        min_required = 50.0  # Minimum işlem eşiği ($50 USDT)
        if self.cash_balance < min_required:
            return False
        return True

    def get_dynamic_position_size_pct(self, confidence: float) -> Tuple[float, str]:
        """Dynamic Kelly-style position sizing based on model prediction confidence:

        - 50% - 70% Confidence: 5% of cash (Standard Sniper)
        - 70% - 80% Confidence: 10% of cash (Strong Signal)
        - 80%+      Confidence: 15% of cash (High Conviction)
        """
        if not getattr(config, "USE_DYNAMIC_SIZING", True):
            return config.POSITION_SIZE_PCT, "STANDART"

        conf = confidence if confidence <= 1.0 else confidence / 100.0

        if conf >= 0.80:
            return config.DYNAMIC_SIZE_TIER_3, "KESİN FIRSAT"   # %15
        elif conf >= 0.70:
            return config.DYNAMIC_SIZE_TIER_2, "GÜÇLÜ SİNYAL"   # %10
        elif conf >= getattr(config, "PROBABILITY_THRESHOLD", 0.50):
            return config.DYNAMIC_SIZE_TIER_1, "STANDART"       # %5
        return 0.0, "YETERSİZ"

    def check_correlation_shield(
        self,
        candidate_symbol: str,
        candidate_candles: pd.DataFrame,
        open_positions_candles: Dict[str, pd.DataFrame],
        max_corr: Optional[float] = None,
        window: Optional[int] = None,
    ) -> Tuple[bool, Optional[str], float]:
        """Portfolio Correlation Shield: Prevents doubling down on correlated market risk.

        Calculates rolling Pearson correlation on the last `window` bars between candidate symbol
        and each currently open position. If correlation exceeds `max_corr` (default: 0.75),
        the new trade is blocked.
        """
        if not self.open_positions:
            return True, None, 0.0

        if not getattr(config, "USE_CORRELATION_SHIELD", True):
            return True, None, 0.0

        if candidate_candles is None or candidate_candles.empty:
            return True, None, 0.0

        max_allowed_corr = max_corr if max_corr is not None else getattr(config, "MAX_CORRELATION_THRESHOLD", 0.75)
        window_size = window if window is not None else getattr(config, "CORRELATION_WINDOW", 30)

        if len(candidate_candles) < 15:
            return True, None, 0.0

        cand_ret = candidate_candles["close"].astype(float).pct_change(1).dropna().tail(window_size)

        for open_sym in list(self.open_positions.keys()):
            if open_sym not in open_positions_candles:
                continue
            open_df = open_positions_candles[open_sym]
            if open_df is None or open_df.empty or len(open_df) < 15:
                continue

            open_ret = open_df["close"].astype(float).pct_change(1).dropna().tail(window_size)

            merged = pd.concat([cand_ret, open_ret], axis=1, join="inner")
            if len(merged) >= 10:
                corr = float(merged.iloc[:, 0].corr(merged.iloc[:, 1]))
            else:
                min_len = min(len(cand_ret), len(open_ret))
                corr = float(np.corrcoef(cand_ret.values[-min_len:], open_ret.values[-min_len:])[0, 1])

            if np.isnan(corr):
                continue

            if corr > max_allowed_corr:
                log.warning(
                    "[PORTFOLIO CORRELATION SHIELD] %s işlemi REDDEDİLDİ! "
                    "Açık pozisyondaki %s ile son %d mumluk korelasyon: %.2f (> %.2f). "
                    "Aynı riski ikiye katlamamak için portföy koruma kalkanı devrede.",
                    candidate_symbol, open_sym, window_size, corr, max_allowed_corr
                )
                return False, open_sym, corr

        return True, None, 0.0

    def open_spot_position(
        self,
        symbol: str,
        current_price: float,
        confidence: float,
        atr_val: Optional[float] = None,
        candidate_candles: Optional[pd.DataFrame] = None,
        open_positions_candles: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> bool:
        """Open a new virtual Spot Long position with dynamic position sizing, slippage & commission."""
        if not self.can_open_position(symbol):
            return False

        # ── KORELASYON KALKANI KONTROLÜ ─────────────────────────────────────
        if candidate_candles is not None and open_positions_candles is not None:
            allowed, corr_sym, corr_val = self.check_correlation_shield(
                symbol, candidate_candles, open_positions_candles
            )
            if not allowed:
                return False

        # Güven bazlı dinamik pozisyon büyüklüğü (Kelly Sizing)
        size_pct, tier_label = self.get_dynamic_position_size_pct(confidence)
        if size_pct <= 0:
            log.warning("Confidence %.1f%% below minimum execution threshold, skipping.", confidence * 100.0)
            return False

        alloc_usdt = self.cash_balance * (size_pct / 100.0)
        alloc_usdt = min(alloc_usdt, self.cash_balance)

        # Kayma payı (Alışta fiyat yukarı kayar)
        entry_price = current_price * (1.0 + (config.SLIPPAGE_PCT / 100.0))
        # Alış komisyonu
        commission = alloc_usdt * (config.COMMISSION_PCT / 100.0)

        # Net coin miktarı
        net_alloc = alloc_usdt - commission
        amount_coins = net_alloc / entry_price

        # ATR veya Sabit % tabanlı SL / TP seviyeleri
        if config.USE_ATR_STOPS and atr_val is not None and atr_val > 0:
            sl_price = entry_price - (atr_val * config.ATR_SL_MULTIPLIER)
            tp_price = entry_price + (atr_val * config.ATR_TP_MULTIPLIER)
            breakeven_target = entry_price + (atr_val * config.BREAKEVEN_TRIGGER_ATR)
        else:
            sl_price = entry_price * (1.0 - (config.STOP_LOSS_PCT / 100.0))
            tp_price = entry_price + (1.0 + (config.TARGET_PROFIT_PCT / 100.0))
            breakeven_target = entry_price * (1.0 + (config.BREAKEVEN_TRIGGER_PCT / 100.0))

        # Pozisyon kaydı
        self.open_positions[symbol] = {
            "symbol": symbol,
            "entry_price": entry_price,
            "amount_coins": amount_coins,
            "size_usdt": alloc_usdt,
            "size_pct": size_pct,
            "tier_label": tier_label,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "breakeven_target": breakeven_target,
            "breakeven_active": False,
            "entry_confidence": confidence,
            "entry_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

        self.cash_balance -= alloc_usdt
        self.last_trade_confidence = confidence * 100.0
        self._save_state()

        log.info(
            "[PAPER PORTFOLIO] [BUY] SPOT BUY -> %-10s | Giris: $%.4f | Boyut: $%.2f (Kasa %%.%d%% [%s]) | SL: $%.4f | TP: $%.4f | Guven: %.1f%%",
            symbol, entry_price, alloc_usdt, int(size_pct), tier_label, sl_price, tp_price, confidence * 100.0,
        )
        return True

    def close_spot_position(
        self,
        symbol: str,
        current_price: float,
        reason: str,
        confidence: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Close an existing spot position, realize PnL, and update cash."""
        if symbol not in self.open_positions:
            return None

        pos = self.open_positions.pop(symbol)

        # Kayma payı (Satışta fiyat aşağı kayar)
        exit_price = current_price * (1.0 - (config.SLIPPAGE_PCT / 100.0))
        gross_return = pos["amount_coins"] * exit_price
        commission = gross_return * (config.COMMISSION_PCT / 100.0)
        net_return = gross_return - commission

        net_pnl_usdt = net_return - pos["size_usdt"]
        net_pnl_pct = (net_pnl_usdt / pos["size_usdt"]) * 100.0

        self.cash_balance += net_return
        if confidence is not None:
            self.last_trade_confidence = confidence * 100.0

        trade_record = {
            "symbol": symbol,
            "entry_price": round(pos["entry_price"], 4),
            "exit_price": round(exit_price, 4),
            "size_usdt": round(pos["size_usdt"], 2),
            "net_pnl_usdt": round(net_pnl_usdt, 2),
            "net_pnl_pct": round(net_pnl_pct, 2),
            "reason": reason,
            "entry_time": pos["entry_time"],
            "exit_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "confidence": round(pos["entry_confidence"] * 100.0, 1),
        }
        self.trade_history.append(trade_record)
        self._save_state()

        pnl_sign = "+" if net_pnl_usdt >= 0 else ""
        action_tag = "[WIN]" if net_pnl_usdt >= 0 else "[LOSS]"
        log.info(
            "[PAPER PORTFOLIO] %s SPOT SELL -> %-10s | Neden: %-15s | Cikis: $%.4f | PnL: %s$%.2f (%s%.2f%%) | Kasa: $%.2f",
            action_tag, symbol, reason, exit_price, pnl_sign, net_pnl_usdt, pnl_sign, net_pnl_pct, self.cash_balance,
        )
        return trade_record

    def update_market_price_and_stops(
        self,
        symbol: str,
        current_price: float,
        atr_val: Optional[float] = None,
    ) -> None:
        """Check TP, SL, and Breakeven adjustments for an open position."""
        if symbol not in self.open_positions:
            return

        pos = self.open_positions[symbol]

        # 1. Take-Profit Kontrolü
        if current_price >= pos["tp_price"]:
            self.close_spot_position(symbol, current_price, reason="TAKE_PROFIT (TP)")
            return

        # 2. Stop-Loss Kontrolü
        if current_price <= pos["sl_price"]:
            self.close_spot_position(symbol, current_price, reason="STOP_LOSS (SL)")
            return

        # 3. Breakeven (Kâr Koruma) Mekanizması
        if config.USE_BREAKEVEN_STOP and not pos.get("breakeven_active", False):
            if current_price >= pos["breakeven_target"]:
                # SL seviyesini giriş fiyatına çek
                pos["sl_price"] = pos["entry_price"]
                pos["breakeven_active"] = True
                self._save_state()
                log.info("[PAPER PORTFOLIO] [BREAKEVEN] ACTIVATED -> %s | SL seviyesi giris fiyatina ($%.4f) tasindi.",
                         symbol, pos["entry_price"])

    def get_total_equity(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """Calculate total current equity (Cash + Market value of open positions)."""
        current_prices = current_prices or {}
        positions_value = 0.0
        for sym, pos in self.open_positions.items():
            price = current_prices.get(sym, pos["entry_price"])
            positions_value += pos["amount_coins"] * price
        return self.cash_balance + positions_value

    def format_status_banner(
        self,
        current_prices: Optional[Dict[str, float]] = None,
        top_radar_conf: Optional[float] = None,
    ) -> str:
        """Return the formatted live status string required for terminal reporting:

        Example:
        "[PAPER PORTFOLIO] Bakiye: $10,240.50 | Açık Pozisyonlar: 2 | Toplam PnL: +2.40% | Son İşlem Güveni: %64.2"
        """
        equity = self.get_total_equity(current_prices)
        total_pnl_pct = ((equity - self.initial_capital) / self.initial_capital) * 100.0
        pnl_sign = "+" if total_pnl_pct >= 0 else ""

        if self.last_trade_confidence > 0:
            conf_str = f"Son İşlem Güveni: %{self.last_trade_confidence:.1f}"
        elif top_radar_conf is not None and top_radar_conf > 0:
            conf_str = f"Piyasa Tepe Güveni: %{top_radar_conf:.1f}"
        else:
            conf_str = f"Son İşlem Güveni: %0.0"

        return (
            f"[PAPER PORTFOLIO] Bakiye: ${equity:,.2f} | "
            f"Açık Pozisyonlar: {len(self.open_positions)} | "
            f"Toplam PnL: {pnl_sign}{total_pnl_pct:.2f}% | "
            f"{conf_str}"
        )
