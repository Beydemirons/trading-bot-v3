"""Backtest engine – simulates trading on historical OHLCV data.

Key quantitative features:
  - Strict lookahead bias prevention: signals on closed candle i-1, execution on open i.
  - Dynamic TP / SL: ATR-based (1.5x SL / 2.0x TP) or fixed percentage.
  - Breakeven Stop: automatically trails Stop Loss to entry price when profit reaches +1.0x ATR.
  - Realistic execution: 0.075% commission + volatility-dependent dynamic slippage.
  - Portfolio simulation: multi-symbol execution with shared capital management.
  - Walk-Forward analysis: in-sample and out-of-sample data partitioning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

import config
from logger_setup import get_logger
from strategy import BaseStrategy, compute_atr

log = get_logger("backtester")


@dataclass
class Trade:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    direction: int          # 1 = LONG, -1 = SHORT
    size: float             # Position size in quote currency
    tp_price: float
    sl_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    is_breakeven: bool = False
    exit_reason: Optional[str] = None


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    equity_dates: list[pd.Timestamp] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Dynamic Slippage & Cost Engine
# ─────────────────────────────────────────────────────────────────────

def _dynamic_slippage_pct(atr_val: float, price: float) -> float:
    """Calculate volatility-adjusted slippage percentage."""
    if not config.DYNAMIC_SLIPPAGE or np.isnan(atr_val) or price == 0:
        return config.SLIPPAGE_PCT
    atr_ratio = atr_val / price
    norm = min(1.0, max(0.0, (atr_ratio - 0.005) / 0.045))
    extra = config.DYNAMIC_SLIPPAGE_MIN + norm * (
        config.DYNAMIC_SLIPPAGE_MAX - config.DYNAMIC_SLIPPAGE_MIN
    )
    return config.SLIPPAGE_PCT + extra


def _apply_cost(
    price: float, direction: int, side: str, slippage_pct: float
) -> float:
    """Apply exchange commission and slippage penalty to execution price."""
    cost_pct = (config.COMMISSION_PCT + slippage_pct) / 100
    if side == "entry":
        return price * (1 + cost_pct) if direction == 1 else price * (1 - cost_pct)
    else:
        return price * (1 - cost_pct) if direction == 1 else price * (1 + cost_pct)


# ─────────────────────────────────────────────────────────────────────
# Single Symbol Backtester
# ─────────────────────────────────────────────────────────────────────

class Backtester:
    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = config.INITIAL_CAPITAL,
        position_size_pct: float = config.POSITION_SIZE_PCT,
        single_position: bool = config.SINGLE_POSITION,
        symbol_label: str = "",
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.single_position = single_position
        self.symbol_label = symbol_label or config.SYMBOL

    def _compute_tp_sl(
        self, entry: float, direction: int, atr_value: float
    ) -> tuple[float, float]:
        """Compute initial Take Profit and Stop Loss levels."""
        if config.USE_ATR_STOPS and not np.isnan(atr_value):
            sl = entry - (atr_value * config.ATR_SL_MULTIPLIER) * direction
            tp = entry + (atr_value * config.ATR_TP_MULTIPLIER) * direction
        else:
            tp = entry * (1 + (config.TAKE_PROFIT_PCT / 100) * direction)
            sl = entry * (1 - (config.STOP_LOSS_PCT / 100) * direction)
        return tp, sl

    def run(self, df: pd.DataFrame) -> BacktestResult:
        signals = self.strategy.generate_signals(df)
        atr = compute_atr(df, config.ATR_PERIOD)

        capital = self.initial_capital
        position: Optional[Trade] = None
        result = BacktestResult()
        result.equity_curve.append(capital)
        result.equity_dates.append(df.index[0])

        for i in range(1, len(df)):
            current = df.iloc[i]
            now = df.index[i]
            atr_val = atr.iloc[i - 1] if i - 1 < len(atr) else np.nan
            slip = _dynamic_slippage_pct(atr_val, current["close"])

            # ── 1. Breakeven & TP/SL Kontrolleri ──────────────────────
            if position is not None:
                closed = False
                exit_price = None
                exit_reason = None

                # Breakeven Tetikleme Seviyesi Hesabı
                if config.USE_BREAKEVEN_STOP and not position.is_breakeven:
                    if config.USE_ATR_STOPS and not np.isnan(atr_val):
                        be_trigger = position.entry_price + (atr_val * config.BREAKEVEN_TRIGGER_ATR * position.direction)
                    else:
                        be_trigger = position.entry_price * (1 + (config.BREAKEVEN_TRIGGER_PCT / 100) * position.direction)
                else:
                    be_trigger = None

                # Mum açılışında zaten breakeven eşiğinin üzerindeyse SL'yi entry'e çek
                if be_trigger is not None and position.direction == 1 and current["open"] >= be_trigger:
                    position.sl_price = max(position.sl_price, position.entry_price)
                    position.is_breakeven = True

                # İntra-bar Fiyat Hareketi Değerlendirmesi (Pessimistic Order Execution)
                if position.direction == 1:
                    # A) Zarar Kes (SL) Kontrolü
                    if current["low"] <= position.sl_price:
                        exit_price = position.sl_price
                        exit_reason = "BREAKEVEN" if position.is_breakeven else "SL"
                        closed = True

                    # B) Kâr Al (TP) Kontrolü
                    elif current["high"] >= position.tp_price:
                        exit_price = position.tp_price
                        exit_reason = "TP"
                        closed = True

                    # C) Mum İçinde Breakeven Eşiğine Ulaşma Kontrolü
                    elif be_trigger is not None and current["high"] >= be_trigger:
                        position.sl_price = max(position.sl_price, position.entry_price)
                        position.is_breakeven = True
                        # Aynı mum içinde kârdan dönüp yeni başabaş SL'ye çarptı mı?
                        if current["low"] <= position.sl_price:
                            exit_price = position.sl_price
                            exit_reason = "BREAKEVEN"
                            closed = True

                else:  # SHORT (İlerideki çift yönlü işlemler için)
                    if be_trigger is not None and current["open"] <= be_trigger:
                        position.sl_price = min(position.sl_price, position.entry_price)
                        position.is_breakeven = True

                    if current["high"] >= position.sl_price:
                        exit_price = position.sl_price
                        exit_reason = "BREAKEVEN" if position.is_breakeven else "SL"
                        closed = True
                    elif current["low"] <= position.tp_price:
                        exit_price = position.tp_price
                        exit_reason = "TP"
                        closed = True
                    elif be_trigger is not None and current["low"] <= be_trigger:
                        position.sl_price = min(position.sl_price, position.entry_price)
                        position.is_breakeven = True
                        if current["high"] >= position.sl_price:
                            exit_price = position.sl_price
                            exit_reason = "BREAKEVEN"
                            closed = True

                # Pozisyon kapandıysa PnL ve Sermaye güncelle
                if closed and exit_price is not None:
                    adj_exit = _apply_cost(exit_price, position.direction, "exit", slip)
                    raw_pnl = (adj_exit - position.entry_price) * position.direction
                    qty = position.size / position.entry_price
                    pnl = raw_pnl * qty
                    position.exit_time = now
                    position.exit_price = adj_exit
                    position.pnl = pnl
                    position.pnl_pct = (raw_pnl / position.entry_price) * 100
                    position.exit_reason = exit_reason
                    capital += pnl
                    result.trades.append(position)
                    log.debug(
                        "CLOSE %s [%s] @ %.4f | PnL=%.2f (%.2f%%)",
                        self.symbol_label, exit_reason, adj_exit, pnl, position.pnl_pct,
                    )
                    position = None

            # ── 2. Giriş ve Strateji Çıkış Sinyalleri ─────────────────
            # Sinyal mutlaka önceki kapanmış mumdan (i-1) alınır (Lookahead Bias yok)
            signal = signals.iloc[i - 1]

            # ALIM (LONG) Girişi
            if signal == 1 and position is None:
                entry_raw = current["open"]
                entry = _apply_cost(entry_raw, 1, "entry", slip)
                alloc = capital * (self.position_size_pct / 100)
                if alloc > 0:
                    tp, sl = self._compute_tp_sl(entry, 1, atr_val)
                    position = Trade(
                        symbol=self.symbol_label,
                        entry_time=now,
                        entry_price=entry,
                        direction=1,
                        size=alloc,
                        tp_price=tp,
                        sl_price=sl,
                    )
                    log.debug(
                        "OPEN LONG %s @ %.4f | TP=%.4f | SL=%.4f | ATR=%.2f",
                        self.symbol_label, entry, tp, sl, atr_val if not np.isnan(atr_val) else 0,
                    )

            # STRATEJİDEN GELEN SATIŞ / ÇIKIŞ SİNYALİ
            elif signal == -1 and position is not None and position.direction == 1:
                adj_exit = _apply_cost(current["open"], 1, "exit", slip)
                raw_pnl = adj_exit - position.entry_price
                qty = position.size / position.entry_price
                pnl = raw_pnl * qty
                position.exit_time = now
                position.exit_price = adj_exit
                position.pnl = pnl
                position.pnl_pct = (raw_pnl / position.entry_price) * 100
                position.exit_reason = "SIGNAL"
                capital += pnl
                result.trades.append(position)
                position = None

            result.equity_curve.append(capital)
            result.equity_dates.append(now)

        # ── 3. Test Sonunda Kalan Açık Pozisyonu Kapat ────────────────
        if position is not None:
            last = df.iloc[-1]
            atr_last = atr.iloc[-1] if len(atr) > 0 else np.nan
            slip = _dynamic_slippage_pct(atr_last, last["close"])
            adj_exit = _apply_cost(last["close"], position.direction, "exit", slip)
            raw_pnl = (adj_exit - position.entry_price) * position.direction
            qty = position.size / position.entry_price
            pnl = raw_pnl * qty
            position.exit_time = df.index[-1]
            position.exit_price = adj_exit
            position.pnl = pnl
            position.pnl_pct = (raw_pnl / position.entry_price) * 100
            position.exit_reason = "END"
            capital += pnl
            result.trades.append(position)
            result.equity_curve.append(capital)
            result.equity_dates.append(df.index[-1])

        return result


# ─────────────────────────────────────────────────────────────────────
# Portfolio Backtester (Multi-Symbol, Shared Capital)
# ─────────────────────────────────────────────────────────────────────

class PortfolioBacktester:
    """Run strategy across multiple symbols with shared capital and position limits."""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = config.INITIAL_CAPITAL,
        max_risk_pct: float = config.MAX_PORTFOLIO_RISK_PCT,
        max_positions: int = config.MAX_OPEN_POSITIONS,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.per_symbol_pct = min(
            config.POSITION_SIZE_PCT,
            max_risk_pct / max(max_positions, 1),
        )
        self.max_positions = max_positions

    def run(self, datasets: dict[str, pd.DataFrame]) -> BacktestResult:
        combined = BacktestResult()
        capital = self.initial_capital
        open_count = 0

        for sym, df in datasets.items():
            size_pct = self.per_symbol_pct if open_count < self.max_positions else 0.0

            bt = Backtester(
                self.strategy,
                initial_capital=capital,
                position_size_pct=size_pct,
                symbol_label=sym,
            )
            res = bt.run(df)

            combined.trades.extend(res.trades)
            if res.equity_curve:
                capital = res.equity_curve[-1]

            open_count += sum(1 for t in res.trades if t.exit_time is None)

        combined.trades.sort(key=lambda t: t.entry_time)

        # Kronolojik işlemlerden konsolide equity curve oluştur
        eq = self.initial_capital
        combined.equity_curve = [eq]
        combined.equity_dates = [
            combined.trades[0].entry_time if combined.trades else pd.Timestamp.now(tz="UTC")
        ]
        for t in combined.trades:
            if t.pnl is not None:
                eq += t.pnl
            combined.equity_curve.append(eq)
            combined.equity_dates.append(t.exit_time or t.entry_time)

        return combined


# ─────────────────────────────────────────────────────────────────────
# Walk-Forward Analysis Partitioning
# ─────────────────────────────────────────────────────────────────────

def walk_forward_split(
    df: pd.DataFrame,
    is_end: str = config.WF_IN_SAMPLE_END,
    oos_start: str = config.WF_OUT_OF_SAMPLE_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into In-Sample (training) and Out-of-Sample (testing)."""
    is_df = df[df.index <= is_end]
    oos_df = df[df.index >= oos_start]
    return is_df, oos_df


# ─────────────────────────────────────────────────────────────────────
# Performance Metrics & Reporting
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(
    result: BacktestResult,
    initial_capital: float,
    label: str = "",
) -> dict:
    trades = result.trades
    if not trades:
        return {"total_trades": 0, "label": label}

    wins = [t for t in trades if t.pnl and t.pnl > 0]
    losses = [t for t in trades if t.pnl and t.pnl <= 0]
    be_trades = [t for t in trades if t.exit_reason == "BREAKEVEN"]
    tp_trades = [t for t in trades if t.exit_reason == "TP"]
    sl_trades = [t for t in trades if t.exit_reason == "SL"]

    total_profit = sum(t.pnl for t in wins)
    total_loss = abs(sum(t.pnl for t in losses)) if losses else 0

    equity = pd.Series(result.equity_curve)
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100
    max_dd = drawdown.min()

    win_durations, loss_durations = [], []
    for t in trades:
        if t.exit_time and t.entry_time:
            d = t.exit_time - t.entry_time
            (win_durations if t.pnl and t.pnl > 0 else loss_durations).append(d)
    all_dur = win_durations + loss_durations
    avg_dur = sum(all_dur, timedelta()) / len(all_dur) if all_dur else timedelta()
    avg_win_dur = sum(win_durations, timedelta()) / len(win_durations) if win_durations else timedelta()
    avg_loss_dur = sum(loss_durations, timedelta()) / len(loss_durations) if loss_durations else timedelta()

    avg_win_amt = total_profit / len(wins) if wins else 0.0
    avg_loss_amt = total_loss / len(losses) if losses else 0.0

    net_pnl = equity.iloc[-1] - initial_capital
    net_roi = (net_pnl / initial_capital) * 100

    date_start = result.equity_dates[0] if result.equity_dates else None
    date_end = result.equity_dates[-1] if result.equity_dates else None
    total_days = (date_end - date_start).days if date_start and date_end else 0
    total_years = round(total_days / 365.25, 2)

    symbols_traded = sorted(set(t.symbol for t in trades))

    return {
        "label": label,
        "symbols": symbols_traded,
        "date_start": str(date_start),
        "date_end": str(date_end),
        "total_days": total_days,
        "total_years": total_years,
        "total_trades": len(trades),
        "total_wins": len(wins),
        "total_losses": len(losses),
        "total_breakeven": len(be_trades),
        "total_tp": len(tp_trades),
        "total_sl": len(sl_trades),
        "win_rate": len(wins) / len(trades) * 100 if trades else 0.0,
        "profit_factor": total_profit / total_loss if total_loss > 0 else float("inf"),
        "max_drawdown_pct": max_dd,
        "net_roi_pct": net_roi,
        "net_pnl": net_pnl,
        "final_capital": equity.iloc[-1],
        "avg_hold_duration": str(avg_dur),
        "avg_win_duration": str(avg_win_dur),
        "avg_loss_duration": str(avg_loss_dur),
        "avg_win_amount": avg_win_amt,
        "avg_loss_amount": avg_loss_amt,
    }


def print_metrics(metrics: dict) -> None:
    label = metrics.get("label", "")
    header = f"  BACKTEST REPORT" + (f" - {label}" if label else "")
    print("\n" + "=" * 64)
    print(header)
    print("=" * 64)
    if metrics["total_trades"] == 0:
        print("  No trades executed.")
        print("=" * 64 + "\n")
        return

    syms = ", ".join(metrics.get("symbols", []))
    if syms:
        print(f"  Symbols             : {syms}")
    print(f"  Period              : {metrics['date_start']}")
    print(f"                     -> {metrics['date_end']}")
    print(f"  Duration            : {metrics['total_days']} days ({metrics['total_years']} years)")
    print("-" * 64)
    print(f"  Total Trades        : {metrics['total_trades']}")
    print(f"  Wins / Losses / BE  : {metrics['total_wins']} / {metrics['total_losses']} / {metrics['total_breakeven']}")
    print(f"  Exit Breakdown      : TP: {metrics['total_tp']} | SL: {metrics['total_sl']} | BE: {metrics['total_breakeven']}")
    print(f"  Win Rate            : {metrics['win_rate']:.2f}%")
    print(f"  Profit Factor       : {metrics['profit_factor']:.2f}")
    print(f"  Max Drawdown        : {metrics['max_drawdown_pct']:.2f}%")
    print("-" * 64)
    print(f"  Net ROI             : {metrics['net_roi_pct']:.2f}%")
    print(f"  Net PnL             : {metrics['net_pnl']:.2f}")
    print(f"  Final Capital       : {metrics['final_capital']:.2f}")
    print("-" * 64)
    print(f"  Avg Win Amount      : {metrics['avg_win_amount']:.2f}")
    print(f"  Avg Loss Amount     : {metrics['avg_loss_amount']:.2f}")
    print(f"  Avg Hold Duration   : {metrics['avg_hold_duration']}")
    print(f"  Avg Win Duration    : {metrics['avg_win_duration']}")
    print(f"  Avg Loss Duration   : {metrics['avg_loss_duration']}")
    print("-" * 64)
    sl_mode = "ATR-based" if config.USE_ATR_STOPS else "Fixed %"
    print(f"  TP/SL Mode          : {sl_mode}")
    if config.USE_ATR_STOPS:
        print(f"  ATR SL / TP Mult    : {config.ATR_SL_MULTIPLIER}x / {config.ATR_TP_MULTIPLIER}x")
    else:
        print(f"  TP / SL             : {config.TAKE_PROFIT_PCT}% / {config.STOP_LOSS_PCT}%")
    be_mode = f"Active (+{config.BREAKEVEN_TRIGGER_ATR}x ATR)" if config.USE_BREAKEVEN_STOP else "Disabled"
    print(f"  Breakeven Stop      : {be_mode}")
    slip_mode = "Dynamic (ATR)" if config.DYNAMIC_SLIPPAGE else "Fixed"
    print(f"  Slippage Mode       : {slip_mode}  |  Commission: {config.COMMISSION_PCT}%")
    print("=" * 64 + "\n")
