"""Chart generation for backtest results."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from backtester import BacktestResult

OUTPUT_DIR = "reports"


def save_report_charts(
    result: BacktestResult,
    df: pd.DataFrame,
    metrics: dict,
    prefix: str = "backtest",
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    fig.suptitle("Backtest Report", fontsize=16, fontweight="bold")

    # ── 1. Equity Curve ──────────────────────────────────────────────
    ax = axes[0]
    eq_dates = result.equity_dates
    eq_values = result.equity_curve[: len(eq_dates)]
    ax.plot(eq_dates, eq_values, linewidth=1.2, color="#2196F3")
    ax.fill_between(eq_dates, eq_values, alpha=0.15, color="#2196F3")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Capital")
    ax.grid(True, alpha=0.3)

    # ── 2. Drawdown ──────────────────────────────────────────────────
    ax = axes[1]
    eq_series = pd.Series(eq_values, index=eq_dates)
    running_max = eq_series.cummax()
    dd = (eq_series - running_max) / running_max * 100
    ax.fill_between(dd.index, dd.values, color="#F44336", alpha=0.4)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)

    # ── 3. Trade PnL Distribution ────────────────────────────────────
    ax = axes[2]
    pnls = [t.pnl for t in result.trades if t.pnl is not None]
    if pnls:
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        ax.bar(range(len(pnls)), pnls, color=colors, width=0.8)
    ax.set_title("Per-Trade PnL")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("PnL")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, f"{prefix}_report.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Report chart saved -> {path}")
