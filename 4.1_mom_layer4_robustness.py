"""
=============================================================
  QUANT STRATEGY: Cross-Sectional Momentum
  Layer 4: Robustness Checks + Final Tearsheet
=============================================================
  Requires: prices_monthly.csv, backtest_results_momentum.csv
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# REUSABLE FUNCTIONS FROM LAYER 3
# ─────────────────────────────────────────────────────────────

def compute_momentum(prices, lookback=12, skip=1):
    return prices.shift(skip) / prices.shift(lookback) - 1

def run_backtest(prices, signals, transaction_cost=0.001):
    monthly_returns = prices.pct_change()
    portfolio_returns = []
    prev_holdings = set()

    for date in signals.index:
        if date not in monthly_returns.index:
            continue
        row           = signals.loc[date]
        curr_holdings = set(row[row == 1].index.tolist())
        valid         = set(prices.loc[date].dropna().index)
        curr_holdings = curr_holdings & valid

        if len(curr_holdings) == 0:
            portfolio_returns.append(0)
            prev_holdings = curr_holdings
            continue

        rets     = monthly_returns.loc[date, list(curr_holdings)].dropna()
        port_ret = rets.mean()

        if len(prev_holdings) > 0:
            entered  = curr_holdings - prev_holdings
            exited   = prev_holdings - curr_holdings
            turnover = (len(entered) + len(exited)) / (2 * len(curr_holdings))
        else:
            turnover = 1.0

        cost      = turnover * transaction_cost * 2
        portfolio_returns.append(port_ret - cost)
        prev_holdings = curr_holdings

    return pd.Series(portfolio_returns, index=signals.index[:len(portfolio_returns)])

def compute_metrics(returns):
    r = returns.dropna()
    if len(r) < 12:
        return {}
    equity     = (1 + r).cumprod()
    years      = len(r) / 12
    ann_ret    = (equity.iloc[-1] ** (1/years) - 1) * 100
    ann_vol    = r.std() * np.sqrt(12) * 100
    sharpe     = (r.mean() * 12) / (r.std() * np.sqrt(12)) if r.std() > 0 else 0
    rolling_max = equity.cummax()
    max_dd     = ((equity - rolling_max) / rolling_max * 100).min()
    calmar     = ann_ret / abs(max_dd) if max_dd != 0 else 0
    downside   = r[r < 0]
    sortino    = (r.mean() * 12) / (downside.std() * np.sqrt(12)) if len(downside) > 1 else 0
    return {
        "ann_ret": round(ann_ret, 2),
        "ann_vol": round(ann_vol, 2),
        "sharpe":  round(sharpe,  3),
        "sortino": round(sortino, 3),
        "max_dd":  round(max_dd,  2),
        "calmar":  round(calmar,  3),
        "equity":  equity,
    }


# ─────────────────────────────────────────────────────────────
# ROBUSTNESS CHECK 1: LOOKBACK WINDOWS
# ─────────────────────────────────────────────────────────────

def test_lookback_windows(prices, windows=[3, 6, 9, 12, 18, 24]):
    """
    Test momentum across different lookback windows.

    Windows tested:
      3-1  = short-term momentum (3 months back, skip 1)
      6-1  = medium-term
      9-1  = medium-term
      12-1 = standard (Jegadeesh & Titman)
      18-1 = longer-term
      24-1 = very long-term (starts to reverse)

    If the strategy only works for 12-1, it's curve-fitted.
    If it works across 6, 9, 12, 18 — it's a real signal.
    """
    print("\n[Layer 4] Testing lookback windows...")
    results = {}

    for lb in windows:
        if lb <= 1:
            continue
        mom      = compute_momentum(prices, lookback=lb, skip=1)
        rankings = mom.rank(axis=1, ascending=False, pct=True)
        signals  = (rankings <= 0.20).astype(int)
        returns  = run_backtest(prices, signals)
        metrics  = compute_metrics(returns)
        if metrics:
            results[f"{lb}-1"] = {**metrics, "returns": returns}
            print(f"  {lb}-1 momentum  |  Sharpe: {metrics['sharpe']:.3f}  "
                  f"Ann Ret: {metrics['ann_ret']:.1f}%  "
                  f"Max DD: {metrics['max_dd']:.1f}%")

    return results


# ─────────────────────────────────────────────────────────────
# ROBUSTNESS CHECK 2: TOP QUINTILE CUTOFF
# ─────────────────────────────────────────────────────────────

def test_portfolio_size(prices, cutoffs=[0.10, 0.20, 0.30, 0.40]):
    """
    Test different portfolio concentration levels.

    Top 10% = very concentrated, ~18 stocks
    Top 20% = standard, ~36 stocks
    Top 30% = diversified, ~55 stocks
    Top 40% = broad, ~73 stocks

    More concentrated = higher potential alpha but higher vol.
    Shows the tradeoff between concentration and diversification.
    """
    print("\n[Layer 4] Testing portfolio concentration...")
    results = {}

    mom      = compute_momentum(prices, lookback=12, skip=1)
    rankings = mom.rank(axis=1, ascending=False, pct=True)

    for cutoff in cutoffs:
        signals = (rankings <= cutoff).astype(int)
        returns = run_backtest(prices, signals)
        metrics = compute_metrics(returns)
        label   = f"Top {int(cutoff*100)}%"
        if metrics:
            results[label] = {**metrics, "returns": returns}
            avg_stocks = signals.sum(axis=1).mean()
            print(f"  {label}  (~{avg_stocks:.0f} stocks)  |  "
                  f"Sharpe: {metrics['sharpe']:.3f}  "
                  f"Ann Ret: {metrics['ann_ret']:.1f}%")

    return results


# ─────────────────────────────────────────────────────────────
# ROBUSTNESS CHECK 3: TRANSACTION COST SENSITIVITY
# ─────────────────────────────────────────────────────────────

def test_transaction_costs(prices, costs=[0.0, 0.001, 0.003, 0.005, 0.010]):
    """
    How sensitive is the strategy to transaction costs?

    0.000 = zero cost (theoretical max)
    0.001 = 0.10% (institutional, what we use)
    0.003 = 0.30% (retail with good broker)
    0.005 = 0.50% (retail average)
    0.010 = 1.00% (high cost — shows strategy breaks down)

    If the strategy only works at 0% cost, it's not real.
    A robust strategy should survive realistic costs.
    """
    print("\n[Layer 4] Testing transaction cost sensitivity...")
    results = {}

    mom      = compute_momentum(prices, lookback=12, skip=1)
    rankings = mom.rank(axis=1, ascending=False, pct=True)
    signals  = (rankings <= 0.20).astype(int)

    for cost in costs:
        returns = run_backtest(prices, signals, transaction_cost=cost)
        metrics = compute_metrics(returns)
        label   = f"{cost*100:.1f}% cost"
        if metrics:
            results[label] = {**metrics, "returns": returns}
            print(f"  {label}  |  Sharpe: {metrics['sharpe']:.3f}  "
                  f"Ann Ret: {metrics['ann_ret']:.1f}%")

    return results


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_robustness(lookback_results, size_results, cost_results):
    """
    4-panel robustness chart:
    1. Equity curves across lookback windows
    2. Sharpe ratio bar chart by lookback
    3. Equity curves across portfolio sizes
    4. Ann return vs transaction cost (degradation curve)
    """
    BG, PANEL    = "#0f0f0f", "#1a1a1a"
    WHITE        = "#e8e8e8"
    GREEN, RED   = "#00ff88", "#ff4466"
    YELLOW, BLUE = "#f5c518", "#4488ff"

    fig = plt.figure(figsize=(16, 16), facecolor=BG)
    gs  = gridspec.GridSpec(4, 1, hspace=0.50)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=7)
        ax.tick_params(colors=WHITE, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(True, color="#222222", linewidth=0.5, alpha=0.8)

    colors = [GREEN, BLUE, YELLOW, "#cc88ff", "#ff8844", RED]

    # ── Panel 1: Equity Curves by Lookback ───────────────
    ax1 = fig.add_subplot(gs[0])
    for (label, data), color in zip(lookback_results.items(), colors):
        eq = data["equity"]
        ax1.plot(eq.index, eq * 100000, color=color, lw=1.2, label=label)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}k"))
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE, ncol=3)
    style_ax(ax1, "Equity Curves by Lookback Window  |  All start at $100k")

    # ── Panel 2: Sharpe by Lookback (bar chart) ───────────
    ax2 = fig.add_subplot(gs[1])
    labels  = list(lookback_results.keys())
    sharpes = [lookback_results[l]["sharpe"] for l in labels]
    bar_colors = [GREEN if s > 0 else RED for s in sharpes]
    bars = ax2.bar(labels, sharpes, color=bar_colors, alpha=0.85, edgecolor="none")
    ax2.axhline(0, color=WHITE, lw=0.8)
    ax2.axhline(1, color=YELLOW, lw=0.8, linestyle="--", alpha=0.6, label="Sharpe=1")
    for bar, val in zip(bars, sharpes):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 val + (0.02 if val >= 0 else -0.08),
                 f"{val:.2f}", ha="center", color=WHITE, fontsize=9)
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    style_ax(ax2, "Sharpe Ratio by Lookback Window  |  Robust if consistently positive")

    # ── Panel 3: Equity Curves by Portfolio Size ──────────
    ax3 = fig.add_subplot(gs[2])
    for (label, data), color in zip(size_results.items(), colors):
        eq = data["equity"]
        ax3.plot(eq.index, eq * 100000, color=color, lw=1.2, label=label)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}k"))
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    style_ax(ax3, "Equity Curves by Portfolio Concentration  |  Top 10% vs Top 40%")

    # ── Panel 4: Return Degradation vs Transaction Costs ──
    ax4 = fig.add_subplot(gs[3])
    cost_labels  = list(cost_results.keys())
    cost_returns = [cost_results[l]["ann_ret"] for l in cost_labels]
    cost_sharpes = [cost_results[l]["sharpe"]  for l in cost_labels]

    ax4.plot(range(len(cost_labels)), cost_returns,
             color=GREEN, lw=2, marker="o", markersize=7, label="Ann. Return %")
    ax4.plot(range(len(cost_labels)), cost_sharpes,
             color=BLUE,  lw=2, marker="s", markersize=7, label="Sharpe Ratio")
    ax4.axhline(0, color=WHITE, lw=0.5, linestyle="--")
    ax4.set_xticks(range(len(cost_labels)))
    ax4.set_xticklabels(cost_labels, color=WHITE, fontsize=8)
    ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    style_ax(ax4, "Strategy Degradation vs Transaction Costs  |  Should survive 0.1-0.3% costs")

    fig.suptitle(
        "Cross-Sectional Momentum — Robustness Checks (Layer 4)  |  2010-2024",
        color=WHITE, fontsize=13, fontweight="bold", y=0.999)

    plt.savefig("robustness_checks.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("\n[Layer 4] Chart saved -> robustness_checks.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────

def print_final_summary(lookback_results, size_results, cost_results):
    print("\n" + "="*65)
    print("  ROBUSTNESS SUMMARY")
    print("="*65)

    print("\n  Lookback Window Results:")
    print(f"  {'Window':<10} {'Ann Ret':>10} {'Sharpe':>10} {'Max DD':>10} {'Calmar':>10}")
    print(f"  {'-'*50}")
    for label, data in lookback_results.items():
        print(f"  {label:<10} {data['ann_ret']:>9.1f}%"
              f" {data['sharpe']:>10.3f}"
              f" {data['max_dd']:>9.1f}%"
              f" {data['calmar']:>10.3f}")

    print("\n  Portfolio Size Results:")
    print(f"  {'Size':<12} {'Ann Ret':>10} {'Sharpe':>10} {'Max DD':>10}")
    print(f"  {'-'*44}")
    for label, data in size_results.items():
        print(f"  {label:<12} {data['ann_ret']:>9.1f}%"
              f" {data['sharpe']:>10.3f}"
              f" {data['max_dd']:>9.1f}%")

    print("\n  Transaction Cost Sensitivity:")
    print(f"  {'Cost':<12} {'Ann Ret':>10} {'Sharpe':>10}")
    print(f"  {'-'*34}")
    for label, data in cost_results.items():
        print(f"  {label:<12} {data['ann_ret']:>9.1f}%"
              f" {data['sharpe']:>10.3f}")

    print("="*65)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("[Layer 4] Loading prices...")
    prices = pd.read_csv("prices_monthly.csv", index_col=0, parse_dates=True)
    print(f"  Loaded {prices.shape}\n")

    # Run all 3 robustness checks
    lookback_results = test_lookback_windows(prices)
    size_results     = test_portfolio_size(prices)
    cost_results     = test_transaction_costs(prices)

    # Summary table
    print_final_summary(lookback_results, size_results, cost_results)

    # Chart
    plot_robustness(lookback_results, size_results, cost_results)

    print("\n[Done] All robustness checks complete.")
    print("  -> robustness_checks.png")
