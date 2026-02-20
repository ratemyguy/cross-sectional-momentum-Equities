"""
=============================================================
  QUANT STRATEGY: Cross-Sectional Momentum
  Layer 3: Backtest Engine + Portfolio Analytics
=============================================================
  Requires: prices_monthly.csv, momentum_signals.csv
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────

def run_backtest(prices, signals, transaction_cost=0.001):
    """
    Monthly rebalancing backtest with transaction cost modeling.

    Each month:
    1. Identify which stocks to hold (signal = 1)
    2. Equal-weight across all held stocks
    3. Calculate portfolio return for the month
    4. Subtract transaction costs on stocks that changed
    5. Compound returns into equity curve

    transaction_cost = 0.001 = 0.10% per trade (one way)
    This is realistic for large-cap US stocks at institutional size.
    Retail investors would pay more (~0.05% with a good broker).

    WHY EQUAL WEIGHT?
    Simplest and most robust. Market-cap weighting would just
    make it a closet index fund. Equal weight gives every
    momentum stock the same shot.
    """
    # Monthly returns for each stock
    monthly_returns = prices.pct_change()

    portfolio_returns = []
    turnover_list     = []
    holdings_list     = []

    prev_holdings = set()

    for date in signals.index:
        if date not in monthly_returns.index:
            continue

        # Current holdings: stocks where signal = 1
        row           = signals.loc[date]
        curr_holdings = set(row[row == 1].index.tolist())

        # Remove stocks with no price data this month
        valid = set(prices.loc[date].dropna().index)
        curr_holdings = curr_holdings & valid

        if len(curr_holdings) == 0:
            portfolio_returns.append(0)
            turnover_list.append(0)
            holdings_list.append(0)
            prev_holdings = curr_holdings
            continue

        # Portfolio return = equal weight average
        rets  = monthly_returns.loc[date, list(curr_holdings)].dropna()
        port_ret = rets.mean()

        # Turnover = fraction of portfolio that changed
        if len(prev_holdings) > 0:
            entered  = curr_holdings - prev_holdings
            exited   = prev_holdings - curr_holdings
            turnover = (len(entered) + len(exited)) / (2 * len(curr_holdings))
        else:
            turnover = 1.0  # first month = full entry

        # Transaction cost: pay on entered AND exited positions
        cost = turnover * transaction_cost * 2

        net_return = port_ret - cost

        portfolio_returns.append(net_return)
        turnover_list.append(turnover)
        holdings_list.append(len(curr_holdings))

        prev_holdings = curr_holdings

    results = pd.DataFrame({
        "return":   portfolio_returns,
        "turnover": turnover_list,
        "holdings": holdings_list,
    }, index=signals.index[:len(portfolio_returns)])

    return results


# ─────────────────────────────────────────────────────────────
# BENCHMARK: BUY AND HOLD SPY
# ─────────────────────────────────────────────────────────────

def build_benchmark(prices):
    """
    Equal-weight buy-and-hold of all stocks.
    This is our benchmark — if momentum can't beat this,
    the signal adds no value over just owning everything.
    """
    monthly_returns = prices.pct_change()
    benchmark = monthly_returns.mean(axis=1)
    return benchmark


# ─────────────────────────────────────────────────────────────
# PERFORMANCE ANALYTICS
# ─────────────────────────────────────────────────────────────

def compute_analytics(returns, label="Strategy", starting_equity=100000):
    """
    Full analytics suite. Every metric here maps to your resume bullet.

    Sharpe Ratio    → risk-adjusted return (higher = better, >1 is good)
    Max Drawdown    → worst peak-to-trough loss (interviewers always ask)
    Turnover        → how much the portfolio changes monthly
    Volatility      → annualized standard deviation of returns
    Calmar Ratio    → annualized return / max drawdown
    """
    r = returns.dropna()

    # Equity curve
    equity = starting_equity * (1 + r).cumprod()

    # Returns
    total_ret  = (equity.iloc[-1] / starting_equity - 1) * 100
    years      = len(r) / 12
    ann_ret    = ((equity.iloc[-1] / starting_equity) ** (1/years) - 1) * 100

    # Risk
    ann_vol    = r.std() * np.sqrt(12) * 100
    sharpe     = (r.mean() * 12) / (r.std() * np.sqrt(12)) if r.std() > 0 else 0

    # Sortino (downside only)
    downside   = r[r < 0]
    sortino    = (r.mean() * 12) / (downside.std() * np.sqrt(12)) if len(downside) > 1 else 0

    # Drawdown
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max * 100
    max_dd      = drawdown.min()

    # Calmar
    calmar     = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate   = (r > 0).mean() * 100

    metrics = {
        "label":      label,
        "total_ret":  round(total_ret,  2),
        "ann_ret":    round(ann_ret,    2),
        "ann_vol":    round(ann_vol,    2),
        "sharpe":     round(sharpe,     3),
        "sortino":    round(sortino,    3),
        "max_dd":     round(max_dd,     2),
        "calmar":     round(calmar,     3),
        "win_rate":   round(win_rate,   1),
        "equity":     equity,
        "drawdown":   drawdown,
    }

    print(f"\n{'='*55}")
    print(f"  PERFORMANCE — {label}")
    print(f"{'='*55}")
    print(f"  Total Return     : {total_ret:.1f}%")
    print(f"  Ann. Return      : {ann_ret:.1f}%")
    print(f"  Ann. Volatility  : {ann_vol:.1f}%")
    print(f"  Sharpe Ratio     : {sharpe:.3f}")
    print(f"  Sortino Ratio    : {sortino:.3f}")
    print(f"  Max Drawdown     : {max_dd:.1f}%")
    print(f"  Calmar Ratio     : {calmar:.3f}")
    print(f"  Monthly Win Rate : {win_rate:.1f}%")
    print(f"  Final Equity     : ${equity.iloc[-1]:,.0f}")
    print(f"{'='*55}")

    return metrics


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_backtest(strat_results, benchmark_returns, strat_metrics,
                  bench_metrics, prices):
    """
    5-panel backtest chart:
    1. Equity curves: strategy vs benchmark
    2. Drawdown comparison
    3. Monthly returns distribution
    4. Rolling 12-month Sharpe
    5. Turnover over time
    """
    BG, PANEL    = "#0f0f0f", "#1a1a1a"
    WHITE        = "#e8e8e8"
    GREEN, RED   = "#00ff88", "#ff4466"
    YELLOW, BLUE = "#f5c518", "#4488ff"

    fig = plt.figure(figsize=(16, 18), facecolor=BG)
    gs  = gridspec.GridSpec(5, 1, hspace=0.50)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=7)
        ax.tick_params(colors=WHITE, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(True, color="#222222", linewidth=0.5, alpha=0.8)

    strat_eq  = strat_metrics["equity"]
    bench_eq  = bench_metrics["equity"]
    strat_dd  = strat_metrics["drawdown"]
    bench_dd  = bench_metrics["drawdown"]

    # ── Panel 1: Equity Curves ────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(strat_eq.index, strat_eq,  color=GREEN,  lw=1.5,
             label=f"Momentum Strategy  (Ann: {strat_metrics['ann_ret']:.1f}%)")
    ax1.plot(bench_eq.index, bench_eq,  color=YELLOW, lw=1.2, linestyle="--",
             label=f"Equal-Weight Benchmark  (Ann: {bench_metrics['ann_ret']:.1f}%)")
    ax1.axhline(100000, color=WHITE, lw=0.5, linestyle=":", alpha=0.4)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}k"))
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    style_ax(ax1, "Equity Curve: Momentum Strategy vs Equal-Weight Benchmark ($100k start)")

    # ── Panel 2: Drawdown ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(strat_dd.index, strat_dd, 0,
                     alpha=0.5, color=RED,   label="Strategy DD")
    ax2.fill_between(bench_dd.index, bench_dd, 0,
                     alpha=0.3, color=YELLOW, label="Benchmark DD")
    ax2.plot(strat_dd.index, strat_dd, color=RED,    lw=0.8)
    ax2.plot(bench_dd.index, bench_dd, color=YELLOW, lw=0.8)
    ax2.axhline(0, color=WHITE, lw=0.5)
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    style_ax(ax2, "Drawdown (%)")

    # ── Panel 3: Monthly Returns Distribution ─────────────
    ax3 = fig.add_subplot(gs[2])
    r = strat_results["return"].dropna()
    ax3.hist(r[r > 0]*100, bins=40, color=GREEN, alpha=0.7,
             label=f"Wins ({(r>0).sum()})", edgecolor="none")
    ax3.hist(r[r <= 0]*100, bins=40, color=RED, alpha=0.7,
             label=f"Losses ({(r<=0).sum()})", edgecolor="none")
    ax3.axvline(0, color=WHITE, lw=1)
    ax3.axvline(r.mean()*100, color=YELLOW, lw=1.5, linestyle="--",
                label=f"Mean: {r.mean()*100:.2f}%")
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    style_ax(ax3, "Monthly Return Distribution (%)")

    # ── Panel 4: Rolling 12-month Sharpe ──────────────────
    ax4 = fig.add_subplot(gs[3])
    r_clean = strat_results["return"].dropna()
    roll_sharpe = r_clean.rolling(12).apply(
        lambda x: (x.mean() * 12) / (x.std() * np.sqrt(12)) if x.std() > 0 else 0
    )
    ax4.plot(roll_sharpe.index, roll_sharpe, color=BLUE, lw=1.2)
    ax4.fill_between(roll_sharpe.index, roll_sharpe, 0,
                     where=roll_sharpe > 0, alpha=0.3, color=GREEN)
    ax4.fill_between(roll_sharpe.index, roll_sharpe, 0,
                     where=roll_sharpe < 0, alpha=0.3, color=RED)
    ax4.axhline(0, color=WHITE, lw=0.8)
    ax4.axhline(1, color=GREEN, lw=0.8, linestyle="--", alpha=0.5, label="Sharpe=1")
    ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    style_ax(ax4, "Rolling 12-Month Sharpe Ratio")

    # ── Panel 5: Turnover ─────────────────────────────────
    ax5 = fig.add_subplot(gs[4])
    tv = strat_results["turnover"].dropna() * 100
    ax5.fill_between(tv.index, tv, alpha=0.5, color=BLUE)
    ax5.plot(tv.index, tv, color=BLUE, lw=0.8)
    ax5.axhline(tv.mean(), color=YELLOW, lw=1, linestyle="--",
                label=f"Avg turnover: {tv.mean():.1f}%")
    ax5.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    style_ax(ax5, "Monthly Portfolio Turnover (%) — Lower = Less Transaction Cost Drag")

    fig.suptitle(
        "Cross-Sectional Momentum Strategy — Backtest Results (Layer 3)  |  2010-2024",
        color=WHITE, fontsize=13, fontweight="bold", y=0.999)

    plt.savefig("backtest_results_momentum.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("[Layer 3] Chart saved -> backtest_results_momentum.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Load data
    print("[Layer 3] Loading data...")
    prices  = pd.read_csv("prices_monthly.csv",   index_col=0, parse_dates=True)
    signals = pd.read_csv("momentum_signals.csv", index_col=0, parse_dates=True)
    print(f"  Prices : {prices.shape}")
    print(f"  Signals: {signals.shape}")

    # Align columns
    common = prices.columns.intersection(signals.columns)
    prices  = prices[common]
    signals = signals[common]

    # 2. Run backtest
    print("\n[Layer 3] Running backtest...")
    results = run_backtest(prices, signals, transaction_cost=0.001)

    # 3. Benchmark
    benchmark = build_benchmark(prices)
    benchmark = benchmark.reindex(results.index)

    # 4. Analytics
    strat_metrics = compute_analytics(results["return"],
                                      label="Momentum Strategy")
    bench_metrics = compute_analytics(benchmark,
                                      label="Equal-Weight Benchmark")

    # 5. Turnover stats
    avg_turnover = results["turnover"].mean() * 100
    avg_holdings = results["holdings"].mean()
    print(f"\n  Avg monthly turnover : {avg_turnover:.1f}%")
    print(f"  Avg stocks held      : {avg_holdings:.1f}")

    # 6. Alpha over benchmark
    alpha = strat_metrics["ann_ret"] - bench_metrics["ann_ret"]
    print(f"\n  Annual Alpha vs Benchmark: {alpha:.2f}%")
    print(f"  Sharpe improvement      : {strat_metrics['sharpe'] - bench_metrics['sharpe']:.3f}")

    # 7. Plot
    plot_backtest(results, benchmark, strat_metrics, bench_metrics, prices)

    # 8. Save
    results.to_csv("backtest_results_momentum.csv")
    print("\n[Done] Saved -> backtest_results_momentum.csv")
