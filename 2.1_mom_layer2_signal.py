"""
=============================================================
  QUANT STRATEGY: Cross-Sectional Momentum
  Layer 2: Momentum Signal + Rankings
=============================================================
  Requires: prices_monthly.csv (from Layer 1)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# MOMENTUM SIGNAL
# ─────────────────────────────────────────────────────────────

def compute_momentum(prices, lookback=12, skip=1):
    """
    12-1 Momentum Signal — Jegadeesh & Titman (1993)

    momentum(t) = price(t - skip) / price(t - lookback) - 1

    WHY SKIP THE LAST MONTH?
    The most recent month shows SHORT-TERM REVERSAL.
    Stocks that just jumped tend to mean-revert in the next month.
    By skipping it, we isolate the intermediate-term trend (months 2-12)
    which is where the real momentum signal lives.

    Example for stock AAPL in December 2024:
      lookback price = price in December 2023  (12 months ago)
      skip price     = price in November 2024  (1 month ago)
      momentum       = (Nov 2024 price / Dec 2023 price) - 1
    """
    momentum = prices.shift(skip) / prices.shift(lookback) - 1
    return momentum


def compute_rankings(momentum):
    """
    Cross-sectional rank: for each month, rank all stocks
    by their momentum score. Rank 1 = highest momentum.

    'Cross-sectional' just means we rank stocks against
    each other within the same time period.
    """
    # Rank from highest to lowest (ascending=False)
    # pct=True gives percentile rank (0 to 1)
    rankings = momentum.rank(axis=1, ascending=False, pct=True)
    return rankings


def compute_signals(rankings, top_pct=0.20):
    """
    Generate buy signals: top 20% of stocks by momentum.

    Top quintile (top 20%) is the standard from the academic
    literature. You can test other cutoffs in robustness checks.

    signal = 1 → hold this stock this month
    signal = 0 → don't hold
    """
    # rankings are percentiles: 0 = best, 1 = worst
    # so top 20% means ranking <= 0.20
    signals = (rankings <= top_pct).astype(int)
    return signals


def compute_quintile_returns(momentum, forward_returns):
    """
    Quintile analysis: split stocks into 5 groups by momentum,
    compute forward returns for each group.

    This is the KEY diagnostic chart for momentum strategies.
    If the strategy works, you should see:
    Q1 (highest momentum) > Q2 > Q3 > Q4 > Q5 (lowest momentum)

    A monotonic pattern = the signal has real predictive power.
    """
    results = {}

    for q in range(1, 6):
        low  = (q - 1) / 5
        high = q / 5
        # stocks in this quintile
        mask = (momentum.rank(axis=1, ascending=False, pct=True) > low) & \
               (momentum.rank(axis=1, ascending=False, pct=True) <= high)
        # equal-weight average forward return for this quintile
        q_returns = forward_returns[mask].mean(axis=1)
        results[f"Q{q}"] = q_returns

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────
# SIGNAL DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

def signal_stats(momentum, signals, prices):
    """
    Print key stats about the signal.
    Know these numbers — interviewers will ask.
    """
    # Forward returns (what happens next month)
    fwd_returns = prices.pct_change().shift(-1)

    # Average number of stocks in portfolio each month
    avg_holdings = signals.sum(axis=1).mean()
    avg_momentum_top = momentum[signals == 1].stack().mean()
    avg_momentum_bot = momentum[signals == 0].stack().mean()

    # Hit rate: does top quintile beat median next month?
    top_q_returns  = fwd_returns[signals == 1].mean(axis=1)
    other_returns  = fwd_returns[signals == 0].mean(axis=1)
    hit_rate = (top_q_returns > other_returns).mean() * 100

    print("\n" + "="*55)
    print("  SIGNAL DIAGNOSTICS")
    print("="*55)
    print(f"  Avg stocks in portfolio : {avg_holdings:.1f} per month")
    print(f"  Avg momentum (top 20%)  : {avg_momentum_top*100:.1f}%")
    print(f"  Avg momentum (rest)     : {avg_momentum_bot*100:.1f}%")
    print(f"  Hit rate vs median      : {hit_rate:.1f}% of months")
    print(f"  Signal coverage         : {signals.notna().mean().mean()*100:.1f}%")
    print("="*55 + "\n")

    return fwd_returns, top_q_returns, other_returns


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_signals(momentum, signals, prices, fwd_returns):
    """
    4-panel chart:
    1. Momentum distribution (what the signal looks like)
    2. Quintile forward returns (does it work?)
    3. Number of holdings over time
    4. Top vs bottom quintile cumulative returns
    """
    BG, PANEL  = "#0f0f0f", "#1a1a1a"
    WHITE      = "#e8e8e8"
    GREEN, RED = "#00ff88", "#ff4466"
    YELLOW, BLUE = "#f5c518", "#4488ff"

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    gs  = gridspec.GridSpec(4, 1, hspace=0.50)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=7)
        ax.tick_params(colors=WHITE, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(True, color="#222222", linewidth=0.5, alpha=0.8)

    # ── Panel 1: Momentum Distribution ───────────────────
    ax1 = fig.add_subplot(gs[0])
    flat_mom = momentum.stack().dropna()
    ax1.hist(flat_mom[flat_mom > -1], bins=80, color=BLUE,
             alpha=0.7, edgecolor="none")
    ax1.axvline(0, color=WHITE, lw=1, linestyle="--")
    ax1.axvline(flat_mom.quantile(0.80), color=GREEN, lw=1.5,
                linestyle="--", label="Top 20% threshold")
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    style_ax(ax1, "Momentum Score Distribution (All Stocks, All Months)")

    # ── Panel 2: Quintile Returns ─────────────────────────
    ax2 = fig.add_subplot(gs[1])
    quintile_rets = compute_quintile_returns(momentum, fwd_returns)
    avg_by_q = quintile_rets.mean() * 100
    colors_q  = [GREEN, "#44ff88", YELLOW, "#ff8844", RED]
    bars = ax2.bar(avg_by_q.index, avg_by_q.values,
                   color=colors_q, alpha=0.85, edgecolor="none")
    ax2.axhline(0, color=WHITE, lw=0.8)
    for bar, val in zip(bars, avg_by_q.values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.2f}%", ha="center", va="bottom",
                 color=WHITE, fontsize=9, fontweight="bold")
    style_ax(ax2, "Avg Monthly Forward Return by Momentum Quintile  |  Q1=Highest Momentum")

    # ── Panel 3: Holdings Over Time ───────────────────────
    ax3 = fig.add_subplot(gs[2])
    holdings = signals.sum(axis=1)
    ax3.fill_between(holdings.index, holdings.values,
                     alpha=0.5, color=BLUE)
    ax3.plot(holdings.index, holdings.values, color=BLUE, lw=1)
    ax3.axhline(holdings.mean(), color=YELLOW, lw=1,
                linestyle="--", label=f"Avg: {holdings.mean():.0f} stocks")
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    style_ax(ax3, "Number of Stocks in Portfolio Each Month")

    # ── Panel 4: Top vs Bottom Quintile Cumulative Return ─
    ax4 = fig.add_subplot(gs[3])
    top_mask = momentum.rank(axis=1, ascending=False, pct=True) <= 0.20
    bot_mask = momentum.rank(axis=1, ascending=False, pct=True) >  0.80

    top_ret  = fwd_returns[top_mask].mean(axis=1).dropna()
    bot_ret  = fwd_returns[bot_mask].mean(axis=1).dropna()

    top_cum  = (1 + top_ret).cumprod() - 1
    bot_cum  = (1 + bot_ret).cumprod() - 1

    ax4.plot(top_cum.index, top_cum * 100, color=GREEN, lw=1.5,
             label="Top Quintile (buy)")
    ax4.plot(bot_cum.index, bot_cum * 100, color=RED,   lw=1.5,
             label="Bottom Quintile (avoid)")
    ax4.axhline(0, color=WHITE, lw=0.5, linestyle="--")
    ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    ax4.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    style_ax(ax4, "Cumulative Return: Top vs Bottom Quintile  |  THIS proves the signal works")

    fig.suptitle("Cross-Sectional Momentum — Signal Analysis (Layer 2)",
                 color=WHITE, fontsize=13, fontweight="bold", y=0.999)

    plt.savefig("momentum_signals.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("[Layer 2] Chart saved -> momentum_signals.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Load monthly prices from Layer 1
    print("[Layer 2] Loading prices...")
    prices = pd.read_csv("prices_monthly.csv", index_col=0, parse_dates=True)
    print(f"[Layer 2] Loaded {prices.shape}")

    # 2. Compute 12-1 momentum
    print("[Layer 2] Computing momentum signals...")
    momentum = compute_momentum(prices, lookback=12, skip=1)

    # 3. Rank stocks cross-sectionally
    rankings = compute_rankings(momentum)

    # 4. Generate buy signals (top 20%)
    signals = compute_signals(rankings, top_pct=0.20)

    # 5. Diagnostics
    fwd_returns, top_q, other_q = signal_stats(momentum, signals, prices)

    # 6. Visualize
    plot_signals(momentum, signals, prices, fwd_returns)

    # 7. Save for Layer 3
    momentum.to_csv("momentum_scores.csv")
    signals.to_csv("momentum_signals.csv")
    print("[Done] Saved -> momentum_scores.csv, momentum_signals.csv")

    print("\nSample signals (last 3 months, first 8 stocks):")
    print(signals.iloc[-3:, :8].to_string())

    print("\nSample momentum scores (last 3 months, first 8 stocks):")
    print((momentum.iloc[-3:, :8] * 100).round(1).to_string())
